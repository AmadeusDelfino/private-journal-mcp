# Search resilience to incompatible / foreign embedding vectors

## Background

`search_journal` loads every `.embedding` file from the project and user
journals, then scores each against the query with cosine similarity
(`src/search.ts` `scoreEntry` → `EmbeddingService.cosineSimilarity`).
Cosine similarity requires both vectors to have the same length and
throws `Vectors must have same length` otherwise (`src/embeddings.ts:110`).

Review finding #1 on PR #1 reproduced the consequence with the real
`dist`: a single `.embedding` whose vector dimension differs from the
query makes the entire `search()` reject. The good entries disappear
too, and the error names no culprit file.

Grounding the finding in the code changed the picture in three ways:

1. **The crash is the narrower of two failure modes.** Vector dimension
   is a function of the model architecture. The old default
   (`all-MiniLM-L6-v2`) and the new default
   (`paraphrase-multilingual-MiniLM-L12-v2`) are **both 384-dim**, so the
   default upgrade never triggers the crash. The dimension crash only
   appears when the configured model changes to a different-dimension
   model (e5/bge = 768, e5-large = 1024) and re-index is incomplete.
   With same-dimension models, vectors from different models are compared
   silently, producing garbage similarities with no error at all. This
   **silent** mode is worse than the noisy crash and the finding does not
   cover it.

2. **The two modes share one root cause:** vectors produced by different
   models coexist on disk. The startup re-index is supposed to heal this
   but has two defects that leave the journal half-migrated:

   - **It cannot observe regen failures.** `generateEmbeddingForEntry`
     (`src/journal.ts:151-187`) wraps its whole body — init, inference,
     save — in a `try/catch` that logs and returns without throwing
     ("Don't throw - embedding failure shouldn't prevent journal
     writing"). So when regen fails (model down, inference error), the
     scan neither aborts nor notices: `count++` (`journal.ts:229`) runs
     anyway, and `server.run` reports "Generated embeddings for N
     entries" when zero were written. The only thing that *does* abort a
     directory today is an `fs`-level error (reading the `.md`, `stat`,
     `readdir`) hitting the **per-`basePath`** `try/catch`
     (`journal.ts:233`) — not the dominant failure class.
   - **Model-load failure spams per entry.** Because init is retried
     lazily inside every `generateEmbeddingForEntry`, a model that can't
     load logs one "Failed to generate embedding for …" per stale entry
     and heals nothing.

   Concurrent writers (two MCP instances with different model env vars
   over the same journal) and restored backups are other, operational
   sources of foreign vectors.

3. **The enforceable invariant is model identity.** Each v2
   `EmbeddingData` carries `version` and `model`. Filtering by
   `entry.version === EMBEDDING_SCHEMA_VERSION && entry.model ===
   currentModel` prevents the dimension throw (a different dimension
   implies a different model) *and* kills the silent
   **different-model** mode. It does **not** catch same-model config
   divergence — `quantized` and the query/doc prefixes change the output
   yet are **not stored** in `EmbeddingData` (`src/embeddings.ts:15-24`),
   so two vectors can share a `model` string and still be incomparable.
   That hole is deliberately left to batch item #4 (see Out-of-scope);
   the design's job is to make the scan and the search guard share one
   predicate so #4 can extend both at once.

### Where re-index happens (why the chosen behavior is safe)

Embedding (re)generation occurs at exactly three points:

- **Startup scan** (`server.run`, `src/server.ts:322-336`):
  `generateMissingEmbeddings()` is `await`ed **before**
  `server.connect(transport)`. By the time the MCP can receive a search,
  the scan has already run. This is the primary heal point.
- **Write path** (`writeThoughts` → `generateEmbeddingForEntry`,
  `src/journal.ts:92`): every new entry is written with the current
  model. Steady-state writes are always current-model.
- **Search** re-indexes nothing — it only embeds the query.

Because the startup scan blocks the connection, in the common case the
"recall-zero window" right after a model change is closed by the time a
search can arrive. But the model-identity ceiling makes un-migrated
entries **invisible**, not merely lower-ranked — so a path where the scan
does not heal degrades recall (fully, on a total model swap where every
in-scope entry is stale; partially, when only some are) and stays that
way until the next restart, because the scan only reruns on restart and
MCP servers run for days. (If the model is genuinely *down*, that is
louder, not silent: `search()` throws at query-embedding time rather than
returning empty.) The cold-cache model-swap case is the real trigger: the
default query init timeout (30s, `embeddings.ts:35`) is too short to
download the ~465MB model, so the scan's init fails and every on-disk
entry is left foreign. This is why Component 2 gives the scan's init a
**longer timeout** so the download can actually complete in-process (see
below); the skip log's re-index hint is the residual recovery path.

This whole safety story depends on the startup scan blocking connect
(`server.ts:322-336`). If that property is ever removed (an upstream
change in that direction has been reported, unverified here), search-time
skipping becomes the *primary* migration-window behavior and this window
widens — a dependency worth tracking.

## Goals

- `search_journal` never crashes because of an incompatible or
  foreign-model vector. Good entries are always returned.
- A vector participates in ranking only when we can attest (from the
  stored `version` + `model`) it came from the current model. This kills
  the noisy crash and the silent **different-model** mode. Same-model
  config divergence (`quantized`/prefix) is a known residual hole left to
  item #4.
- The scan's staleness check and the search guard use **one shared
  compatibility predicate**, so item #4 extends both in lockstep instead
  of letting them drift.
- The startup re-index does not leave the journal half-migrated: one
  failing entry never aborts the rest of the directory, the scan can
  observe failures, and the reported count reflects real successes.
- When entries are skipped, the log names enough to act (count + sample
  paths + a re-index hint), without spamming.

## Non-goals

- **On-the-fly self-heal during search.** Search stays a read-only
  operation. Healing is the migration's job (decision A). Regenerating
  vectors inside a query would add non-deterministic latency and make
  search mutate disk.
- **Preserving the v1 whole-entry fallback at search time.** v1 files
  (no `model` field) are skipped by provenance (decision ii). The
  migration converts v1 → v2; we do the most we can to migrate, but do
  not shape the architecture around a migration-catastrophe hypothesis.
- **A persisted migration state machine / verification pass.** The scan
  is idempotent and retries on every startup; a resumable state machine
  is YAGNI here.
- **Large-journal startup latency.** A model swap on a big journal makes
  the blocking startup scan long, risking a client connect timeout. This
  is a pre-existing concern; the mitigation is to pay the migration
  outside the client (`node dist/index.js` once with the new env). Not
  addressed here.
- **Concurrent writers.** Two MCP instances with divergent model env vars
  over the same journal is an operational issue the code cannot fully
  prevent. The skip guard degrades gracefully; we do not try to
  coordinate the writers.

## Design

### The shared compatibility predicate

Introduce one predicate, used by both the scan and the search guard:

```
isCompatible(entry) =
  entry.version === EMBEDDING_SCHEMA_VERSION &&
  entry.model === embeddingService.getModelName()
```

`getModelName()` is an **instance** method (`src/embeddings.ts:50`) — call
it on the service, not the class. This is today's staleness key
(`journal.ts:216-217`) lifted into a single named function so the search
guard and the scan cannot drift, and so item #4 (fold `quantized`/prefix
into the key) touches one place. A model referenced by two identifiers
(local path vs hub id) will fail strict string equality and be treated as
foreign — an acceptable over-skip that a re-index resolves.

### The two guards

- **Model-identity ceiling (correctness).** In the `search()` scoring
  path, an entry is *scorable* only if `isCompatible(entry)`. v1 files
  have no `version`/`model`, so they fail and are skipped by provenance;
  foreign v2 entries (divergent `model`) skip the same way. No separate
  "is v2" test is needed.
- **Dimension floor (crash safety, backstop).** Before each
  `cosineSimilarity` call, pre-check
  `Array.isArray(v) && v.length === queryEmbedding.length`; skip the
  offending vector otherwise. The `Array.isArray` half matters: a corrupt
  file with `embedding: null`/missing must not throw
  `Cannot read properties of null` *inside the guard*. With the ceiling
  in place a genuinely different-dimension vector is already foreign and
  skipped upstream — so this floor only fires on same-model corruption.
  It makes the throw unreachable **without** masking the invariant: the
  `throw` in `cosineSimilarity` stays as a correct programming-error
  signal for its callers.

Both guards live in the `search()` scoring path, where the query vector
exists — **not** in the shared `loadEmbeddingsFromPath`. `listRecent`
also loads embeddings but never scores; it is a chronological browse, so
it must keep listing foreign/v1 entries and is left untouched.

### Component 1 — Resilient scoring (`src/search.ts`)

In `search()`, after loading and filtering, partition entries into
scorable (`isCompatible`) vs foreign before the `.map` that scores:

- Foreign (v2 with divergent `model`, or v1) → excluded from results,
  counted, and sampled in the skip log.
- Scorable → scored via `scoreEntry`, with the dimension floor guarding
  each comparison. A vector that fails the floor is excluded from that
  entry's scoring; if it leaves the entry with no scorable vector, the
  entry is excluded from results. Floor-skips are corruption, a distinct
  class the scan heals where detectable (structural damage — Component 2;
  same-model wrong-length vectors are an accepted residual), so they
  are **not** added to the foreign-model skip count/log.

`scoreEntry` keeps its two branches for scorable (current-model) entries:
`sectionEmbeddings` when present, whole-entry `embedding` as the fallback
for a current-model entry that has no section vectors. What changes is
that v1 files never reach it — they are foreign by provenance and skipped
upstream — so the whole-entry branch is no longer a path for
*unknown-model* vectors.

**The skip log must name the real file, not the stored path.**
`loadEmbeddingsFromPath` spreads `{...embeddingData, type}`
(`src/search.ts:286-289`), discarding the actual on-disk `.embedding`
location; `entry.path` is whatever was written at creation time
(`journal.ts:179`). For a restored/moved journal — an explicit motivating
source — that path is stale or, for a file lacking `path`, `undefined`.
So `loadEmbeddingsFromPath` must attach the on-disk path as a distinct
field (harmless to `listRecent`), and the log uses that.

**Observability (decision A):** when `skipped > 0`, emit a single
`console.error` per search call, e.g.:

```
search_journal: skipped 3 entries (incompatible embedding model);
run a re-index. Examples: <disk-path-1>, <disk-path-2>, <disk-path-3>
```

Sample capped at the first ~3 on-disk paths. Cadence: once per search
call; self-clears once the re-index heals the entries. Note the count is
computed **after** the sections/dateRange filter, so it reflects foreign
entries *within this query's scope*, not the whole journal — it may vary
across queries. Expected to never appear in a healthy setup.

### Component 2 — Robust migration (`src/journal.ts`)

The current `generateEmbeddingForEntry` swallows every error and never
throws (`journal.ts:183-186`). That non-throwing guarantee is correct for
the **write path** (`writeThoughts`, `journal.ts:92`): a failed embedding
must not break journaling. But it means the scan can neither isolate a
per-entry failure nor tell "one entry failed" from "model is down," and
`count++` fires on silent failures so the reported count lies. So the
mechanism has to change, not just get a `try/catch` wrapper.

Rework (decision B, corrected):

- **A regen path the scan can observe.** Extract the regen body into a
  variant that **throws** on failure. `writeThoughts` keeps calling the
  swallowing wrapper (its guarantee is unchanged); the scan calls the
  throwing variant so it can catch per-entry.
- **Per-entry `try/catch` around the whole per-file body** — the `.md`
  read (`journal.ts:226`), timestamp extraction, and regen — not just the
  regen call. An unreadable `.md` (EACCES, broken symlink) is exactly the
  `fs`-level failure that today escapes to the per-`basePath` catch
  (`journal.ts:233`) and aborts the rest of the directory; wrapping the
  full body is what actually delivers Goal 4. A failing entry logs once
  and the scan continues. Idempotent — the next startup retries.
- **Truthful count.** Success means an `.embedding` was **actually
  written**. Increment only then, so `server.run`'s "Generated embeddings
  for N entries" is honest. An empty-text entry (`journal.ts:159-161`
  returns without writing) is neither counted nor an error — otherwise it
  inflates the count and re-logs on every boot.
- **Delete the orphan `.embedding` of an empty source** (decided
  2026-07-13: unrecoverable state). Regen can never rewrite an
  `.embedding` whose `.md` has no embeddable text, so a stale one would
  stay foreign forever — re-flagged by every scan and named by the search
  skip log with a "run a re-index" hint that cannot heal it. The regen
  path removes the file instead (`force`: a no-op when absent). The write
  path shares the code, harmlessly: a fresh write has no pre-existing
  `.embedding`, so the delete is a no-op there (an all-empty-string
  `writeThoughts` call *can* produce an empty `.md` — `hasUserContent`
  only checks `!== undefined`). Residual, accepted: the empty `.md` itself is
  still re-flagged on every boot (one "Generating/refreshing" line and a
  model init) because staleness is decided before the text is extracted.
- **Heal structurally-corrupt vectors.** The scan's `needsRegen` becomes
  `!isCompatible(existing) || !vectorsOk(existing)`, where `vectorsOk`
  requires the whole-entry `embedding` to be an array **and**
  `sectionEmbeddings` to be an array whose every entry carries an array
  vector. A file that is version+model-compatible but carries a
  `null`/missing/non-array vector — whole-entry or per-section — is
  regenerated rather than left for the search floor to skip forever.
  (Truncated/invalid JSON already triggers regen via the parse-failure
  path, `journal.ts:220-221`.) Residual, accepted: a same-model
  **wrong-length** vector is not detectable here — the expected dimension
  is a property of the loaded model — so the search floor excludes it but
  the scan cannot heal it; reaching that state requires hand-edited or
  cross-stamped files.
- **Explicit model init with a longer timeout**, lazily on the first
  entry that needs regen (preserves "nothing to migrate → never load the
  465MB model"). The scan uses a generous init timeout (e.g. 120s)
  instead of the 30s query default (`initTimeoutMs`, `embeddings.ts:35`),
  because that is the *only* variant that lets a cold-cache download
  finish in-process — transformers.js (v2.17.2) neither resumes nor
  dedupes downloads (`node_modules/@xenova/transformers/src/utils/hub.js`:
  cache checked once at :422, full fetch + buffer + `cache.put` at
  :505-541), so an immediate short retry would just launch a second
  concurrent 465MB download and fail again, only later. No retry. If the
  single generous attempt fails, emit **one** scan-level `console.error`
  ("embedding model unavailable — aborting re-index") and abort by
  returning the partial success count (do **not** throw). This
  distinguishes "model is down" from "this entry failed" and keeps
  `server.run`'s count honest. Implementation note: the timeout is set on
  the shared `EmbeddingService` singleton; leaving it raised is benign —
  if the scan aborted, a later query's init then also gets the longer
  window to land the download and self-heal.

**Costs and consequences to accept and document:**

- The generous timeout means the scan (which already blocks `connect`,
  `server.ts:322-336`) can block up to ~120s on a cold model swap. That
  is the large-journal startup-latency non-goal; the documented mitigation
  is to pay the migration out-of-band (`node dist/index.js` once with the
  new env). **cwd caveat:** the **project** journal resolves from the
  current working directory (`src/paths.ts`), so running the out-of-band
  migration from the wrong cwd migrates a different project journal — the
  skip log would then persist for project entries while the user journal
  heals.
- If the init still fails (offline, cache-dir unwritable), the scan aborts
  and un-migrated entries stay invisible until a restart heals them; the
  skip log's re-index hint is the recovery path.

`EmbeddingService` already exposes the needed surface: `getModelName()`
(`src/embeddings.ts:50`) and `initialize()` (`:54`).

## Error handling summary

| Situation | Behavior |
|---|---|
| Foreign-model v2 vector at search | Skipped from ranking; counted; on-disk path sampled in one log line |
| v1 vector at search | Skipped from ranking (fails `isCompatible` — no version/model) |
| Same-model `null`/missing/non-array vector (whole-entry or section) | Floor (`Array.isArray` + length) excludes it from results (not in foreign count); scan regenerates the file → self-heals |
| Same-model wrong-length vector | Floor excludes it at search; scan cannot detect it (needs the model's dimension) — accepted residual |
| Stale `.embedding` whose `.md` has no embeddable text | Scan deletes the orphan (unrecoverable state); not counted, not an error |
| All in-scope entries foreign | Empty results + the skip log; recall degraded until re-index (partial if only some are stale) |
| One entry fails to regen in scan | Logged once; scan continues; count not incremented |
| Model init fails during scan (single generous-timeout attempt) | One scan-level log; scan aborts by returning the partial count (no throw); un-migrated entries invisible until restart |
| `listRecent` with foreign/v1 vectors | Still listed (no scoring, no filter) |

## Testing

### Existing tests this design changes (must be handled, not silently broken)

The model-identity ceiling changes pinned behavior. Both are addressed
head-on, with the rationale recorded — not deleted to make the suite pass:

- **`tests/search.test.ts:47-65`** ("legacy v1 entry … scored via
  whole-entry vector", added on this branch by `aa585a2`) asserts a v1
  file scores and returns. Decision (ii) inverts it: the test is
  **replaced** by one asserting a v1 file is skipped from ranking. The
  commit message cites `aa585a2` and the reason (v1 is foreign by
  provenance; the startup scan converts v1→v2; cross-model v1 scores were
  already silent garbage after the default-model swap).
- **`tests/search.test.ts` `writeEmbedding` fixture (line 18)** writes
  `model: 'test'`, which is now foreign under the ceiling and would make
  the flagship "best section" test's `results[0]` undefined. The fixture
  is updated to write the current model (`getModelName()`), so it stays a
  test of ranking, not of the guard.

### New tests (TDD — red observed before each)

Resilient scoring (`tests/search.test.ts`):

- Mixed dimensions on disk (one foreign 768-dim among current-model
  entries): `search()` returns the good entries, skips the bad one, does
  **not** throw; assert the skip log names the foreign entry's on-disk
  path.
- Same-dimension, different `model`: the foreign entry is absent from
  ranking (kills the silent different-model mode).
- Corrupt vector (`embedding: null`): skipped, no throw (exercises the
  `Array.isArray` half of the floor).
- v1 entry present: skipped from ranking.
- `listRecent` with a foreign/v1 vector present: still lists it.

Robust migration:

- One entry's regen fails (mock `generateEmbedding` to throw for a single
  entry — now observable via the throwing regen path): the remaining
  entries still migrate, and the returned count counts only the successes.
  The count half is red under current code (the swallow lets `count++`
  fire for the failed entry, `journal.ts:229`); the "remaining still
  migrate" half is only meaningfully testable once the regen path throws.
- An unreadable `.md` in the first day-dir (`fs` failure): the remaining
  day-dirs/entries in that journal still migrate — proves the per-file
  catch wraps the `.md` read, not just the regen (Goal 4).
- Empty-text entry: not counted, not logged as an error, no `.embedding`
  written — and its stale orphan `.embedding` is deleted.
- Current-model file with a structurally-corrupt vector (`null`
  whole-entry; `null` inside a section entry): regenerated and counted.
- Model init fails (single generous-timeout attempt): exactly **one**
  scan-level abort log and **zero** per-entry failure logs (the assertion
  must be scoped this way — each init attempt itself logs
  "Loading…"/"Failed to load…", `embeddings.ts:73,81`); the scan returns
  the partial count rather than throwing.

All new logs asserted via the `console.error` spy so suite output stays
pristine (consistent with the fix shipped in commit `0ee7bbe`).

## Out-of-scope follow-ups (recorded, not built)

- **Item #4 — `quantized`/prefix in the compatibility key.** Same-model
  config divergence stays silent. This design's contribution is the
  single shared `isCompatible` predicate; #4 adds the fields to
  `EmbeddingData` and extends that one predicate, healing scan and search
  together.
- Large-journal blocking-scan client-timeout mitigation (pay migration
  out-of-band; mind the cwd caveat above).
- Concurrent-writer coordination.
- Dependency to watch: the safety story relies on the startup scan
  blocking connect (`server.ts:322-336`). If that is ever removed,
  search-time skipping becomes the primary migration-window behavior and
  the recall-zero considerations above get more severe.
