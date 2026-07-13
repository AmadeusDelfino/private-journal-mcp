# Dream: daily consolidation of recurring journal themes

## Background

The journal accumulates entries continuously (`process_thoughts` →
`src/journal.ts`), and `search_journal` can retrieve them semantically, but
nothing ever *looks back over the corpus as a whole* to notice what keeps
recurring. A technical detail written five times across three weeks stays five
disconnected entries; it never gets promoted into something future sessions
benefit from without a human going and searching for it.

"Dream" is a reflection ritual, inspired by agent memory-consolidation loops:
periodically review what has been journaled, detect themes that recur, and
consolidate the strong ones into durable memory. The practical goals, in the
project owner's words:

- **(A) Feed future sessions.** The dream writes a synthesis entry back into the
  journal, indexed like any other entry, so it enriches later `search_journal`
  and context. Nobody has to read it actively.
- **(C) Consolidate memory.** A recurring detail (e.g. registered ~5× over
  weeks) gets promoted into a durable memory of the kind Claude already keeps
  (file-memory / a proposed `CLAUDE.md` fact).
- **(D) The mechanism is the point.** This is a playground fork, not a
  production feature. Correctness and cleanliness still matter; scope stays
  minimal.

### Two facts about the current server that shape the whole design

1. **The server has no daemon.** It is a pure MCP stdio server
   (`src/index.ts`, `PrivateJournalServer.run` in `src/server.ts`): it is
   spawned by an MCP client and dies when the client disconnects. There is no
   loop, cron, or long-lived process. So a "once a day" trigger cannot originate
   *inside* the server — it must come from outside.

2. **The server has no LLM.** Its only runtime dependencies are the MCP SDK and
   `@xenova/transformers`, and transformers is used here **only for embeddings**
   (feature extraction), never text generation. The server can *measure
   similarity* between thoughts but cannot *write* a natural-language synthesis.
   The party that "thinks" is either an LLM (Claude, in a session) or vector
   math (clustering, counting, novelty).

These two facts force the split-brain architecture below and rule out an
in-server generative dream (which would also require sending journal contents to
an external API — a direct violation of the project's "nothing leaves your
machine" promise — or bundling a heavy, low-quality local generation model).

## Recurrence model

The dream is *triggered* on a cadence (daily, conceptually) but *measures*
recurrence over a rolling window of the corpus, not over a single day's entries.
A daily-only scope would never see "5× over three weeks," which is exactly the
pattern worth consolidating. So: **today's batch is the trigger to re-evaluate;
the window (last N days, or all-time) is the material.**

No dedicated dream-state ledger is needed. Idempotency across repeated runs
comes from two places already in the system:

- **De-dup against existing memory.** Before promoting, the ritual checks
  whether a memory already covers the theme (Claude can read the memory
  directory and `CLAUDE.md`). Already-covered → skip.
- **Dream entries are marked and excluded** from the recurrence corpus (see the
  feedback-loop note below), so re-running never re-counts prior dreams.

## Architecture: two halves

```
[you / later: cron via `claude -p "dream"`]
     │  invokes the ritual
     ▼
[dream ritual  ──  a skill, runs inside a Claude session]
     │ 1. calls find_recurring_themes  (read-only)
     ▼
[MCP server: recurrence engine]  ← reads the existing .embedding corpus
     │ returns recurring themes + evidence
     │          (distinct-entry count, date span, excerpts, source paths)
     ▼
[ritual decides, per strong theme:]
     ├─ (A) writes a "dream entry" back to the journal (record_dream) → marked, so it never feeds the loop
     └─ (C) drafts a memory candidate; promotion is propose-only (see Safety model)
```

**Ownership.** The repository owns two new MCP tools —
`find_recurring_themes` (read) and `record_dream` (write). The ritual itself is
a Claude skill (like the owner's existing `/route-memory`) that orchestrates the
two; it lives in the owner's Claude config, not in this repo. The repo exposes
primitives; the "dream" narrative is Claude-side.

**Containment.** `find_recurring_themes` is strictly read-only and offline — a
pure function of the `.embedding` files on disk. Every write (journal entry and
memory) is performed by the Claude ritual, never by the server.

## Component 1 — the recurrence engine

A new module (e.g. `src/recurrence.ts`) that operates on **already-computed
embeddings**, not on text. This is the key testability win: it takes vectors +
metadata as input, so it needs no model and no transformers mock.

**Input.** Every section vector (`sectionEmbeddings` in `EmbeddingData`,
`src/embeddings.ts`) from entries in the window, loaded from the same
`.embedding` corpus that `search_journal` reads, minus:

- entries marked as dream entries (feedback-loop guard, below);
- vectors that fail the existing `EmbeddingService.isCompatible` check
  (model + `EMBEDDING_SCHEMA_VERSION`) — mixing dimensions or models would
  produce garbage similarities, the same failure class the dimension-resilience
  work addressed.

Each vector keeps its metadata: source entry path, date, section type, and the
chunk text.

**Clustering.** Threshold-based "leader/medoid" clustering (no pre-chosen `k`),
iterating in **fixed chronological order** so the result is deterministic and
testable:

1. take an unclustered chunk;
2. gather all chunks within cosine ≥ τ of it;
3. elect the **medoid** (the most central member) and re-gather everything
   within τ of the medoid;
4. that group is one theme; mark its members used; repeat.

Medoid rather than single-linkage on purpose: single-linkage *chains*
(A~B, B~C, but A and C unrelated) and merges distinct themes; medoid keeps
clusters tight and yields the representative excerpt for free.

**Counting the "5×".** A theme's strength is the number of **distinct source
entries** it draws from, not the number of chunks. Two chunks from the same
entry count once. Promotion threshold: `minEntries` (default 5). Secondary
guard: `minDays` (default 2) — a theme must appear on ≥2 distinct days, so a
single-day repetition does not masquerade as recurrence-over-time.

*Example.* 7 chunks land in one cluster, from 5 files, across 3 days →
`distinctEntries: 5`, `occurrences: 7`, `distinctDays: 3` → passes (≥5 entries,
≥2 days) → becomes a candidate.

**Output per theme**, ranked by strength (distinct entries; tie-broken by
cohesion):

| field | purpose |
|---|---|
| `representativeExcerpt` | medoid chunk text — what the ritual uses to write the memory |
| `supportingExcerpts` | 2–3 nearest neighbours (extra evidence) |
| `distinctEntries`, `occurrences`, `distinctDays`, `dateSpan` | strength and window of the pattern |
| `sectionDistribution` | e.g. `{technical_insights: 4, project_notes: 3}` — helps the ritual route (technical → file-memory/`CLAUDE.md`; user_context → a fact about the human) |
| `cohesion` | mean cosine to the medoid (how tight) |
| `sourcePaths`, `type` | traceability + project/user origin |

**Multilingual, for free.** Because the current default model
(`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim) produces multilingual
per-section embeddings, a theme expressed in Portuguese one day and English
another clusters together. The dream sees patterns *across* language.

**Scale.** Clustering is O(n²) in the number of chunks in the window. For a
personal journal (hundreds to low thousands of chunks) this is instant, and the
window bounds it. Blocking/ANN is deferred (YAGNI).

## Component 2 — the `find_recurring_themes` tool

A neutral primitive: the server *finds patterns*; it does not "dream." The name,
narrative, and cadence live in the ritual. This also makes the tool reusable
outside the dream ("what have I been chewing on this week?").

Read-only, offline. Parameters (all defaulted; none required):

| param | default | meaning |
|---|---|---|
| `days` | 30 | look-back window; `0` = all-time |
| `minEntries` | 5 | distinct entries for a theme to count |
| `minDays` | 2 | minimum distinct days |
| `threshold` | *to calibrate* | cosine cutoff τ (see below) |
| `sections` | all | restrict to given section types |
| `type` | both | project / user / both |
| `limit` | 20 | max themes returned (ranked by strength) |
| `preview` | false | calibration mode (see below) |

**On `threshold` (τ).** This is the most sensitive knob and its correct value
depends on the specific embedding model; it is **not** guessed here. Two honest
provisions:

1. calibrate empirically during implementation — run against the real journal
   and observe where good clusters form;
2. **`preview: true`** returns statistics only — "at τ=0.72, 14 clusters formed,
   largest with 9 entries" — without excerpts, so τ can be swept quickly and a
   sensible default fixed.

**Return format.** Text in the style of `search_journal`'s handler output: a
header (window, echoed params, entries scanned) followed by numbered themes with
the fields above. Claude reads and acts on it; a human can read it directly too.

## Component 3 — the `record_dream` tool

The write side. Minimal tool that writes the dream entry into the journal with
`dream: true` in the front-matter and embeds it normally. It has two properties:

- **searchable** — goal (A): it feeds future sessions via `search_journal`;
- **excluded from recurrence** — the engine skips entries whose embedding is
  marked `dream`, killing the feedback loop where a dream entry would become raw
  material for the next dream and inflate/duplicate clusters.

To let the engine filter purely from `.embedding` files (it reads vectors, not
`.md`), `record_dream` sets an optional `dream: true` field on the written
`EmbeddingData` in addition to the `.md` front-matter. Only the recurrence
engine reads this field; `search_journal` treats dream entries as ordinary
searchable entries.

A dedicated tool rather than extending `process_thoughts`: it keeps the existing
tool clean and makes the marker an explicit responsibility of the dreamer rather
than a bolted-on parameter. Dream entries default to the user journal (they are
meta-reflection about the person).

## Component 4 — the ritual

A Claude skill (invocable as e.g. `/dream`, or `claude -p "dream"` later). Steps:

1. call `find_recurring_themes` with the chosen window and thresholds;
2. write the dream entry via `record_dream` — Claude's narrative over what
   recurred, loose threads, what stood out (this is goal A, and where the prose
   synthesis the server can't do gets written);
3. for each strong theme, run what is effectively an **automated
   `/route-memory`**: de-dup against existing memories / `CLAUDE.md` (Claude
   sees both), then route by the theme's nature — durable rule/preference →
   `CLAUDE.md`; stable project fact → file-memory; ephemeral → left in the
   journal only.

### Safety model — promotion is propose-only (model i)

The owner's memory rules permit auto-writing file-memory but require proposing
`CLAUDE.md` edits. Taken literally, the dream *could* auto-write file-memory. It
will not, in the MVP. Because the dream is autonomous and new, and memory
auto-loads into every future session and shapes behaviour, silent pollution is a
real risk. The MVP therefore **proposes** every promotion:

- The dream detects and drafts; the dream entry lists each candidate (name,
  description, drafted body, target store, link to the evidence entries).
- The human approves the ones worth keeping. The dream **never rewrites
  long-term memory unattended.** This matches the spirit of `/route-memory`
  ("I propose, you decide").

This is not a weaker cron story: a scheduled run simply prepares the tray;
approval is a few seconds the next morning.

**Evolution path** (explicitly out of scope now): model (iii) — auto-write only
very-high-confidence themes (many distinct entries + high cohesion + clearly not
already covered) to file-memory, everything else stays a candidate, `CLAUDE.md`
always proposed.

## Testing

- **Recurrence engine — pure function over pre-computed vectors.** Unit tests
  inject synthetic vectors directly; no model, no transformers mock. Cases:
  5 near-identical vectors from 5 files across 3 days → one theme,
  `distinctEntries=5`; threshold boundaries (4 entries → nothing; 5 → a theme;
  all same day → blocked by `minDays`); a `dream: true` entry does not count; an
  incompatible vector (`isCompatible=false`) is skipped; medoid selection and
  cohesion computation. Deterministic (fixed chronological iteration).
- **`find_recurring_themes` handler** — param parsing/defaults, read-only
  behaviour, output formatting, in the style of the existing `search_journal`
  tests.
- **`record_dream`** — writes a `dream: true` entry, embeds it, and the engine
  excludes it on a subsequent scan (embeddings mocked as the current tests do).
- **The ritual (skill)** is a procedure, not repo code; the suite does not cover
  it. It is validated by running it (dry-run). TDD-as-code does not apply to
  this piece — it lives outside the repo.

TDD applies to all three code components: tests first, starting with the
engine's clustering contract.

## Out of scope (YAGNI), with the evolution trail

- **cron/daemon** — MVP is a requested action (manual invocation) only; later,
  `claude -p` driven by cron.
- **LLM in the server** — never (privacy).
- **auto-writing memory** — no (safety model i); later → model (iii), tiered by
  confidence.
- **dream-state/ledger file** — unnecessary (de-dup against existing memory +
  the dream marker).
- **ANN/blocking clustering** — O(n²) is fine at personal scale.
- **τ calibration** — an implementation step (run `preview` against the real
  journal), not a code feature.

## Open items

- Fix a sensible default for `threshold` (τ) after empirical calibration against
  `paraphrase-multilingual-MiniLM-L12-v2` on the real journal.
