# Search Resilience to Foreign Embedding Vectors — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `search_journal` never crash on (and never silently use) an embedding vector produced by a different model, and harden the startup migration so it heals reliably.

**Architecture:** One shared compatibility predicate (`version` + `model`) gates scoring; a dimension floor is a corruption backstop; the startup scan gets an observable (throwing) regen path, a truthful success count, per-file isolation, a generous init timeout, and deletes the orphan `.embedding` of an empty source. Search stays read-only; healing is the migration's job.

**Tech Stack:** TypeScript (CommonJS via ts-jest), Node `fs/promises`, `@xenova/transformers` (mocked in tests via `tests/setup.ts`), Jest.

**Design spec:** `docs/superpowers/specs/2026-07-13-dimension-mismatch-resilience-design.md` (read it first).

## Global Constraints

- **Never crash `search()` on a bad vector.** A mismatched/corrupt/foreign vector is skipped, never thrown past.
- **The `throw` in `EmbeddingService.cosineSimilarity` (`src/embeddings.ts:110`) stays** — guard at the call site, do not weaken the primitive.
- **`writeThoughts`' embedding failure must stay non-fatal** (`src/journal.ts:92` path): journaling must not break when embedding fails.
- **One shared predicate.** The scan's staleness check and the search guard both go through `EmbeddingService.isCompatible`. Do not inline two copies.
- **Test output stays pristine.** Every expected `console.error` is asserted via the global spy in `tests/setup.ts` (`jest.mocked(console.error)`); no leaked log blocks (convention from commit `0ee7bbe`).
- **TDD throughout** — write the failing test, watch it fail, implement minimally, watch it pass, commit. `npx jest` runs the suite; `npx tsc --noEmit` typechecks.
- **The plan file lives under `docs/`, which is git-ignored globally — stage it and every spec with `git add -f`.** Source and test files under `src/`/`tests/` add normally.
- **Branch & PR (decided 2026-07-13):** all work happens on a NEW branch cut from `feat/multilingual-per-section-embeddings` (e.g. `fix/dimension-mismatch-resilience`). When done, open a PR whose **base is `feat/multilingual-per-section-embeddings`** — not `main`. Do not commit directly to the PR #1 branch.

---

## File Structure

- `src/embeddings.ts` — add `EmbeddingService.isCompatible(entry)`; the shared predicate. (Task 1)
- `src/search.ts` — dimension floor in `scoreEntry` (Task 2); model-identity partition + disk-path skip log in `search()`, plus `diskPath` on loaded rows (Task 3).
- `src/journal.ts` — extract a throwing regen path; rework `generateMissingEmbeddings` (Task 4).
- `tests/embeddings.test.ts` — `isCompatible` unit test (Task 1); migration hardening tests (Task 4).
- `tests/search.test.ts` — floor test (Task 2); ceiling/skip-log/`listRecent` tests + fix the two existing tests the ceiling changes (Task 3).

---

## Task 1: Shared compatibility predicate `EmbeddingService.isCompatible`

**Files:**
- Modify: `src/embeddings.ts` (add a method to the `EmbeddingService` class, near `getModelName` at `:50`)
- Test: `tests/embeddings.test.ts`

**Interfaces:**
- Produces: `isCompatible(entry: { version?: number; model?: string }): boolean` on `EmbeddingService`. Returns `true` iff `entry.version === EMBEDDING_SCHEMA_VERSION && entry.model === this.modelName`. v1 rows (no `version`/`model`) and foreign-model rows return `false`.

- [ ] **Step 1: Write the failing test**

Add to `tests/embeddings.test.ts` (top import already has `EmbeddingService`; add `EMBEDDING_SCHEMA_VERSION`):

```typescript
// at top with the other imports:
import { EmbeddingService, EMBEDDING_SCHEMA_VERSION } from '../src/embeddings';

// as a new top-level describe (outside the existing one is fine):
describe('EmbeddingService.isCompatible', () => {
  test('true only for current-model current-version entries', () => {
    const svc = EmbeddingService.getInstance();
    const model = svc.getModelName();

    expect(svc.isCompatible({ version: EMBEDDING_SCHEMA_VERSION, model })).toBe(true);
    expect(svc.isCompatible({ version: EMBEDDING_SCHEMA_VERSION, model: 'other-model' })).toBe(false);
    expect(svc.isCompatible({})).toBe(false);                              // v1: no version/model
    expect(svc.isCompatible({ version: 1, model })).toBe(false);           // stale version
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest tests/embeddings.test.ts -t "isCompatible"`
Expected: FAIL — `svc.isCompatible is not a function`.

- [ ] **Step 3: Implement the method**

In `src/embeddings.ts`, inside `class EmbeddingService`, right after `getModelName()` (`:50-52`):

```typescript
  isCompatible(entry: { version?: number; model?: string }): boolean {
    return entry.version === EMBEDDING_SCHEMA_VERSION && entry.model === this.modelName;
  }
```

(`EMBEDDING_SCHEMA_VERSION` is already declared and exported at the top of the file; `this.modelName` is the private field at `:29-30`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest tests/embeddings.test.ts -t "isCompatible"`
Expected: PASS.

- [ ] **Step 5: Typecheck + commit**

Run: `npx tsc --noEmit` → no errors.

```bash
git add src/embeddings.ts tests/embeddings.test.ts
git commit -m "feat(embeddings): add isCompatible predicate (version + model)"
```

---

## Task 2: Dimension floor in `scoreEntry` (never crash)

**Files:**
- Modify: `src/search.ts` — `scoreEntry` (`:121-136`)
- Test: `tests/search.test.ts`

**Interfaces:**
- Consumes: nothing new.
- Produces: `scoreEntry` unchanged signature (`{ score: number; matchedSection?: string }`) but returns `score: -Infinity` when no vector is dimension-compatible with the query, and never calls `cosineSimilarity` on a mismatched/non-array vector.

This task fixes the literal reported crash without the model ceiling yet. It only touches current-model corruption, so it does not disturb the existing `model: 'test'` / v1 tests (they still score under the old behavior). The ceiling arrives in Task 3.

- [ ] **Step 1: Write the failing test**

Add to `tests/search.test.ts`. Extend the existing embeddings import at the top of the file (line 8) so it also brings in the service — `EmbeddingService` lives in `src/embeddings` and is NOT re-exported by `src/search`:

```typescript
// line 8 becomes:
import { EMBEDDING_SCHEMA_VERSION, EmbeddingService } from '../src/embeddings';
```

New describe:

```typescript
describe('dimension floor (crash safety)', () => {
  let dir: string;
  const model = EmbeddingService.getInstance().getModelName();

  const writeEntry = async (name: string, sectionEmbeddings: any[], embedding: any = [0.1, 0.2, 0.3, 0.4, 0.5]) => {
    const day = path.join(dir, '2026-07-08');
    await fs.mkdir(day, { recursive: true });
    await fs.writeFile(path.join(day, `${name}.md`), '## X\n\nbody', 'utf8');
    await fs.writeFile(path.join(day, `${name}.embedding`), JSON.stringify({
      version: 2, model, embedding, sectionEmbeddings,
      text: 'body', sections: sectionEmbeddings.map((s: any) => s.section),
      timestamp: Date.now(), path: path.join(day, `${name}.md`),
    }), 'utf8');
  };

  beforeEach(async () => { dir = await fs.mkdtemp(path.join(os.tmpdir(), 'floor-test-')); });
  afterEach(async () => { await fs.rm(dir, { recursive: true, force: true }); });

  test('a wrong-dimension section is skipped, good entries still returned, no throw', async () => {
    // query mock is 5-dim ([0.1..0.5], tests/setup.ts)
    await writeEntry('good', [{ section: 'A', text: 'a', embedding: [0.1, 0.2, 0.3, 0.4, 0.5] }]);
    await writeEntry('bad', [{ section: 'B', text: 'b', embedding: [1, 2, 3] }]); // 3-dim → floor skips

    const svc = new SearchService(dir, path.join(dir, 'no-user'));
    const results = await svc.search('anything', { type: 'project', minScore: -1 });

    expect(results.map(r => r.path.includes('good.md'))).toContain(true);
    expect(results.some(r => r.path.includes('bad.md'))).toBe(false); // excluded, not crashed
  });

  test('a null/missing vector does not throw inside the guard', async () => {
    await writeEntry('nullvec', [{ section: 'C', text: 'c', embedding: null }]);
    const svc = new SearchService(dir, path.join(dir, 'no-user'));
    await expect(svc.search('anything', { type: 'project', minScore: -1 })).resolves.toEqual([]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx jest tests/search.test.ts -t "dimension floor"`
Expected: FAIL — the `bad`/`nullvec` cases throw `Vectors must have same length` / `Cannot read properties of null` and reject the whole `search()`.

- [ ] **Step 3: Implement the floor**

Replace `scoreEntry` in `src/search.ts` (`:121-136`) with:

```typescript
  private scoreEntry(
    queryEmbedding: number[],
    entry: EmbeddingData
  ): { score: number; matchedSection?: string } {
    const dimOk = (v: unknown): v is number[] =>
      Array.isArray(v) && v.length === queryEmbedding.length;

    if (entry.sectionEmbeddings && entry.sectionEmbeddings.length > 0) {
      let best = -Infinity;
      let matchedSection: string | undefined;
      for (const se of entry.sectionEmbeddings) {
        if (!dimOk(se.embedding)) continue; // corruption backstop: skip, never throw
        const s = this.embeddingService.cosineSimilarity(queryEmbedding, se.embedding);
        if (s > best) { best = s; matchedSection = se.section; }
      }
      return { score: best, matchedSection };
    }
    // Legacy fallback: whole-entry vector
    if (!dimOk(entry.embedding)) return { score: -Infinity };
    return { score: this.embeddingService.cosineSimilarity(queryEmbedding, entry.embedding) };
  }
```

Entries that score `-Infinity` are dropped by the existing `.filter(result => result.score >= minScore)` in `search()` (`:114`) — even at `minScore: -1`.

- [ ] **Step 4: Run test to verify it passes**

Run: `npx jest tests/search.test.ts -t "dimension floor"`
Expected: PASS.

- [ ] **Step 5: Full suite + typecheck + commit**

Run: `npx jest` → all green (existing tests unaffected). `npx tsc --noEmit` → clean.

```bash
git add src/search.ts tests/search.test.ts
git commit -m "fix(search): dimension floor in scoreEntry — skip incompatible vectors instead of crashing"
```

---

## Task 3: Model-identity ceiling + disk-path skip log (never use foreign)

**Files:**
- Modify: `src/search.ts` — `loadEmbeddingsFromPath` (`:264-304`), `search()` (`:50-119`), and the row type used by both
- Test: `tests/search.test.ts` — new tests, plus fix the two existing tests the ceiling changes

**Interfaces:**
- Consumes: `EmbeddingService.isCompatible` (Task 1); `scoreEntry` (Task 2).
- Produces: loaded rows now carry `diskPath: string` (the actual on-disk `.embedding` path). `search()` excludes non-`isCompatible` rows from ranking, counts them, and logs one line naming up to 3 `diskPath`s when the count is > 0.

- [ ] **Step 1: Fix the two existing tests the ceiling changes, and add the new ceiling tests (write them red)**

In `tests/search.test.ts`:

(a) The shared `writeEmbedding` helper (`:13-23`) hardcodes `model: 'test'`, which becomes foreign under the ceiling. Change that one line to the current model. At the top of that `describe` add `const model = EmbeddingService.getInstance().getModelName();` (import already added in Task 2), and in the helper's JSON change `model: 'test',` to `model,`.

(b) Replace the existing v1 test (`:47-65`, "legacy v1 entry … scored via whole-entry vector", added by `aa585a2`) with its inverse — a v1 file is now skipped by provenance:

```typescript
  test('legacy v1 entry (no version/model) is skipped from ranking', async () => {
    // Rationale: decision (ii) in the design spec. v1 is foreign by provenance;
    // the startup scan converts v1 -> v2. Replaces the aa585a2 fallback assertion.
    const day = path.join(projectDir, '2026-07-08');
    await fs.mkdir(day, { recursive: true });
    await fs.writeFile(path.join(day, 'legacy.md'), '## X\n\nbody', 'utf8');
    await fs.writeFile(path.join(day, 'legacy.embedding'), JSON.stringify({
      embedding: [0.1, 0.2, 0.3, 0.4, 0.5], text: 'body', sections: ['X'],
      timestamp: Date.now(), path: path.join(day, 'legacy.md'),
    }), 'utf8');

    const svc = new SearchService(projectDir, path.join(projectDir, 'no-user'));
    const results = await svc.search('anything', { type: 'project', minScore: -1 });

    expect(results).toHaveLength(0);
    expect(jest.mocked(console.error)).toHaveBeenCalledWith(
      expect.stringContaining('skipped 1'),
    );
  });
```

(c) Add ceiling + skip-log tests, plus the `listRecent` provenance-blindness pin required by the spec, as a new describe:

```typescript
describe('model-identity ceiling', () => {
  let dir: string;
  const model = EmbeddingService.getInstance().getModelName();

  const writeEntry = async (name: string, entryModel: string) => {
    const day = path.join(dir, '2026-07-08');
    await fs.mkdir(day, { recursive: true });
    await fs.writeFile(path.join(day, `${name}.md`), '## X\n\nbody', 'utf8');
    await fs.writeFile(path.join(day, `${name}.embedding`), JSON.stringify({
      version: 2, model: entryModel,
      embedding: [0.1, 0.2, 0.3, 0.4, 0.5],
      sectionEmbeddings: [{ section: 'A', text: 'a', embedding: [0.1, 0.2, 0.3, 0.4, 0.5] }],
      text: 'body', sections: ['A'], timestamp: Date.now(), path: path.join(day, `${name}.md`),
    }), 'utf8');
  };

  beforeEach(async () => { dir = await fs.mkdtemp(path.join(os.tmpdir(), 'ceiling-test-')); });
  afterEach(async () => { await fs.rm(dir, { recursive: true, force: true }); });

  test('same-dimension foreign-model entry is excluded from ranking and logged with its on-disk path', async () => {
    await writeEntry('native', model);
    await writeEntry('foreign', 'some-other-model'); // same 5-dim, different model

    const svc = new SearchService(dir, path.join(dir, 'no-user'));
    const results = await svc.search('anything', { type: 'project', minScore: -1 });

    expect(results).toHaveLength(1);
    expect(results[0].path).toContain('native.md');
    expect(jest.mocked(console.error)).toHaveBeenCalledWith(
      expect.stringContaining('foreign.embedding'), // on-disk path, not the stored path
    );
  });

  test('no skip log when everything is current-model', async () => {
    await writeEntry('native', model);
    const svc = new SearchService(dir, path.join(dir, 'no-user'));
    await svc.search('anything', { type: 'project', minScore: -1 });
    expect(jest.mocked(console.error)).not.toHaveBeenCalledWith(
      expect.stringContaining('skipped'),
    );
  });

  test('listRecent still lists foreign/v1 entries (browse is provenance-blind)', async () => {
    // Spec: the ceiling filters RANKING only; chronological browsing must
    // keep showing entries whose vectors are foreign.
    await writeEntry('native', model);
    await writeEntry('foreign', 'some-other-model');

    const svc = new SearchService(dir, path.join(dir, 'no-user'));
    const recent = await svc.listRecent({ type: 'project' });

    expect(recent).toHaveLength(2);
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest tests/search.test.ts`
Expected: FAIL — foreign entries still score (no ceiling), no skip log emitted, `foreign.embedding` never logged. Exception: the `listRecent` pin passes immediately — it guards deliberately-unchanged behavior (the ceiling must not leak into chronological browsing), so it is green before and after.

- [ ] **Step 3: Attach `diskPath` at load**

In `src/search.ts`, add a row type alias just below the imports (after `import ... paths.js;`, ~`:8`):

```typescript
type LoadedEmbedding = EmbeddingData & { type: 'project' | 'user'; diskPath: string };
```

Change `loadEmbeddingsFromPath` (`:264-304`): update its return type to `Promise<LoadedEmbedding[]>`, the internal `const embeddings` to `LoadedEmbedding[]`, and the push (`:289`) to include the on-disk path:

```typescript
            embeddings.push({ ...embeddingData, type, diskPath: embeddingPath });
```

(`embeddingPath` is already computed at `:286`.)

- [ ] **Step 4: Partition + skip log in `search()`**

In `search()`, change the `allEmbeddings` declaration (`:63`) to `const allEmbeddings: LoadedEmbedding[] = [];`. Then replace the scoring block (`:97-118`, from `// Calculate similarities and sort` through `return results;`) with:

```typescript
    // Only score entries we can attest came from the current model. Foreign /
    // v1 rows are skipped (read-only: healing is the migration's job) and
    // surfaced once, naming their on-disk paths.
    const scorable = filtered.filter(e => this.embeddingService.isCompatible(e));
    const foreign = filtered.filter(e => !this.embeddingService.isCompatible(e));

    if (foreign.length > 0) {
      const sample = foreign.slice(0, 3).map(e => e.diskPath).join(', ');
      console.error(
        `search_journal: skipped ${foreign.length} ` +
        `entr${foreign.length === 1 ? 'y' : 'ies'} (incompatible embedding model); ` +
        `run a re-index. Examples: ${sample}`
      );
    }

    const results: SearchResult[] = scorable
      .map(embedding => {
        const { score, matchedSection } = this.scoreEntry(queryEmbedding, embedding);
        const excerpt = this.generateExcerpt(embedding.text, query);
        return {
          path: embedding.path,
          score,
          text: embedding.text,
          sections: embedding.sections,
          timestamp: embedding.timestamp,
          excerpt,
          type: embedding.type,
          matchedSection,
        };
      })
      .filter(result => result.score >= minScore)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);

    return results;
```

(`listRecent`'s `allEmbeddings` at `:145` may stay `Array<EmbeddingData & { type: 'project' | 'user' }>` — a `LoadedEmbedding` is assignable to it; it does not read `diskPath`.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx jest tests/search.test.ts`
Expected: PASS (new ceiling tests, replaced v1 test, and the fixed `model:'test'`→`model` "best section" test).

- [ ] **Step 6: Full suite + typecheck + commit**

Run: `npx jest` → all green. `npx tsc --noEmit` → clean.

```bash
git add src/search.ts tests/search.test.ts
git commit -m "fix(search): model-identity ceiling + disk-path skip log; skip foreign/v1 vectors from ranking"
```

---

## Task 4: Harden the startup migration scan

**Files:**
- Modify: `src/journal.ts` — extract a throwing regen path from `generateEmbeddingForEntry` (`:151-187`); rework `generateMissingEmbeddings` (`:189-241`)
- Test: `tests/embeddings.test.ts`

**Interfaces:**
- Consumes: `EmbeddingService.isCompatible` (Task 1); `EmbeddingService.initialize()` (`:54`); `initTimeoutMs` (`:35`).
- Produces:
  - `private async regenerateEmbedding(filePath, content, timestamp): Promise<boolean>` — throws on real failure, returns `false` when nothing was written (empty text — in that case any stale `.embedding` orphan is deleted: unrecoverable state, decided 2026-07-13), `true` when an `.embedding` was saved.
  - `generateEmbeddingForEntry` keeps its `Promise<void>` swallowing contract (delegates to `regenerateEmbedding`).
  - `generateMissingEmbeddings(): Promise<number>` — count reflects only real writes; per-file failures are isolated; a failed model init aborts by returning the partial count (no throw); a structurally-corrupt vector (whole-entry **or** per-section `null`/non-array) counts as stale and triggers regen.

- [ ] **Step 1: Write the failing tests**

Add to `tests/embeddings.test.ts` (imports `fs`, `path`, `os`, `pipeline`, `EmbeddingService`, `JournalManager` already present). These live inside the existing top-level `describe` so they get its `journalManager` (`new JournalManager(projectTempDir)`) and temp dirs.

```typescript
  describe('generateMissingEmbeddings hardening', () => {
    const staleEmbedding = (mdPath: string) => JSON.stringify({
      version: 1, model: 'old-model', embedding: [0, 0, 0],
      text: 'x', sections: ['X'], timestamp: Date.now(), path: mdPath,
    });

    const writeStale = async (name: string, body: string) => {
      const day = path.join(projectTempDir, '2026-07-08');
      await fs.mkdir(day, { recursive: true });
      const mdPath = path.join(day, `${name}.md`);
      await fs.writeFile(mdPath, `## X\n\n${body}`, 'utf8');
      await fs.writeFile(path.join(day, `${name}.embedding`), staleEmbedding(mdPath), 'utf8');
      return mdPath;
    };

    afterEach(() => { EmbeddingService.resetInstance(); });

    test('one entry failing to regen does not stop the others; count is only successes', async () => {
      await writeStale('good', 'good body');
      await writeStale('boom', 'boom body');

      const svc = EmbeddingService.getInstance();
      jest.spyOn(svc, 'generateEmbedding').mockImplementation(async (text: string) => {
        if (text.includes('boom')) throw new Error('inference failed');
        return [0.1, 0.2, 0.3, 0.4, 0.5];
      });

      const count = await journalManager.generateMissingEmbeddings();

      expect(count).toBe(1); // only 'good' migrated
      const goodEmb = JSON.parse(await fs.readFile(
        path.join(projectTempDir, '2026-07-08', 'good.embedding'), 'utf8'));
      expect(goodEmb.version).toBe(2);
      expect(goodEmb.model).toBe(svc.getModelName());
    });

    test('an unreadable .md isolates to that file; the rest still migrate (Goal 4)', async () => {
      const boomMd = await writeStale('boom', 'boom body');
      await writeStale('good', 'good body');

      const realReadFile = jest.requireActual('fs/promises').readFile;
      const readSpy = jest.spyOn(fs, 'readFile').mockImplementation(((p: any, ...a: any[]) =>
        String(p) === boomMd
          ? Promise.reject(Object.assign(new Error('EACCES'), { code: 'EACCES' }))
          : realReadFile(p, ...a)) as any);

      try {
        const count = await journalManager.generateMissingEmbeddings();
        expect(count).toBe(1); // 'good' migrated despite 'boom.md' being unreadable
      } finally {
        // jest.config.cjs sets no restoreMocks — restore by hand or the spy
        // leaks into every later test in this file.
        readSpy.mockRestore();
      }
    });

    test('empty-text entry is not counted, writes no .embedding, and its stale orphan is deleted', async () => {
      const day = path.join(projectTempDir, '2026-07-08');
      await fs.mkdir(day, { recursive: true });
      await fs.writeFile(path.join(day, 'empty.md'), '   \n', 'utf8'); // no sections, blank body
      await fs.writeFile(path.join(day, 'empty.embedding'),
        staleEmbedding(path.join(day, 'empty.md')), 'utf8');

      const count = await journalManager.generateMissingEmbeddings();

      expect(count).toBe(0);
      // Regen can never rewrite an .embedding for an empty source, so a stale
      // one would be re-flagged and search-skip-logged forever. The scan
      // deletes the orphan (unrecoverable state — decided 2026-07-13).
      await expect(fs.access(path.join(day, 'empty.embedding')))
        .rejects.toMatchObject({ code: 'ENOENT' });
    });

    test('current-model file with a corrupt vector (whole-entry or section) is regenerated', async () => {
      const writeCorrupt = async (name: string, vectors: { embedding: unknown; sectionEmbeddings: unknown }) => {
        const day = path.join(projectTempDir, '2026-07-08');
        await fs.mkdir(day, { recursive: true });
        const mdPath = path.join(day, `${name}.md`);
        await fs.writeFile(mdPath, `## X\n\n${name} body`, 'utf8');
        await fs.writeFile(path.join(day, `${name}.embedding`), JSON.stringify({
          version: 2, model: EmbeddingService.getInstance().getModelName(),
          text: 'x', sections: ['X'], timestamp: Date.now(), path: mdPath,
          ...vectors,
        }), 'utf8');
      };
      await writeCorrupt('corrupt-whole', {
        embedding: null, // structurally corrupt whole-entry vector
        sectionEmbeddings: [{ section: 'X', text: 'x', embedding: [0.1, 0.2, 0.3, 0.4, 0.5] }],
      });
      await writeCorrupt('corrupt-section', {
        embedding: [0.1, 0.2, 0.3, 0.4, 0.5],
        sectionEmbeddings: [{ section: 'X', text: 'x', embedding: null }], // corrupt section vector
      });

      const count = await journalManager.generateMissingEmbeddings();

      expect(count).toBe(2); // both healed, honestly counted
      for (const name of ['corrupt-whole', 'corrupt-section']) {
        const healed = JSON.parse(await fs.readFile(
          path.join(projectTempDir, '2026-07-08', `${name}.embedding`), 'utf8'));
        expect(Array.isArray(healed.embedding)).toBe(true);
        expect(healed.sectionEmbeddings.every((se: any) => Array.isArray(se.embedding))).toBe(true);
      }
    });

    test('model init failure aborts with one scan-level log, no per-file spam, partial count', async () => {
      // Relies on the OUTER beforeEach recreating journalManager per test:
      // after the previous test's resetInstance(), it captures a fresh,
      // UNinitialized singleton. Converting that beforeEach to beforeAll
      // breaks this test (the captured instance would already be initialized).
      await writeStale('a', 'a body');
      await writeStale('b', 'b body');

      EmbeddingService.resetInstance();
      const transformers = require('@xenova/transformers');
      const savedPipeline = transformers.pipeline;
      transformers.pipeline = jest.fn().mockRejectedValue(new Error('offline'));

      try {
        const count = await journalManager.generateMissingEmbeddings();
        expect(count).toBe(0); // aborted before any write

        const errorCalls = jest.mocked(console.error).mock.calls.map(c => String(c[0]));
        expect(errorCalls.filter(m => m.includes('aborting re-index'))).toHaveLength(1);
        expect(errorCalls.some(m => m.startsWith('Failed to migrate'))).toBe(false);
      } finally {
        transformers.pipeline = savedPipeline;
      }
    });
  });
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npx jest tests/embeddings.test.ts -t "generateMissingEmbeddings hardening"`
Expected: FAIL. Under current code: the "one entry failing" test gets `count` = 2 (the swallow lets `count++` fire for `boom`); the unreadable-`.md` test aborts the rest of that journal's scan at the per-`basePath` catch, so `count` = 0; the empty-text test gets `count` = 1 and the stale `.embedding` survives; the corrupt-vector test gets `count` = 0 (version+model match, so the current scan never re-inspects the vectors); the init-failure test emits no `aborting re-index` line.

- [ ] **Step 3: Extract the throwing regen path**

In `src/journal.ts`, replace `generateEmbeddingForEntry` (`:151-187`) with a throwing core plus the swallowing wrapper:

```typescript
  private async regenerateEmbedding(
    filePath: string,
    content: string,
    timestamp: Date
  ): Promise<boolean> {
    const { text, sections, sectionChunks } = this.embeddingService.extractSearchableText(content);

    if (text.trim().length === 0) {
      // Regen can never rewrite an .embedding for an empty source, so a stale
      // one would stay foreign forever (re-flagged by every scan, named by the
      // search skip log). Unrecoverable: delete the orphan (no-op when absent).
      await fs.rm(filePath.replace(/\.md$/, '.embedding'), { force: true });
      return false; // nothing to embed; not a success, not an error
    }

    const embedding = await this.embeddingService.generateEmbedding(text, 'document');

    const sectionEmbeddings: SectionEmbedding[] = [];
    for (const chunk of sectionChunks) {
      const emb = await this.embeddingService.generateEmbedding(chunk.body, 'document');
      sectionEmbeddings.push({ section: chunk.section, text: chunk.body, embedding: emb });
    }

    const embeddingData: EmbeddingData = {
      version: EMBEDDING_SCHEMA_VERSION,
      model: this.embeddingService.getModelName(),
      embedding,
      sectionEmbeddings,
      text,
      sections,
      timestamp: timestamp.getTime(),
      path: filePath,
    };

    await this.embeddingService.saveEmbedding(filePath, embeddingData);
    return true;
  }

  private async generateEmbeddingForEntry(
    filePath: string,
    content: string,
    timestamp: Date
  ): Promise<void> {
    try {
      await this.regenerateEmbedding(filePath, content, timestamp);
    } catch (error) {
      // Don't throw - embedding failure shouldn't prevent journal writing
      console.error(`Failed to generate embedding for ${filePath}:`, error);
    }
  }
```

(`writeThoughtsToLocation` at `:92` keeps calling `generateEmbeddingForEntry`, so the write path is unchanged.)

- [ ] **Step 4: Rework the scan**

Replace the **entire body** of `generateMissingEmbeddings` — everything between the method signature (`:189`) and its closing brace (`:241`), i.e. from `let count = 0;` (`:190`) through the existing `return count;` (`:240`) **inclusive** — with the block below. The block ends with its own `return count;`; leaving the old one behind would create unreachable code that neither `tsc` nor this repo's ESLint config flags.

```typescript
    let count = 0;
    let modelReady = false;
    const paths = [this.projectJournalPath, this.userJournalPath];

    for (const basePath of paths) {
      try {
        const dayDirs = await fs.readdir(basePath);

        for (const dayDir of dayDirs) {
          const dayPath = path.join(basePath, dayDir);
          const stat = await fs.stat(dayPath);

          if (!stat.isDirectory() || !dayDir.match(/^\d{4}-\d{2}-\d{2}$/)) {
            continue;
          }

          const files = await fs.readdir(dayPath);
          const mdFiles = files.filter(file => file.endsWith('.md'));

          for (const mdFile of mdFiles) {
            try {
              const mdPath = path.join(dayPath, mdFile);
              const embeddingPath = mdPath.replace(/\.md$/, '.embedding');

              let needsRegen = false;
              try {
                const raw = await fs.readFile(embeddingPath, 'utf8');
                const existing = JSON.parse(raw);
                // Structural check covers the whole-entry vector AND every
                // section vector; a same-model wrong-length vector is not
                // detectable here (needs the model's dimension) — accepted.
                const vectorsOk = Array.isArray(existing.embedding) &&
                  Array.isArray(existing.sectionEmbeddings) &&
                  existing.sectionEmbeddings.every((se) => Array.isArray(se?.embedding));
                if (!this.embeddingService.isCompatible(existing) || !vectorsOk) {
                  needsRegen = true;
                }
              } catch {
                needsRegen = true; // missing or unreadable
              }

              if (!needsRegen) {
                continue;
              }

              if (!modelReady) {
                // Generous timeout: the query default (30s) is too short to
                // download a cold ~465MB model, and transformers.js neither
                // resumes nor dedupes, so a short retry would only start a
                // second concurrent download. One long attempt or bust.
                this.embeddingService.initTimeoutMs = 120_000;
                try {
                  await this.embeddingService.initialize();
                  modelReady = true;
                } catch (error) {
                  console.error('embedding model unavailable — aborting re-index:', error);
                  return count; // abort the whole scan; report real successes so far
                }
              }

              console.error(`Generating/refreshing embedding for ${mdPath}`);
              const content = await fs.readFile(mdPath, 'utf8');
              const timestamp = this.extractTimestampFromPath(mdPath) || new Date();
              const wrote = await this.regenerateEmbedding(mdPath, content, timestamp);
              if (wrote) {
                count++;
              }
            } catch (error) {
              // Per-file isolation: one bad file never aborts the directory.
              console.error(`Failed to migrate ${mdFile}:`, error);
            }
          }
        }
      } catch (error) {
        if ((error as NodeJS.ErrnoException)?.code !== 'ENOENT') {
          console.error(`Failed to scan ${basePath} for missing embeddings:`, error);
        }
      }
    }

    return count;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `npx jest tests/embeddings.test.ts -t "generateMissingEmbeddings hardening"`
Expected: PASS (all five).

- [ ] **Step 6: Guard the existing migration test still asserts its log**

The prior console-hygiene test asserts the `Generating/refreshing embedding for` line (from commit `0ee7bbe`); it is preserved above. Run the whole embeddings suite:

Run: `npx jest tests/embeddings.test.ts`
Expected: PASS, output pristine (no leaked `console.error` blocks — the `aborting re-index`, `Failed to migrate`, and `Generating/refreshing` lines are asserted or suppressed by the global spy).

- [ ] **Step 7: Full suite + typecheck + commit**

Run: `npx jest` → all green. `npx tsc --noEmit` → clean. `npm run lint` → clean (`src/` only).

```bash
git add src/journal.ts tests/embeddings.test.ts
git commit -m "fix(journal): observable throwing regen, truthful count, per-file isolation, generous init timeout in migration scan"
```

---

## Final verification (after all tasks)

- [ ] `npx jest` — full suite green; count matches baseline + new tests; no leaked `console.error` blocks.
- [ ] `npx tsc --noEmit` — clean.
- [ ] `npm run lint` — clean.
- [ ] Manual sanity per the `verify` skill: point a dev MCP at a journal copy, set `PRIVATE_JOURNAL_EMBED_MODEL` to a different-dimension model (e.g. an e5-base), do a partial/interrupted start, then run a search — confirm it returns the still-current entries with a single skip log naming a real path, instead of crashing.
- [ ] Open the PR: head = this work branch, base = `feat/multilingual-per-section-embeddings` (decided 2026-07-13; follow superpowers:finishing-a-development-branch).
