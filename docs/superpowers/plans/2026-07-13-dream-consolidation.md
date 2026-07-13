# Dream Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the dream-consolidation design (`docs/superpowers/specs/2026-07-13-dream-consolidation-design.md`): a read-only `find_recurring_themes` MCP tool that detects recurring themes by clustering journal section embeddings, a `record_dream` MCP tool that writes marked consolidation entries excluded from future scans, and a `/dream` ritual command that orchestrates them with propose-only memory promotion.

**Architecture:** A new pure-function module `src/recurrence.ts` (medoid clustering over pre-computed section vectors — no model, no I/O) is fed by `SearchService.collectEmbeddings` (made public) and wired into `src/server.ts` as thin MCP handlers. `record_dream` reuses the existing journal write/embed pipeline in `src/journal.ts`, with `dream: true` in the `.md` front-matter as the durable marker, mirrored onto the `.embedding` file on every (re)generation. The ritual is a slash command in the owner's Claude config, outside this repo.

**Tech Stack:** TypeScript (ESM, `strict`), MCP SDK 0.4.0 (stdio only — no InMemoryTransport, so handlers stay thin glue over unit-tested pure functions), Jest + ts-jest with the transformers mock in `tests/setup.ts`.

## Global Constraints

- The server stays LLM-free and fully offline; journal contents never leave the machine (spec, "Two facts").
- No daemon/cron code in the repo; the tools are invoked externally (spec, "Two facts").
- `find_recurring_themes` is strictly read-only — a pure function of the `.embedding` files on disk.
- Memory promotion is propose-only (safety model i); enforced by the ritual text, never by repo code.
- Tool defaults, copied from the spec: `days` 30 (0 = all-time), `minEntries` 5, `minDays` 2, `limit` 20, `type` 'both', `preview` false. `threshold` defaults to `DEFAULT_THRESHOLD` (0.65 provisional; Task 7 calibrates it empirically).
- Clustering iterates in fixed chronological order (deterministic); O(n²) is accepted at personal-journal scale (no ANN/blocking).
- Dream entries are searchable like any entry but never feed the recurrence corpus.
- ESM: imports inside `src/` need the `.js` suffix; tests import without it (ts-jest `moduleNameMapper` strips it).
- `tests/setup.ts` mocks the model: every `generateEmbedding` call returns `[0.1, 0.2, 0.3, 0.4, 0.5]` (5-dim), and `console.error` is spied/silenced.
- Commits use the repo's conventional prefixes (`feat:`, `fix:`, `test:`, `docs:`). NEVER add Co-Authored-By or any AI-credit trailer.
- The user's GLOBAL gitignore ignores `docs/`: any commit touching `docs/**` must stage with `git add -f`.
- TDD: run the failing test and see it fail before implementing, every cycle.

## File Structure

| File | Responsibility |
|---|---|
| `src/recurrence.ts` (create) | Recurrence engine: types, `safeCosine`, `clusterChunks` (leader/medoid), `findRecurringThemes`, `gatherThemeChunks` (corpus → chunks with filters), `parseThemeParams`, `formatThemesOutput`, `DEFAULT_THRESHOLD` |
| `src/embeddings.ts` (modify) | Add optional `dream?: boolean` to `EmbeddingData`; extract `cosineSimilarity` as a standalone exported function (class method delegates) |
| `src/search.ts` (modify) | Export `LoadedEmbedding`; make `collectEmbeddings` public (recurrence must inherit its coincident-roots dedup) |
| `src/journal.ts` (modify) | Extract `formatFrontmatter` helper; add `writeDream`; stamp `dream: true` onto `EmbeddingData` when the `.md` front-matter carries it (survives re-indexing) |
| `src/server.ts` (modify) | Register `find_recurring_themes` and `record_dream` tools + thin handlers |
| `tests/recurrence.test.ts` (create) | Engine, gather, params, formatter, and on-disk end-to-end tests |
| `tests/dream.test.ts` (create) | Dream marker propagation, `writeDream`, searchable-but-excluded integration |
| `jest.config.cjs` (modify) | Add `src/recurrence.ts` to `collectCoverageFrom` |
| `README.md` (modify) | Document the two new tools under "## MCP Tools" |
| `~/.claude/commands/dream.md` (create, OUTSIDE repo) | The ritual (Component 4), matching the house style of `route-memory.md` |

---

### Task 1: Dream marker on embedding data

The `.md` front-matter is the durable dream marker; every embedding (re)generation mirrors it onto the `.embedding` JSON so the recurrence engine can filter without reading `.md` files — and so the flag survives re-indexing after a model switch.

**Files:**
- Modify: `src/embeddings.ts` (interface `EmbeddingData`)
- Modify: `src/journal.ts` (`regenerateEmbedding`)
- Test: `tests/dream.test.ts` (create)

**Interfaces:**
- Consumes: existing `JournalManager.generateMissingEmbeddings(): Promise<number>`, `EmbeddingService.extractSearchableText`.
- Produces: `EmbeddingData.dream?: boolean` — present and `true` only for dream entries; key absent otherwise. Later tasks (gather in Task 3, `writeDream` in Task 6) rely on exactly this.

- [ ] **Step 1: Write the failing tests**

Create `tests/dream.test.ts`:

```typescript
// ABOUTME: Tests for dream entries - the dream marker on embeddings and the writeDream pipeline
// ABOUTME: Dream entries are searchable like any entry but never feed the recurrence corpus

import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import { JournalManager } from '../src/journal';

describe('dream marker on embeddings', () => {
  let projectDir: string;
  let userDir: string;
  let originalHome: string | undefined;

  beforeEach(async () => {
    projectDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dream-project-'));
    userDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dream-user-'));
    originalHome = process.env.HOME;
    process.env.HOME = userDir;
  });

  afterEach(async () => {
    if (originalHome !== undefined) {
      process.env.HOME = originalHome;
    } else {
      delete process.env.HOME;
    }
    await fs.rm(projectDir, { recursive: true, force: true });
    await fs.rm(userDir, { recursive: true, force: true });
  });

  test('generateMissingEmbeddings stamps dream: true from the front-matter', async () => {
    // A dream .md whose .embedding is missing (e.g. after a model switch wiped
    // it): the regenerated .embedding must re-derive the flag from the .md.
    const day = path.join(projectDir, '2026-07-10');
    await fs.mkdir(day, { recursive: true });
    await fs.writeFile(
      path.join(day, '10-00-00-000001.md'),
      '---\ntitle: "10:00:00 AM - July 10, 2026"\ndate: 2026-07-10T10:00:00.000Z\ntimestamp: 1783692000000\ndream: true\n---\n\n## Dream\n\nrecurring themes consolidated\n',
      'utf8'
    );

    const manager = new JournalManager(projectDir);
    const count = await manager.generateMissingEmbeddings();

    expect(count).toBe(1);
    const data = JSON.parse(
      await fs.readFile(path.join(day, '10-00-00-000001.embedding'), 'utf8')
    );
    expect(data.dream).toBe(true);
  });

  test('non-dream entries carry no dream key', async () => {
    const day = path.join(projectDir, '2026-07-10');
    await fs.mkdir(day, { recursive: true });
    await fs.writeFile(
      path.join(day, '11-00-00-000001.md'),
      '---\ntitle: "11:00:00 AM - July 10, 2026"\ndate: 2026-07-10T11:00:00.000Z\ntimestamp: 1783695600000\n---\n\n## Project Notes\n\nplain note\n',
      'utf8'
    );

    const manager = new JournalManager(projectDir);
    await manager.generateMissingEmbeddings();

    const data = JSON.parse(
      await fs.readFile(path.join(day, '11-00-00-000001.embedding'), 'utf8')
    );
    expect('dream' in data).toBe(false);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx jest tests/dream.test.ts -v`
Expected: FAIL — `expect(data.dream).toBe(true)` receives `undefined` (the flag is never written). The second test may already pass (key absent today); that is fine — it pins the contract.

- [ ] **Step 3: Implement**

In `src/embeddings.ts`, extend the interface (add one line to `EmbeddingData`):

```typescript
export interface EmbeddingData {
  version: number;                       // EMBEDDING_SCHEMA_VERSION
  model: string;                         // model that produced these vectors
  embedding: number[];                   // whole-entry vector (legacy/fallback)
  sectionEmbeddings: SectionEmbedding[];
  text: string;
  sections: string[];
  timestamp: number;
  path: string;
  dream?: boolean;                       // dream entries: searchable, but never feed recurrence
}
```

In `src/journal.ts`, inside `regenerateEmbedding`, after the `extractSearchableText` call and the existing empty-text early return, derive the flag and spread it into the `EmbeddingData` literal:

```typescript
    // The .md front-matter is the durable dream marker; mirror it onto the
    // .embedding so the recurrence engine can filter without reading .md
    // files. Re-derived on every regen so the flag survives re-indexing.
    const frontmatter = content.match(/^---\n([\s\S]*?)\n---\n/);
    const isDream = frontmatter !== null && /^dream: true$/m.test(frontmatter[1]);
```

and change the `embeddingData` literal's last lines to:

```typescript
    const embeddingData: EmbeddingData = {
      version: EMBEDDING_SCHEMA_VERSION,
      model: this.embeddingService.getModelName(),
      embedding,
      sectionEmbeddings,
      text,
      sections,
      timestamp: timestamp.getTime(),
      path: filePath,
      ...(isDream ? { dream: true } : {}),
    };
```

- [ ] **Step 4: Run the tests and the full suite**

Run: `npx jest tests/dream.test.ts -v` → PASS.
Run: `npm test` → all green (no existing behavior changed).

- [ ] **Step 5: Commit**

```bash
git add src/embeddings.ts src/journal.ts tests/dream.test.ts
git commit -m "feat: mark dream entries in embedding data and preserve the flag on regen"
```

---

### Task 2: Recurrence engine — medoid clustering over section vectors

Pure function over injected vectors: no model, no mock, no I/O. Deterministic given its input order (callers pass chronologically sorted chunks — Task 3 guarantees that).

**Files:**
- Create: `src/recurrence.ts`
- Modify: `src/embeddings.ts` (extract standalone `cosineSimilarity`)
- Modify: `jest.config.cjs` (coverage)
- Test: `tests/recurrence.test.ts` (create)

**Interfaces:**
- Consumes: `cosineSimilarity(a: number[], b: number[]): number` — new standalone export from `src/embeddings.ts` (throws on length mismatch, returns 0 on zero norm; the class method delegates to it).
- Produces (all exported from `src/recurrence.ts`; Tasks 3–6 use these exact names):
  - `interface ThemeChunk { vector: number[]; entryPath: string; date: string; timestamp: number; section: string; text: string; type: 'project' | 'user' }`
  - `interface RecurringTheme { representativeExcerpt: string; supportingExcerpts: string[]; distinctEntries: number; occurrences: number; distinctDays: number; dateSpan: { start: string; end: string }; sectionDistribution: Record<string, number>; cohesion: number; sourcePaths: string[]; type: 'project' | 'user' | 'both' }`
  - `interface RecurrenceStats { chunksScanned: number; entriesScanned: number; clustersFormed: number; largestClusterEntries: number; themesQualifying: number }`
  - `interface EngineOptions { threshold: number; minEntries: number; minDays: number; limit: number }`
  - `interface RecurrenceResult { themes: RecurringTheme[]; stats: RecurrenceStats }`
  - `findRecurringThemes(chunks: ThemeChunk[], options: EngineOptions): RecurrenceResult`

- [ ] **Step 1: Extract standalone cosineSimilarity (refactor, no behavior change)**

In `src/embeddings.ts`, add a module-level function above the class (exact body moved from the method):

```typescript
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new Error('Vectors must have same length');
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) {
    return 0;
  }

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}
```

and replace the class method body with delegation:

```typescript
  cosineSimilarity(a: number[], b: number[]): number {
    return cosineSimilarity(a, b);
  }
```

Run: `npm test` → all green (pure extraction; `tests/embeddings.test.ts` exercises the method).

- [ ] **Step 2: Write the failing engine tests**

Create `tests/recurrence.test.ts`. Vector helpers: `v(θ°) = [cos θ, sin θ, 0]`, so `cos(v(a), v(b)) = cos(a − b)`. At τ = 0.9 the join radius is ≈ 25.84°; at τ = 0.89 it is ≈ 27.13°.

```typescript
// ABOUTME: Unit tests for the recurrence engine and its supporting pure functions
// ABOUTME: Injects synthetic vectors directly - no model, no transformers mock needed

import {
  findRecurringThemes,
  ThemeChunk,
  EngineOptions,
} from '../src/recurrence';

const chunk = (over: Partial<ThemeChunk> = {}): ThemeChunk => ({
  vector: [1, 0, 0],
  entryPath: '/j/2026-07-01/a.md',
  date: '2026-07-01',
  timestamp: 1,
  section: 'Technical Insights',
  text: 'the same recurring insight',
  type: 'user',
  ...over,
});

const opts = (over: Partial<EngineOptions> = {}): EngineOptions => ({
  threshold: 0.9,
  minEntries: 5,
  minDays: 2,
  limit: 20,
  ...over,
});

// Angle-parameterized unit vectors: cos(v(a), v(b)) === cos(a - b).
const NEAR = [0.99, 0.14, 0]; // vs [1,0,0]: cos ≈ 0.990

describe('findRecurringThemes - clustering and counting', () => {
  // Spec example: 7 chunks from 5 files across 3 days -> one theme,
  // distinctEntries 5, occurrences 7, distinctDays 3.
  const specExample = (): ThemeChunk[] => [
    chunk({ entryPath: '/j/2026-07-01/e1.md', date: '2026-07-01', timestamp: 1, section: 'Technical Insights', text: 't1' }),
    chunk({ entryPath: '/j/2026-07-01/e1.md', date: '2026-07-01', timestamp: 1, section: 'Project Notes', text: 't2', vector: NEAR }),
    chunk({ entryPath: '/j/2026-07-01/e2.md', date: '2026-07-01', timestamp: 2, section: 'Technical Insights', text: 't3' }),
    chunk({ entryPath: '/j/2026-07-02/e3.md', date: '2026-07-02', timestamp: 3, section: 'Project Notes', text: 't4', vector: NEAR }),
    chunk({ entryPath: '/j/2026-07-02/e3.md', date: '2026-07-02', timestamp: 3, section: 'Technical Insights', text: 't5' }),
    chunk({ entryPath: '/j/2026-07-03/e4.md', date: '2026-07-03', timestamp: 4, section: 'Technical Insights', text: 't6' }),
    chunk({ entryPath: '/j/2026-07-03/e5.md', date: '2026-07-03', timestamp: 5, section: 'Project Notes', text: 't7', vector: NEAR }),
    chunk({ entryPath: '/j/2026-07-02/e6.md', date: '2026-07-02', timestamp: 6, section: 'Reflections', text: 'noise', vector: [0, 1, 0] }),
  ];

  test('5 entries across 3 days form one theme with full evidence', () => {
    const { themes, stats } = findRecurringThemes(specExample(), opts());

    expect(themes).toHaveLength(1);
    const t = themes[0];
    expect(t.distinctEntries).toBe(5);
    expect(t.occurrences).toBe(7);
    expect(t.distinctDays).toBe(3);
    expect(t.dateSpan).toEqual({ start: '2026-07-01', end: '2026-07-03' });
    expect(t.sectionDistribution).toEqual({ 'Technical Insights': 4, 'Project Notes': 3 });
    expect(t.sourcePaths).toEqual([
      '/j/2026-07-01/e1.md',
      '/j/2026-07-01/e2.md',
      '/j/2026-07-02/e3.md',
      '/j/2026-07-03/e4.md',
      '/j/2026-07-03/e5.md',
    ]);
    expect(t.type).toBe('user');
    // Medoid: the [1,0,0] group is denser (4 vs 3), earliest [1,0,0] chunk wins.
    expect(t.representativeExcerpt).toBe('t1');
    // Nearest neighbours of the medoid: the other [1,0,0] chunks, chronological.
    expect(t.supportingExcerpts).toEqual(['t3', 't5', 't6']);
    expect(t.cohesion).toBeCloseTo(0.995, 2);

    expect(stats).toEqual({
      chunksScanned: 8,
      entriesScanned: 6,
      clustersFormed: 1, // singletons don't count
      largestClusterEntries: 5,
      themesQualifying: 1,
    });
  });

  test('4 distinct entries do not qualify (minEntries)', () => {
    const chunks = [1, 2, 3, 4].map(i =>
      chunk({ entryPath: `/j/2026-07-0${i}/e${i}.md`, date: `2026-07-0${i}`, timestamp: i })
    );
    expect(findRecurringThemes(chunks, opts()).themes).toHaveLength(0);
  });

  test('5 entries on a single day are blocked by minDays', () => {
    const chunks = [1, 2, 3, 4, 5].map(i =>
      chunk({ entryPath: `/j/2026-07-01/e${i}.md`, timestamp: i })
    );
    expect(findRecurringThemes(chunks, opts()).themes).toHaveLength(0);
    expect(findRecurringThemes(chunks, opts({ minDays: 1 })).themes).toHaveLength(1);
  });

  test('chunks from the same entry count once toward distinctEntries', () => {
    const chunks = [1, 2, 3].map(i =>
      chunk({ entryPath: '/j/2026-07-01/e1.md', timestamp: 1, section: `S${i}`, text: `c${i}` })
    ).concat([
      chunk({ entryPath: '/j/2026-07-02/e2.md', date: '2026-07-02', timestamp: 2, text: 'c4' }),
      chunk({ entryPath: '/j/2026-07-02/e2.md', date: '2026-07-02', timestamp: 2, section: 'Other', text: 'c5' }),
    ]);
    const { themes } = findRecurringThemes(chunks, opts({ minEntries: 2 }));
    expect(themes).toHaveLength(1);
    expect(themes[0].distinctEntries).toBe(2);
    expect(themes[0].occurrences).toBe(5);
  });

  test('empty input yields empty result with zeroed stats', () => {
    expect(findRecurringThemes([], opts())).toEqual({
      themes: [],
      stats: { chunksScanned: 0, entriesScanned: 0, clustersFormed: 0, largestClusterEntries: 0, themesQualifying: 0 },
    });
  });
});

describe('findRecurringThemes - medoid behavior', () => {
  const v = (deg: number): number[] => [
    Math.cos((deg * Math.PI) / 180),
    Math.sin((deg * Math.PI) / 180),
    0,
  ];
  const at = (deg: number, i: number, text: string): ThemeChunk =>
    chunk({ vector: v(deg), entryPath: `/j/2026-07-01/e${i}.md`, timestamp: i, text });

  test('chained neighbours do not merge (medoid, not single-linkage)', () => {
    // A~B (20°) and B~C (20°) but A vs C is 40° (cos 0.766 < 0.9): C must not
    // ride the chain into A's cluster.
    const chunks = [at(0, 1, 'A'), at(20, 2, 'B'), at(40, 3, 'C')];
    const { themes, stats } = findRecurringThemes(chunks, opts({ minEntries: 2, minDays: 1 }));

    expect(themes).toHaveLength(1);
    expect(themes[0].occurrences).toBe(2);
    const excerpts = [themes[0].representativeExcerpt, ...themes[0].supportingExcerpts];
    expect(excerpts).not.toContain('C');
    expect(stats.clustersFormed).toBe(1);
  });

  test('the most central member is elected medoid', () => {
    // A=0°, B=20°, C=25°: mean sims A≈0.923, B≈0.968, C≈0.951 -> medoid B.
    const chunks = [at(0, 1, 'A'), at(20, 2, 'B'), at(25, 3, 'C')];
    const { themes } = findRecurringThemes(chunks, opts({ minEntries: 3, minDays: 1 }));

    expect(themes).toHaveLength(1);
    expect(themes[0].representativeExcerpt).toBe('B');
    // Supporting sorted by similarity to the medoid: C (cos 5°) then A (cos 20°).
    expect(themes[0].supportingExcerpts).toEqual(['C', 'A']);
    expect(themes[0].cohesion).toBeCloseTo(0.97, 2);
  });

  test('a member outside the medoid radius is dropped and later forms its own cluster', () => {
    // τ=0.89 (radius ≈27.13°). Leader X=0° gathers Q=-26° and P1..P4=24..27°.
    // The dense P-side pulls the medoid to P1=24°; Q (Δ50° from P1) is dropped
    // on re-gather, remains unclustered, and is processed later as a singleton.
    // No chunk is lost or double-clustered. Q sits mid-array to pin
    // order-independence of the drop.
    const chunks = [
      at(0, 1, 'X'),
      at(-26, 2, 'Q'),
      at(24, 3, 'P1'),
      at(25, 4, 'P2'),
      at(26, 5, 'P3'),
      at(27, 6, 'P4'),
    ];
    const { themes, stats } = findRecurringThemes(
      chunks,
      opts({ threshold: 0.89, minEntries: 2, minDays: 1 })
    );

    expect(themes).toHaveLength(1);
    expect(themes[0].occurrences).toBe(5); // X + P1..P4; Q excluded
    expect(themes[0].representativeExcerpt).toBe('P1');
    expect(themes[0].sourcePaths).not.toContain('/j/2026-07-01/e2.md');
    expect(stats.chunksScanned).toBe(6);
    expect(stats.clustersFormed).toBe(1); // Q's singleton doesn't count
  });

  test('degenerate vectors isolate quietly instead of throwing', () => {
    const chunks = [
      chunk({ entryPath: '/j/2026-07-01/e1.md', timestamp: 1, text: 'good1' }),
      chunk({ entryPath: '/j/2026-07-02/e2.md', date: '2026-07-02', timestamp: 2, text: 'good2' }),
      chunk({ entryPath: '/j/2026-07-02/e3.md', date: '2026-07-02', timestamp: 3, text: 'short', vector: [1, 0] }),
      chunk({ entryPath: '/j/2026-07-03/e4.md', date: '2026-07-03', timestamp: 4, text: 'zero', vector: [0, 0, 0] }),
    ];
    const { themes, stats } = findRecurringThemes(chunks, opts({ minEntries: 2, minDays: 1 }));

    expect(themes).toHaveLength(1);
    expect(themes[0].occurrences).toBe(2);
    expect(stats.chunksScanned).toBe(4);
  });
});

describe('findRecurringThemes - ranking and limit', () => {
  const entryChunks = (n: number, vector: number[], tag: string, startTs: number): ThemeChunk[] =>
    Array.from({ length: n }, (_, i) =>
      chunk({
        vector,
        entryPath: `/j/2026-07-0${(i % 2) + 1}/${tag}${i}.md`,
        date: `2026-07-0${(i % 2) + 1}`,
        timestamp: startTs + i,
        text: `${tag}${i}`,
      })
    );

  test('themes rank by distinct entries, strongest first; limit truncates but themesQualifying does not', () => {
    const big = entryChunks(3, [1, 0, 0], 'big', 1);
    const small = entryChunks(2, [0, 1, 0], 'small', 10);
    const options = opts({ minEntries: 2, minDays: 1, limit: 1 });

    const { themes, stats } = findRecurringThemes([...big, ...small], options);

    expect(themes).toHaveLength(1);
    expect(themes[0].distinctEntries).toBe(3);
    expect(stats.themesQualifying).toBe(2);
  });

  test('equal strength ties break by cohesion', () => {
    const tight = entryChunks(2, [0, 0, 1], 'tight', 1); // identical vectors: cohesion 1
    const loose = [
      chunk({ vector: [1, 0, 0], entryPath: '/j/2026-07-01/l1.md', timestamp: 10, text: 'l1' }),
      chunk({ vector: NEAR, entryPath: '/j/2026-07-02/l2.md', date: '2026-07-02', timestamp: 11, text: 'l2' }),
    ];
    const { themes } = findRecurringThemes([...tight, ...loose], opts({ minEntries: 2, minDays: 1 }));

    expect(themes).toHaveLength(2);
    expect(themes[0].representativeExcerpt).toBe('tight0');
  });

  test('a theme mixing project and user entries reports type both', () => {
    const chunks = [
      chunk({ entryPath: '/j/2026-07-01/u.md', timestamp: 1, type: 'user' }),
      chunk({ entryPath: '/j/2026-07-02/p.md', date: '2026-07-02', timestamp: 2, type: 'project' }),
    ];
    const { themes } = findRecurringThemes(chunks, opts({ minEntries: 2, minDays: 1 }));
    expect(themes[0].type).toBe('both');
  });
});
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `npx jest tests/recurrence.test.ts -v`
Expected: FAIL — `Cannot find module '../src/recurrence'`.

- [ ] **Step 4: Implement the engine**

Create `src/recurrence.ts`:

```typescript
// ABOUTME: Recurrence engine detecting themes that recur across journal entries
// ABOUTME: Pure functions over pre-computed section vectors - no model, no I/O

import { cosineSimilarity } from './embeddings.js';

export const DEFAULT_THRESHOLD = 0.65; // provisional; calibrated against the real corpus (plan Task 7)

export interface ThemeChunk {
  vector: number[];
  entryPath: string; // on-disk .md path
  date: string;      // YYYY-MM-DD (the entry's day directory)
  timestamp: number;
  section: string;
  text: string;
  type: 'project' | 'user';
}

export interface RecurringTheme {
  representativeExcerpt: string;
  supportingExcerpts: string[];
  distinctEntries: number;
  occurrences: number;
  distinctDays: number;
  dateSpan: { start: string; end: string };
  sectionDistribution: Record<string, number>;
  cohesion: number;
  sourcePaths: string[];
  type: 'project' | 'user' | 'both';
}

export interface RecurrenceStats {
  chunksScanned: number;
  entriesScanned: number;
  clustersFormed: number;        // clusters with >= 2 chunks, before qualification
  largestClusterEntries: number; // max distinctEntries over all clusters
  themesQualifying: number;      // themes passing minEntries/minDays, before limit
}

export interface EngineOptions {
  threshold: number;
  minEntries: number;
  minDays: number;
  limit: number;
}

export interface RecurrenceResult {
  themes: RecurringTheme[];
  stats: RecurrenceStats;
}

// Length-mismatched or degenerate vectors must isolate quietly (similarity 0),
// not throw like the service's cosineSimilarity: one corrupt vector may not
// take down a whole scan.
function safeCosine(a: number[], b: number[]): number {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length === 0 || a.length !== b.length) {
    return 0;
  }
  return cosineSimilarity(a, b);
}

interface Cluster {
  medoid: number;    // index into chunks
  members: number[]; // indices, ascending (= chronological when input is sorted)
}

// Leader/medoid clustering in fixed input order (callers pass chronologically
// sorted chunks, so results are deterministic). For each unused chunk: gather
// everything within τ of it, elect the medoid (most central member), then
// re-gather around the medoid. Because cosine is symmetric the leader is
// always within τ of its group's medoid, so the leader is never dropped; a
// dropped non-leader member always has a later index and seeds its own
// cluster when the loop reaches it.
function clusterChunks(chunks: ThemeChunk[], threshold: number): Cluster[] {
  const used = new Set<number>();
  const clusters: Cluster[] = [];

  for (let leader = 0; leader < chunks.length; leader++) {
    if (used.has(leader)) continue;

    const group: number[] = [];
    for (let j = 0; j < chunks.length; j++) {
      if (used.has(j)) continue;
      if (j === leader || safeCosine(chunks[leader].vector, chunks[j].vector) >= threshold) {
        group.push(j);
      }
    }

    let medoid = group[0];
    let bestMean = -Infinity;
    for (const m of group) {
      let sum = 0;
      for (const g of group) {
        if (g !== m) sum += safeCosine(chunks[m].vector, chunks[g].vector);
      }
      const mean = group.length > 1 ? sum / (group.length - 1) : 1;
      if (mean > bestMean) {
        bestMean = mean;
        medoid = m;
      }
    }

    const members: number[] = [];
    for (let j = 0; j < chunks.length; j++) {
      if (used.has(j)) continue;
      if (j === medoid || safeCosine(chunks[medoid].vector, chunks[j].vector) >= threshold) {
        members.push(j);
      }
    }
    for (const j of members) used.add(j);
    clusters.push({ medoid, members });
  }

  return clusters;
}

function buildTheme(chunks: ThemeChunk[], cluster: Cluster): RecurringTheme {
  const members = cluster.members.map(i => chunks[i]);
  const medoidChunk = chunks[cluster.medoid];

  const entryPaths: string[] = [];
  for (const m of members) {
    if (!entryPaths.includes(m.entryPath)) entryPaths.push(m.entryPath);
  }
  const days = new Set(members.map(m => m.date));
  const dates = members.map(m => m.date).sort();

  const sectionDistribution: Record<string, number> = {};
  for (const m of members) {
    sectionDistribution[m.section] = (sectionDistribution[m.section] || 0) + 1;
  }

  const others = cluster.members.filter(i => i !== cluster.medoid);
  const cohesion =
    others.length > 0
      ? others.reduce((sum, i) => sum + safeCosine(medoidChunk.vector, chunks[i].vector), 0) /
        others.length
      : 1;

  const supportingExcerpts = others
    .map(i => ({ i, sim: safeCosine(medoidChunk.vector, chunks[i].vector) }))
    .sort((a, b) => b.sim - a.sim) // stable: ties stay chronological
    .slice(0, 3)
    .map(({ i }) => chunks[i].text);

  const types = new Set(members.map(m => m.type));

  return {
    representativeExcerpt: medoidChunk.text,
    supportingExcerpts,
    distinctEntries: entryPaths.length,
    occurrences: members.length,
    distinctDays: days.size,
    dateSpan: { start: dates[0], end: dates[dates.length - 1] },
    sectionDistribution,
    cohesion,
    sourcePaths: entryPaths,
    type: types.size > 1 ? 'both' : members[0].type,
  };
}

export function findRecurringThemes(
  chunks: ThemeChunk[],
  options: EngineOptions
): RecurrenceResult {
  const { threshold, minEntries, minDays, limit } = options;

  const clusters = clusterChunks(chunks, threshold);
  const themed = clusters.map(cluster => ({ cluster, theme: buildTheme(chunks, cluster) }));

  const qualifying = themed
    .filter(({ theme }) => theme.distinctEntries >= minEntries && theme.distinctDays >= minDays)
    .sort(
      (a, b) =>
        b.theme.distinctEntries - a.theme.distinctEntries ||
        b.theme.cohesion - a.theme.cohesion ||
        a.cluster.medoid - b.cluster.medoid
    );

  const stats: RecurrenceStats = {
    chunksScanned: chunks.length,
    entriesScanned: new Set(chunks.map(c => c.entryPath)).size,
    clustersFormed: themed.filter(({ theme }) => theme.occurrences >= 2).length,
    largestClusterEntries: themed.reduce(
      (max, { theme }) => Math.max(max, theme.distinctEntries),
      0
    ),
    themesQualifying: qualifying.length,
  };

  return { themes: qualifying.slice(0, limit).map(({ theme }) => theme), stats };
}
```

- [ ] **Step 5: Add coverage tracking**

In `jest.config.cjs`, add `'src/recurrence.ts',` to `collectCoverageFrom` (after the `'src/search.ts',` line).

- [ ] **Step 6: Run the tests**

Run: `npx jest tests/recurrence.test.ts -v` → PASS.
Run: `npm test` → all green.

- [ ] **Step 7: Commit**

```bash
git add src/recurrence.ts src/embeddings.ts jest.config.cjs tests/recurrence.test.ts
git commit -m "feat: recurrence engine with medoid clustering over section embeddings"
```

---

### Task 3: gatherThemeChunks — corpus filters and chronological ordering

Turns loaded `.embedding` records into engine input, applying the spec's exclusions: dream entries, incompatible vectors, out-of-window entries, unwanted sections, structural corruption. Sorts chronologically so clustering is deterministic regardless of filesystem enumeration order.

**Files:**
- Modify: `src/search.ts:9` (export the `LoadedEmbedding` type)
- Modify: `src/recurrence.ts` (add `GatherOptions`, `gatherThemeChunks`)
- Test: `tests/recurrence.test.ts`

**Interfaces:**
- Consumes: `LoadedEmbedding = EmbeddingData & { type: 'project' | 'user'; diskPath: string }` (newly exported from `src/search.ts`); `EmbeddingData.dream?: boolean` from Task 1.
- Produces: `interface GatherOptions { now: number; days: number; sections?: string[]; isCompatible: (entry: { version?: number; model?: string }) => boolean }` and `gatherThemeChunks(embeddings: LoadedEmbedding[], options: GatherOptions): ThemeChunk[]`. Task 5's handler passes `(e) => embeddingService.isCompatible(e)` as the predicate.

- [ ] **Step 1: Write the failing tests**

Append to `tests/recurrence.test.ts` (and extend its imports):

```typescript
import { gatherThemeChunks, GatherOptions } from '../src/recurrence';
import { LoadedEmbedding } from '../src/search';
import { EMBEDDING_SCHEMA_VERSION } from '../src/embeddings';
```

```typescript
describe('gatherThemeChunks', () => {
  // Loosely typed on purpose (house style of the existing search tests): the
  // corruption cases deliberately violate the LoadedEmbedding shape.
  const emb = (over: Record<string, any> = {}): LoadedEmbedding =>
    ({
      version: EMBEDDING_SCHEMA_VERSION,
      model: 'test-model',
      embedding: [1, 0, 0],
      sectionEmbeddings: [
        { section: 'Technical Insights', text: 'insight', embedding: [1, 0, 0] },
      ],
      text: 'insight',
      sections: ['Technical Insights'],
      timestamp: Date.parse('2026-07-10T10:00:00Z'),
      path: '/stored/elsewhere.md',
      type: 'user',
      diskPath: '/journal/2026-07-10/a.embedding',
      ...over,
    }) as LoadedEmbedding;

  const gopts = (over: Partial<GatherOptions> = {}): GatherOptions => ({
    now: Date.parse('2026-07-13T12:00:00Z'),
    days: 30,
    isCompatible: e => e.version === EMBEDDING_SCHEMA_VERSION && e.model === 'test-model',
    ...over,
  });

  test('explodes section embeddings into chunks with disk-derived path and date', () => {
    const chunks = gatherThemeChunks(
      [
        emb({
          sectionEmbeddings: [
            { section: 'Technical Insights', text: 'a', embedding: [1, 0, 0] },
            { section: 'Project Notes', text: 'b', embedding: [0, 1, 0] },
          ],
        }),
      ],
      gopts()
    );

    expect(chunks).toHaveLength(2);
    expect(chunks[0]).toEqual({
      vector: [1, 0, 0],
      entryPath: '/journal/2026-07-10/a.md', // from diskPath, not the stored path
      date: '2026-07-10',                    // from the day directory
      timestamp: Date.parse('2026-07-10T10:00:00Z'),
      section: 'Technical Insights',
      text: 'a',
      type: 'user',
    });
    expect(chunks[1].section).toBe('Project Notes');
  });

  test('dream entries never feed the corpus', () => {
    expect(gatherThemeChunks([emb({ dream: true })], gopts())).toHaveLength(0);
  });

  test('incompatible entries are excluded via the predicate', () => {
    const chunks = gatherThemeChunks(
      [emb(), emb({ model: 'other-model', diskPath: '/journal/2026-07-10/b.embedding' })],
      gopts()
    );
    expect(chunks).toHaveLength(1);
  });

  test('the window filters by timestamp; days 0 means all-time', () => {
    const old = emb({
      timestamp: Date.parse('2026-05-01T10:00:00Z'),
      diskPath: '/journal/2026-05-01/old.embedding',
    });
    expect(gatherThemeChunks([emb(), old], gopts({ days: 30 }))).toHaveLength(1);
    expect(gatherThemeChunks([emb(), old], gopts({ days: 0 }))).toHaveLength(2);
  });

  test('sections filter matches case-insensitively by substring, per chunk', () => {
    const e = emb({
      sectionEmbeddings: [
        { section: 'Technical Insights', text: 'a', embedding: [1, 0, 0] },
        { section: 'Project Notes', text: 'b', embedding: [0, 1, 0] },
      ],
    });
    const chunks = gatherThemeChunks([e], gopts({ sections: ['technical'] }));
    expect(chunks).toHaveLength(1);
    expect(chunks[0].section).toBe('Technical Insights');
  });

  test('structural corruption is skipped, never thrown', () => {
    const broken = [
      emb({ sectionEmbeddings: { length: 1 }, diskPath: '/journal/2026-07-10/b.embedding' }),
      emb({ sectionEmbeddings: [null], diskPath: '/journal/2026-07-10/c.embedding' }),
      emb({
        sectionEmbeddings: [{ section: 'X', text: 'x', embedding: [] }],
        diskPath: '/journal/2026-07-10/d.embedding',
      }),
      emb({ timestamp: undefined, diskPath: '/journal/2026-07-10/e.embedding' }),
    ];
    expect(gatherThemeChunks([emb(), ...broken], gopts())).toHaveLength(1);
  });

  test('chunks come out chronologically sorted regardless of input order', () => {
    const mk = (iso: string, name: string) =>
      emb({ timestamp: Date.parse(iso), diskPath: `/journal/${iso.slice(0, 10)}/${name}.embedding` });
    const shuffled = [
      mk('2026-07-12T10:00:00Z', 'c'),
      mk('2026-07-10T10:00:00Z', 'a'),
      mk('2026-07-11T10:00:00Z', 'b'),
    ];
    const chunks = gatherThemeChunks(shuffled, gopts());
    expect(chunks.map(c => c.date)).toEqual(['2026-07-10', '2026-07-11', '2026-07-12']);
  });

  test('same-timestamp entries tie-break by path; sections keep their in-entry order', () => {
    const ts = Date.parse('2026-07-10T10:00:00Z');
    const second = emb({
      timestamp: ts,
      diskPath: '/journal/2026-07-10/zz.embedding',
      sectionEmbeddings: [
        { section: 'S1', text: 'z1', embedding: [1, 0, 0] },
        { section: 'S2', text: 'z2', embedding: [1, 0, 0] },
      ],
    });
    const first = emb({ timestamp: ts, diskPath: '/journal/2026-07-10/aa.embedding' });
    const chunks = gatherThemeChunks([second, first], gopts());
    expect(chunks.map(c => c.text)).toEqual(['insight', 'z1', 'z2']);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx jest tests/recurrence.test.ts -v`
Expected: FAIL — `gatherThemeChunks` is not exported (and `LoadedEmbedding` is not exported from `../src/search`).

- [ ] **Step 3: Implement**

In `src/search.ts:9`, export the type:

```typescript
export type LoadedEmbedding = EmbeddingData & { type: 'project' | 'user'; diskPath: string };
```

In `src/recurrence.ts`, add near the top:

```typescript
import * as path from 'path';
import { LoadedEmbedding } from './search.js';
```

and append:

```typescript
export interface GatherOptions {
  now: number;
  days: number; // 0 = all-time
  sections?: string[];
  isCompatible: (entry: { version?: number; model?: string }) => boolean;
}

// Corpus -> engine input. Applies the spec's exclusions (dream entries,
// incompatible vectors, out-of-window, unwanted sections, structural
// corruption) and sorts chronologically so clustering is deterministic
// regardless of filesystem enumeration order.
export function gatherThemeChunks(
  embeddings: LoadedEmbedding[],
  options: GatherOptions
): ThemeChunk[] {
  const { now, days, sections, isCompatible } = options;
  const cutoff = days > 0 ? now - days * 24 * 60 * 60 * 1000 : -Infinity;

  const chunks: ThemeChunk[] = [];
  for (const entry of embeddings) {
    if (entry.dream === true) continue; // dreams never feed the loop
    if (!isCompatible(entry)) continue; // foreign model/schema -> garbage similarities
    if (typeof entry.timestamp !== 'number' || entry.timestamp < cutoff) continue;
    if (!Array.isArray(entry.sectionEmbeddings)) continue;

    const entryPath = entry.diskPath.replace(/\.embedding$/, '.md');
    const date = path.basename(path.dirname(entry.diskPath));

    for (const se of entry.sectionEmbeddings) {
      if (!se || typeof se.section !== 'string' || typeof se.text !== 'string') continue;
      if (!Array.isArray(se.embedding) || se.embedding.length === 0) continue;
      if (sections && sections.length > 0) {
        const wanted = sections.some(s => se.section.toLowerCase().includes(s.toLowerCase()));
        if (!wanted) continue;
      }
      chunks.push({
        vector: se.embedding,
        entryPath,
        date,
        timestamp: entry.timestamp,
        section: se.section,
        text: se.text,
        type: entry.type,
      });
    }
  }

  // Stable sort: within an entry, section order is preserved.
  return chunks.sort(
    (a, b) => a.timestamp - b.timestamp || a.entryPath.localeCompare(b.entryPath)
  );
}
```

- [ ] **Step 4: Run the tests**

Run: `npx jest tests/recurrence.test.ts -v` → PASS. Run `npm test` → all green.

- [ ] **Step 5: Commit**

```bash
git add src/recurrence.ts src/search.ts tests/recurrence.test.ts
git commit -m "feat: gather theme chunks with window, section, dream and compatibility filters"
```

---

### Task 4: Parameter parsing and text output

`parseThemeParams` centralizes the spec's defaults with the same `typeof`-guard style the existing handlers use. `formatThemesOutput` renders the search_journal-style text: header with echoed params and scan counts, then numbered themes — or statistics only in `preview` mode.

**Files:**
- Modify: `src/recurrence.ts`
- Test: `tests/recurrence.test.ts`

**Interfaces:**
- Consumes: `RecurrenceResult`, `RecurrenceStats`, `DEFAULT_THRESHOLD` (Task 2).
- Produces: `interface ThemeParams { days: number; minEntries: number; minDays: number; threshold: number; sections?: string[]; type: 'project' | 'user' | 'both'; limit: number; preview: boolean }`, `parseThemeParams(args: Record<string, unknown>): ThemeParams`, `formatThemesOutput(result: RecurrenceResult, params: ThemeParams): string`. Task 5's handler calls exactly these three names.

- [ ] **Step 1: Write the failing tests**

Append to `tests/recurrence.test.ts` (extend the recurrence import with `parseThemeParams, formatThemesOutput, DEFAULT_THRESHOLD, RecurringTheme, RecurrenceResult`):

```typescript
describe('parseThemeParams', () => {
  test('applies the spec defaults on an empty argument object', () => {
    expect(parseThemeParams({})).toEqual({
      days: 30,
      minEntries: 5,
      minDays: 2,
      threshold: DEFAULT_THRESHOLD,
      sections: undefined,
      type: 'both',
      limit: 20,
      preview: false,
    });
  });

  test('honors explicit values, including days 0 (all-time)', () => {
    const params = parseThemeParams({
      days: 0,
      minEntries: 3,
      minDays: 1,
      threshold: 0.8,
      sections: ['technical_insights', 42],
      type: 'project',
      limit: 5,
      preview: true,
    });
    expect(params.days).toBe(0);
    expect(params.minEntries).toBe(3);
    expect(params.minDays).toBe(1);
    expect(params.threshold).toBe(0.8);
    expect(params.sections).toEqual(['technical_insights']); // non-strings dropped
    expect(params.type).toBe('project');
    expect(params.limit).toBe(5);
    expect(params.preview).toBe(true);
  });

  test('falls back on wrong-typed values', () => {
    const params = parseThemeParams({ days: 'ten', type: 'everything', preview: 'yes' });
    expect(params.days).toBe(30);
    expect(params.type).toBe('both');
    expect(params.preview).toBe(false);
  });
});

describe('formatThemesOutput', () => {
  const theme: RecurringTheme = {
    representativeExcerpt: 'the recurring insight about embeddings',
    supportingExcerpts: ['second sighting', 'third sighting'],
    distinctEntries: 5,
    occurrences: 7,
    distinctDays: 3,
    dateSpan: { start: '2026-07-01', end: '2026-07-03' },
    sectionDistribution: { 'Technical Insights': 4, 'Project Notes': 3 },
    cohesion: 0.912345,
    sourcePaths: ['/j/2026-07-01/e1.md', '/j/2026-07-03/e5.md'],
    type: 'user',
  };
  const result: RecurrenceResult = {
    themes: [theme],
    stats: {
      chunksScanned: 12,
      entriesScanned: 6,
      clustersFormed: 2,
      largestClusterEntries: 5,
      themesQualifying: 1,
    },
  };
  const params = parseThemeParams({});

  test('renders a header with the window, echoed params and scan counts', () => {
    const text = formatThemesOutput(result, params);
    expect(text).toContain('last 30 days');
    expect(text).toContain(`threshold ${DEFAULT_THRESHOLD}`);
    expect(text).toContain('minEntries 5');
    expect(text).toContain('minDays 2');
    expect(text).toContain('6 entries');
    expect(text).toContain('12 section chunks');
  });

  test('renders numbered themes with evidence, excerpts and sources', () => {
    const text = formatThemesOutput(result, params);
    expect(text).toContain('1. [5 entries / 7 occurrences / 3 days] 2026-07-01 → 2026-07-03 (user, cohesion 0.91)');
    expect(text).toContain('Sections: Technical Insights ×4, Project Notes ×3');
    expect(text).toContain('Theme: the recurring insight about embeddings');
    expect(text).toContain('Also: second sighting');
    expect(text).toContain('Sources: /j/2026-07-01/e1.md, /j/2026-07-03/e5.md');
  });

  test('days 0 reads as all time', () => {
    expect(formatThemesOutput(result, parseThemeParams({ days: 0 }))).toContain('all time');
  });

  test('long excerpts are truncated (600 representative / 200 supporting)', () => {
    const long: RecurrenceResult = {
      ...result,
      themes: [
        { ...theme, representativeExcerpt: 'R'.repeat(700), supportingExcerpts: ['S'.repeat(300)] },
      ],
    };
    const text = formatThemesOutput(long, params);
    expect(text).toContain('R'.repeat(600) + '...');
    expect(text).not.toContain('R'.repeat(601));
    expect(text).toContain('S'.repeat(200) + '...');
    expect(text).not.toContain('S'.repeat(201));
  });

  test('empty result says so, still reporting the scan', () => {
    const empty: RecurrenceResult = {
      themes: [],
      stats: { chunksScanned: 3, entriesScanned: 2, clustersFormed: 0, largestClusterEntries: 1, themesQualifying: 0 },
    };
    const text = formatThemesOutput(empty, params);
    expect(text).toContain('No recurring themes found');
    expect(text).toContain('2 entries');
  });

  test('preview mode reports statistics only, no excerpts', () => {
    const text = formatThemesOutput(result, parseThemeParams({ preview: true }));
    expect(text).toContain('2 clusters formed');
    expect(text).toContain('Largest cluster spans 5 distinct entries');
    expect(text).toContain('1 theme passes');
    expect(text).not.toContain('Theme:');
    expect(text).not.toContain('the recurring insight');
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx jest tests/recurrence.test.ts -v`
Expected: FAIL — `parseThemeParams` / `formatThemesOutput` not exported.

- [ ] **Step 3: Implement**

Append to `src/recurrence.ts`:

```typescript
export interface ThemeParams {
  days: number;
  minEntries: number;
  minDays: number;
  threshold: number;
  sections?: string[];
  type: 'project' | 'user' | 'both';
  limit: number;
  preview: boolean;
}

export function parseThemeParams(args: Record<string, unknown>): ThemeParams {
  return {
    days: typeof args.days === 'number' ? args.days : 30,
    minEntries: typeof args.minEntries === 'number' ? args.minEntries : 5,
    minDays: typeof args.minDays === 'number' ? args.minDays : 2,
    threshold: typeof args.threshold === 'number' ? args.threshold : DEFAULT_THRESHOLD,
    sections: Array.isArray(args.sections)
      ? args.sections.filter((s): s is string => typeof s === 'string')
      : undefined,
    type:
      args.type === 'project' || args.type === 'user' || args.type === 'both'
        ? args.type
        : 'both',
    limit: typeof args.limit === 'number' ? args.limit : 20,
    preview: args.preview === true,
  };
}

function truncate(text: string, max: number): string {
  const clean = text.replace(/\s+/g, ' ').trim();
  return clean.length > max ? clean.slice(0, max) + '...' : clean;
}

export function formatThemesOutput(result: RecurrenceResult, params: ThemeParams): string {
  const { themes, stats } = result;
  const window = params.days > 0 ? `last ${params.days} days` : 'all time';
  const header =
    `Recurring themes (${window}, threshold ${params.threshold}, ` +
    `minEntries ${params.minEntries}, minDays ${params.minDays}): ` +
    `scanned ${stats.entriesScanned} entries (${stats.chunksScanned} section chunks), ` +
    `${stats.clustersFormed} clusters formed.`;

  if (params.preview) {
    return (
      `${header}\n` +
      `Largest cluster spans ${stats.largestClusterEntries} distinct entries; ` +
      `${stats.themesQualifying} theme${stats.themesQualifying === 1 ? ' passes' : 's pass'} the thresholds.`
    );
  }

  if (themes.length === 0) {
    return `${header}\nNo recurring themes found.`;
  }

  const body = themes
    .map((t, i) => {
      const distribution = Object.entries(t.sectionDistribution)
        .map(([section, count]) => `${section} ×${count}`)
        .join(', ');
      const lines = [
        `${i + 1}. [${t.distinctEntries} entries / ${t.occurrences} occurrences / ${t.distinctDays} days] ` +
          `${t.dateSpan.start} → ${t.dateSpan.end} (${t.type}, cohesion ${t.cohesion.toFixed(2)})`,
        `   Sections: ${distribution}`,
        `   Theme: ${truncate(t.representativeExcerpt, 600)}`,
        ...t.supportingExcerpts.map(e => `   Also: ${truncate(e, 200)}`),
        `   Sources: ${t.sourcePaths.join(', ')}`,
      ];
      return lines.join('\n');
    })
    .join('\n\n');

  return `${header}\n\n${body}`;
}
```

- [ ] **Step 4: Run the tests**

Run: `npx jest tests/recurrence.test.ts -v` → PASS. Run `npm test` → all green.

- [ ] **Step 5: Commit**

```bash
git add src/recurrence.ts tests/recurrence.test.ts
git commit -m "feat: parameter parsing and text output for recurring themes"
```

---

### Task 5: Expose the find_recurring_themes MCP tool

Thin handler: collect (reusing `SearchService`'s loader so the coincident-roots dedup applies to recurrence too — otherwise distinct-entry counts would double) → gather → find → format. SDK 0.4.0 has no in-memory transport, so — like every existing handler — the glue itself has no direct unit test; the composition seam it calls is covered end-to-end over a real on-disk corpus.

**Files:**
- Modify: `src/search.ts:270` (make `collectEmbeddings` public)
- Modify: `src/server.ts` (tool registration + handler)
- Modify: `README.md` (document the tool)
- Test: `tests/recurrence.test.ts`

**Interfaces:**
- Consumes: `SearchService.collectEmbeddings(type: 'project' | 'user' | 'both'): Promise<LoadedEmbedding[]>` (now public), `EmbeddingService.getInstance()`, `isCompatible`, and Task 2–4 exports.
- Produces: the `find_recurring_themes` MCP tool (schema below). No new TypeScript exports.

- [ ] **Step 1: Write the failing end-to-end test**

Append to `tests/recurrence.test.ts` (add imports: `import * as fs from 'fs/promises'; import * as os from 'os'; import { SearchService } from '../src/search'; import { EmbeddingService } from '../src/embeddings';` — keep the existing `path` import if already present, otherwise add it):

```typescript
describe('end to end over an on-disk corpus (the handler composition seam)', () => {
  let dir: string;
  const model = EmbeddingService.getInstance().getModelName();

  const writeEmbedding = async (
    day: string,
    name: string,
    over: Record<string, unknown> = {}
  ) => {
    const dayDir = path.join(dir, day);
    await fs.mkdir(dayDir, { recursive: true });
    await fs.writeFile(path.join(dayDir, `${name}.md`), '## X\n\nbody', 'utf8');
    await fs.writeFile(
      path.join(dayDir, `${name}.embedding`),
      JSON.stringify({
        version: EMBEDDING_SCHEMA_VERSION,
        model,
        embedding: [0.1, 0.2, 0.3, 0.4, 0.5],
        sectionEmbeddings: [
          { section: 'Technical Insights', text: `note ${name}`, embedding: [0.1, 0.2, 0.3, 0.4, 0.5] },
        ],
        text: `note ${name}`,
        sections: ['Technical Insights'],
        timestamp: Date.parse(`${day}T10:00:00Z`),
        path: path.join(dayDir, `${name}.md`),
        ...over,
      }),
      'utf8'
    );
  };

  beforeEach(async () => {
    dir = await fs.mkdtemp(path.join(os.tmpdir(), 'recurrence-e2e-'));
  });
  afterEach(async () => {
    await fs.rm(dir, { recursive: true, force: true });
  });

  test('collect -> gather -> find -> format detects a theme; dream and foreign entries stay out; nothing is written', async () => {
    await writeEmbedding('2026-07-01', 'e1');
    await writeEmbedding('2026-07-01', 'e2');
    await writeEmbedding('2026-07-02', 'e3');
    await writeEmbedding('2026-07-03', 'e4');
    await writeEmbedding('2026-07-03', 'e5');
    await writeEmbedding('2026-07-02', 'dream', { dream: true });
    await writeEmbedding('2026-07-02', 'foreign', { model: 'some-other-model' });

    const before = (await fs.readdir(dir, { recursive: true })).sort();

    const svc = new SearchService(dir, path.join(dir, 'no-user'));
    const embeddings = await svc.collectEmbeddings('project');
    const es = EmbeddingService.getInstance();
    const chunks = gatherThemeChunks(embeddings, {
      now: Date.now(),
      days: 0,
      isCompatible: e => es.isCompatible(e),
    });
    const result = findRecurringThemes(chunks, {
      threshold: 0.9,
      minEntries: 5,
      minDays: 2,
      limit: 20,
    });

    expect(result.themes).toHaveLength(1);
    expect(result.themes[0].distinctEntries).toBe(5);
    expect(result.stats.entriesScanned).toBe(5); // dream + foreign excluded before counting

    const text = formatThemesOutput(result, parseThemeParams({ days: 0 }));
    expect(text).toContain('1. [5 entries');
    expect(text).not.toContain('dream.md');
    expect(text).not.toContain('foreign.md');

    // Read-only: the scan created, modified and deleted nothing.
    const after = (await fs.readdir(dir, { recursive: true })).sort();
    expect(after).toEqual(before);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx jest tests/recurrence.test.ts -v`
Expected: FAIL — TypeScript error: property `collectEmbeddings` is private.

- [ ] **Step 3: Make collectEmbeddings public**

In `src/search.ts:270`, change `private async collectEmbeddings(` to `async collectEmbeddings(` and extend the comment above it — replace its last sentence block with:

```typescript
  // Collect embeddings honoring the type filter. When both journal roots resolve
  // to the same physical directory (e.g. PRIVATE_JOURNAL_PATH is set, or CWD ===
  // HOME), scanning both would return every entry twice — halving the effective
  // limit and surfacing each hit side by side. In that case scan once, labeling
  // by the requested type (both/user -> 'user', since PRIVATE_JOURNAL_PATH is the
  // personal journal). Public: find_recurring_themes reuses this loader so the
  // dedup applies to recurrence counting too.
```

Run: `npx jest tests/recurrence.test.ts -v` → PASS.

- [ ] **Step 4: Register the tool in the server**

In `src/server.ts`, extend the imports:

```typescript
import { EmbeddingService } from './embeddings.js';
import {
  DEFAULT_THRESHOLD,
  findRecurringThemes,
  formatThemesOutput,
  gatherThemeChunks,
  parseThemeParams,
} from './recurrence.js';
```

In the `ListToolsRequestSchema` handler, append after the `read_recent_entries` tool object:

```typescript
        {
          name: 'find_recurring_themes',
          description: "Detect themes that recur across journal entries by clustering their section embeddings. Read-only and fully offline: returns each recurring theme with evidence (distinct entries, date span, representative excerpts, source paths). Dream entries are excluded from the scanned corpus.",
          inputSchema: {
            type: 'object',
            properties: {
              days: {
                type: 'number',
                description: "Look-back window in days; 0 means all-time (default: 30)",
                default: 30,
              },
              minEntries: {
                type: 'number',
                description: "Minimum distinct entries for a theme to qualify (default: 5)",
                default: 5,
              },
              minDays: {
                type: 'number',
                description: "Minimum distinct days a theme must span (default: 2)",
                default: 2,
              },
              threshold: {
                type: 'number',
                description: "Cosine similarity cutoff for clustering",
                default: DEFAULT_THRESHOLD,
              },
              sections: {
                type: 'array',
                items: { type: 'string' },
                description: "Restrict to section types (e.g., ['technical_insights'])",
              },
              type: {
                type: 'string',
                enum: ['project', 'user', 'both'],
                description: "Scan project notes, user notes, or both (default: both)",
                default: 'both',
              },
              limit: {
                type: 'number',
                description: "Maximum themes returned, ranked by strength (default: 20)",
                default: 20,
              },
              preview: {
                type: 'boolean',
                description: "Return clustering statistics only, without excerpts — for sweeping thresholds (default: false)",
                default: false,
              },
            },
            required: [],
          },
        },
```

In the `CallToolRequestSchema` handler, insert before the final `throw new Error(\`Unknown tool: ...\`)`:

```typescript
      if (request.params.name === 'find_recurring_themes') {
        const params = parseThemeParams(args ?? {});

        try {
          const embeddings = await this.searchService.collectEmbeddings(params.type);
          const embeddingService = EmbeddingService.getInstance();
          const chunks = gatherThemeChunks(embeddings, {
            now: Date.now(),
            days: params.days,
            sections: params.sections,
            isCompatible: (entry) => embeddingService.isCompatible(entry),
          });
          const result = findRecurringThemes(chunks, params);
          return {
            content: [
              {
                type: 'text',
                text: formatThemesOutput(result, params),
              },
            ],
          };
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
          throw new Error(`Failed to find recurring themes: ${errorMessage}`);
        }
      }
```

- [ ] **Step 5: Document in README**

In `README.md`, insert after the `### \`list_recent_entries\`` block (its last line is `- **days**: Days back to search (default: 30)`, README.md:142) and before `## File Structure`:

```markdown
### `find_recurring_themes`
Detect themes that recur across entries by clustering their section embeddings (read-only, fully offline):
- **days**: Look-back window in days, 0 = all-time (default: 30)
- **minEntries**: Distinct entries a theme needs to qualify (default: 5)
- **minDays**: Distinct days a theme must span (default: 2)
- **threshold**: Cosine similarity cutoff for clustering (default: 0.65)
- **sections**: Restrict to specific categories
- **type**: Scan scope - 'project', 'user', or 'both' (default: 'both')
- **limit**: Maximum themes returned (default: 20)
- **preview**: Statistics only, for sweeping thresholds (default: false)

Dream entries (see `record_dream`) are excluded from the scanned corpus.
```

- [ ] **Step 6: Build, lint, full suite**

Run: `npm run build && npm run lint && npm test`
Expected: all green (the build catches any handler-glue type error the tests can't reach).

- [ ] **Step 7: Commit**

```bash
git add src/search.ts src/server.ts README.md tests/recurrence.test.ts
git commit -m "feat: expose find_recurring_themes MCP tool"
```

---

### Task 6: writeDream and the record_dream MCP tool

The write side. `writeDream` reuses the journal pipeline: same timestamped layout, `dream: true` front-matter (which Task 1 already mirrors onto the `.embedding`), body wrapped in a `## Dream` section so the standard section-chunking embeds it. Dream entries go to the **user** journal (meta-reflection about the person). A small front-matter helper is extracted so `formatThoughts` and `writeDream` share it byte-identically.

**Files:**
- Modify: `src/journal.ts` (extract `formatFrontmatter`, add `writeDream`)
- Modify: `src/server.ts` (tool registration + handler)
- Modify: `README.md`
- Test: `tests/dream.test.ts`

**Interfaces:**
- Consumes: Task 1's dream detection in `regenerateEmbedding`; `gatherThemeChunks`/`findRecurringThemes` (Tasks 2–3) and `SearchService`/`EmbeddingService` for the integration test.
- Produces: `JournalManager.writeDream(content: string): Promise<string>` — returns the absolute `.md` path of the written entry; the `record_dream` MCP tool.

- [ ] **Step 1: Write the failing tests**

Append to `tests/dream.test.ts` (extend imports):

```typescript
import { SearchService } from '../src/search';
import { EmbeddingService } from '../src/embeddings';
import { findRecurringThemes, gatherThemeChunks } from '../src/recurrence';
```

```typescript
describe('writeDream', () => {
  let projectDir: string;
  let userDir: string;

  beforeEach(async () => {
    projectDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dream-write-project-'));
    userDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dream-write-user-'));
  });
  afterEach(async () => {
    await fs.rm(projectDir, { recursive: true, force: true });
    await fs.rm(userDir, { recursive: true, force: true });
  });

  test('writes a dream entry into the user journal with dream front-matter and a Dream section', async () => {
    const manager = new JournalManager(projectDir, userDir);
    const filePath = await manager.writeDream('The fork work keeps circling embedding compatibility.');

    expect(filePath.startsWith(userDir + path.sep)).toBe(true);
    expect(filePath).toMatch(/\d{4}-\d{2}-\d{2}\/\d{2}-\d{2}-\d{2}-\d{6}\.md$/);

    const content = await fs.readFile(filePath, 'utf8');
    expect(content).toMatch(/^---\n/);
    expect(content).toContain('\ndream: true\n');
    expect(content).toContain('## Dream');
    expect(content).toContain('keeps circling embedding compatibility');
    expect(content).toMatch(/title: "/);
    expect(content).toMatch(/timestamp: \d+/);
  });

  test('embeds the dream entry with the dream flag and a Dream section vector', async () => {
    const manager = new JournalManager(projectDir, userDir);
    const filePath = await manager.writeDream('dream body');

    const data = JSON.parse(await fs.readFile(filePath.replace(/\.md$/, '.embedding'), 'utf8'));
    expect(data.dream).toBe(true);
    expect(data.sectionEmbeddings).toHaveLength(1);
    expect(data.sectionEmbeddings[0].section).toBe('Dream');
    expect(Array.isArray(data.sectionEmbeddings[0].embedding)).toBe(true);
  });
});

describe('dream exclusion end to end', () => {
  let projectDir: string;
  let userDir: string;

  beforeEach(async () => {
    projectDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dream-e2e-project-'));
    userDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dream-e2e-user-'));
  });
  afterEach(async () => {
    await fs.rm(projectDir, { recursive: true, force: true });
    await fs.rm(userDir, { recursive: true, force: true });
  });

  test('a recorded dream is searchable but never feeds a subsequent recurrence scan', async () => {
    const manager = new JournalManager(projectDir, userDir);
    await manager.writeThoughts({ reflections: 'first recurring thought' });
    await manager.writeThoughts({ reflections: 'second recurring thought' });
    const dreamPath = await manager.writeDream('the consolidation of those thoughts');

    const svc = new SearchService(projectDir, userDir);

    // Goal (A): dream entries are ordinary searchable entries.
    const hits = await svc.search('consolidation', { type: 'user', minScore: -1 });
    expect(hits.some(h => h.path === dreamPath)).toBe(true);

    // Feedback-loop guard: the engine never sees the dream.
    const embeddings = await svc.collectEmbeddings('user');
    const es = EmbeddingService.getInstance();
    const chunks = gatherThemeChunks(embeddings, {
      now: Date.now(),
      days: 0,
      isCompatible: e => es.isCompatible(e),
    });
    const { themes } = findRecurringThemes(chunks, {
      threshold: 0.9,
      minEntries: 2,
      minDays: 1,
      limit: 20,
    });

    expect(themes).toHaveLength(1);
    expect(themes[0].occurrences).toBe(2); // only the two thoughts are material
    expect(themes[0].sourcePaths).not.toContain(dreamPath);
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx jest tests/dream.test.ts -v`
Expected: FAIL — `manager.writeDream is not a function`.

- [ ] **Step 3: Implement writeDream (with the front-matter helper)**

In `src/journal.ts`, add the helper (below `formatTimestamp`):

```typescript
  private formatFrontmatter(timestamp: Date, extraLines: string[] = []): string {
    const timeDisplay = timestamp.toLocaleTimeString('en-US', {
      hour12: true,
      hour: 'numeric',
      minute: '2-digit',
      second: '2-digit'
    });
    const dateDisplay = timestamp.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
    const lines = [
      `title: "${timeDisplay} - ${dateDisplay}"`,
      `date: ${timestamp.toISOString()}`,
      `timestamp: ${timestamp.getTime()}`,
      ...extraLines,
    ];
    return `---\n${lines.join('\n')}\n---\n`;
  }
```

Rewrite `formatThoughts` to use it — delete its `timeDisplay`/`dateDisplay` declarations and replace the trailing template-literal `return` with (byte-identical output; the existing journal format tests must stay green):

```typescript
    return `${this.formatFrontmatter(timestamp)}\n${sections.join('\n\n')}\n`;
```

Add `writeDream` (after `writeThoughts`):

```typescript
  async writeDream(content: string): Promise<string> {
    const timestamp = new Date();
    const dateString = this.formatDate(timestamp);
    const timeString = this.formatTimestamp(timestamp);

    // Dream entries are meta-reflection about the person: user journal.
    const dayDirectory = path.join(this.userJournalPath, dateString);
    const filePath = path.join(dayDirectory, `${timeString}.md`);

    await this.ensureDirectoryExists(dayDirectory);

    const formattedEntry = `${this.formatFrontmatter(timestamp, ['dream: true'])}\n## Dream\n\n${content}\n`;
    await fs.writeFile(filePath, formattedEntry, 'utf8');

    await this.generateEmbeddingForEntry(filePath, formattedEntry, timestamp);
    return filePath;
  }
```

Run: `npx jest tests/dream.test.ts tests/journal.test.ts -v` → PASS (both files).

- [ ] **Step 4: Register the record_dream tool**

In `src/server.ts`, append to the `ListToolsRequestSchema` tools array (after `find_recurring_themes`):

```typescript
        {
          name: 'record_dream',
          description: "Record a dream entry — a consolidation written after reviewing recurring themes (see find_recurring_themes). Stored in the user journal and searchable like any entry, but excluded from future recurrence scans.",
          inputSchema: {
            type: 'object',
            properties: {
              content: {
                type: 'string',
                description: "The dream narrative in markdown: what recurred (with evidence), loose threads, and any memory-promotion candidates",
              },
            },
            required: ['content'],
          },
        },
```

And in the `CallToolRequestSchema` handler, after the `find_recurring_themes` branch:

```typescript
      if (request.params.name === 'record_dream') {
        if (!args || typeof args.content !== 'string' || args.content.trim().length === 0) {
          throw new Error('content is required and must be a non-empty string');
        }

        try {
          const filePath = await this.journalManager.writeDream(args.content);
          return {
            content: [
              {
                type: 'text',
                text: `Dream recorded at ${filePath}`,
              },
            ],
          };
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
          throw new Error(`Failed to record dream: ${errorMessage}`);
        }
      }
```

- [ ] **Step 5: Document in README**

In `README.md`, insert after the `### \`find_recurring_themes\`` block added in Task 5:

```markdown
### `record_dream`
Record a consolidation entry ("dream") written after reviewing recurring themes:
- **content** (required): The dream narrative in markdown

Dream entries live in the user journal and are searchable like any other entry, but they never feed `find_recurring_themes` — preventing feedback loops.
```

- [ ] **Step 6: Build, lint, full suite**

Run: `npm run build && npm run lint && npm test`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add src/journal.ts src/server.ts README.md tests/dream.test.ts
git commit -m "feat: record_dream tool writing dream entries excluded from recurrence"
```

---

### Task 7: Calibrate the default threshold against the real corpus

The spec's one open item. The engine is fully offline (no model load — `isCompatible` only compares strings), so this runs instantly against the real journal. The real user journal is `/Users/amadeus.delfino/.private_journal` (underscore — set via `PRIVATE_JOURNAL_PATH`), ~24 entries as of 2026-07-13.

**Files:**
- Create: `<scratchpad>/calibrate.mjs` (throwaway; NOT committed)
- Modify: `src/recurrence.ts` (the `DEFAULT_THRESHOLD` constant)
- Modify: `README.md` (the documented default)
- Modify: `docs/superpowers/specs/2026-07-13-dream-consolidation-design.md` (close the open item)

**Interfaces:**
- Consumes: `dist/` build of Tasks 1–6.
- Produces: the calibrated `DEFAULT_THRESHOLD` value (code, README and spec all agree on it).

- [ ] **Step 1: Build and write the sweep script**

Run: `npm run build`. Then write `calibrate.mjs` in the session scratchpad directory (do not commit it):

```javascript
import { SearchService } from '/Users/amadeus.delfino/IdeaProjects/private-journal-mcp/dist/search.js';
import { EmbeddingService } from '/Users/amadeus.delfino/IdeaProjects/private-journal-mcp/dist/embeddings.js';
import {
  gatherThemeChunks,
  findRecurringThemes,
  formatThemesOutput,
  parseThemeParams,
} from '/Users/amadeus.delfino/IdeaProjects/private-journal-mcp/dist/recurrence.js';

const svc = new SearchService(
  '/Users/amadeus.delfino/IdeaProjects/private-journal-mcp/.private-journal',
  '/Users/amadeus.delfino/.private_journal'
);
const embeddings = await svc.collectEmbeddings('both');
const es = EmbeddingService.getInstance();
const chunks = gatherThemeChunks(embeddings, {
  now: Date.now(),
  days: 0,
  isCompatible: (e) => es.isCompatible(e),
});
console.log(`corpus: ${embeddings.length} embedding files, ${chunks.length} section chunks\n`);

for (let i = 0; i <= 7; i++) {
  const threshold = 0.5 + i * 0.05;
  // minEntries 3 for the sweep: the corpus is ~24 entries, the default 5 would
  // hide the cluster structure we are trying to observe.
  const { stats } = findRecurringThemes(chunks, { threshold, minEntries: 3, minDays: 2, limit: 20 });
  console.log(
    `τ=${threshold.toFixed(2)}  clusters(≥2 chunks)=${stats.clustersFormed}  ` +
    `largest=${stats.largestClusterEntries} entries  qualifying(≥3e,≥2d)=${stats.themesQualifying}`
  );
}

const CANDIDATE = Number(process.argv[2] || 0.65);
const result = findRecurringThemes(chunks, { threshold: CANDIDATE, minEntries: 3, minDays: 2, limit: 5 });
console.log('\n' + formatThemesOutput(result, parseThemeParams({ days: 0, threshold: CANDIDATE, minEntries: 3 })));
```

- [ ] **Step 2: Sweep and inspect**

Run: `node <scratchpad>/calibrate.mjs` — read the sweep table. Then re-run with 2–3 candidate values (`node <scratchpad>/calibrate.mjs 0.60`, `0.70`, …) and read the printed themes: at a good τ the excerpts inside one theme are visibly about the same thing, and unrelated topics do not share a cluster. Too low → one mega-cluster (largest ≈ corpus size); too high → only near-duplicates cluster (all singletons).

Pick the τ where the top themes are semantically coherent. Sanity-expect the neighborhood of 0.60–0.75 for `paraphrase-multilingual-MiniLM-L12-v2`; trust the observed clusters, not the guess.

- [ ] **Step 3: Fix the default everywhere**

With the chosen value (call it `τ*`):
- `src/recurrence.ts`: set `DEFAULT_THRESHOLD = τ*` and update its comment to `// calibrated 2026-07-13 against the real corpus (paraphrase-multilingual-MiniLM-L12-v2)`.
- `README.md`: update `(default: 0.65)` on the threshold line to the chosen value.
- Spec `docs/superpowers/specs/2026-07-13-dream-consolidation-design.md`, "Open items" section: replace the single bullet with:

```markdown
- ~~Fix a sensible default for `threshold` (τ)~~ — calibrated 2026-07-13 against
  the real corpus (~24 entries, paraphrase-multilingual-MiniLM-L12-v2):
  default fixed at τ = <chosen value>.
```

(`<chosen value>` is the measured result of Step 2 — the one deliberate runtime substitution in this plan.)

- [ ] **Step 4: Verify nothing depended on the provisional value**

Run: `npm test` → all green (tests reference `DEFAULT_THRESHOLD` by import, never by literal). Run `grep -rn "0\.65" src/ tests/ README.md` → no stale mentions (ignore coincidental vector components in tests, which use values like 0.65 only if you introduced them — none exist in this plan).

- [ ] **Step 5: Commit**

```bash
git add src/recurrence.ts README.md
git add -f docs/superpowers/specs/2026-07-13-dream-consolidation-design.md
git commit -m "feat: calibrate default recurrence threshold against the real corpus"
```

---

### Task 8: The /dream ritual command (outside the repo) and handoff

Component 4. A slash command in the owner's Claude config, matching the house style of `~/.claude/commands/route-memory.md` (front-matter with `description` + `argument-hint`, Portuguese, numbered ritual steps). Not repo code: no repo commit, and the test suite does not cover it (per the spec, it is validated by running it).

**Files:**
- Create: `/Users/amadeus.delfino/.claude/commands/dream.md`

**Interfaces:**
- Consumes: the `find_recurring_themes` and `record_dream` MCP tools (Tasks 5–6) and the owner's memory doctrine (global CLAUDE.md "Learning and Memory Management", referenced — not duplicated — exactly like route-memory.md does).
- Produces: the invocable `/dream` ritual.

- [ ] **Step 1: Write the command file**

Create `/Users/amadeus.delfino/.claude/commands/dream.md`:

```markdown
---
description: Ritual do sonho — varre o journal por temas recorrentes (find_recurring_themes), consolida-os numa entrada-sonho (record_dream) e propõe promoções de memória. Propose-only — nenhuma memória é escrita sem aprovação. Rode manualmente quando quiser consolidar (futuramente via cron).
argument-hint: [janela em dias ou foco opcional]
---

Você vai executar o ritual do sonho: olhar o corpus do journal como um todo, notar o que fica recorrendo e consolidar. NÃO é resumo de sessão — é consolidação sobre o histórico.

Argumento opcional: $ARGUMENTS
(Número → use como `days`. Texto → foco ao interpretar os temas. Vazio → defaults.)

## Passo 1 — Detectar
Chame `find_recurring_themes` (defaults; `days` do argumento se houver). Resultado vazio ou ruidoso? Ajuste `threshold` (±0.05) ou `days` usando `preview: true` para checar a estrutura antes de rodar cheio. Se seguir sem temas, escreva uma entrada-sonho curta registrando isso e PARE.

## Passo 2 — Sonhar (record_dream)
Escreva a síntese via `record_dream` — prosa sua, não dump da ferramenta:
- por tema forte: a essência em 1–3 frases + evidência (distinctEntries, dateSpan, seções);
- fios soltos e o que se destacou fora dos clusters;
- a lista de candidatos a memória do Passo 3 (nome, descrição, corpo rascunhado, store alvo, paths de evidência).

## Passo 3 — Propor promoções (propose-only)
Para cada tema forte, um mini /route-memory:
1. **Dedup:** leia o MEMORY.md + arquivos relacionados do file-memory e releia o CLAUDE.md global/do projeto. Já coberto → pule (dizendo qual memória cobre).
2. **Route pela natureza do tema** (critério: seção "Learning and Memory Management" do CLAUDE.md global — referencie, não reproduza): regra/preferência durável → CLAUDE.md; fato estável de projeto → file-memory; efêmero → fica só no journal.
3. **Apresente a lista:** `tema (1 linha) | evidência (N entradas, M dias) | store sugerido | rascunho`.

REGRA DURA (modelo i da spec): o sonho NUNCA escreve memória de longo prazo sem aprovação do Amadeus — nem file-memory. Detecta, rascunha, propõe; ele decide. Só grave após o "ok", e aí siga os gates por store do /route-memory.

## Passo 4 — Reportar
Resumo curto: temas achados (com força), entrada-sonho gravada (path), candidatos propostos vs. pulados por dedup, e o que aguarda decisão.
```

- [ ] **Step 2: Validate the ritual (dry-run limits) and hand off**

The suite does not cover this component. Mechanical validation happened in Task 7 (the engine ran against the real corpus). The full ritual needs the new tools visible in a Claude session, which requires re-pinning the user's MCP config to the post-merge SHA — an action for Amadeus. Report in the final summary:

1. The MCP config still pins the pre-dream SHA via npx; after merge, re-pin to the new SHA to expose `find_recurring_themes` / `record_dream`.
2. First `/dream` run should be attended (it writes a real dream entry to the real journal; memory promotion is propose-only regardless).

---

## Final verification (after all tasks)

- [ ] `npm run build && npm run lint && npm test` — everything green, output pristine.
- [ ] Grep-verify each commit's claim against the files as they are now (per CLAUDE.md: build+test passing is not proof the intended changes landed).
- [ ] Use superpowers:verification-before-completion, then superpowers:requesting-code-review, then superpowers:finishing-a-development-branch (branch: `feat/dream-consolidation`, PRs target `main` of the fork).

## Spec-coverage map (self-review record)

| Spec section | Where |
|---|---|
| Goal A — dream feeds future sessions via search | Task 6 (searchable assert) |
| Goal C — recurring detail promoted to durable memory | Task 8 (ritual, propose-only) |
| No daemon / no LLM in server | By construction; engine offline (Tasks 2–3), no cron code anywhere |
| Window model (trigger vs. material) | `days` param (Tasks 3–5) |
| No ledger; idempotency via memory de-dup + dream marker | Task 8 step 3.1 + Tasks 1/3/6 |
| Engine over pre-computed vectors, no model | Task 2 (pure), Task 3 (filters: dream, isCompatible) |
| Leader/medoid clustering, deterministic, anti-chaining | Task 2 (chain test, drop test, sort in Task 3) |
| distinct entries + minDays counting; 7-chunks/5-files/3-days example | Task 2 (spec-example test) |
| Theme output fields (excerpt, supporting, counts, span, distribution, cohesion, paths, type) | Task 2 |
| find_recurring_themes params + defaults | Task 4 (parse), Task 5 (schema) |
| threshold calibration + preview mode | Task 4 (preview format), Task 7 (calibration) |
| Output format in search_journal style | Task 4 (formatter tests) |
| record_dream: front-matter + EmbeddingData flag, user journal, searchable, excluded | Tasks 1 and 6 |
| Ritual as owner-config skill; validated by running, not by suite | Task 8 |
| Safety model (i): propose-only | Task 8 (RULE block) |
| Out-of-scope list | Nothing built for them; O(n²) accepted (Task 2) |
