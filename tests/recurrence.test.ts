// ABOUTME: Unit tests for the recurrence engine and its supporting pure functions
// ABOUTME: Injects synthetic vectors directly - no model, no transformers mock needed

import * as path from 'path';
import * as fs from 'fs/promises';
import * as os from 'os';
import {
  findRecurringThemes,
  ThemeChunk,
  EngineOptions,
  gatherThemeChunks,
  GatherOptions,
  parseThemeParams,
  formatThemesOutput,
  DEFAULT_THRESHOLD,
  RecurringTheme,
  RecurrenceResult,
} from '../src/recurrence';
import { LoadedEmbedding, SearchService } from '../src/search';
import { EMBEDDING_SCHEMA_VERSION, EmbeddingService } from '../src/embeddings';

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
    // A~B (20°) and B~C (20°) but A vs C is 40° (cos 0.766 < 0.9). With the
    // leader A, the gathered ball is {A,B}; the medoid election ties A and B
    // (equal means) and strict `>` keeps the earliest — A — so the re-gather
    // stays anchored at A and C stays out. This pins the tie-break; the
    // general anti-chaining invariant (membership is one hop from the medoid,
    // never transitive) is pinned by the bridge test below.
    const chunks = [at(0, 1, 'A'), at(20, 2, 'B'), at(40, 3, 'C')];
    const { themes, stats } = findRecurringThemes(chunks, opts({ minEntries: 2, minDays: 1 }));

    expect(themes).toHaveLength(1);
    expect(themes[0].occurrences).toBe(2);
    const excerpts = [themes[0].representativeExcerpt, ...themes[0].supportingExcerpts];
    expect(excerpts).not.toContain('C');
    expect(stats.clustersFormed).toBe(1);
  });

  test('a bridge member within τ of the medoid joins, but nothing chains beyond the medoid radius', () => {
    // Spec model: cluster membership is "within τ of the MEDOID" (one hop from
    // the center), never transitive. Leader A gathers {A,B,N}; the medoid
    // re-centers on B. C sits outside A's ball but within τ of B (Δ20°), so it
    // joins — bounded medoid-radius growth. D is within τ of C (Δ20°) —
    // single-linkage would chain D in via C — but D is outside τ of the medoid
    // B (Δ40°), so it stays out.
    const chunks = [at(0, 1, 'A'), at(20, 2, 'B'), at(22, 3, 'N'), at(40, 4, 'C'), at(60, 5, 'D')];
    const { themes, stats } = findRecurringThemes(chunks, opts({ minEntries: 2, minDays: 1 }));

    expect(themes).toHaveLength(1);
    expect(themes[0].representativeExcerpt).toBe('B'); // medoid re-centers off the leader
    expect(themes[0].occurrences).toBe(4); // A, B, N, C — not D
    // A and C are equidistant from B; their relative order is float noise.
    expect(themes[0].supportingExcerpts).toHaveLength(3);
    expect(themes[0].supportingExcerpts[0]).toBe('N'); // nearest neighbour first
    expect([...themes[0].supportingExcerpts].sort()).toEqual(['A', 'C', 'N']);
    expect(stats.clustersFormed).toBe(1); // D ends as an uncounted singleton
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

  test('full ties (distinct entries and cohesion) rank the earlier cluster first', () => {
    const first = entryChunks(2, [0, 0, 1], 'first', 1);
    const second = entryChunks(2, [0, 1, 0], 'second', 10);
    const { themes } = findRecurringThemes([...first, ...second], opts({ minEntries: 2, minDays: 1 }));

    expect(themes).toHaveLength(2);
    expect(themes[0].representativeExcerpt).toBe('first0');
    expect(themes[1].representativeExcerpt).toBe('second0');
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
