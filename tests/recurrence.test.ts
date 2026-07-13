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
