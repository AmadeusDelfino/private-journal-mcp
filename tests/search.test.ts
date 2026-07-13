// ABOUTME: Unit tests for max-over-sections ranking in SearchService
// ABOUTME: Verifies entries are scored by their best-matching section, not the whole-entry average

import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import { SearchService } from '../src/search';
import { EMBEDDING_SCHEMA_VERSION, EmbeddingService } from '../src/embeddings';

describe('max-over-sections ranking', () => {
  let projectDir: string;

  const writeEmbedding = async (name: string, sectionEmbeddings: any[]) => {
    const day = path.join(projectDir, '2026-07-08');
    await fs.mkdir(day, { recursive: true });
    await fs.writeFile(path.join(day, `${name}.md`), '## X\n\nbody', 'utf8');
    await fs.writeFile(path.join(day, `${name}.embedding`), JSON.stringify({
      version: EMBEDDING_SCHEMA_VERSION, model: 'test',
      embedding: [0, 0, 1], sectionEmbeddings,
      text: 'body', sections: sectionEmbeddings.map(s => s.section),
      timestamp: Date.now(), path: path.join(day, `${name}.md`),
    }), 'utf8');
  };

  beforeEach(async () => { projectDir = await fs.mkdtemp(path.join(os.tmpdir(), 'search-test-')); });
  afterEach(async () => { await fs.rm(projectDir, { recursive: true, force: true }); });

  test('entry scored by its best section; matchedSection reported', async () => {
    // NOTE: tests/setup.ts mocks the model to return [0.1,0.2,0.3,0.4,0.5] for the QUERY.
    // Craft section vectors so one aligns with the (normalized) query direction.
    await writeEmbedding('good', [
      { section: 'A', text: 'a', embedding: [0.1, 0.2, 0.3, 0.4, 0.5] }, // parallel to query → cos ~1
      { section: 'B', text: 'b', embedding: [1, 0, 0, 0, 0] },
    ]);
    await writeEmbedding('bad', [
      { section: 'C', text: 'c', embedding: [-0.1, -0.2, -0.3, -0.4, -0.5] }, // opposite → cos ~-1
    ]);

    const svc = new SearchService(projectDir, path.join(projectDir, 'no-user'));
    const results = await svc.search('anything', { type: 'project', minScore: -1 });

    expect(results[0].path).toContain('good.md');
    expect(results[0].matchedSection).toBe('A');
    expect(results[0].score).toBeGreaterThan(results[1].score);
  });

  test('legacy v1 entry (no sectionEmbeddings) scored via whole-entry vector', async () => {
    const day = path.join(projectDir, '2026-07-08');
    await fs.mkdir(day, { recursive: true });
    await fs.writeFile(path.join(day, 'legacy.md'), '## X\n\nbody', 'utf8');
    await fs.writeFile(path.join(day, 'legacy.embedding'), JSON.stringify({
      // v1 format: no version, no model, no sectionEmbeddings
      embedding: [0.1, 0.2, 0.3, 0.4, 0.5], // parallel to query → cos ~1
      text: 'body', sections: ['X'],
      timestamp: Date.now(), path: path.join(day, 'legacy.md'),
    }), 'utf8');

    const svc = new SearchService(projectDir, path.join(projectDir, 'no-user'));
    const results = await svc.search('anything', { type: 'project', minScore: -1 });

    expect(results).toHaveLength(1);
    expect(results[0].path).toContain('legacy.md');
    expect(results[0].score).toBeCloseTo(1, 5);
    expect(results[0].matchedSection).toBeUndefined();
  });
});

describe('dimension floor (crash safety)', () => {
  let dir: string;
  const model = EmbeddingService.getInstance().getModelName();

  const writeEntry = async (name: string, sectionEmbeddings: any[], embedding: any = [0.1, 0.2, 0.3, 0.4, 0.5]) => {
    const day = path.join(dir, '2026-07-08');
    await fs.mkdir(day, { recursive: true });
    await fs.writeFile(path.join(day, `${name}.md`), '## X\n\nbody', 'utf8');
    await fs.writeFile(path.join(day, `${name}.embedding`), JSON.stringify({
      version: 2, model, embedding, sectionEmbeddings,
      text: 'body', sections: sectionEmbeddings.map((s: any) => s?.section),
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

  test('a null section entry is skipped, the good section still scores', async () => {
    await writeEntry('nullsec', [null, { section: 'A', text: 'a', embedding: [0.1, 0.2, 0.3, 0.4, 0.5] }]);

    const svc = new SearchService(dir, path.join(dir, 'no-user'));
    const results = await svc.search('anything', { type: 'project', minScore: -1 });

    expect(results).toHaveLength(1);
    expect(results[0].path).toContain('nullsec.md');
    expect(results[0].matchedSection).toBe('A');
  });

  test('entries with a bad whole-entry vector and no sections are excluded, no throw', async () => {
    // No sectionEmbeddings key at all → legacy fallback branch; version/model current
    // so Task 3's model ceiling won't be what excludes them.
    const day = path.join(dir, '2026-07-08');
    await fs.mkdir(day, { recursive: true });
    await fs.writeFile(path.join(day, 'wrongdim.md'), '## X\n\nbody', 'utf8');
    await fs.writeFile(path.join(day, 'wrongdim.embedding'), JSON.stringify({
      version: 2, model,
      embedding: [1, 2, 3], // 3-dim vs 5-dim query → floor excludes
      text: 'body', sections: ['X'],
      timestamp: Date.now(), path: path.join(day, 'wrongdim.md'),
    }), 'utf8');
    await fs.writeFile(path.join(day, 'nullwhole.md'), '## X\n\nbody', 'utf8');
    await fs.writeFile(path.join(day, 'nullwhole.embedding'), JSON.stringify({
      version: 2, model,
      embedding: null,
      text: 'body', sections: ['X'],
      timestamp: Date.now(), path: path.join(day, 'nullwhole.md'),
    }), 'utf8');

    const svc = new SearchService(dir, path.join(dir, 'no-user'));
    await expect(svc.search('anything', { type: 'project', minScore: -1 })).resolves.toEqual([]);
  });
});
