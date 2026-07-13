// ABOUTME: Unit tests for max-over-sections ranking in SearchService
// ABOUTME: Verifies entries are scored by their best-matching section, not the whole-entry average

import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import { SearchService } from '../src/search';
import { EMBEDDING_SCHEMA_VERSION, EmbeddingService } from '../src/embeddings';

describe('max-over-sections ranking', () => {
  let projectDir: string;
  const model = EmbeddingService.getInstance().getModelName();

  const writeEmbedding = async (name: string, sectionEmbeddings: any[]) => {
    const day = path.join(projectDir, '2026-07-08');
    await fs.mkdir(day, { recursive: true });
    await fs.writeFile(path.join(day, `${name}.md`), '## X\n\nbody', 'utf8');
    await fs.writeFile(path.join(day, `${name}.embedding`), JSON.stringify({
      version: EMBEDDING_SCHEMA_VERSION, model,
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
});

describe('dimension floor (crash safety)', () => {
  let dir: string;
  const model = EmbeddingService.getInstance().getModelName();

  const writeEntry = async (name: string, sectionEmbeddings: any[], embedding: any = [0.1, 0.2, 0.3, 0.4, 0.5]) => {
    const day = path.join(dir, '2026-07-08');
    await fs.mkdir(day, { recursive: true });
    await fs.writeFile(path.join(day, `${name}.md`), '## X\n\nbody', 'utf8');
    await fs.writeFile(path.join(day, `${name}.embedding`), JSON.stringify({
      version: EMBEDDING_SCHEMA_VERSION, model, embedding, sectionEmbeddings,
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
      version: EMBEDDING_SCHEMA_VERSION, model,
      embedding: [1, 2, 3], // 3-dim vs 5-dim query → floor excludes
      text: 'body', sections: ['X'],
      timestamp: Date.now(), path: path.join(day, 'wrongdim.md'),
    }), 'utf8');
    await fs.writeFile(path.join(day, 'nullwhole.md'), '## X\n\nbody', 'utf8');
    await fs.writeFile(path.join(day, 'nullwhole.embedding'), JSON.stringify({
      version: EMBEDDING_SCHEMA_VERSION, model,
      embedding: null,
      text: 'body', sections: ['X'],
      timestamp: Date.now(), path: path.join(day, 'nullwhole.md'),
    }), 'utf8');

    const svc = new SearchService(dir, path.join(dir, 'no-user'));
    await expect(svc.search('anything', { type: 'project', minScore: -1 })).resolves.toEqual([]);
  });

  test('a non-array sectionEmbeddings container falls through to the legacy whole-entry branch', async () => {
    // Hand-crafted entry: sectionEmbeddings is a non-array object with a
    // truthy numeric length (`{ length: 1 }`). The old condition
    // `entry.sectionEmbeddings && entry.sectionEmbeddings.length > 0` was
    // truthy here too, reaching the for...of and throwing "is not iterable",
    // which rejected the whole search(). The floor must guard with
    // Array.isArray and fall through to the legacy whole-entry-vector branch.
    const day = path.join(dir, '2026-07-08');
    await fs.mkdir(day, { recursive: true });
    await fs.writeFile(path.join(day, 'nonarray.md'), '## X\n\nbody', 'utf8');
    await fs.writeFile(path.join(day, 'nonarray.embedding'), JSON.stringify({
      version: EMBEDDING_SCHEMA_VERSION, model,
      embedding: [0.1, 0.2, 0.3, 0.4, 0.5],
      sectionEmbeddings: { length: 1 },
      text: 'body', sections: ['X'],
      timestamp: Date.now(), path: path.join(day, 'nonarray.md'),
    }), 'utf8');

    const svc = new SearchService(dir, path.join(dir, 'no-user'));
    const results = await svc.search('anything', { type: 'project', minScore: -1 });

    expect(results).toHaveLength(1);
    expect(results[0].path).toContain('nonarray.md');
  });
});

describe('model-identity ceiling', () => {
  let dir: string;
  const model = EmbeddingService.getInstance().getModelName();

  const writeEntry = async (name: string, entryModel: string) => {
    const day = path.join(dir, '2026-07-08');
    await fs.mkdir(day, { recursive: true });
    await fs.writeFile(path.join(day, `${name}.md`), '## X\n\nbody', 'utf8');
    await fs.writeFile(path.join(day, `${name}.embedding`), JSON.stringify({
      version: EMBEDDING_SCHEMA_VERSION, model: entryModel,
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
