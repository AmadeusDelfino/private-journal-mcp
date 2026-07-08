// ABOUTME: Unit tests for max-over-sections ranking in SearchService
// ABOUTME: Verifies entries are scored by their best-matching section, not the whole-entry average

import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import { SearchService } from '../src/search';
import { EMBEDDING_SCHEMA_VERSION } from '../src/embeddings';

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
