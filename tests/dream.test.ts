// ABOUTME: Tests for dream entries - the dream marker on embeddings and the writeDream pipeline
// ABOUTME: Dream entries are searchable like any entry but never feed the recurrence corpus

import * as fs from 'fs/promises';
import * as path from 'path';
import * as os from 'os';
import { JournalManager } from '../src/journal';
import { SearchService } from '../src/search';
import { EmbeddingService } from '../src/embeddings';
import { findRecurringThemes, gatherThemeChunks } from '../src/recurrence';

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
