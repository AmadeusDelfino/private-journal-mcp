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
