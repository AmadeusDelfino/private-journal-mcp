# Multilingual + Per-Section Embeddings for private-journal-mcp — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve semantic-search recall of the private-journal MCP by making the embedding model configurable (multilingual default), loading full-precision weights, and embedding each journal section as its own vector instead of one blob per entry.

**Architecture:** Fork of `obra/private-journal-mcp` @ `8e97230` (= tag `v2.0.1`). Changes are isolated to `src/embeddings.ts`, `src/journal.ts`, `src/search.ts`, `src/types.ts`. The embedding-config changes (model/quantized/prefix via env) are authored as clean, isolated commits so they can be proposed upstream as a small PR ("plan A′"). The per-section refactor is fork-local for now and bumps the on-disk `.embedding` schema, with automatic migration on startup.

**Tech Stack:** TypeScript (ESM), `@xenova/transformers` v2 (Transformers.js, runs ONNX on CPU), Jest + ts-jest, MCP SDK.

---

## Context — how we got here (read this first; the next session has zero memory of it)

Amadeus e Claude configuraram o `private-journal-mcp` como memória cross-project. Setup atual (já funcionando, **não mexer** a não ser pra dev):

- **MCP registrado** em `~/.claude.json` (escopo user):
  - `command: npx`, `args: ["github:obra/private-journal-mcp#8e97230e827496998427f093aa1208831acac3fe"]` (pinado em v2.0.1 por segurança de supply chain).
  - `env.PRIVATE_JOURNAL_PATH = ~/.private_journal`
  - `env.NODE_EXTRA_CA_CERTS = ~/certs/corporate-ca.pem`
- **Journal em disco:** `~/.private_journal/<YYYY-MM-DD>/<HH-MM-SS-micros>.md` + um `.embedding` (JSON) ao lado de cada `.md`.

### Gotcha de ambiente (CRÍTICO pra este repo também)
A máquina está atrás de **um proxy corporativo com inspeção de TLS**. `git`/`curl` funcionam (usam o keychain do macOS); **Node/npm NÃO usam o keychain** → `npm ci`, `npm install` e o download de pesos do modelo pelo Transformers.js quebram com `SELF_SIGNED_CERT_IN_CHAIN` a menos que você aponte o Node pra CA:

```bash
export NODE_EXTRA_CA_CERTS="$HOME/certs/corporate-ca.pem"
```

Exporte isso no shell **antes** de qualquer `npm`/`node` neste repo. (Também já está no `~/.zshrc:57`, mas nem todo contexto herda o `.zshrc`.)

### Por que estamos forkando (achados da leitura do fonte @ v2.0.1)
Gargalos de **qualidade de retrieval** (não de performance — o hardware está ocioso; Transformers.js roda em CPU e um journal é minúsculo). Em ordem de impacto:

1. **Um vetor por entrada inteira.** `JournalManager.generateEmbeddingForEntry` (`src/journal.ts:151`) chama `extractSearchableText` (`src/embeddings.ts:128`), que junta TODAS as seções (`Reflections` + `User Context` + `Technical Insights`…) num texto só, e gera **um** embedding mean-pooled (`src/embeddings.ts:79`). Média de tópicos distintos → dilui qualquer query específica. **Maior alavanca.**
2. **Modelo quantizado (int8) por padrão.** `pipeline('feature-extraction', modelName)` (`src/embeddings.ts:56`) sem opções → Transformers.js v2 carrega `model_quantized.onnx`. Menor fidelidade.
3. **Modelo English-centric.** `modelName = 'Xenova/all-MiniLM-L6-v2'` (`src/embeddings.ts:18`), hardcoded. Amadeus escreve em **português** → busca semântica em PT e cross-lingual PT↔EN sofre.
4. **Teto de 256 tokens** do MiniLM → entradas longas truncadas.

Modelos ONNX multilíngues confirmados no HF (todos com `model.onnx` full-precision, verificado via HF API):
`Xenova/paraphrase-multilingual-MiniLM-L12-v2` (384-dim, **sem prefixo**), `Xenova/multilingual-e5-base` e `-small` (768/384-dim, prefixos `query:`/`passage:`), `Xenova/bge-m3` (1024-dim, 8k ctx), `Xenova/gte-base`.

**Decisão de default:** `Xenova/paraphrase-multilingual-MiniLM-L12-v2` — multilíngue, 384-dim (mesma dimensão do atual → migração trivial), e **não exige prefixo** (correção sem risco). Suporte a prefixo fica implementado e configurável pra quem quiser trocar por e5/bge.

### Escopo desta sessão
Fazer **A** (o fork, este plano) e depois **A′** (PR upstream com o subconjunto model/quantized/prefix). **Sem Ollama** (o C/reranker fica pra outro dia). A′ é ação externa (PR): **não abrir sem o OK explícito do Amadeus.**

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/embeddings.ts` | Model loading, embedding generation, `.embedding` I/O, markdown parsing | Model/quantized/prefix via env; `generateEmbedding(text, kind)`; per-section extraction; new `EmbeddingData` shape + `EMBEDDING_SCHEMA_VERSION`; `getModelName()` |
| `src/journal.ts` | Writing entries, generating/saving embeddings, missing-embedding scan | Generate per-section embeddings; migrate stale `.embedding` on scan |
| `src/search.ts` | Load embeddings, score vs query, rank | Score = best section (max-over-sections); report `matchedSection`; fallback to whole-entry vector for legacy files |
| `src/types.ts` | Shared interfaces | (unchanged; `EmbeddingData` lives in `embeddings.ts`) |
| `tests/embeddings.test.ts` | Unit tests | Add tests for config, prefixes, per-section extraction |
| `tests/search.test.ts` | Unit tests (NEW) | Max-over-sections ranking, legacy fallback |

**Testing note (important):** `tests/setup.ts` mocks `@xenova/transformers` so `pipeline` returns a fixed 5-dim vector regardless of input. Therefore unit tests assert **structure and call-args** (e.g. "pipeline was called with `{ quantized: false }`", "extractor received the `query: `-prefixed string", "an entry with 3 sections produced 3 section-embeddings"), **not** semantic quality. Real retrieval quality (does multilingual improve PT recall?) is validated **manually** in Task 8 against the real model.

`EmbeddingService` is a singleton that reads `process.env` at construction. Any test that changes an env var MUST call `EmbeddingService.resetInstance()` afterward (method already exists at `src/embeddings.ts:31`).

---

## Task 1: Fork setup & green baseline

**Files:** none modified (setup only)

- [ ] **Step 1: Fork + clone**

```bash
cd ~/IdeaProjects
gh repo fork obra/private-journal-mcp --clone=true --remote=true
cd private-journal-mcp
git checkout -b feat/multilingual-per-section-embeddings
```

- [ ] **Step 2: Install deps (needs the corporate CA)**

```bash
export NODE_EXTRA_CA_CERTS="$HOME/certs/corporate-ca.pem"
npm ci
```
Expected: install completes without `SELF_SIGNED_CERT_IN_CHAIN`.

- [ ] **Step 3: Build + baseline tests must be green BEFORE any change**

```bash
npm run build
npm test
```
Expected: `tsc` succeeds; Jest passes (existing suites: `embeddings`, `journal`, `paths`).

- [ ] **Step 4: Move this plan into the fork and commit it**

```bash
mkdir -p docs/superpowers/plans
mv ~/private-journal-mcp-fork-plan.md docs/superpowers/plans/2026-07-08-multilingual-per-section-embeddings.md
git add docs/superpowers/plans/2026-07-08-multilingual-per-section-embeddings.md
git commit -m "docs: add multilingual + per-section embeddings plan"
```

- [ ] **Step 5: Wire the LOCAL build into Claude Code for dogfooding (separate dev entry, keep prod pinned)**

```bash
claude mcp add-json private-journal-dev '{"type":"stdio","command":"node","args":["'"$PWD"'/dist/index.js"],"env":{"PRIVATE_JOURNAL_PATH":"/path/to/.private_journal","NODE_EXTRA_CA_CERTS":"/path/to/corporate-ca.pem"}}' -s user
claude mcp get private-journal-dev
```
Expected: `✓ Connected`. Note: `.md` files are the source of truth; `.embedding` files are derived and safe to delete/regenerate. For extra safety during dev you may point `PRIVATE_JOURNAL_PATH` at a copy of `~/.private_journal`.

---

## Task 2: Configurable model + full-precision (`quantized: false`)

**Files:**
- Modify: `src/embeddings.ts:18` (modelName), `src/embeddings.ts:56` (pipeline call)
- Test: `tests/embeddings.test.ts`

- [ ] **Step 1: Write failing tests**

Add to `tests/embeddings.test.ts` (inside the top-level `describe`):

```ts
import { pipeline } from '@xenova/transformers';

describe('embedding model configuration', () => {
  const ENV_KEYS = ['PRIVATE_JOURNAL_EMBED_MODEL', 'PRIVATE_JOURNAL_EMBED_QUANTIZED'];
  let saved: Record<string, string | undefined>;

  beforeEach(() => {
    saved = Object.fromEntries(ENV_KEYS.map(k => [k, process.env[k]]));
    ENV_KEYS.forEach(k => delete process.env[k]);
    (pipeline as jest.Mock).mockClear();
    EmbeddingService.resetInstance();
  });

  afterEach(() => {
    ENV_KEYS.forEach(k => { if (saved[k] === undefined) delete process.env[k]; else process.env[k] = saved[k]!; });
    EmbeddingService.resetInstance();
  });

  test('defaults to the multilingual model, full precision', async () => {
    await EmbeddingService.getInstance().generateEmbedding('hi');
    expect(pipeline).toHaveBeenCalledWith(
      'feature-extraction',
      'Xenova/paraphrase-multilingual-MiniLM-L12-v2',
      { quantized: false }
    );
  });

  test('honors PRIVATE_JOURNAL_EMBED_MODEL and PRIVATE_JOURNAL_EMBED_QUANTIZED', async () => {
    process.env.PRIVATE_JOURNAL_EMBED_MODEL = 'Xenova/multilingual-e5-base';
    process.env.PRIVATE_JOURNAL_EMBED_QUANTIZED = 'true';
    EmbeddingService.resetInstance();
    await EmbeddingService.getInstance().generateEmbedding('hi');
    expect(pipeline).toHaveBeenCalledWith(
      'feature-extraction',
      'Xenova/multilingual-e5-base',
      { quantized: true }
    );
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test -- tests/embeddings.test.ts`
Expected: FAIL (pipeline called with 2 args / wrong model name).

- [ ] **Step 3: Implement**

In `src/embeddings.ts`, replace the hardcoded field (line 18) and the pipeline call (line 56):

```ts
  private readonly modelName =
    process.env.PRIVATE_JOURNAL_EMBED_MODEL || 'Xenova/paraphrase-multilingual-MiniLM-L12-v2';
  private readonly quantized = process.env.PRIVATE_JOURNAL_EMBED_QUANTIZED === 'true'; // default: false
```

```ts
      this.extractor = await Promise.race([
        pipeline('feature-extraction', this.modelName, { quantized: this.quantized }),
        timeoutPromise,
      ]);
```

Also add a public accessor (used later by migration):

```ts
  getModelName(): string {
    return this.modelName;
  }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `npm test -- tests/embeddings.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/embeddings.ts tests/embeddings.test.ts
git commit -m "feat(embeddings): configurable model via env, default multilingual, full-precision"
```

---

## Task 3: Query/document prefixes (enables e5/bge models)

**Files:**
- Modify: `src/embeddings.ts` (`generateEmbedding` signature + prefix logic)
- Modify: `src/search.ts:59` (pass `'query'`), `src/journal.ts:163` (pass `'document'`)
- Test: `tests/embeddings.test.ts`

- [ ] **Step 1: Write failing test**

Add inside the `describe('embedding model configuration', ...)` block:

```ts
test('applies query/document prefixes from env', async () => {
  process.env.PRIVATE_JOURNAL_EMBED_QUERY_PREFIX = 'query: ';
  process.env.PRIVATE_JOURNAL_EMBED_DOC_PREFIX = 'passage: ';
  EmbeddingService.resetInstance();

  // Capture the text passed to the extractor
  const calls: string[] = [];
  (pipeline as jest.Mock).mockResolvedValueOnce(
    jest.fn(async (text: string) => { calls.push(text); return { data: new Float32Array([0.1, 0.2, 0.3]) }; })
  );

  const svc = EmbeddingService.getInstance();
  await svc.generateEmbedding('hello', 'query');
  await svc.generateEmbedding('world', 'document');

  expect(calls).toEqual(['query: hello', 'passage: world']);
});
```
Remember to add the two prefix keys to `ENV_KEYS` in this describe block so they get cleaned up.

- [ ] **Step 2: Run to verify fail**

Run: `npm test -- tests/embeddings.test.ts`
Expected: FAIL (prefixes not applied; `generateEmbedding` has no `kind` param).

- [ ] **Step 3: Implement**

In `src/embeddings.ts`, add fields near `quantized`:

```ts
  private readonly queryPrefix = process.env.PRIVATE_JOURNAL_EMBED_QUERY_PREFIX ?? '';
  private readonly docPrefix = process.env.PRIVATE_JOURNAL_EMBED_DOC_PREFIX ?? '';
```

Change `generateEmbedding` (line 69) to:

```ts
  async generateEmbedding(text: string, kind: 'query' | 'document' = 'document'): Promise<number[]> {
    if (!this.extractor) {
      await this.initialize();
    }
    if (!this.extractor) {
      throw new Error('Embedding model not initialized');
    }
    const prefix = kind === 'query' ? this.queryPrefix : this.docPrefix;
    const input = prefix ? prefix + text : text;
    try {
      const result = await this.extractor(input, { pooling: 'mean', normalize: true });
      return Array.from(result.data as Float32Array);
    } catch (error) {
      console.error('Failed to generate embedding:', error);
      throw error;
    }
  }
```

In `src/search.ts:59` change to `const queryEmbedding = await this.embeddingService.generateEmbedding(query, 'query');`

In `src/journal.ts:163` change the whole-entry call to `const embedding = await this.embeddingService.generateEmbedding(text, 'document');` (the per-section calls added in Task 5 also use `'document'`).

- [ ] **Step 4: Run tests**

Run: `npm test -- tests/embeddings.test.ts`
Expected: PASS (and the existing embedding/search tests still pass — default prefixes are empty, so behavior is unchanged for the default model).

- [ ] **Step 5: Commit**

```bash
git add src/embeddings.ts src/search.ts src/journal.ts tests/embeddings.test.ts
git commit -m "feat(embeddings): optional query/document prefixes for asymmetric models (e5/bge)"
```

---

## Task 4: Per-section extraction (`sectionChunks`)

**Files:**
- Modify: `src/embeddings.ts:128` (`extractSearchableText` return shape — additive)
- Test: `tests/embeddings.test.ts`

- [ ] **Step 1: Write failing test**

```ts
test('extractSearchableText returns per-section chunks (headers + bodies)', () => {
  const md = `---
title: "x"
---

## Reflections

alpha reflection body

## Technical Insights

beta insight body`;
  const { text, sections, sectionChunks } = EmbeddingService.getInstance().extractSearchableText(md);
  expect(sections).toEqual(['Reflections', 'Technical Insights']);
  expect(text).toContain('alpha reflection body');
  expect(sectionChunks).toEqual([
    { section: 'Reflections', body: 'alpha reflection body' },
    { section: 'Technical Insights', body: 'beta insight body' },
  ]);
});
```

- [ ] **Step 2: Run to verify fail**

Run: `npm test -- tests/embeddings.test.ts`
Expected: FAIL (`sectionChunks` is undefined).

- [ ] **Step 3: Implement**

Replace `extractSearchableText` (lines 128-149) with (keeps `text` and `sections` identical to before — existing test at line 55 must still pass):

```ts
  extractSearchableText(markdownContent: string): {
    text: string;
    sections: string[];
    sectionChunks: { section: string; body: string }[];
  } {
    const withoutFrontmatter = markdownContent.replace(/^---\n.*?\n---\n/s, '');

    const sections: string[] = [];
    const sectionMatches = withoutFrontmatter.match(/^## (.+)$/gm);
    if (sectionMatches) {
      sections.push(...sectionMatches.map(match => match.replace('## ', '')));
    }

    // Split into (header, body) chunks
    const sectionChunks: { section: string; body: string }[] = [];
    const parts = withoutFrontmatter.split(/^## (.+)$/gm); // [pre, name1, body1, name2, body2, ...]
    for (let i = 1; i < parts.length; i += 2) {
      const section = parts[i].trim();
      const body = (parts[i + 1] ?? '').replace(/\n{3,}/g, '\n\n').trim();
      if (body.length > 0) {
        sectionChunks.push({ section, body });
      }
    }

    const cleanText = withoutFrontmatter
      .replace(/^## .+$/gm, '')
      .replace(/\n{3,}/g, '\n\n')
      .trim();

    return { text: cleanText, sections, sectionChunks };
  }
```

- [ ] **Step 4: Run tests**

Run: `npm test -- tests/embeddings.test.ts`
Expected: PASS (new test + the pre-existing `extracts searchable text from markdown` test).

- [ ] **Step 5: Commit**

```bash
git add src/embeddings.ts tests/embeddings.test.ts
git commit -m "feat(embeddings): parse per-section chunks in extractSearchableText"
```

---

## Task 5: Generate & store per-section embeddings (schema v2 + migration hook)

**Files:**
- Modify: `src/embeddings.ts` (`EmbeddingData` interface + `EMBEDDING_SCHEMA_VERSION` + `SectionEmbedding`)
- Modify: `src/journal.ts:151` (`generateEmbeddingForEntry`)
- Test: `tests/embeddings.test.ts`

- [ ] **Step 1: Write failing test**

```ts
test('writing an entry stores one embedding per non-empty section', async () => {
  await journalManager.writeThoughts({
    reflections: 'reflection body',
    technical_insights: 'insight body',
  });
  // find the .embedding file under the user temp dir
  const userRoot = path.join(userTempDir, '.private-journal');
  const day = (await fs.readdir(userRoot)).find(d => /^\d{4}-\d{2}-\d{2}$/.test(d))!;
  const embFile = (await fs.readdir(path.join(userRoot, day))).find(f => f.endsWith('.embedding'))!;
  const data = JSON.parse(await fs.readFile(path.join(userRoot, day, embFile), 'utf8'));

  expect(data.version).toBe(2);
  expect(typeof data.model).toBe('string');
  expect(data.sectionEmbeddings.map((s: any) => s.section)).toEqual(['Reflections', 'Technical Insights']);
  expect(Array.isArray(data.sectionEmbeddings[0].embedding)).toBe(true);
});
```

- [ ] **Step 2: Run to verify fail**

Run: `npm test -- tests/embeddings.test.ts`
Expected: FAIL (`version`/`sectionEmbeddings` undefined).

- [ ] **Step 3: Implement**

In `src/embeddings.ts`, add near the top (after imports):

```ts
export const EMBEDDING_SCHEMA_VERSION = 2;

export interface SectionEmbedding {
  section: string;
  text: string;
  embedding: number[];
}
```

Extend `EmbeddingData`:

```ts
export interface EmbeddingData {
  version: number;                       // EMBEDDING_SCHEMA_VERSION
  model: string;                         // model that produced these vectors
  embedding: number[];                   // whole-entry vector (legacy/fallback)
  sectionEmbeddings: SectionEmbedding[];
  text: string;
  sections: string[];
  timestamp: number;
  path: string;
}
```

In `src/journal.ts`, update the import (line 7) to include the new symbols, then replace `generateEmbeddingForEntry` (lines 151-178):

```ts
  private async generateEmbeddingForEntry(
    filePath: string,
    content: string,
    timestamp: Date
  ): Promise<void> {
    try {
      const { text, sections, sectionChunks } = this.embeddingService.extractSearchableText(content);
      if (text.trim().length === 0) {
        return;
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
    } catch (error) {
      console.error(`Failed to generate embedding for ${filePath}:`, error);
    }
  }
```

Import line (`src/journal.ts:7`) becomes:

```ts
import { EmbeddingService, EmbeddingData, SectionEmbedding, EMBEDDING_SCHEMA_VERSION } from './embeddings.js';
```

- [ ] **Step 4: Run tests**

Run: `npm test -- tests/embeddings.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/embeddings.ts src/journal.ts tests/embeddings.test.ts
git commit -m "feat(embeddings): store per-section vectors (schema v2 with model + version)"
```

---

## Task 6: Rank by best-matching section (max-over-sections)

**Files:**
- Modify: `src/search.ts` (`SearchResult` + scoring in `search`)
- Test: `tests/search.test.ts` (NEW)

- [ ] **Step 1: Write failing test**

Create `tests/search.test.ts`:

```ts
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
});
```

- [ ] **Step 2: Run to verify fail**

Run: `npm test -- tests/search.test.ts`
Expected: FAIL (`matchedSection` undefined; scoring uses whole-entry vector).

- [ ] **Step 3: Implement**

In `src/search.ts`, add `matchedSection?: string;` to the `SearchResult` interface (after line 16).

Add a private helper to `SearchService`:

```ts
  private scoreEntry(
    queryEmbedding: number[],
    entry: EmbeddingData
  ): { score: number; matchedSection?: string } {
    if (entry.sectionEmbeddings && entry.sectionEmbeddings.length > 0) {
      let best = -Infinity;
      let matchedSection: string | undefined;
      for (const se of entry.sectionEmbeddings) {
        const s = this.embeddingService.cosineSimilarity(queryEmbedding, se.embedding);
        if (s > best) { best = s; matchedSection = se.section; }
      }
      return { score: best, matchedSection };
    }
    // Legacy fallback: whole-entry vector
    return { score: this.embeddingService.cosineSimilarity(queryEmbedding, entry.embedding) };
  }
```

Replace the `.map` scoring block inside `search` (lines 97-111) so it uses the helper:

```ts
    const results: SearchResult[] = filtered
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
```

(Optional polish: in `src/server.ts` where results are formatted (~lines 199/254), append `matchedSection` to the output line so the model sees which section matched.)

- [ ] **Step 4: Run tests**

Run: `npm test`
Expected: PASS (new search test + all existing suites; legacy entries without `sectionEmbeddings` still work via fallback).

- [ ] **Step 5: Commit**

```bash
git add src/search.ts tests/search.test.ts
git commit -m "feat(search): rank entries by best-matching section, report matchedSection"
```

---

## Task 7: Auto-migrate stale embeddings on startup

**Files:**
- Modify: `src/journal.ts:180` (`generateMissingEmbeddings`)
- Test: `tests/embeddings.test.ts`

- [ ] **Step 1: Write failing test**

```ts
test('regenerates embeddings whose model/version is stale', async () => {
  await journalManager.writeThoughts({ observations: 'obs body' });
  const userRoot = path.join(userTempDir, '.private-journal');
  const day = (await fs.readdir(userRoot)).find(d => /^\d{4}-\d{2}-\d{2}$/.test(d))!;
  const embPath = path.join(userRoot, day, (await fs.readdir(path.join(userRoot, day))).find(f => f.endsWith('.embedding'))!);

  // Simulate an old-format file (v1: no version/model/sectionEmbeddings)
  await fs.writeFile(embPath, JSON.stringify({ embedding: [0, 0, 0], text: 'obs body', sections: ['Observations'], timestamp: Date.now(), path: embPath.replace('.embedding', '.md') }), 'utf8');

  const count = await journalManager.generateMissingEmbeddings();
  expect(count).toBe(1);
  const migrated = JSON.parse(await fs.readFile(embPath, 'utf8'));
  expect(migrated.version).toBe(2);
  expect(migrated.sectionEmbeddings.length).toBeGreaterThan(0);
});
```

- [ ] **Step 2: Run to verify fail**

Run: `npm test -- tests/embeddings.test.ts`
Expected: FAIL (current scan only checks file existence → count 0, file unchanged).

- [ ] **Step 3: Implement**

In `src/journal.ts`, inside `generateMissingEmbeddings`, replace the `try { await fs.access(embeddingPath); } catch { ...generate... }` block (lines 203-213) with a version/model check:

```ts
            let needsRegen = false;
            try {
              const raw = await fs.readFile(embeddingPath, 'utf8');
              const existing = JSON.parse(raw);
              if (existing.version !== EMBEDDING_SCHEMA_VERSION ||
                  existing.model !== this.embeddingService.getModelName()) {
                needsRegen = true;
              }
            } catch {
              needsRegen = true; // missing or unreadable
            }

            if (needsRegen) {
              console.error(`Generating/refreshing embedding for ${mdPath}`);
              const content = await fs.readFile(mdPath, 'utf8');
              const timestamp = this.extractTimestampFromPath(mdPath) || new Date();
              await this.generateEmbeddingForEntry(mdPath, content, timestamp);
              count++;
            }
```

- [ ] **Step 4: Run tests**

Run: `npm test`
Expected: PASS (full suite).

- [ ] **Step 5: Lint + commit**

```bash
npm run lint
git add src/journal.ts tests/embeddings.test.ts
git commit -m "feat(journal): auto-regenerate stale embeddings (model/version mismatch)"
```

---

## Task 8: Manual validation against the real model (no unit test — model is mocked)

**Files:** none

- [ ] **Step 1: Build**

```bash
export NODE_EXTRA_CA_CERTS="$HOME/certs/corporate-ca.pem"
npm run build
```

- [ ] **Step 2: Force a re-index of the real journal**

The startup scan (Task 7) will regenerate anything stale, but to be explicit you can wipe derived vectors (`.md` are untouched):

```bash
find ~/.private_journal -name '*.embedding' -delete
```

- [ ] **Step 3: Restart Claude Code, then exercise `private-journal-dev`**

Ask Claude (in a session using the dev server) to run several **Portuguese** searches that paraphrase existing entries (do NOT reuse the entry's exact words), e.g. a PT paraphrase of an existing technical note. Confirm:
- The right entry ranks #1.
- `matchedSection` points at the relevant section.
- First run logs `Generating/refreshing embedding ...` (migration) and downloads the model once (needs the CA).

- [ ] **Step 4: Record results**

Append a short "Validation results" note to this plan file (scores/rankings before vs after, model used). Commit:

```bash
git add docs/superpowers/plans/2026-07-08-multilingual-per-section-embeddings.md
git commit -m "docs: record retrieval validation results"
```

- [ ] **Step 5 (optional): try a heavier model**

To A/B a stronger model, set on the dev server env and re-index:
`PRIVATE_JOURNAL_EMBED_MODEL=Xenova/multilingual-e5-base`, `PRIVATE_JOURNAL_EMBED_QUERY_PREFIX="query: "`, `PRIVATE_JOURNAL_EMBED_DOC_PREFIX="passage: "`.

---

## Task 9 (Plan A′): Prepare the upstream PR — DO NOT OPEN WITHOUT AMADEUS'S GO-AHEAD

**Scope of the PR:** only the broadly-useful, low-risk commits — Task 2 (configurable model + `quantized`) and Task 3 (prefixes). The per-section refactor (Tasks 4-7) changes the on-disk schema and is a bigger discussion; propose it separately or open an issue first.

- [ ] **Step 1: Branch off upstream main and cherry-pick the two commits**

```bash
git fetch upstream
git checkout -b upstream-configurable-model upstream/main
git cherry-pick <task2-sha> <task3-sha>   # get SHAs from `git log --oneline`
npm ci && npm run build && npm test && npm run lint
```

- [ ] **Step 2: Draft PR description**

Write `docs/pr-body.md`: what (env-configurable model, full-precision default toggle, optional query/doc prefixes), why (multilingual/PT recall; MiniLM is English-only and quantized by default), backward-compat (defaults keep English users working; new envs are opt-in EXCEPT the default model — call this out and offer to keep `all-MiniLM-L6-v2` as the default if the maintainer prefers).

- [ ] **Step 3: STOP. Ask Amadeus to approve opening the PR (external action).**

Only after explicit approval:

```bash
git push -u origin upstream-configurable-model
gh pr create --repo obra/private-journal-mcp --base main --head <your-fork>:upstream-configurable-model --title "Configurable embedding model (multilingual) + full-precision + prefixes" --body-file docs/pr-body.md
```

---

## Validation results (Task 8 — executed 2026-07-08)

**Method (adaptação conservadora):** em vez de apagar os `.embedding` do journal real, a validação rodou contra **cópias** de `~/.private_journal` (2 entradas, ambas em PT), via um driver MCP stdio (`initialize` → `tools/call search_journal`, `limit: 5, type: 'both'`, minScore default 0.1). "Antes" = v2.0.1 (`8e97230`) buildada em worktree isolado, embeddings v1 existentes (all-MiniLM-L6-v2, quantized, vetor único). "Depois" = esta branch (`9a26019`): na inicialização o scan migrou os dois arquivos v1 → v2 automaticamente (log `Generating/refreshing embedding for ...` para ambos; arquivo final: `version: 2`, `model: Xenova/paraphrase-multilingual-MiniLM-L12-v2`, sectionEmbeddings por seção, 384 dims). O journal real permanece v1 e migra sozinho no primeiro start do `private-journal-dev` (nenhuma deleção necessária — Task 7 cobre).

**Queries (paráfrases, sem reusar palavras exatas das entradas):**

| # | Query | Antes (v2.0.1) | Depois (fork) |
|---|---|---|---|
| 1 | paráfrase PT de uma nota técnica (A) | ✓ entrada certa #1, score 0.289 | ✓ certa #1, **0.506**, matched `Observations` |
| 2 | paráfrase PT de uma nota de trabalho (B) | ✗ entrada ERRADA #1 (0.201); a certa ficou abaixo do minScore | ✓ certa #1, **0.441**, matched `Observations` (a seção que contém a nota correspondente) |
| 3 | paráfrase PT de uma reflexão de processo (C) | ✗ entrada ERRADA #1 (0.224); a certa filtrada | ✓ certa #1, **0.405**, matched `Reflections` (exatamente a seção da reflexão) |
| 4 | query EN para a nota técnica (A) (EN→PT cross-lingual) | ✓ certa #1, 0.360 | ✓ certa #1, **0.492**, matched `Observations` |

**Leitura:** recall passou de 2/4 para **4/4**; separação certa-vs-errada ficou muito mais discriminativa (ex.: Q4: 0.492 vs 0.102). O `matchedSection` apontou a seção relevante em todos os casos. **Decisão:** o default `paraphrase-multilingual-MiniLM-L12-v2` é suficiente — não trocar para `multilingual-e5-base` (Step 5 opcional não executado).

**Caveats observados:**
- Q3 venceu por margem apertada (0.405 vs 0.385 da entrada errada) — com corpus de 2 entradas é sinal fraco; reavaliar quando o journal crescer.
- **Quirk pré-existente (também na v2.0.1 e no prod):** com `PRIVATE_JOURNAL_PATH` setado, project path e user path resolvem para o MESMO diretório → cada entrada aparece duplicada (uma como `project`, outra como `user`) nos resultados. Fora do escopo deste plano; candidato a fix futuro (dedup por realpath).
- Modelo full-precision (465MB) cacheado em `<repo>/node_modules/@xenova/transformers/.cache/` — o `private-journal-dev` reusa; um futuro `npm ci` apaga o cache e re-baixa (precisa de `NODE_EXTRA_CA_CERTS`).

---

## Open questions / decisions to confirm during execution

- **Default model:** plan uses `paraphrase-multilingual-MiniLM-L12-v2` (no prefix, 384-dim). If PT recall still feels weak in Task 8, switch default to `multilingual-e5-base` (+ prefixes) — better quality, 768-dim, needs re-index.
- **Upstream default:** the PR changes the default model. Maintainer may want the default to stay `all-MiniLM-L6-v2` with multilingual as opt-in. Be ready to concede that in review.
- **`bge-m3` (1024-dim, 8k ctx)** is the "max quality multilingual" option but heavier (has `.onnx_data` external weights) — try only if you want to push it.
- **Excerpt/`generateExcerpt`** still uses whole-entry `text`; fine to leave. Could later switch to the matched section's body.
