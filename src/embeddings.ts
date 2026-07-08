// ABOUTME: Local embedding service using transformers for semantic journal search
// ABOUTME: Provides text embedding generation and similarity computation utilities

import { pipeline, FeatureExtractionPipeline } from '@xenova/transformers';
import * as fs from 'fs/promises';

export interface EmbeddingData {
  embedding: number[];
  text: string;
  sections: string[];
  timestamp: number;
  path: string;
}

export class EmbeddingService {
  private static instance: EmbeddingService;
  private extractor: FeatureExtractionPipeline | null = null;
  private readonly modelName =
    process.env.PRIVATE_JOURNAL_EMBED_MODEL || 'Xenova/paraphrase-multilingual-MiniLM-L12-v2';
  private readonly quantized = process.env.PRIVATE_JOURNAL_EMBED_QUANTIZED === 'true'; // default: false
  private initPromise: Promise<void> | null = null;
  initTimeoutMs = 30_000;

  private constructor() {}

  static getInstance(): EmbeddingService {
    if (!EmbeddingService.instance) {
      EmbeddingService.instance = new EmbeddingService();
    }
    return EmbeddingService.instance;
  }

  static resetInstance(): void {
    EmbeddingService.instance = undefined as unknown as EmbeddingService;
  }

  getModelName(): string {
    return this.modelName;
  }

  async initialize(): Promise<void> {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = this.doInitialize();
    return this.initPromise;
  }

  private async doInitialize(): Promise<void> {
    let timeoutId: ReturnType<typeof setTimeout>;
    const timeoutPromise = new Promise<never>((_, reject) => {
      timeoutId = setTimeout(() => reject(new Error(
        `Model loading timed out after ${this.initTimeoutMs / 1000}s. ` +
        `This may be caused by stale lock files. ` +
        `Try: rm -rf ~/.cache/huggingface/hub/.locks/`
      )), this.initTimeoutMs);
    });
    try {
      console.error('Loading embedding model...');
      this.extractor = await Promise.race([
        pipeline('feature-extraction', this.modelName, { quantized: this.quantized }),
        timeoutPromise,
      ]);
      console.error('Embedding model loaded successfully');
    } catch (error) {
      this.initPromise = null; // Allow retry on next call
      console.error('Failed to load embedding model:', error);
      throw error;
    } finally {
      clearTimeout(timeoutId!);
    }
  }

  async generateEmbedding(text: string): Promise<number[]> {
    if (!this.extractor) {
      await this.initialize();
    }

    if (!this.extractor) {
      throw new Error('Embedding model not initialized');
    }

    try {
      const result = await this.extractor(text, { pooling: 'mean', normalize: true });
      return Array.from(result.data);
    } catch (error) {
      console.error('Failed to generate embedding:', error);
      throw error;
    }
  }

  cosineSimilarity(a: number[], b: number[]): number {
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

  async saveEmbedding(filePath: string, embeddingData: EmbeddingData): Promise<void> {
    const embeddingPath = filePath.replace(/\.md$/, '.embedding');
    await fs.writeFile(embeddingPath, JSON.stringify(embeddingData, null, 2), 'utf8');
  }

  async loadEmbedding(filePath: string): Promise<EmbeddingData | null> {
    const embeddingPath = filePath.replace(/\.md$/, '.embedding');
    
    try {
      const content = await fs.readFile(embeddingPath, 'utf8');
      return JSON.parse(content);
    } catch (error) {
      if ((error as NodeJS.ErrnoException)?.code === 'ENOENT') {
        return null; // File doesn't exist
      }
      throw error;
    }
  }

  extractSearchableText(markdownContent: string): { text: string; sections: string[] } {
    // Remove YAML frontmatter
    const withoutFrontmatter = markdownContent.replace(/^---\n.*?\n---\n/s, '');
    
    // Extract sections
    const sections: string[] = [];
    const sectionMatches = withoutFrontmatter.match(/^## (.+)$/gm);
    if (sectionMatches) {
      sections.push(...sectionMatches.map(match => match.replace('## ', '')));
    }

    // Clean up markdown for embedding
    const cleanText = withoutFrontmatter
      .replace(/^## .+$/gm, '') // Remove section headers
      .replace(/\n{3,}/g, '\n\n') // Normalize whitespace
      .trim();

    return {
      text: cleanText,
      sections
    };
  }
}