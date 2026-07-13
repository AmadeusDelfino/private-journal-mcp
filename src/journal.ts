// ABOUTME: Core journal writing functionality for MCP server
// ABOUTME: Handles file system operations, timestamps, and markdown formatting

import * as fs from 'fs/promises';
import * as path from 'path';
import { resolveUserJournalPath } from './paths.js';
import { EmbeddingService, EmbeddingData, SectionEmbedding, EMBEDDING_SCHEMA_VERSION } from './embeddings.js';

export class JournalManager {
  private projectJournalPath: string;
  private userJournalPath: string;
  private embeddingService: EmbeddingService;

  constructor(projectJournalPath: string, userJournalPath?: string) {
    this.projectJournalPath = projectJournalPath;
    this.userJournalPath = userJournalPath || resolveUserJournalPath();
    this.embeddingService = EmbeddingService.getInstance();
  }

  async writeThoughts(thoughts: {
    reflections?: string;
    observations?: string;
    project_notes?: string;
    user_context?: string;
    technical_insights?: string;
    world_knowledge?: string;
  }): Promise<void> {
    const timestamp = new Date();
    
    // Split thoughts into project-local and user-global
    const projectThoughts = { project_notes: thoughts.project_notes };
    const userThoughts = {
      reflections: thoughts.reflections,
      observations: thoughts.observations,
      user_context: thoughts.user_context,
      technical_insights: thoughts.technical_insights,
      world_knowledge: thoughts.world_knowledge
    };
    
    // Write project notes to project directory
    if (projectThoughts.project_notes) {
      await this.writeThoughtsToLocation(projectThoughts, timestamp, this.projectJournalPath);
    }
    
    // Write user thoughts to user directory
    const hasUserContent = Object.values(userThoughts).some(value => value !== undefined);
    if (hasUserContent) {
      await this.writeThoughtsToLocation(userThoughts, timestamp, this.userJournalPath);
    }
  }

  private formatDate(date: Date): string {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
  }

  private formatTimestamp(date: Date): string {
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');
    const microseconds = String(date.getMilliseconds() * 1000 + Math.floor(Math.random() * 1000)).padStart(6, '0');
    return `${hours}-${minutes}-${seconds}-${microseconds}`;
  }

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

  private async writeThoughtsToLocation(
    thoughts: {
      reflections?: string;
      observations?: string;
      project_notes?: string;
      user_context?: string;
      technical_insights?: string;
      world_knowledge?: string;
    },
    timestamp: Date,
    basePath: string
  ): Promise<void> {
    const dateString = this.formatDate(timestamp);
    const timeString = this.formatTimestamp(timestamp);
    
    const dayDirectory = path.join(basePath, dateString);
    const fileName = `${timeString}.md`;
    const filePath = path.join(dayDirectory, fileName);

    await this.ensureDirectoryExists(dayDirectory);
    
    const formattedEntry = this.formatThoughts(thoughts, timestamp);
    await fs.writeFile(filePath, formattedEntry, 'utf8');

    // Generate and save embedding
    await this.generateEmbeddingForEntry(filePath, formattedEntry, timestamp);
  }

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

  private formatThoughts(thoughts: {
    reflections?: string;
    observations?: string;
    project_notes?: string;
    user_context?: string;
    technical_insights?: string;
    world_knowledge?: string;
  }, timestamp: Date): string {
    const sections = [];

    if (thoughts.reflections) {
      sections.push(`## Reflections\n\n${thoughts.reflections}`);
    }

    if (thoughts.observations) {
      sections.push(`## Observations\n\n${thoughts.observations}`);
    }

    if (thoughts.project_notes) {
      sections.push(`## Project Notes\n\n${thoughts.project_notes}`);
    }

    if (thoughts.user_context) {
      sections.push(`## User Context\n\n${thoughts.user_context}`);
    }

    if (thoughts.technical_insights) {
      sections.push(`## Technical Insights\n\n${thoughts.technical_insights}`);
    }

    if (thoughts.world_knowledge) {
      sections.push(`## World Knowledge\n\n${thoughts.world_knowledge}`);
    }

    return `${this.formatFrontmatter(timestamp)}\n${sections.join('\n\n')}\n`;
  }

  private async regenerateEmbedding(
    filePath: string,
    content: string,
    timestamp: Date
  ): Promise<boolean> {
    const { text, sections, sectionChunks } = this.embeddingService.extractSearchableText(content);

    if (text.trim().length === 0) {
      // Regen can never rewrite an .embedding for an empty source, so a stale
      // one would stay foreign forever (re-flagged by every scan, named by the
      // search skip log). Unrecoverable: delete the orphan (no-op when absent).
      await fs.rm(filePath.replace(/\.md$/, '.embedding'), { force: true });
      return false; // nothing to embed; not a success, not an error
    }

    // The .md front-matter is the durable dream marker; mirror it onto the
    // .embedding so the recurrence engine can filter without reading .md
    // files. Re-derived on every regen so the flag survives re-indexing.
    const frontmatter = content.match(/^---\n([\s\S]*?)\n---\n/);
    const isDream = frontmatter !== null && /^dream: true$/m.test(frontmatter[1]);

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
      ...(isDream ? { dream: true } : {}),
    };

    await this.embeddingService.saveEmbedding(filePath, embeddingData);
    return true;
  }

  private async generateEmbeddingForEntry(
    filePath: string,
    content: string,
    timestamp: Date
  ): Promise<void> {
    try {
      await this.regenerateEmbedding(filePath, content, timestamp);
    } catch (error) {
      // Don't throw - embedding failure shouldn't prevent journal writing
      console.error(`Failed to generate embedding for ${filePath}:`, error);
    }
  }

  async generateMissingEmbeddings(): Promise<number> {
    let count = 0;
    let modelReady = false;
    const paths = [this.projectJournalPath, this.userJournalPath];

    for (const basePath of paths) {
      try {
        const dayDirs = await fs.readdir(basePath);

        for (const dayDir of dayDirs) {
          const dayPath = path.join(basePath, dayDir);
          const stat = await fs.stat(dayPath);

          if (!stat.isDirectory() || !dayDir.match(/^\d{4}-\d{2}-\d{2}$/)) {
            continue;
          }

          const files = await fs.readdir(dayPath);
          const mdFiles = files.filter(file => file.endsWith('.md'));

          for (const mdFile of mdFiles) {
            try {
              const mdPath = path.join(dayPath, mdFile);
              const embeddingPath = mdPath.replace(/\.md$/, '.embedding');

              let needsRegen = false;
              try {
                const raw = await fs.readFile(embeddingPath, 'utf8');
                const existing = JSON.parse(raw);
                // Structural check covers the whole-entry vector AND every
                // section vector; a same-model wrong-length vector is not
                // detectable here (needs the model's dimension) — accepted.
                const vectorsOk = Array.isArray(existing.embedding) &&
                  Array.isArray(existing.sectionEmbeddings) &&
                  existing.sectionEmbeddings.every(
                    (se: { embedding?: unknown } | null | undefined) => Array.isArray(se?.embedding)
                  );
                if (!this.embeddingService.isCompatible(existing) || !vectorsOk) {
                  needsRegen = true;
                }
              } catch {
                needsRegen = true; // missing or unreadable
              }

              if (!needsRegen) {
                continue;
              }

              if (!modelReady) {
                // Generous timeout: the query default (30s) is too short to
                // download a cold ~465MB model, and transformers.js neither
                // resumes nor dedupes, so a short retry would only start a
                // second concurrent download. One long attempt or bust.
                this.embeddingService.initTimeoutMs = 120_000;
                try {
                  await this.embeddingService.initialize();
                  modelReady = true;
                } catch (error) {
                  console.error('embedding model unavailable — aborting re-index:', error);
                  return count; // abort the whole scan; report real successes so far
                }
              }

              console.error(`Generating/refreshing embedding for ${mdPath}`);
              const content = await fs.readFile(mdPath, 'utf8');
              const timestamp = this.extractTimestampFromPath(mdPath) || new Date();
              const wrote = await this.regenerateEmbedding(mdPath, content, timestamp);
              if (wrote) {
                count++;
              }
            } catch (error) {
              // Per-file isolation: one bad file never aborts the directory.
              console.error(`Failed to migrate ${mdFile}:`, error);
            }
          }
        }
      } catch (error) {
        if ((error as NodeJS.ErrnoException)?.code !== 'ENOENT') {
          console.error(`Failed to scan ${basePath} for missing embeddings:`, error);
        }
      }
    }

    return count;
  }

  private extractTimestampFromPath(filePath: string): Date | null {
    const filename = path.basename(filePath, '.md');
    const match = filename.match(/^(\d{2})-(\d{2})-(\d{2})-\d{6}$/);
    
    if (!match) return null;
    
    const [, hours, minutes, seconds] = match;
    const dirName = path.basename(path.dirname(filePath));
    const dateMatch = dirName.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    
    if (!dateMatch) return null;
    
    const [, year, month, day] = dateMatch;
    return new Date(parseInt(year), parseInt(month) - 1, parseInt(day), 
                   parseInt(hours), parseInt(minutes), parseInt(seconds));
  }

  private async ensureDirectoryExists(dirPath: string): Promise<void> {
    try {
      await fs.access(dirPath);
    } catch (error) {
      try {
        await fs.mkdir(dirPath, { recursive: true });
      } catch (mkdirError) {
        throw new Error(`Failed to create journal directory at ${dirPath}: ${mkdirError instanceof Error ? mkdirError.message : mkdirError}`);
      }
    }
  }
}