// ABOUTME: Recurrence engine detecting themes that recur across journal entries
// ABOUTME: Pure functions over pre-computed section vectors - no model, no I/O

import * as path from 'path';
import { cosineSimilarity } from './embeddings.js';
import { LoadedEmbedding } from './search.js';

export const DEFAULT_THRESHOLD = 0.65; // provisional; calibrated against the real corpus (plan Task 7)

export interface ThemeChunk {
  vector: number[];
  entryPath: string; // on-disk .md path
  date: string;      // YYYY-MM-DD (the entry's day directory)
  timestamp: number;
  section: string;
  text: string;
  type: 'project' | 'user';
}

export interface RecurringTheme {
  representativeExcerpt: string;
  supportingExcerpts: string[];
  distinctEntries: number;
  occurrences: number;
  distinctDays: number;
  dateSpan: { start: string; end: string };
  sectionDistribution: Record<string, number>;
  cohesion: number;
  sourcePaths: string[];
  type: 'project' | 'user' | 'both';
}

export interface RecurrenceStats {
  chunksScanned: number;
  entriesScanned: number;
  clustersFormed: number;        // clusters with >= 2 chunks, before qualification
  largestClusterEntries: number; // max distinctEntries over all clusters
  themesQualifying: number;      // themes passing minEntries/minDays, before limit
}

export interface EngineOptions {
  threshold: number;
  minEntries: number;
  minDays: number;
  limit: number;
}

export interface RecurrenceResult {
  themes: RecurringTheme[];
  stats: RecurrenceStats;
}

// Length-mismatched or degenerate vectors must isolate quietly (similarity 0),
// not throw like the service's cosineSimilarity: one corrupt vector may not
// take down a whole scan.
function safeCosine(a: number[], b: number[]): number {
  if (!Array.isArray(a) || !Array.isArray(b) || a.length === 0 || a.length !== b.length) {
    return 0;
  }
  return cosineSimilarity(a, b);
}

interface Cluster {
  medoid: number;    // index into chunks
  members: number[]; // indices, ascending (= chronological when input is sorted)
}

// Leader/medoid clustering in fixed input order (callers pass chronologically
// sorted chunks, so results are deterministic). For each unused chunk: gather
// everything within τ of it, elect the medoid (most central member), then
// re-gather around the medoid. Because cosine is symmetric the leader is
// always within τ of its group's medoid, so the leader is never dropped; a
// dropped non-leader member always has a later index and seeds its own
// cluster when the loop reaches it.
function clusterChunks(chunks: ThemeChunk[], threshold: number): Cluster[] {
  const used = new Set<number>();
  const clusters: Cluster[] = [];

  for (let leader = 0; leader < chunks.length; leader++) {
    if (used.has(leader)) continue;

    const group: number[] = [];
    for (let j = 0; j < chunks.length; j++) {
      if (used.has(j)) continue;
      if (j === leader || safeCosine(chunks[leader].vector, chunks[j].vector) >= threshold) {
        group.push(j);
      }
    }

    let medoid = group[0];
    let bestMean = -Infinity;
    for (const m of group) {
      let sum = 0;
      for (const g of group) {
        if (g !== m) sum += safeCosine(chunks[m].vector, chunks[g].vector);
      }
      const mean = group.length > 1 ? sum / (group.length - 1) : 1;
      if (mean > bestMean) {
        bestMean = mean;
        medoid = m;
      }
    }

    const members: number[] = [];
    for (let j = 0; j < chunks.length; j++) {
      if (used.has(j)) continue;
      if (j === medoid || safeCosine(chunks[medoid].vector, chunks[j].vector) >= threshold) {
        members.push(j);
      }
    }
    for (const j of members) used.add(j);
    clusters.push({ medoid, members });
  }

  return clusters;
}

function buildTheme(chunks: ThemeChunk[], cluster: Cluster): RecurringTheme {
  const members = cluster.members.map(i => chunks[i]);
  const medoidChunk = chunks[cluster.medoid];

  const entryPaths: string[] = [];
  for (const m of members) {
    if (!entryPaths.includes(m.entryPath)) entryPaths.push(m.entryPath);
  }
  const days = new Set(members.map(m => m.date));
  const dates = members.map(m => m.date).sort();

  const sectionDistribution: Record<string, number> = {};
  for (const m of members) {
    sectionDistribution[m.section] = (sectionDistribution[m.section] || 0) + 1;
  }

  const others = cluster.members.filter(i => i !== cluster.medoid);
  const cohesion =
    others.length > 0
      ? others.reduce((sum, i) => sum + safeCosine(medoidChunk.vector, chunks[i].vector), 0) /
        others.length
      : 1;

  const supportingExcerpts = others
    .map(i => ({ i, sim: safeCosine(medoidChunk.vector, chunks[i].vector) }))
    .sort((a, b) => b.sim - a.sim) // stable: ties stay chronological
    .slice(0, 3)
    .map(({ i }) => chunks[i].text);

  const types = new Set(members.map(m => m.type));

  return {
    representativeExcerpt: medoidChunk.text,
    supportingExcerpts,
    distinctEntries: entryPaths.length,
    occurrences: members.length,
    distinctDays: days.size,
    dateSpan: { start: dates[0], end: dates[dates.length - 1] },
    sectionDistribution,
    cohesion,
    sourcePaths: entryPaths,
    type: types.size > 1 ? 'both' : members[0].type,
  };
}

export function findRecurringThemes(
  chunks: ThemeChunk[],
  options: EngineOptions
): RecurrenceResult {
  const { threshold, minEntries, minDays, limit } = options;

  const clusters = clusterChunks(chunks, threshold);
  const themed = clusters.map(cluster => ({ cluster, theme: buildTheme(chunks, cluster) }));

  const qualifying = themed
    .filter(({ theme }) => theme.distinctEntries >= minEntries && theme.distinctDays >= minDays)
    .sort(
      (a, b) =>
        b.theme.distinctEntries - a.theme.distinctEntries ||
        b.theme.cohesion - a.theme.cohesion ||
        a.cluster.medoid - b.cluster.medoid
    );

  const stats: RecurrenceStats = {
    chunksScanned: chunks.length,
    entriesScanned: new Set(chunks.map(c => c.entryPath)).size,
    clustersFormed: themed.filter(({ theme }) => theme.occurrences >= 2).length,
    largestClusterEntries: themed.reduce(
      (max, { theme }) => Math.max(max, theme.distinctEntries),
      0
    ),
    themesQualifying: qualifying.length,
  };

  return { themes: qualifying.slice(0, limit).map(({ theme }) => theme), stats };
}

export interface GatherOptions {
  now: number;
  days: number; // 0 = all-time
  sections?: string[];
  isCompatible: (entry: { version?: number; model?: string }) => boolean;
}

// Corpus -> engine input. Applies the spec's exclusions (dream entries,
// incompatible vectors, out-of-window, unwanted sections, structural
// corruption) and sorts chronologically so clustering is deterministic
// regardless of filesystem enumeration order.
export function gatherThemeChunks(
  embeddings: LoadedEmbedding[],
  options: GatherOptions
): ThemeChunk[] {
  const { now, days, sections, isCompatible } = options;
  const cutoff = days > 0 ? now - days * 24 * 60 * 60 * 1000 : -Infinity;

  const chunks: ThemeChunk[] = [];
  for (const entry of embeddings) {
    if (entry.dream === true) continue; // dreams never feed the loop
    if (!isCompatible(entry)) continue; // foreign model/schema -> garbage similarities
    if (typeof entry.timestamp !== 'number' || entry.timestamp < cutoff) continue;
    if (!Array.isArray(entry.sectionEmbeddings)) continue;

    const entryPath = entry.diskPath.replace(/\.embedding$/, '.md');
    const date = path.basename(path.dirname(entry.diskPath));

    for (const se of entry.sectionEmbeddings) {
      if (!se || typeof se.section !== 'string' || typeof se.text !== 'string') continue;
      if (!Array.isArray(se.embedding) || se.embedding.length === 0) continue;
      if (sections && sections.length > 0) {
        const wanted = sections.some(s => se.section.toLowerCase().includes(s.toLowerCase()));
        if (!wanted) continue;
      }
      chunks.push({
        vector: se.embedding,
        entryPath,
        date,
        timestamp: entry.timestamp,
        section: se.section,
        text: se.text,
        type: entry.type,
      });
    }
  }

  // Stable sort: within an entry, section order is preserved.
  return chunks.sort(
    (a, b) => a.timestamp - b.timestamp || a.entryPath.localeCompare(b.entryPath)
  );
}
