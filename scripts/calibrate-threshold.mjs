#!/usr/bin/env node
// Recalibration helper for the recurrence threshold (τ) used by
// find_recurring_themes. Read-only and fully offline: it sweeps τ over the
// existing .embedding corpus and prints cluster statistics, then renders the
// themes at one candidate τ so their coherence can be judged by eye. No model
// is loaded and nothing is written.
//
// Usage (from the repo root):
//   npm run build
//   [PRIVATE_JOURNAL_PATH=/path/to/journal] npm run calibrate [-- <candidateTau>]
//
// Journal paths resolve exactly like the server's: PRIVATE_JOURNAL_PATH
// overrides everything; otherwise CWD/.private-journal (project) and
// ~/.private-journal (user) are scanned.
//
// Reading the sweep: too low a τ collapses unrelated topics into one
// mega-cluster (largest ≈ corpus size); too high leaves only near-duplicates
// (all singletons). Pick the τ where the printed themes are semantically
// coherent — excerpts within one theme visibly about the same thing, and
// unrelated topics not sharing a cluster. Then update DEFAULT_THRESHOLD in
// src/recurrence.ts (and the README default) to the chosen value.
import { SearchService } from '../dist/search.js';
import { EmbeddingService } from '../dist/embeddings.js';
import {
  DEFAULT_THRESHOLD,
  findRecurringThemes,
  formatThemesOutput,
  gatherThemeChunks,
  parseThemeParams,
} from '../dist/recurrence.js';

const svc = new SearchService();
const embeddings = await svc.collectEmbeddings('both');
const es = EmbeddingService.getInstance();
const chunks = gatherThemeChunks(embeddings, {
  now: Date.now(),
  days: 0,
  isCompatible: (e) => es.isCompatible(e),
});
console.log(`corpus: ${embeddings.length} embedding files, ${chunks.length} section chunks (model: ${es.getModelName()})\n`);

// minEntries 3 in the sweep: on a small corpus the real default (5) hides the
// cluster structure the sweep is meant to expose.
console.log('τ sweep (all-time window, minEntries 3, minDays 2):');
for (let i = 0; i <= 7; i++) {
  const threshold = Math.round((0.5 + i * 0.05) * 100) / 100;
  const { stats } = findRecurringThemes(chunks, { threshold, minEntries: 3, minDays: 2, limit: 20 });
  console.log(
    `  τ=${threshold.toFixed(2)}  clusters(>=2 chunks)=${stats.clustersFormed}  ` +
      `largest=${stats.largestClusterEntries} entries  qualifying=${stats.themesQualifying}`
  );
}

const candidate = Number(process.argv[2] ?? DEFAULT_THRESHOLD);
console.log(`\nThemes at candidate τ=${candidate} (minEntries 3 — judge coherence by eye):\n`);
const inspect = findRecurringThemes(chunks, { threshold: candidate, minEntries: 3, minDays: 2, limit: 5 });
console.log(formatThemesOutput(inspect, parseThemeParams({ days: 0, threshold: candidate, minEntries: 3 })));

const real = findRecurringThemes(chunks, { threshold: candidate, minEntries: 5, minDays: 2, limit: 20 });
console.log(`\nAt the tool's real defaults (minEntries 5, minDays 2): ${real.stats.themesQualifying} theme(s) qualify.`);
