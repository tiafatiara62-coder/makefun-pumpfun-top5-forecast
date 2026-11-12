// pumpfun-top5-forecast-01 entry (Node 22, standard library only): a deterministic
// statistical forecaster of the top 5 new pump.fun coins by market cap at the seven
// answer reads. No model calls, no network, no dependencies.
//
// Idea
// -----
// For every coin in the history, fit, on log10(market cap in SOL) against time:
//   - a drift (log-cap slope per hour, from first to last observation, shrunk and clamped),
//   - a volatility (per-hour, from the first differences of log-cap around the drift,
//     shrunk towards a cross-sectional prior so a coin with 1-2 observations does
//     not read as nearly deterministic),
//   - a small per-hour crash hazard (a coin can collapse out of the chart).
// Then a fixed-seed Monte Carlo projects every candidate coin to each answer read:
// the drift decays exponentially (tau = 3h, so no coin grows forever), the noise grows
// like sqrt(dt), and the crash hazard grows with dt. At each simulated read the coins
// are ranked, and the place and size-bin counts are averaged over all simulations and
// all seven reads. Coins the history never ranked (and coins born after the cut) cannot
// be named, so a substantial share of every place (~30%) is left unnamed for them,
// and a place's named probabilities are capped at 0.70.
//
// Everything is deterministic: one fixed PRNG seed, fixed iteration orders, no clock,
// no Math.random. Two runs on the same history produce byte-identical output.
//
// Usage (local): mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// --- Tuned constants (fixed in advance, not tuned on the practice answer) ----------
const NSIMS = 3000; // Monte Carlo paths; 3000 x 7 reads keeps the MC error ~0.003 on p=0.2.
const SEED = 0xc0ffee; // fixed PRNG seed: the run is deterministic.
const DRIFT_TAU_H = 3.0; // drift decays as exp(-dt/tau): no coin compounds forever.
const DRIFT_CLAMP = 1.5; // log10(cap) per hour, either way.
const VOL_FLOOR = 0.12; // per-hour volatility floor (log10 units).
const VOL_CAP = 1.2; // per-hour volatility cap.
const VOL_PRIOR = 0.3; // cross-sectional prior vol: with 1-2 observations a coin's own
// estimate is unreliable, so it is shrunk towards this prior (below).
const VOL_PRIOR_WEIGHT = 2; // pseudo-observations behind the prior.
const VOL_DEFAULT = VOL_PRIOR; // per-hour volatility with a single observation.
const CRASH_HAZARD_PER_H = 0.04; // chance per hour a coin collapses.
const CRASH_DROP_MIN = 0.8; // log10 drop on a crash.
const CRASH_DROP_MAX = 2.0;
const CURVE_DRIFT_FACTOR = 0.7; // a coin still on its bonding curve is growth-limited.
const MAX_CANDIDATES = 300; // coins carried into the Monte Carlo, by last log-cap.
const PLACE_NAMED_CAP = 0.7; // a place's named probabilities add up to at most this.
const PLACE_MIN_P = 0.01; // only name a row at least this probable (below it, the
// unnamed share serves the coin better than a tiny named p would).
const SIZE_NAMED_CAP = 0.92; // a place's named size bins add up to at most this.
const SIZE_MIN_P = 0.004; // only name a size bin at least this probable.
const MICRO = 1_000_000;

interface MarkRow {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  venue: string;
  created_at: string;
  complete: boolean;
  mayhem: boolean;
}

interface Obs {
  t: number; // hours since t0
  y: number; // log10(market cap in SOL)
  venue: string;
}

interface Coin {
  mint: string;
  obs: Obs[];
  lastY: number;
  lastT: number;
  lastVenue: string;
  drift: number; // log10 per hour, shrunk and clamped
  vol: number; // per-hour log10 volatility
}

/** mulberry32: a small deterministic PRNG. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal from the PRNG (Box-Muller, with a cached spare draw). */
function makeGaussian(rand: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u1 = 0;
    while (u1 === 0) u1 = rand(); // avoid log(0)
    const u2 = rand();
    const r = Math.sqrt(-2 * Math.log(u1));
    const theta = 2 * Math.PI * u2;
    spare = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  t0: string;
  places: number;
  output: string;
  reads: string[];
  size: { edges: number[] };
};
const PLACES = task.places;
const EDGES: number[] = task.size.edges;
const READ_TIMES = task.reads.map((s) => Date.parse(s));
const T0 = Date.parse(task.t0);

const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '');
const marks: Array<{ at: number; rows: MarkRow[] }> = [];
for (const line of lines) {
  const m = JSON.parse(line) as { ok: boolean; at: string; rows: MarkRow[] | null };
  if (m.ok === true && Array.isArray(m.rows) && m.rows.length > 0) {
    marks.push({ at: Date.parse(m.at), rows: m.rows });
  }
}

const HOUR = 3_600_000;
const logcapOf = (lamports: string | number): number => Math.log10(Number(BigInt(lamports))) - 9;

// --- Build each coin's observation series ---------------------------------------
const byMint = new Map<string, Coin>();
for (const mark of marks) {
  const t = (mark.at - T0) / HOUR;
  for (const row of mark.rows) {
    let c = byMint.get(row.mint);
    if (c === undefined) {
      c = { mint: row.mint, obs: [], lastY: 0, lastT: 0, lastVenue: row.venue, drift: 0, vol: VOL_DEFAULT };
      byMint.set(row.mint, c);
    }
    c.obs.push({ t, y: logcapOf(row.market_cap_lamports), venue: row.venue });
    c.lastY = logcapOf(row.market_cap_lamports);
    c.lastT = t;
    c.lastVenue = row.venue;
  }
}

const lastMark = marks[marks.length - 1];
const lastMarkTime = lastMark !== undefined ? lastMark.at : T0;
const lastMarkMints = new Set<string>();
if (lastMark !== undefined) for (const row of lastMark.rows) lastMarkMints.add(row.mint);

// --- Candidate selection: coins in the last mark, plus coins seen recently --------
// A coin that left the last mark's chart can still surge back; keep recent dropouts.
const candidates: Coin[] = [];
for (const c of byMint.values()) {
  const inLast = lastMarkMints.has(c.mint);
  const recent = c.lastT >= (lastMarkTime - T0) / HOUR - 0.5;
  if (inLast || (recent && c.obs.length >= 2)) candidates.push(c);
}
candidates.sort((a, b) => b.lastY - a.lastY || (a.mint < b.mint ? -1 : a.mint > b.mint ? 1 : 0));
const kept = candidates.slice(0, MAX_CANDIDATES);

// --- Per-coin drift and volatility ----------------------------------------------
for (const c of kept) {
  const obs = c.obs;
  if (obs.length >= 2) {
    const first = obs[0]!;
    const last = obs[obs.length - 1]!;
    const span = last.t - first.t;
    let drift = 0;
    if (span > 0) drift = (last.y - first.y) / span;
    // Shrink towards 0: few observations mean little evidence of a trend.
    drift *= obs.length / (obs.length + 2);
    if (drift > DRIFT_CLAMP) drift = DRIFT_CLAMP;
    if (drift < -DRIFT_CLAMP) drift = -DRIFT_CLAMP;
    if (c.lastVenue === 'curve') drift *= CURVE_DRIFT_FACTOR;
    c.drift = drift;
    // Volatility from first differences around the drift (Brownian-motion estimator),
    // shrunk towards the cross-sectional prior: with 1-2 differences the raw estimate
    // is mostly luck (one quiet 10-minute interval can read as near-zero volatility,
    // which would fake huge confidence 13 hours out).
    let s2 = 0;
    let k = 0;
    for (let i = 1; i < obs.length; i++) {
      const dt = obs[i]!.t - obs[i - 1]!.t;
      if (dt <= 0) continue;
      const d = obs[i]!.y - obs[i - 1]!.y - drift * dt;
      s2 += (d * d) / dt;
      k++;
    }
    let vol = Math.sqrt((s2 + VOL_PRIOR_WEIGHT * VOL_PRIOR * VOL_PRIOR) / (k + VOL_PRIOR_WEIGHT));
    if (!(vol > 0) || !Number.isFinite(vol)) vol = VOL_DEFAULT;
    if (vol < VOL_FLOOR) vol = VOL_FLOOR;
    if (vol > VOL_CAP) vol = VOL_CAP;
    c.vol = vol;
  } else {
    c.drift = 0;
    c.vol = VOL_DEFAULT;
  }
}

// --- Monte Carlo -----------------------------------------------------------------
const n = kept.length;
const nReads = READ_TIMES.length;
const placeCounts: number[][] = Array.from({ length: PLACES }, () => new Array<number>(n).fill(0));
const sizeCounts: number[][] = Array.from({ length: PLACES }, () => new Array<number>(EDGES.length + 1).fill(0));
const totalSamples = NSIMS * nReads;

const rand = mulberry32(SEED);
const gauss = makeGaussian(rand);

const lastY = kept.map((c) => c.lastY);
const drift = kept.map((c) => c.drift);
const vol = kept.map((c) => c.vol);
const dts = READ_TIMES.map((rt) => Math.max(0, (rt - lastMarkTime) / HOUR));

// Precompute per-read, per-coin mean and sd of the projected log-cap.
const mean: number[][] = [];
const sd: number[][] = [];
for (let r = 0; r < nReads; r++) {
  const dt = dts[r]!;
  const growth = DRIFT_TAU_H * (1 - Math.exp(-dt / DRIFT_TAU_H));
  const s = Math.sqrt(dt);
  const mRow: number[] = new Array<number>(n);
  const sRow: number[] = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    mRow[i] = lastY[i]! + drift[i]! * growth;
    sRow[i] = vol[i]! * s;
  }
  mean.push(mRow);
  sd.push(sRow);
}

const proj: number[] = new Array<number>(n);
const order: number[] = new Array<number>(n);
for (let i = 0; i < n; i++) order[i] = i;

function binOf(y: number): number {
  if (y < EDGES[0]!) return 0;
  for (let b = 1; b <= EDGES.length; b++) {
    if (b === EDGES.length) return EDGES.length;
    if (y < EDGES[b]!) return b;
  }
  return EDGES.length;
}

for (let sim = 0; sim < NSIMS; sim++) {
  for (let r = 0; r < nReads; r++) {
    const dt = dts[r]!;
    const pCrash = dt > 0 ? 1 - Math.exp(-CRASH_HAZARD_PER_H * dt) : 0;
    const mRow = mean[r]!;
    const sRow = sd[r]!;
    for (let i = 0; i < n; i++) {
      let v = mRow[i]! + gauss() * sRow[i]!;
      if (pCrash > 0 && rand() < pCrash) {
        v -= CRASH_DROP_MIN + rand() * (CRASH_DROP_MAX - CRASH_DROP_MIN);
      }
      proj[i] = v;
    }
    // Rank by projected log-cap, ties to the mint's bytes ascending (as the scorer does).
    order.sort((a, b) => {
      const d = proj[b]! - proj[a]!;
      if (d !== 0) return d;
      const ma = kept[a]!.mint;
      const mb = kept[b]!.mint;
      return ma < mb ? -1 : ma > mb ? 1 : 0;
    });
    for (let place = 1; place <= PLACES; place++) {
      const idx = order[place - 1];
      if (idx === undefined) break;
      placeCounts[place - 1]![idx]! += 1;
      const b = binOf(proj[idx]!);
      sizeCounts[place - 1]![b]! += 1;
    }
  }
}

// --- Post-processing -------------------------------------------------------------
const floor6 = (p: number): number => Math.floor(p * MICRO) / MICRO;

interface ProbRow {
  mint: string;
  place: number;
  p: number;
}
const probabilities: ProbRow[] = [];
for (let place = 1; place <= PLACES; place++) {
  const counts = placeCounts[place - 1]!;
  const entries: Array<{ mint: string; p: number }> = [];
  for (let i = 0; i < n; i++) {
    const p = counts[i]! / totalSamples;
    if (p >= PLACE_MIN_P) entries.push({ mint: kept[i]!.mint, p });
  }
  entries.sort((a, b) => b.p - a.p || (a.mint < b.mint ? -1 : a.mint > b.mint ? 1 : 0));
  const top = entries.slice(0, 100);
  let sum = 0;
  for (const e of top) sum += e.p;
  const scale = sum > PLACE_NAMED_CAP ? PLACE_NAMED_CAP / sum : 1;
  for (const e of top) {
    const p = floor6(e.p * scale);
    if (p > 0) probabilities.push({ mint: e.mint, place, p });
  }
}

// Expected place per coin (6 = outside the top 5) for the top5 tie-break.
const expectedPlace = new Map<string, number>();
for (let i = 0; i < n; i++) {
  let ep = 6;
  let covered = 0;
  for (let place = 1; place <= PLACES; place++) {
    const p = placeCounts[place - 1]![i]! / totalSamples;
    ep -= p * (6 - place);
    covered += p;
  }
  void covered;
  expectedPlace.set(kept[i]!.mint, ep);
}
const top5: string[] = [...expectedPlace.entries()]
  .sort((a, b) => a[1] - b[1] || (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
  .slice(0, PLACES)
  .map(([mint]) => mint);
// Pad if the history held fewer than 5 coins (cannot happen on a real history).
for (let i = top5.length; i < PLACES; i++) top5.push('A'.repeat(44 - i));

interface SizeRow {
  place: number;
  bin: number;
  p: number;
}
const sizes: SizeRow[] = [];
for (let place = 1; place <= PLACES; place++) {
  const counts = sizeCounts[place - 1]!;
  const entries: Array<{ bin: number; p: number }> = [];
  for (let b = 0; b <= EDGES.length; b++) {
    const p = counts[b]! / totalSamples;
    if (p >= SIZE_MIN_P) entries.push({ bin: b, p });
  }
  entries.sort((a, b) => b.p - a.p || a.bin - b.bin);
  let sum = 0;
  for (const e of entries) sum += e.p;
  const scale = sum > SIZE_NAMED_CAP ? SIZE_NAMED_CAP / sum : 1;
  for (const e of entries) {
    const p = floor6(e.p * scale);
    if (p > 0) sizes.push({ place, bin: e.bin, p });
  }
}

writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top5, sizes })}\n`);
