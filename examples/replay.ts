/**
 * Simulated routing replay. Four synthetic arms with fixed success rates and
 * per-call costs; the same seeded outcome stream is fed to the bandit and to
 * four fixed policies. Nothing here is a production measurement.
 *
 *   node examples/replay.ts [seed] [calls]
 */
import { Bandit, DailyBudget, MemoryBudgetStore, Router, costDiscountedReward, seededUniform } from '../src/index.ts';

const seed = Number(process.argv[2] ?? 20261115);
const calls = Number(process.argv[3] ?? 20_000);
if (!Number.isInteger(seed) || !Number.isInteger(calls) || calls <= 0) {
  console.error('usage: node examples/replay.ts [integer seed] [positive integer calls]');
  process.exit(2);
}

interface SimArm {
  armKey: string;
  provider: string;
  successRate: number;
  costMicros: number;
}

const ARMS: SimArm[] = [
  { armKey: 'alpha:frontier', provider: 'alpha', successRate: 0.97, costMicros: 30_000 },
  { armKey: 'beta:mid', provider: 'beta', successRate: 0.93, costMicros: 4_000 },
  { armKey: 'gamma:small', provider: 'gamma', successRate: 0.8, costMicros: 400 },
  { armKey: 'delta:flaky', provider: 'delta', successRate: 0.45, costMicros: 100 },
];
const byKey = new Map(ARMS.map((a) => [a.armKey, a]));
const expected = (a: SimArm): number => a.successRate * costDiscountedReward(true, a.costMicros);
const oracle = ARMS.reduce((best, a) => (expected(a) > expected(best) ? a : best));

/** One uniform per call, shared by every policy, so they face the same luck. */
const luck = Array.from({ length: calls }, seededUniform(seed));

interface Tally {
  name: string;
  successes: number;
  costMicros: number;
  reward: number;
  picks: Record<string, number>;
}

function tally(name: string): Tally {
  return { name, successes: 0, costMicros: 0, reward: 0, picks: {} };
}

function play(t: Tally, arm: SimArm, i: number): boolean {
  const success = luck[i]! < arm.successRate;
  t.successes += success ? 1 : 0;
  t.costMicros += arm.costMicros;
  t.reward += costDiscountedReward(success, arm.costMicros);
  t.picks[arm.armKey] = (t.picks[arm.armKey] ?? 0) + 1;
  return success;
}

function fixed(name: string, pick: (i: number) => SimArm): Tally {
  const t = tally(name);
  for (let i = 0; i < calls; i++) play(t, pick(i), i);
  return t;
}

const uniformPick = seededUniform(seed + 1);
const baselines = [
  fixed('always frontier', () => ARMS[0]!),
  fixed('always cheapest', () => ARMS[3]!),
  fixed('uniform random', () => ARMS[Math.floor(uniformPick() * ARMS.length)]!),
  fixed(`oracle (${oracle.armKey})`, () => oracle),
];

function runBandit(name: string, costCeilingMicros?: number): Tally {
  const bandit = new Bandit({ rng: seededUniform(seed + 2), costCeilingMicros });
  const t = tally(name);
  for (let i = 0; i < calls; i++) {
    const arm = byKey.get(bandit.select('chat', ARMS.map((a) => a.armKey))!.armKey)!;
    const success = play(t, arm, i);
    bandit.update('chat', arm.armKey, { success, costMicros: arm.costMicros });
  }
  return t;
}
const banditTally = runBandit('bandit (this library)');

const usd = (micros: number): string => `$${(micros / 1e6).toFixed(2)}`;
const oracleReward = baselines[3]!.reward;
console.log(`seed ${seed}, ${calls} calls, synthetic arms (simulation, not production data)\n`);
console.log('arm              success  cost/call  expected reward');
for (const a of ARMS) {
  console.log(
    `${a.armKey.padEnd(16)} ${(a.successRate * 100).toFixed(0).padStart(6)}%  ${`$${(a.costMicros / 1e6).toFixed(4)}`.padStart(9)}  ${expected(a).toFixed(4)}`,
  );
}
console.log('\npolicy                    success   total cost  cost/success   regret vs oracle');
for (const t of [...baselines, banditTally]) {
  console.log(
    `${t.name.padEnd(25)} ${((t.successes / calls) * 100).toFixed(1).padStart(6)}%  ${usd(t.costMicros).padStart(10)}  ${`$${(t.costMicros / 1e6 / t.successes).toFixed(5)}`.padStart(12)}  ${(oracleReward - t.reward).toFixed(1).padStart(10)}`,
  );
}
console.log('\nbandit picks:', JSON.stringify(banditTally.picks));

/* The reward's cost ceiling sets how hard the router leans toward cheap arms. */
const tight = runBandit('tight', 50_000);
console.log('\nsame stream with costCeilingMicros = 50_000 (a $0.05 call halves a success):');
console.log(`  success ${((tight.successes / calls) * 100).toFixed(1)}%, total cost ${usd(tight.costMicros)}`);
console.log('  picks:', JSON.stringify(tight.picks));

/* Part 2: the same bandit behind the policy gate, with a daily cap on one provider. */
const capUsd = 5;
let breachAt = -1;
let step = 0;
const gated = new Bandit({ rng: seededUniform(seed + 3) });
const budget = new DailyBudget({
  store: new MemoryBudgetStore(),
  limitsUsd: { beta: capUsd },
  now: () => Date.UTC(2026, 0, 1),
  onBreach: () => {
    breachAt = step;
  },
});
const router = new Router({ bandit: gated, budget, arms: ARMS });
const gatedTally = tally('gated');
let refusals = 0;
for (step = 0; step < calls; step++) {
  const decision = await router.route('chat');
  refusals += decision.refusals.length;
  if (!decision.chosen) throw new Error(`no routable arm at call ${step}`);
  const arm = byKey.get(decision.chosen.armKey)!;
  const success = play(gatedTally, arm, step);
  await router.record('chat', arm.armKey, { success, costMicros: arm.costMicros });
}
const betaSpend = (await budget.snapshot())[0]!.spentMicros;
console.log(`\nwith a ${usd(capUsd * 1e6)}/day cap on provider "beta":`);
console.log(`  cap crossed at call ${breachAt}; beta spend held at ${usd(betaSpend)}`);
console.log(`  ${refusals} routing attempts refused at the budget stage and rerouted`);
console.log(`  success ${((gatedTally.successes / calls) * 100).toFixed(1)}%, total cost ${usd(gatedTally.costMicros)}`);
console.log('  picks:', JSON.stringify(gatedTally.picks));
