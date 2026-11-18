import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  Bandit,
  DailyBudget,
  MemoryBudgetStore,
  Router,
  applyObservation,
  betaQuantileMc,
  classifyDeadArm,
  costDiscountedReward,
  createBetaSampler,
  defaultArm,
  policyGate,
  seededUniform,
} from '../src/index.ts';
import type { PolicyRefusal } from '../src/index.ts';

const DAY = 86_400_000;

test('beta sampler matches the analytic mean and variance', () => {
  const s = createBetaSampler(seededUniform(7));
  for (const [a, b] of [[2, 5], [0.5, 0.5], [40, 3]] as const) {
    const n = 40_000;
    let sum = 0;
    let sq = 0;
    for (let i = 0; i < n; i++) {
      const x = s.sampleBeta(a, b);
      assert.ok(x >= 0 && x <= 1);
      sum += x;
      sq += x * x;
    }
    const mean = sum / n;
    const variance = sq / n - mean * mean;
    assert.ok(Math.abs(mean - a / (a + b)) < 0.01, `mean Beta(${a},${b}) = ${mean}`);
    const expectedVar = (a * b) / ((a + b) ** 2 * (a + b + 1));
    assert.ok(Math.abs(variance - expectedVar) < 0.005, `var Beta(${a},${b}) = ${variance}`);
  }
});

test('sampler rejects invalid shapes instead of returning NaN', () => {
  const s = createBetaSampler(seededUniform(1));
  assert.throws(() => s.sampleGamma(0), RangeError);
  assert.throws(() => s.sampleGamma(Number.NaN), RangeError);
  assert.throws(() => betaQuantileMc(s, -1, 2), RangeError);
});

test('seeded generator is reproducible and stays inside (0, 1)', () => {
  const a = seededUniform(42);
  const b = seededUniform(42);
  for (let i = 0; i < 1000; i++) {
    const x = a();
    assert.equal(x, b());
    assert.ok(x > 0 && x < 1);
  }
  assert.notEqual(seededUniform(1)(), seededUniform(2)());
});

test('reward: failure is worth zero, success falls as cost rises', () => {
  assert.equal(costDiscountedReward(false, 0), 0);
  assert.equal(costDiscountedReward(false, 10), 0);
  const free = costDiscountedReward(true, 0);
  const cheap = costDiscountedReward(true, 400);
  const dear = costDiscountedReward(true, 30_000);
  assert.ok(free > cheap && cheap > dear && dear > 0 && free < 1);
  assert.equal(costDiscountedReward(true, 500_000), 0.5);
  assert.equal(costDiscountedReward(true, 2_000, { costCeilingMicros: 2_000 }), 0.5);
  assert.throws(() => costDiscountedReward(true, -1), RangeError);
  assert.throws(() => costDiscountedReward(true, Number.NaN), RangeError);
});

test('arm update: discount, floor at 1, overflow halving, no mutation', () => {
  const arm = defaultArm();
  const next = applyObservation(arm, { success: true, costMicros: 0 }, 5, 1);
  assert.deepEqual(arm, defaultArm());
  assert.equal(next.alpha, 0.995 + 1);
  assert.equal(next.beta, 1);
  assert.equal(next.sampleCount, 1);
  assert.equal(next.lastUpdatedAt, 5);

  const big = applyObservation({ ...arm, alpha: 10_050, beta: 400 }, { success: true, costMicros: 0 }, 0, 1);
  assert.equal(big.alpha, 5000);
  assert.equal(big.beta, 199);

  assert.throws(() => applyObservation(arm, { success: true, costMicros: 0 }, 0, 1.2), RangeError);
});

test('cold start: every arm gets its guaranteed pulls even when one arm never fails', () => {
  for (const seed of [3, 4, 5, 6, 7]) {
    const bandit = new Bandit({ rng: seededUniform(seed) });
    const arms = ['a', 'b', 'c', 'd'];
    for (let i = 0; i < 40; i++) {
      const pick = bandit.select('chat', arms)!;
      bandit.update('chat', pick.armKey, { success: pick.armKey === 'a', costMicros: 100 });
    }
    const stats = bandit.stats('chat');
    assert.equal(stats.length, 4);
    for (const s of stats) assert.ok(s.sampleCount >= 5, `seed ${seed} ${s.armKey}=${s.sampleCount}`);
  }
  const noBoost = new Bandit({ rng: seededUniform(3), coldStartBoost: 0 });
  for (let i = 0; i < 40; i++) {
    const pick = noBoost.select('chat', ['a', 'b', 'c', 'd'])!;
    noBoost.update('chat', pick.armKey, { success: pick.armKey === 'a', costMicros: 100 });
  }
  assert.ok(
    noBoost.stats('chat').some((s) => s.sampleCount < 5),
    'without the boost at least one arm is starved, so the test above can fail',
  );
});

test('bandit converges on the arm with the best cost-discounted success rate', () => {
  const env: Record<string, { p: number; cost: number }> = {
    frontier: { p: 0.97, cost: 30_000 },
    mid: { p: 0.93, cost: 4_000 },
    small: { p: 0.8, cost: 400 },
    flaky: { p: 0.45, cost: 100 },
  };
  const world = seededUniform(11);
  const bandit = new Bandit({ rng: seededUniform(12) });
  const picks: Record<string, number> = {};
  for (let i = 0; i < 4000; i++) {
    const arm = bandit.select('chat', Object.keys(env))!.armKey;
    const e = env[arm]!;
    bandit.update('chat', arm, { success: world() < e.p, costMicros: e.cost });
    if (i >= 2000) picks[arm] = (picks[arm] ?? 0) + 1;
  }
  assert.ok((picks.mid ?? 0) + (picks.frontier ?? 0) > 1700, JSON.stringify(picks));
  assert.ok((picks.flaky ?? 0) < 40, JSON.stringify(picks));
});

test('same seed, same decisions', () => {
  const run = (): string => {
    const world = seededUniform(5);
    const bandit = new Bandit({ rng: seededUniform(6) });
    let trace = '';
    for (let i = 0; i < 300; i++) {
      const arm = bandit.select('t', ['x', 'y', 'z'])!.armKey;
      bandit.update('t', arm, { success: world() < 0.6, costMicros: 1000 });
      trace += arm;
    }
    return trace;
  };
  assert.equal(run(), run());
});

test('dead arms are skipped until the TTL passes, then revive clean', () => {
  let now = 1_000;
  const bandit = new Bandit({ rng: seededUniform(9), now: () => now });
  bandit.update('chat', 'a', { success: true, costMicros: 0 });
  bandit.update('code', 'a', { success: true, costMicros: 0 });
  bandit.update('chat', 'b', { success: true, costMicros: 0 });
  assert.deepEqual(bandit.markDead('a', 'gone').sort(), ['chat', 'code']);
  assert.deepEqual(bandit.rank('chat', ['a', 'b']).map((s) => s.armKey), ['b']);
  assert.equal(bandit.select('chat', ['a']), null);
  now += DAY - 1;
  assert.equal(bandit.isDead('chat', 'a'), true);
  now += 2;
  assert.equal(bandit.isDead('chat', 'a'), false);
  bandit.update('chat', 'a', { success: true, costMicros: 0 });
  assert.equal(bandit.state.chat!.a!.deadAt, undefined);
});

test('state round-trips through JSON', () => {
  const bandit = new Bandit({ rng: seededUniform(2) });
  bandit.update('chat', 'a', { success: true, costMicros: 250, fallbacks: 1 });
  const resumed = new Bandit({ state: JSON.parse(JSON.stringify(bandit.state)) });
  assert.deepEqual(resumed.stats('chat'), bandit.stats('chat'));
});

test('reserved object keys are refused and never reach the prototype', () => {
  const bandit = new Bandit({ rng: seededUniform(1) });
  for (const bad of ['__proto__', 'constructor', 'prototype']) {
    assert.throws(() => bandit.update(bad, 'a', { success: true, costMicros: 0 }), TypeError);
    assert.throws(() => bandit.update('chat', bad, { success: true, costMicros: 0 }), TypeError);
    assert.throws(() => bandit.rank('chat', [bad]), TypeError);
    assert.throws(() => bandit.markDead(bad, 'gone'), TypeError);
  }
  assert.equal(({} as Record<string, unknown>).alpha, undefined);
  bandit.update('toString', 'valueOf', { success: true, costMicros: 0 });
  assert.equal(bandit.stats('toString')[0]!.sampleCount, 1);
});

test('dead-arm classifier: permanent faults yes, transient faults no', () => {
  assert.equal(classifyDeadArm(new Error('API error (410): gone'))?.reason, 'gone');
  assert.equal(classifyDeadArm('The model `x-1` has been deprecated')?.reason, 'model_deprecated');
  assert.equal(classifyDeadArm({ message: 'model_not_found' })?.reason, 'model_not_found');
  assert.equal(classifyDeadArm({ error: 'Invalid API key provided' })?.reason, 'auth_revoked');
  for (const transient of ['HTTP 429 rate limit exceeded', 'status 503 overloaded', 'request timed out', '']) {
    assert.equal(classifyDeadArm(transient), null, transient);
  }
});

test('policy gate refuses in stage order and names the stage', async () => {
  const c = { armKey: 'p:m', provider: 'p' };
  const stageOf = async (ctx: Parameters<typeof policyGate>[1]) => {
    const v = await policyGate(c, ctx);
    return v.allowed ? 'allowed' : `${v.stage}/${v.scope}`;
  };
  const everything = {
    bannedProviders: new Set(['p']),
    disabledArms: new Set(['p:m']),
    isArmDead: () => true,
    isCircuitOpen: () => true,
    isWithinBudget: () => false,
    admit: () => ({ allowed: false, stage: 'admission', reason: 'full', scope: 'provider' }) as PolicyRefusal,
  };
  assert.equal(await stageOf(everything), 'provider_policy/provider');
  assert.equal(await stageOf({ ...everything, bannedProviders: undefined }), 'provider_policy/arm');
  assert.equal(await stageOf({ allowedProviders: new Set(['q']) }), 'provider_policy/provider');
  assert.equal(await stageOf({ isArmDead: () => true, isCircuitOpen: () => true }), 'dead_or_circuit/arm');
  assert.equal(await stageOf({ isCircuitOpen: () => true, isWithinBudget: () => false }), 'dead_or_circuit/provider');
  assert.equal(await stageOf({ isWithinBudget: async () => false, admit: everything.admit }), 'budget/provider');
  assert.equal(await stageOf({ admit: everything.admit }), 'admission/provider');
  assert.equal(await stageOf({}), 'allowed');

  const seen: string[] = [];
  await policyGate(c, { isWithinBudget: () => false, onRefusal: (_, r) => seen.push(r.reason) });
  assert.deepEqual(seen, ['daily_budget_exhausted:p']);
});

test('daily budget: trips at the cap, notifies once, resets at the UTC day boundary', async () => {
  let now = Date.UTC(2026, 0, 1, 23, 0, 0);
  const breaches: string[] = [];
  const ttls: number[] = [];
  const store = new MemoryBudgetStore();
  const put = store.put.bind(store);
  store.put = async (k, v, o?: { expirationTtl?: number }) => {
    ttls.push(o?.expirationTtl ?? -1);
    return put(k, v);
  };
  const budget = new DailyBudget({
    store,
    limitsUsd: { p: 1 },
    now: () => now,
    onBreach: (b) => void breaches.push(`${b.provider}@${b.day}`),
  });
  assert.equal(await budget.isWithinBudget('p'), true);
  assert.equal(await budget.isWithinBudget('uncapped'), true);
  await budget.recordSpend('p', 600_000);
  assert.equal(await budget.isWithinBudget('p'), true);
  await budget.recordSpend('p', 400_000);
  assert.equal(await budget.isWithinBudget('p'), false);
  await budget.recordSpend('p', 50_000);
  assert.deepEqual(breaches, ['p@2026-01-01']);
  assert.equal(ttls[0], 3600 + 3600);
  assert.deepEqual(await budget.snapshot(), [
    { provider: 'p', spentMicros: 1_050_000, limitMicros: 1_000_000, breached: true },
  ]);
  now += 2 * 3600 * 1000;
  assert.equal(await budget.isWithinBudget('p'), true);
  assert.equal(await budget.spentMicros('p'), 0);

  await assert.rejects(budget.recordSpend('p', -5), RangeError);
  await store.put('budget:p:2026-01-02', 'garbage');
  await assert.rejects(budget.isWithinBudget('p'), /corrupt counter/);
  assert.throws(() => new DailyBudget({ store, limitsUsd: { p: Number.NaN } }), RangeError);
});

test('router: falls through refused arms, skips a blocked provider, kills dead arms', async () => {
  const bandit = new Bandit({ rng: seededUniform(4), coldStartMinPulls: 0 });
  const budget = new DailyBudget({ store: new MemoryBudgetStore(), limitsUsd: { dear: 0.01 } });
  const router = new Router({
    bandit,
    budget,
    arms: [
      { armKey: 'dear:big', provider: 'dear' },
      { armKey: 'dear:bigger', provider: 'dear' },
      { armKey: 'cheap:small', provider: 'cheap' },
    ],
  });
  for (let i = 0; i < 30; i++) {
    bandit.update('chat', 'dear:big', { success: true, costMicros: 0 });
    bandit.update('chat', 'dear:bigger', { success: true, costMicros: 0 });
    bandit.update('chat', 'cheap:small', { success: false, costMicros: 0 });
  }
  assert.equal((await router.route('chat')).chosen?.provider, 'dear');

  await router.record('chat', 'dear:big', { success: true, costMicros: 20_000 });
  const afterBreach = await router.route('chat');
  assert.equal(afterBreach.chosen?.armKey, 'cheap:small');
  assert.equal(afterBreach.refusals.length, 1, 'second arm on the blocked provider is not re-checked');
  assert.equal(afterBreach.refusals[0]!.refusal.stage, 'budget');

  const gone = { success: false, costMicros: 0, error: new Error('HTTP 410 model retired') };
  await router.record('chat', 'cheap:small', gone);
  assert.equal(bandit.isDead('chat', 'cheap:small'), false, 'one matching error is not enough');
  await router.record('chat', 'cheap:small', { success: false, costMicros: 0, error: 'timeout' });
  await router.record('chat', 'cheap:small', gone);
  assert.equal(bandit.isDead('chat', 'cheap:small'), false, 'a different outcome resets the count');
  await router.record('chat', 'cheap:small', gone);
  assert.equal(bandit.isDead('chat', 'cheap:small'), true);
  assert.throws(() => new Router({ bandit, arms: [{ armKey: 'a', provider: 'p' }], deadArmConfirmations: 0 }), RangeError);
  const none = await router.route('chat');
  assert.equal(none.chosen, null);

  await assert.rejects(router.record('chat', 'nope', { success: true, costMicros: 0 }), TypeError);
  assert.throws(() => new Router({ bandit, arms: [] }), TypeError);
});
