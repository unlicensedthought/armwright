import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Bandit, Cascade, CircuitBreaker, Router, seededUniform } from '../src/index.ts';

test('circuit breaker: opens on consecutive failures, half-opens, recloses or reopens', () => {
  let now = 0;
  const changes: string[] = [];
  const cb = new CircuitBreaker({
    failureThreshold: 3,
    openMs: 1000,
    now: () => now,
    onStateChange: (p, s) => changes.push(`${p}:${s}`),
  });
  cb.record('p', false);
  cb.record('p', false);
  cb.record('p', true);
  cb.record('p', false);
  cb.record('p', false);
  assert.equal(cb.state('p'), 'closed', 'a success resets the streak');
  cb.record('p', false);
  assert.equal(cb.isOpen('p'), true);
  assert.equal(cb.isOpen('other'), false);
  now = 999;
  assert.equal(cb.state('p'), 'open');
  now = 1000;
  assert.equal(cb.state('p'), 'half_open');
  assert.equal(cb.isOpen('p'), false);
  cb.record('p', false);
  assert.equal(cb.state('p'), 'open', 'one failed probe reopens for a full window');
  now = 2000;
  cb.record('p', true);
  assert.equal(cb.state('p'), 'closed');
  assert.deepEqual(changes, ['p:open', 'p:open', 'p:closed']);
  assert.throws(() => new CircuitBreaker({ failureThreshold: 0 }), RangeError);
  assert.throws(() => new CircuitBreaker({ openMs: 0 }), RangeError);
});

interface Task {
  kind: string;
  tokens: number;
  failedAttempts?: number;
}

const cascade = new Cascade<Task>({
  tiers: [
    { name: 'triage', arms: ['a:tiny'] },
    { name: 'workhorse', arms: ['a:mid', 'b:mid'] },
    { name: 'reasoning', arms: ['c:big'] },
  ],
  rules: [
    { tier: 'reasoning', reason: 'long context', when: (t) => t.tokens > 100_000 },
    { tier: 'triage', reason: 'simple chore', when: (t) => t.kind === 'echo' },
  ],
  defaultTier: 'workhorse',
});

test('cascade: first matching rule wins, default otherwise, same input same answer', () => {
  assert.deepEqual(cascade.decide({ kind: 'echo', tokens: 10 }), {
    tier: 'triage',
    arms: ['a:tiny'],
    reason: 'simple chore',
  });
  assert.equal(cascade.decide({ kind: 'echo', tokens: 200_000 }).tier, 'reasoning');
  assert.equal(cascade.decide({ kind: 'feature', tokens: 10 }).reason, 'default');
  assert.deepEqual(cascade.decide({ kind: 'x', tokens: 1 }), cascade.decide({ kind: 'x', tokens: 1 }));
});

test('cascade: failures escalate one tier per two attempts and stop at the top', () => {
  const at = (failedAttempts: number) => cascade.decide({ kind: 'echo', tokens: 1, failedAttempts });
  assert.equal(at(1).tier, 'triage');
  assert.equal(at(1).escalatedFrom, undefined);
  assert.equal(at(2).tier, 'workhorse');
  assert.equal(at(2).escalatedFrom, 'triage');
  assert.equal(at(4).tier, 'reasoning');
  assert.equal(at(40).tier, 'reasoning');
  assert.throws(() => at(-1), RangeError);
  assert.throws(() => at(1.5), RangeError);
});

test('cascade: bad configuration is refused at construction', () => {
  const tiers = [{ name: 't', arms: ['a'] }];
  assert.throws(() => new Cascade({ tiers: [], rules: [], defaultTier: 't' }), TypeError);
  assert.throws(() => new Cascade({ tiers: [...tiers, ...tiers], rules: [], defaultTier: 't' }), TypeError);
  assert.throws(() => new Cascade({ tiers: [{ name: 't', arms: [] }], rules: [], defaultTier: 't' }), TypeError);
  assert.throws(() => new Cascade({ tiers, rules: [], defaultTier: 'nope' }), TypeError);
  assert.throws(
    () => new Cascade({ tiers, rules: [{ tier: 'nope', reason: 'r', when: () => true }], defaultTier: 't' }),
    TypeError,
  );
  assert.throws(() => new Cascade({ tiers, rules: [], defaultTier: 't', escalateAfterFailures: 0 }), RangeError);
});

test('router: a tier restricts the choice, and an open circuit reroutes within it', async () => {
  let now = 0;
  const bandit = new Bandit({ rng: seededUniform(8), coldStartMinPulls: 0, now: () => now });
  const breaker = new CircuitBreaker({ failureThreshold: 2, openMs: 1000, now: () => now });
  const router = new Router({
    bandit,
    breaker,
    arms: [
      { armKey: 'a:tiny', provider: 'a' },
      { armKey: 'a:mid', provider: 'a' },
      { armKey: 'b:mid', provider: 'b' },
      { armKey: 'c:big', provider: 'c' },
    ],
  });
  for (let i = 0; i < 40; i++) {
    bandit.update('chat', 'c:big', { success: true, costMicros: 0 });
    bandit.update('chat', 'a:mid', { success: true, costMicros: 0 });
    bandit.update('chat', 'b:mid', { success: false, costMicros: 0 });
  }
  const tier = cascade.decide({ kind: 'feature', tokens: 10 });
  for (let i = 0; i < 20; i++) {
    assert.notEqual((await router.route('chat', tier.arms)).chosen?.armKey, 'c:big');
  }
  await assert.rejects(router.route('chat', ['nope']), TypeError);

  await router.record('chat', 'a:mid', { success: false, costMicros: 0, error: 'HTTP 503' });
  await router.record('chat', 'a:mid', { success: false, costMicros: 0, error: 'HTTP 503' });
  assert.equal(breaker.isOpen('a'), true);
  const rerouted = await router.route('chat', tier.arms);
  assert.equal(rerouted.chosen?.armKey, 'b:mid');
  assert.equal(rerouted.refusals[0]?.refusal.reason, 'circuit_open:a');

  const retired = { success: false, costMicros: 0, error: 'HTTP 410 gone' };
  await router.record('chat', 'b:mid', retired);
  await router.record('chat', 'b:mid', retired);
  assert.equal(breaker.isOpen('b'), false, 'dead-arm errors do not open the provider circuit');
  assert.equal(bandit.isDead('chat', 'b:mid'), true);

  now = 1000;
  assert.equal((await router.route('chat', tier.arms)).chosen?.armKey, 'a:mid');
});
