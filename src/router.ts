import type { Observation } from './arm.ts';
import { Bandit } from './bandit.ts';
import type { DailyBudget } from './budget.ts';
import type { CircuitBreaker } from './circuit.ts';
import { classifyDeadArm } from './dead-arm.ts';
import { policyGate } from './policy.ts';
import type { PolicyContext, PolicyRefusal, RouteCandidate } from './policy.ts';

export interface RouterOptions {
  bandit: Bandit;
  /** Every arm the router may pick, with the provider that serves it. */
  arms: readonly RouteCandidate[];
  budget?: DailyBudget;
  /**
   * Opens on consecutive transient failures per provider. Errors classified as
   * a dead arm do not count: a retired model says nothing about the provider.
   */
  breaker?: CircuitBreaker;
  policy?: Omit<PolicyContext, 'isArmDead' | 'isWithinBudget'>;
  /**
   * Consecutive permanent-looking errors before an arm is marked dead. Default 2.
   * Provider error text is untrusted, so one matching message never kills an arm,
   * and any other outcome in between resets the count.
   */
  deadArmConfirmations?: number;
}

export interface RouteDecision {
  /** The first candidate, in bandit order, that passed the policy gate. Null if none did. */
  chosen: RouteCandidate | null;
  refusals: Array<{ candidate: RouteCandidate; refusal: PolicyRefusal }>;
}

export interface Outcome extends Observation {
  /** The provider error, when the call failed. Used to detect arms that are permanently gone. */
  error?: unknown;
}

/** Bandit ranking, then the policy gate, then outcome recording into both. */
export class Router {
  private readonly bandit: Bandit;
  private readonly budget?: DailyBudget;
  private readonly policy: RouterOptions['policy'];
  private readonly byKey: Map<string, RouteCandidate>;
  private readonly breaker?: CircuitBreaker;
  private readonly deadArmConfirmations: number;
  private readonly deadStrikes = new Map<string, number>();

  constructor(opts: RouterOptions) {
    if (opts.arms.length === 0) throw new TypeError('Router: at least one arm is required');
    this.deadArmConfirmations = opts.deadArmConfirmations ?? 2;
    if (!Number.isInteger(this.deadArmConfirmations) || this.deadArmConfirmations < 1) {
      throw new RangeError('Router: deadArmConfirmations must be an integer >= 1');
    }
    this.bandit = opts.bandit;
    this.budget = opts.budget;
    this.breaker = opts.breaker;
    this.policy = opts.policy;
    this.byKey = new Map(opts.arms.map((a) => [a.armKey, a]));
    if (this.byKey.size !== opts.arms.length) throw new TypeError('Router: duplicate armKey');
  }

  /**
   * @param armKeys Restrict this decision to a subset of the router's arms,
   *   for example the arms of a cascade tier. Defaults to every arm.
   */
  async route(taskType: string, armKeys?: readonly string[]): Promise<RouteDecision> {
    const refusals: RouteDecision['refusals'] = [];
    const blockedProviders = new Set<string>();
    const budget = this.budget;
    const breaker = this.breaker;
    for (const k of armKeys ?? []) {
      if (!this.byKey.has(k)) throw new TypeError(`route: unknown armKey ${k}`);
    }
    for (const { armKey } of this.bandit.rank(taskType, armKeys ?? [...this.byKey.keys()])) {
      const candidate = this.byKey.get(armKey)!;
      if (blockedProviders.has(candidate.provider)) continue;
      const verdict = await policyGate(candidate, {
        isCircuitOpen: breaker ? (p) => breaker.isOpen(p) : undefined,
        ...this.policy,
        isArmDead: (k) => this.bandit.isDead(taskType, k),
        isWithinBudget: budget ? (p) => budget.isWithinBudget(p) : undefined,
      });
      if (verdict.allowed) return { chosen: candidate, refusals };
      refusals.push({ candidate, refusal: verdict });
      if (verdict.scope === 'provider') blockedProviders.add(candidate.provider);
    }
    return { chosen: null, refusals };
  }

  async record(taskType: string, armKey: string, outcome: Outcome): Promise<void> {
    const candidate = this.byKey.get(armKey);
    if (!candidate) throw new TypeError(`record: unknown armKey ${armKey}`);
    this.bandit.update(taskType, armKey, outcome);
    if (outcome.costMicros > 0) await this.budget?.recordSpend(candidate.provider, outcome.costMicros);
    const dead =
      !outcome.success && outcome.error !== undefined ? classifyDeadArm(outcome.error) : null;
    if (!dead) this.breaker?.record(candidate.provider, outcome.success);
    if (!dead) {
      this.deadStrikes.delete(armKey);
      return;
    }
    const strikes = (this.deadStrikes.get(armKey) ?? 0) + 1;
    if (strikes >= this.deadArmConfirmations) {
      this.deadStrikes.delete(armKey);
      this.bandit.markDead(armKey, dead.reason);
    } else {
      this.deadStrikes.set(armKey, strikes);
    }
  }
}
