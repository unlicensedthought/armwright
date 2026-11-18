export interface CascadeTier {
  name: string;
  /** Arm keys the bandit may choose between when this tier is selected. */
  arms: readonly string[];
}

export interface CascadeRule<C> {
  tier: string;
  reason: string;
  when: (ctx: C) => boolean;
}

export interface CascadeContext {
  /** Attempts at this task that have already failed. Drives escalation. */
  failedAttempts?: number;
}

export interface CascadeOptions<C extends CascadeContext> {
  /** Cheapest first. Escalation moves toward the end of this list. */
  tiers: readonly CascadeTier[];
  /** First matching rule wins. */
  rules: readonly CascadeRule<C>[];
  defaultTier: string;
  /** Failed attempts per one-tier escalation. Default 2. */
  escalateAfterFailures?: number;
}

export interface CascadeDecision {
  tier: string;
  arms: readonly string[];
  reason: string;
  /** Set when attempt history moved the task above the tier its rule chose. */
  escalatedFrom?: string;
}

/**
 * Deterministic tier gate that sits above the bandit: rules on task metadata
 * pick a cost tier, repeated failures escalate it, and the bandit then chooses
 * among that tier's arms only. Makes no model calls and costs nothing to run.
 */
export class Cascade<C extends CascadeContext> {
  private readonly tiers: readonly CascadeTier[];
  private readonly index: Map<string, number>;
  private readonly rules: readonly CascadeRule<C>[];
  private readonly defaultTier: string;
  private readonly escalateAfter: number;

  constructor(opts: CascadeOptions<C>) {
    if (opts.tiers.length === 0) throw new TypeError('Cascade: at least one tier is required');
    this.index = new Map(opts.tiers.map((t, i) => [t.name, i]));
    if (this.index.size !== opts.tiers.length) throw new TypeError('Cascade: duplicate tier name');
    for (const t of opts.tiers) {
      if (t.arms.length === 0) throw new TypeError(`Cascade: tier ${t.name} has no arms`);
    }
    for (const name of [opts.defaultTier, ...opts.rules.map((r) => r.tier)]) {
      if (!this.index.has(name)) throw new TypeError(`Cascade: unknown tier ${name}`);
    }
    this.escalateAfter = opts.escalateAfterFailures ?? 2;
    if (!Number.isInteger(this.escalateAfter) || this.escalateAfter < 1) {
      throw new RangeError('Cascade: escalateAfterFailures must be an integer >= 1');
    }
    this.tiers = opts.tiers;
    this.rules = opts.rules;
    this.defaultTier = opts.defaultTier;
  }

  decide(ctx: C): CascadeDecision {
    const failed = ctx.failedAttempts ?? 0;
    if (!Number.isInteger(failed) || failed < 0) {
      throw new RangeError(`Cascade: failedAttempts must be an integer >= 0, got ${failed}`);
    }
    const rule = this.rules.find((r) => r.when(ctx));
    const base = rule?.tier ?? this.defaultTier;
    const baseIdx = this.index.get(base)!;
    const idx = Math.min(this.tiers.length - 1, baseIdx + Math.floor(failed / this.escalateAfter));
    const tier = this.tiers[idx]!;
    const reason = rule?.reason ?? 'default';
    if (idx === baseIdx) return { tier: tier.name, arms: tier.arms, reason };
    return {
      tier: tier.name,
      arms: tier.arms,
      reason: `${reason}; escalated after ${failed} failed attempts`,
      escalatedFrom: base,
    };
  }
}
