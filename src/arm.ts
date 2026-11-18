export const MICROS_PER_USD = 1_000_000;

export type DeadArmReason = 'model_not_found' | 'model_deprecated' | 'auth_revoked' | 'gone';

export interface Arm {
  /** Discounted sum of rewards, floored at 1. */
  alpha: number;
  /** Discounted sum of (1 - reward), floored at 1. */
  beta: number;
  sampleCount: number;
  successCount: number;
  totalCostMicros: number;
  /** Failover hops accumulated across observations. */
  totalFallbacks: number;
  lastUpdatedAt: number;
  deadReason?: DeadArmReason;
  deadAt?: number;
}

export interface RewardOptions {
  /** Cost at which a success is worth half a reward. Default $0.50. */
  costCeilingMicros?: number;
  /** Floor so a free call does not divide by near-zero. Default 100 micros. */
  costFloorMicros?: number;
}

export interface ArmUpdateOptions {
  /** Per-observation discount on the posterior, so old evidence fades. Default 0.995. */
  discount?: number;
  /** When either parameter passes this, both are halved. Default 10_000. */
  overflowCap?: number;
}

export function defaultArm(): Arm {
  return {
    alpha: 1,
    beta: 1,
    sampleCount: 0,
    successCount: 0,
    totalCostMicros: 0,
    totalFallbacks: 0,
    lastUpdatedAt: 0,
  };
}

/**
 * The one place cost enters arm selection: `success ? C / (C + max(cost, floor)) : 0`.
 * A failure is worth nothing however cheap it was; a success is worth less the more it cost.
 * The posterior accumulates this, so the score applies no second cost multiplier.
 */
export function costDiscountedReward(
  success: boolean,
  costMicros: number,
  opts: RewardOptions = {},
): number {
  if (!Number.isFinite(costMicros) || costMicros < 0) {
    throw new RangeError(`costDiscountedReward: costMicros must be finite and >= 0, got ${costMicros}`);
  }
  if (!success) return 0;
  const ceiling = opts.costCeilingMicros ?? 500_000;
  const floor = opts.costFloorMicros ?? 100;
  return ceiling / (ceiling + Math.max(costMicros, floor));
}

export function posteriorMean(arm: Pick<Arm, 'alpha' | 'beta'>): number {
  return arm.alpha / (arm.alpha + arm.beta);
}

export interface Observation {
  success: boolean;
  costMicros: number;
  /** Failover hops this call needed before it resolved. */
  fallbacks?: number;
}

/** Pure posterior update. Returns a new arm; never mutates the input. */
export function applyObservation(
  arm: Arm,
  obs: Observation,
  now: number,
  reward: number,
  opts: ArmUpdateOptions = {},
): Arm {
  if (!(reward >= 0 && reward <= 1)) {
    throw new RangeError(`applyObservation: reward must be in [0, 1], got ${reward}`);
  }
  const discount = opts.discount ?? 0.995;
  const cap = opts.overflowCap ?? 10_000;
  let alpha = Math.max(1, arm.alpha * discount + reward);
  let beta = Math.max(1, arm.beta * discount + (1 - reward));
  if (alpha > cap || beta > cap) {
    alpha = Math.max(1, Math.floor(alpha / 2));
    beta = Math.max(1, Math.floor(beta / 2));
  }
  return {
    ...arm,
    alpha,
    beta,
    sampleCount: arm.sampleCount + 1,
    successCount: arm.successCount + (obs.success ? 1 : 0),
    totalCostMicros: arm.totalCostMicros + obs.costMicros,
    totalFallbacks: arm.totalFallbacks + (obs.fallbacks ?? 0),
    lastUpdatedAt: now,
  };
}
