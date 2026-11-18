import {
  applyObservation,
  costDiscountedReward,
  defaultArm,
  posteriorMean,
} from './arm.ts';
import type { Arm, ArmUpdateOptions, DeadArmReason, Observation, RewardOptions } from './arm.ts';
import { createBetaSampler, cryptoUniform } from './sampler.ts';
import type { BetaSampler, UniformRng } from './sampler.ts';

/** taskType -> armKey -> arm. Plain JSON, so any store can persist it. */
export type BanditState = Record<string, Record<string, Arm>>;

export interface BanditOptions extends RewardOptions, ArmUpdateOptions {
  rng?: UniformRng;
  now?: () => number;
  state?: BanditState;
  /** An arm with fewer pulls than this gets an additive score boost. Default 10. */
  coldStartMinPulls?: number;
  /**
   * Size of that boost at zero pulls; decays linearly to 0 at coldStartMinPulls. Default 2.
   * A warm arm scores at most 1, so a new arm is certain to be tried while its
   * boost is above 1: the first coldStartMinPulls * (1 - 1/boost) pulls (5 by
   * default). After that the boost only tilts the draw.
   */
  coldStartBoost?: number;
  /** Share of the score that depends on the arm's failover-hop history. Default 0.15. */
  fallbackWeight?: number;
  /** Observations before the failover dimension stops being neutral. Default 20. */
  fallbackMinObservations?: number;
  /** A dead arm becomes selectable again after this long. Default 24h. */
  deadArmTtlMs?: number;
}

export interface ScoredArm {
  armKey: string;
  score: number;
  posteriorMean: number;
  sampleCount: number;
}

export interface ArmStats {
  armKey: string;
  sampleCount: number;
  successRate: number;
  posteriorMean: number;
  avgCostMicros: number;
  dead: boolean;
}

const UNSAFE_KEYS = new Set(['__proto__', 'constructor', 'prototype']);

function assertSafeKey(key: string): void {
  if (typeof key !== 'string' || UNSAFE_KEYS.has(key)) {
    throw new TypeError(`armwright: ${JSON.stringify(key)} is not a valid task type or arm key`);
  }
}

/**
 * Online Thompson-sampling bandit over model arms, per task type, with cost
 * folded into the reward. Synchronous and storage-free: read `state` to
 * persist it, pass it back in to resume.
 */
export class Bandit {
  readonly state: BanditState;
  private readonly sampler: BetaSampler;
  private readonly now: () => number;
  private readonly opts: BanditOptions;

  constructor(opts: BanditOptions = {}) {
    this.opts = opts;
    this.state = opts.state ?? {};
    this.sampler = createBetaSampler(opts.rng ?? cryptoUniform);
    this.now = opts.now ?? Date.now;
  }

  private own(taskType: string, armKey: string): Arm | undefined {
    assertSafeKey(taskType);
    assertSafeKey(armKey);
    const arms = Object.hasOwn(this.state, taskType) ? this.state[taskType] : undefined;
    return arms && Object.hasOwn(arms, armKey) ? arms[armKey] : undefined;
  }

  private arm(taskType: string, armKey: string): Arm {
    return this.own(taskType, armKey) ?? defaultArm();
  }

  isDead(taskType: string, armKey: string): boolean {
    const arm = this.own(taskType, armKey);
    if (arm?.deadAt === undefined) return false;
    return this.now() - arm.deadAt < (this.opts.deadArmTtlMs ?? 24 * 60 * 60 * 1000);
  }

  score(taskType: string, armKey: string): number {
    const arm = this.arm(taskType, armKey);
    const draw = this.sampler.sampleBeta(arm.alpha, arm.beta);
    const fallbackWeight = this.opts.fallbackWeight ?? 0.15;
    const warm = arm.sampleCount >= (this.opts.fallbackMinObservations ?? 20);
    const avgFallbacks = warm ? arm.totalFallbacks / arm.sampleCount : 0;
    const fallbackFactor = warm ? Math.max(1 - avgFallbacks * 0.2, 0) : 0.5;
    const minPulls = this.opts.coldStartMinPulls ?? 10;
    const coldFraction = minPulls > 0 ? Math.max(0, 1 - arm.sampleCount / minPulls) : 0;
    return (
      draw * (1 - fallbackWeight + fallbackWeight * fallbackFactor) +
      (this.opts.coldStartBoost ?? 2) * coldFraction
    );
  }

  /** Every live candidate, best score first. One posterior draw per arm per call. */
  rank(taskType: string, candidates: readonly string[]): ScoredArm[] {
    if (!taskType) throw new TypeError('rank: taskType is required');
    const scored: ScoredArm[] = [];
    for (const armKey of new Set(candidates)) {
      if (this.isDead(taskType, armKey)) continue;
      const arm = this.arm(taskType, armKey);
      scored.push({
        armKey,
        score: this.score(taskType, armKey),
        posteriorMean: posteriorMean(arm),
        sampleCount: arm.sampleCount,
      });
    }
    return scored.sort((a, b) => b.score - a.score);
  }

  select(taskType: string, candidates: readonly string[]): ScoredArm | null {
    return this.rank(taskType, candidates)[0] ?? null;
  }

  /** Record an outcome. Returns the reward credited to the arm. */
  update(taskType: string, armKey: string, obs: Observation): number {
    if (!taskType || !armKey) throw new TypeError('update: taskType and armKey are required');
    const reward = costDiscountedReward(obs.success, obs.costMicros, this.opts);
    const next = applyObservation(this.arm(taskType, armKey), obs, this.now(), reward, this.opts);
    if (!Object.hasOwn(this.state, taskType)) this.state[taskType] = {};
    const arms = this.state[taskType]!;
    if (next.deadAt !== undefined && !this.isDead(taskType, armKey)) {
      delete next.deadAt;
      delete next.deadReason;
    }
    arms[armKey] = next;
    return reward;
  }

  /** Suppress an arm in every task type that has seen it. Returns those task types. */
  markDead(armKey: string, reason: DeadArmReason): string[] {
    assertSafeKey(armKey);
    const touched: string[] = [];
    for (const [taskType, arms] of Object.entries(this.state)) {
      const arm = Object.hasOwn(arms, armKey) ? arms[armKey] : undefined;
      if (!arm) continue;
      arms[armKey] = { ...arm, deadReason: reason, deadAt: this.now() };
      touched.push(taskType);
    }
    return touched;
  }

  stats(taskType: string): ArmStats[] {
    return Object.entries(this.state[taskType] ?? {}).map(([armKey, arm]) => ({
      armKey,
      sampleCount: arm.sampleCount,
      successRate: arm.sampleCount ? arm.successCount / arm.sampleCount : 0,
      posteriorMean: posteriorMean(arm),
      avgCostMicros: arm.sampleCount ? arm.totalCostMicros / arm.sampleCount : 0,
      dead: this.isDead(taskType, armKey),
    }));
  }
}
