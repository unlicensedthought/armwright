export type PolicyStage = 'provider_policy' | 'dead_or_circuit' | 'budget' | 'admission';

/** Whether a refusal rules out one arm or every arm on the provider. */
export type RefusalScope = 'arm' | 'provider';

export interface RouteCandidate {
  armKey: string;
  provider: string;
}

export interface PolicyRefusal {
  allowed: false;
  stage: PolicyStage;
  reason: string;
  scope: RefusalScope;
  retryAfterMs?: number;
}

export type PolicyVerdict = { allowed: true } | PolicyRefusal;

export interface PolicyContext {
  /** When set, only these providers may be dispatched to. */
  allowedProviders?: ReadonlySet<string>;
  bannedProviders?: ReadonlySet<string>;
  disabledArms?: ReadonlySet<string>;
  isArmDead?: (armKey: string) => boolean;
  isCircuitOpen?: (provider: string) => boolean;
  isWithinBudget?: (provider: string) => boolean | Promise<boolean>;
  /** Last word: concurrency or rate admission. May return its own refusal. */
  admit?: (candidate: RouteCandidate) => PolicyVerdict | Promise<PolicyVerdict>;
  onRefusal?: (candidate: RouteCandidate, refusal: PolicyRefusal) => void;
}

/**
 * The fixed order every candidate passes before dispatch: static policy, then
 * liveness, then money, then admission. Cheapest and most permanent checks run
 * first, and the stage that refused is named so a caller can tell "never" from
 * "not now".
 */
export async function policyGate(
  candidate: RouteCandidate,
  ctx: PolicyContext,
): Promise<PolicyVerdict> {
  const refuse = (r: Omit<PolicyRefusal, 'allowed'>): PolicyRefusal => {
    const refusal: PolicyRefusal = { allowed: false, ...r };
    ctx.onRefusal?.(candidate, refusal);
    return refusal;
  };
  const { armKey, provider } = candidate;

  if (ctx.bannedProviders?.has(provider)) {
    return refuse({ stage: 'provider_policy', reason: `provider_banned:${provider}`, scope: 'provider' });
  }
  if (ctx.allowedProviders && !ctx.allowedProviders.has(provider)) {
    return refuse({ stage: 'provider_policy', reason: `provider_not_allowed:${provider}`, scope: 'provider' });
  }
  if (ctx.disabledArms?.has(armKey)) {
    return refuse({ stage: 'provider_policy', reason: 'arm_disabled', scope: 'arm' });
  }
  if (ctx.isArmDead?.(armKey)) {
    return refuse({ stage: 'dead_or_circuit', reason: 'arm_dead', scope: 'arm' });
  }
  if (ctx.isCircuitOpen?.(provider)) {
    return refuse({ stage: 'dead_or_circuit', reason: `circuit_open:${provider}`, scope: 'provider' });
  }
  if (ctx.isWithinBudget && !(await ctx.isWithinBudget(provider))) {
    return refuse({ stage: 'budget', reason: `daily_budget_exhausted:${provider}`, scope: 'provider' });
  }
  if (ctx.admit) {
    const verdict = await ctx.admit(candidate);
    if (!verdict.allowed) {
      ctx.onRefusal?.(candidate, verdict);
      return verdict;
    }
  }
  return { allowed: true };
}
