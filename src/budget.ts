import { MICROS_PER_USD } from './arm.ts';

/** The subset of a key-value store the breaker needs. Cloudflare KV satisfies it as-is. */
export interface BudgetStore {
  get(key: string): Promise<string | null>;
  put(key: string, value: string, opts?: { expirationTtl?: number }): Promise<void>;
}

export class MemoryBudgetStore implements BudgetStore {
  private readonly data = new Map<string, string>();
  async get(key: string): Promise<string | null> {
    return this.data.get(key) ?? null;
  }
  async put(key: string, value: string): Promise<void> {
    this.data.set(key, value);
  }
}

export interface BudgetBreach {
  provider: string;
  day: string;
  spentMicros: number;
  limitMicros: number;
}

export interface ProviderBudgetStatus {
  provider: string;
  spentMicros: number;
  limitMicros: number;
  breached: boolean;
}

export interface DailyBudgetOptions {
  store: BudgetStore;
  /** Daily cap per provider, in USD. A provider with no entry is uncapped. */
  limitsUsd: Record<string, number>;
  now?: () => number;
  /** Called once per provider per UTC day, on the call that crosses the cap. */
  onBreach?: (breach: BudgetBreach) => void | Promise<void>;
  keyPrefix?: string;
}

/**
 * Per-provider daily spend breaker. Spend accumulates under a key that includes
 * the UTC date, so the counter resets at the day boundary with no reset job.
 *
 * Increments are read-modify-write. On a store without atomic increment,
 * concurrent writers can undercount, so treat the cap as a runaway-loop stop
 * rather than an accounting ledger.
 */
export class DailyBudget {
  private readonly store: BudgetStore;
  private readonly limitsUsd: Record<string, number>;
  private readonly now: () => number;
  private readonly onBreach?: DailyBudgetOptions['onBreach'];
  private readonly prefix: string;

  constructor(opts: DailyBudgetOptions) {
    for (const [provider, usd] of Object.entries(opts.limitsUsd)) {
      if (!Number.isFinite(usd) || usd < 0) {
        throw new RangeError(`DailyBudget: limit for ${provider} must be finite and >= 0, got ${usd}`);
      }
    }
    this.store = opts.store;
    this.limitsUsd = opts.limitsUsd;
    this.now = opts.now ?? Date.now;
    this.onBreach = opts.onBreach;
    this.prefix = opts.keyPrefix ?? 'budget';
  }

  private day(): string {
    return new Date(this.now()).toISOString().slice(0, 10);
  }

  private key(provider: string): string {
    return `${this.prefix}:${provider}:${this.day()}`;
  }

  private limitMicros(provider: string): number | null {
    const usd = this.limitsUsd[provider];
    return usd === undefined ? null : Math.round(usd * MICROS_PER_USD);
  }

  async spentMicros(provider: string): Promise<number> {
    const raw = await this.store.get(this.key(provider));
    if (raw === null) return 0;
    const n = Number(raw);
    if (!Number.isFinite(n) || n < 0) {
      throw new Error(`DailyBudget: corrupt counter at ${this.key(provider)}: ${raw.slice(0, 40)}`);
    }
    return n;
  }

  async isWithinBudget(provider: string): Promise<boolean> {
    const limit = this.limitMicros(provider);
    if (limit === null) return true;
    return (await this.spentMicros(provider)) < limit;
  }

  async recordSpend(provider: string, costMicros: number): Promise<number> {
    if (!Number.isFinite(costMicros) || costMicros < 0) {
      throw new RangeError(`recordSpend: costMicros must be finite and >= 0, got ${costMicros}`);
    }
    const before = await this.spentMicros(provider);
    const after = before + costMicros;
    const msIntoDay = this.now() % 86_400_000;
    const ttl = Math.ceil((86_400_000 - msIntoDay) / 1000) + 3600;
    await this.store.put(this.key(provider), String(after), { expirationTtl: ttl });
    const limit = this.limitMicros(provider);
    if (limit !== null && before < limit && after >= limit) {
      await this.onBreach?.({ provider, day: this.day(), spentMicros: after, limitMicros: limit });
    }
    return after;
  }

  async snapshot(): Promise<ProviderBudgetStatus[]> {
    const out: ProviderBudgetStatus[] = [];
    for (const provider of Object.keys(this.limitsUsd)) {
      const limitMicros = this.limitMicros(provider) ?? 0;
      const spentMicros = await this.spentMicros(provider);
      out.push({ provider, spentMicros, limitMicros, breached: spentMicros >= limitMicros });
    }
    return out;
  }
}
