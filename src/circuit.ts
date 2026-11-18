export interface CircuitBreakerOptions {
  /** Consecutive failures that open a provider's circuit. Default 5. */
  failureThreshold?: number;
  /** How long an open circuit refuses traffic before letting one probe through. Default 60s. */
  openMs?: number;
  now?: () => number;
  onStateChange?: (provider: string, state: CircuitState) => void;
}

export type CircuitState = 'closed' | 'open' | 'half_open';

interface Entry {
  failures: number;
  openedAt: number | null;
}

/**
 * Per-provider consecutive-failure breaker. Closed until `failureThreshold`
 * failures in a row, then open for `openMs`, then half-open: traffic is allowed
 * again, the first success closes the circuit and the first failure reopens it
 * for another full window.
 */
export class CircuitBreaker {
  private readonly entries = new Map<string, Entry>();
  private readonly threshold: number;
  private readonly openMs: number;
  private readonly now: () => number;
  private readonly onStateChange?: CircuitBreakerOptions['onStateChange'];

  constructor(opts: CircuitBreakerOptions = {}) {
    this.threshold = opts.failureThreshold ?? 5;
    this.openMs = opts.openMs ?? 60_000;
    if (!Number.isInteger(this.threshold) || this.threshold < 1) {
      throw new RangeError('CircuitBreaker: failureThreshold must be an integer >= 1');
    }
    if (!Number.isFinite(this.openMs) || this.openMs <= 0) {
      throw new RangeError('CircuitBreaker: openMs must be finite and > 0');
    }
    this.now = opts.now ?? Date.now;
    this.onStateChange = opts.onStateChange;
  }

  state(provider: string): CircuitState {
    const e = this.entries.get(provider);
    if (!e || e.openedAt === null) return 'closed';
    return this.now() - e.openedAt < this.openMs ? 'open' : 'half_open';
  }

  isOpen(provider: string): boolean {
    return this.state(provider) === 'open';
  }

  record(provider: string, success: boolean): void {
    const before = this.state(provider);
    const e = this.entries.get(provider) ?? { failures: 0, openedAt: null };
    if (success) {
      e.failures = 0;
      e.openedAt = null;
    } else {
      e.failures += 1;
      if (before === 'half_open' || e.failures >= this.threshold) e.openedAt = this.now();
    }
    this.entries.set(provider, e);
    const after = this.state(provider);
    if (after !== before) this.onStateChange?.(provider, after);
  }
}
