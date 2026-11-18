import type { DeadArmReason } from './arm.ts';

export interface DeadArmVerdict {
  reason: DeadArmReason;
  providerError: string;
}

const STATUS_PATTERNS = [
  /\((\d{3})\)/,
  /status\s*:?\s*(\d{3})\b/i,
  /HTTP\s*(\d{3})\b/i,
  /\bcode\s*:?\s*(\d{3})\b/i,
];

const MODEL_NOT_FOUND = [
  'model_not_found',
  'model not found',
  'model_does_not_exist',
  'model does not exist',
  'does not exist or you do not have access',
  'unknown model',
];

const MODEL_DEPRECATED = [
  'model is deprecated',
  'has been deprecated',
  'is no longer supported',
  'this model is retired',
  'model has been retired',
  'is decommissioned',
  'no longer available',
];

const AUTH_REVOKED = [
  'api key has been revoked',
  'api key is invalid',
  'api key revoked',
  'invalid api key',
  'key has been disabled',
  'key has been deactivated',
  'account suspended',
  'account has been disabled',
];

function extractMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === 'string') return error;
  if (typeof error === 'object' && error !== null) {
    const o = error as Record<string, unknown>;
    if (typeof o.message === 'string') return o.message;
    if (typeof o.error === 'string') return o.error;
  }
  return String(error);
}

function extractStatus(msg: string): number | null {
  for (const re of STATUS_PATTERNS) {
    const m = msg.match(re);
    if (m?.[1]) {
      const code = Number.parseInt(m[1], 10);
      if (code >= 100 && code < 600) return code;
    }
  }
  return null;
}

/**
 * Decide whether a provider error means the arm will keep failing until someone
 * changes configuration (retired model, revoked key, HTTP 410), as opposed to a
 * transient fault. Rate limits, timeouts and 5xx return null: those belong to
 * a circuit breaker, not to arm death.
 */
export function classifyDeadArm(error: unknown): DeadArmVerdict | null {
  const msg = extractMessage(error);
  if (!msg) return null;
  const lower = msg.toLowerCase();
  const providerError = msg.slice(0, 300);
  if (extractStatus(msg) === 410) return { reason: 'gone', providerError };
  if (MODEL_DEPRECATED.some((s) => lower.includes(s))) return { reason: 'model_deprecated', providerError };
  if (MODEL_NOT_FOUND.some((s) => lower.includes(s))) return { reason: 'model_not_found', providerError };
  if (AUTH_REVOKED.some((s) => lower.includes(s))) return { reason: 'auth_revoked', providerError };
  return null;
}
