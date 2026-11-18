export type UniformRng = () => number;

export interface BetaSampler {
  sampleNormal(): number;
  sampleGamma(shape: number): number;
  sampleBeta(alpha: number, beta: number): number;
}

/** Uniform draw strictly inside (0, 1) from the platform CSPRNG. */
export function cryptoUniform(): number {
  const buf = new Uint32Array(1);
  crypto.getRandomValues(buf);
  return ((buf[0] ?? 0) + 1) / 0x1_0000_0001;
}

/** Seeded uniform generator (mulberry32). For replays and tests, not for production draws. */
export function seededUniform(seed: number): UniformRng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (((t ^ (t >>> 14)) >>> 0) + 0.5) / 4294967296;
  };
}

function openUnit(u: number): number {
  return u > 0 ? u : Number.MIN_VALUE;
}

/**
 * Normal (Box-Muller), Gamma (Marsaglia-Tsang, with the shape<1 boost) and Beta
 * (ratio of Gammas) samplers over an injected uniform source.
 */
export function createBetaSampler(rng: UniformRng): BetaSampler {
  const sampleNormal = (): number => {
    const u1 = openUnit(rng());
    const u2 = rng();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };

  const sampleGamma = (shape: number): number => {
    if (!(shape > 0) || !Number.isFinite(shape)) {
      throw new RangeError(`sampleGamma: shape must be a finite number > 0, got ${shape}`);
    }
    if (shape < 1) {
      return sampleGamma(shape + 1) * openUnit(rng()) ** (1 / shape);
    }
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x: number;
      let v: number;
      do {
        x = sampleNormal();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = openUnit(rng());
      if (u < 1 - 0.0331 * (x * x) * (x * x)) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  };

  const sampleBeta = (alpha: number, beta: number): number => {
    const x = sampleGamma(alpha);
    const y = sampleGamma(beta);
    return x / (x + y);
  };

  return { sampleNormal, sampleGamma, sampleBeta };
}

/** Monte Carlo estimate of the p-th quantile of Beta(alpha, beta). */
export function betaQuantileMc(
  sampler: BetaSampler,
  alpha: number,
  beta: number,
  p = 0.05,
  samples = 2_000,
): number {
  if (!Number.isFinite(alpha) || !Number.isFinite(beta) || alpha <= 0 || beta <= 0) {
    throw new RangeError(`betaQuantileMc: alpha and beta must be finite and > 0, got ${alpha}, ${beta}`);
  }
  if (p <= 0) return 0;
  if (p >= 1) return 1;
  const draws = new Array<number>(samples);
  for (let i = 0; i < samples; i++) draws[i] = sampler.sampleBeta(alpha, beta);
  draws.sort((a, b) => a - b);
  const idx = Math.min(samples - 1, Math.max(0, Math.floor(p * samples)));
  return draws[idx] ?? 0;
}
