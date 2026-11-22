# Changelog

## 0.1.0 - 2026-11-18

- Thompson sampling per task type, with each (task type, model) pair a Beta posterior.
- Cost enters the reward once: a success is worth `C / (C + cost)`, a failure 0.
- Fixed policy gate before dispatch (static policy, liveness, budget, admission); a refusal names its stage.
- Per-provider daily budget breakers, a per-provider circuit breaker, and dead-arm detection.
- Deterministic tier gate above the bandit that makes no model calls.
- `npm run demo` replays a seeded 20,000-call stream against four synthetic arms; it is a simulation, not a production measurement.
- Zero runtime dependencies; runs on Node 22.18+ and Cloudflare Workers.
