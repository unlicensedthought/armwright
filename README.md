# armwright

An online, cost-aware router for LLM model arms. It learns which model to send
each kind of task to from the outcomes of real calls, and it counts what each
call cost when it does.

- **Thompson sampling per task type.** Each (task type, model) pair is a Beta
  posterior. Routing is one draw per candidate.
- **Cost lives in the reward, once.** A success is worth `C / (C + cost)`; a
  failure is worth 0. There is no second cost multiplier at selection time.
- **A fixed policy gate before dispatch.** Static policy, then liveness, then
  budget, then admission. A refusal names its stage and whether it rules out
  one arm or the whole provider.
- **Per-provider daily budget breakers** on any key-value store.
- **A deterministic tier gate above the bandit.** Rules on task metadata pick a
  cost tier, repeated failures escalate it, and the bandit chooses only among
  that tier's arms. It makes no model calls.
- **A per-provider circuit breaker** for transient failures: open after N in a
  row, one probe after a cool-off.
- **Dead-arm detection.** Two consecutive errors that say the model is retired
  or the key is revoked take the arm out for 24 hours; a rate limit or a
  timeout does not. Provider error text is untrusted, so one message is never
  enough.

Zero runtime dependencies. Runs on Node 22.18+ and on Cloudflare Workers.

Write-ups: [Cost in the reward, once](https://unlicensedthought.com/writing/cost-in-the-reward-once/) · [3,380 consecutive provider failures were a policy failure, not a transient one](https://unlicensedthought.com/writing/3380-provider-failures-in-a-row-were-a-policy-failure/)

## The numbers

`npm run demo` replays one seeded stream of 20,000 calls against four synthetic
arms. Every policy sees the same luck. **This is a simulation. None of it is a
production measurement.**

| Arm | Success rate | Cost per call |
|---|---|---|
| `alpha:frontier` | 97% | $0.0300 |
| `beta:mid` | 93% | $0.0040 |
| `gamma:small` | 80% | $0.0004 |
| `delta:flaky` | 45% | $0.0001 |

| Policy | Success | Total cost | Cost per success |
|---|---|---|---|
| Always the frontier arm | 97.0% | $600.00 | $0.0309 |
| Always the cheapest arm | 44.5% | $2.00 | $0.0002 |
| Uniform random | 78.7% | $172.93 | $0.0110 |
| Oracle (told the best arm in advance) | 92.9% | $80.00 | $0.0043 |
| **armwright, defaults** | **94.5%** | **$296.43** | **$0.0157** |
| **armwright, `costCeilingMicros: 50_000`** | **92.2%** | **$76.47** | **$0.0041** |

Seed 20261115. On seeds 1, 2 and 3 the default run costs $266 to $312 at 94.3%
to 94.8% success, and the tight-ceiling run costs $76 to $77 at 92.2% to 92.4%.

Two things to read off this:

1. With defaults the router spends about half what always-frontier spends and
   gives up 2.5 points of success. It does **not** settle on one arm: it splits
   58/42 between `beta:mid` and `alpha:frontier`, because under the default
   reward those two arms are 0.8% apart and the posterior forgets fast enough
   (see below) that it never resolves a gap that small.
2. The cost ceiling is the knob. At `$0.50` a `$0.03` call loses 6% of its
   reward, so price barely matters at these prices. Set the ceiling near your
   real per-call prices and the router leans hard toward the cheaper arm.

With a $5/day cap on provider `beta`, the same run crosses the cap at call
3,741, holds `beta` at exactly $5.00, and reroutes the remaining traffic.

## Use

```ts
import { Bandit, DailyBudget, MemoryBudgetStore, Router } from 'armwright';

const bandit = new Bandit({ costCeilingMicros: 50_000 });
const budget = new DailyBudget({
  store: new MemoryBudgetStore(), // or a Cloudflare KV namespace
  limitsUsd: { openai: 25, groq: 5 },
  onBreach: (b) => console.warn(JSON.stringify({ event: 'budget_breach', ...b })),
});
const router = new Router({
  bandit,
  budget,
  arms: [
    { armKey: 'openai:big', provider: 'openai' },
    { armKey: 'groq:small', provider: 'groq' },
  ],
});

const { chosen, refusals } = await router.route('summarize');
if (!chosen) throw new Error(`nothing routable: ${JSON.stringify(refusals)}`);

// ...call the model, then tell the router what happened:
await router.record('summarize', chosen.armKey, {
  success: true,
  costMicros: 1_840, // millionths of a dollar
});
```

To keep cheap work on cheap models, put a cascade in front and pass its arms to
`route`:

```ts
import { Cascade, CircuitBreaker } from 'armwright';

const cascade = new Cascade<{ kind: string; tokens: number; failedAttempts?: number }>({
  tiers: [
    { name: 'triage', arms: ['groq:small'] },
    { name: 'workhorse', arms: ['groq:small', 'openai:big'] },
  ],
  rules: [{ tier: 'triage', reason: 'simple chore', when: (t) => t.kind === 'echo' }],
  defaultTier: 'workhorse',
});

const tier = cascade.decide({ kind: 'echo', tokens: 40, failedAttempts: 0 });
const decision = await router.route('summarize', tier.arms);
```

Pass `breaker: new CircuitBreaker()` to the router to stop sending traffic to a
provider that is failing right now.

`bandit.state` is plain JSON. Persist it however you like and pass it back as
`new Bandit({ state })`.

You decide what "success" means. The router only needs a boolean and a cost.

## How it decides

**Reward.** `success ? C / (C + max(cost, floor)) : 0`, with `C` defaulting to
$0.50 and the floor to $0.0001.

**Update.** `alpha = max(1, alpha * 0.995 + reward)`,
`beta = max(1, beta * 0.995 + (1 - reward))`. The 0.995 discount caps the
posterior's effective memory at about 200 observations, so an arm that degrades
is noticed quickly. The price is resolution: two arms within a point of each
other will keep sharing traffic.

**Score.** One Beta draw, scaled by a failover factor (15% of the score, neutral
until the arm has 20 observations), plus a cold-start boost of 2.0 that decays
linearly to zero over the arm's first 10 pulls. A warm arm scores at most 1, so a
new arm is certain to be tried for its first 5 pulls; after that the boost only
tilts the draw.

**Gate.** Candidates are tried in score order. A provider-scoped refusal skips
that provider's other arms without re-checking them.

## What it does not do

- **It does not look at the prompt.** Routing is per task type, not per query.
  There is no complexity classifier and no contextual features.
- **It does not judge quality.** A wrong answer you report as a success is
  learned as a success.
- **It does not optimise latency.** Latency is not in the reward.
- **The budget counter is not atomic.** It is read-modify-write, so concurrent
  writers on a store without atomic increment can undercount. It stops runaway
  loops; it is not a ledger.
- **The circuit breaker and dead-arm strike counts live in memory.** They reset
  when the process restarts and are not shared between instances.
- **Nothing here calls a model.** You dispatch; the router picks and learns.

## Development

```
npm install
npm test          # node --test, 20 tests
npm run typecheck
npm run demo      # node examples/replay.ts [seed] [calls]
```

## Licence

Apache-2.0.
