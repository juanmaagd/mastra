# Persisted signal starvation reproduction

This regression test demonstrates that persisting an idle signal and immediately starting a run on the same thread can starve the Node.js event loop.

## Run

Use Node.js 22 and install the workspace dependencies, then run:

```sh
pnpm build:core
pnpm --filter @mastra/core exec vitest run src/agent/__tests__/agent-signals.test.ts -t "allows a run immediately after persisting an idle signal"
```

## Current behavior

The child reproduction process does not exit and is terminated with `SIGKILL` after five seconds. The test fails with an assertion showing `code: null` and `signal: "SIGKILL"`.

## Expected behavior

The immediate contender yields long enough for persisted-signal cleanup to complete, then exits normally with code 0.
