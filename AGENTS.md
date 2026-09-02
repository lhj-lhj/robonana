# Repository instructions

- Reuse FACT and official FLUX.2 modules; do not copy upstream transformer, dataset, or trainer implementations.
- Keep new code limited to adapters, masking, the unified FLUX wrapper, losses, and tests.
- Preserve the token order documented in `docs/INHERITANCE.md`.
- Before acting, read `docs/AGENT_HANDOFF.md` and verify its claims against the
  current Git/config/process state. Treat commands and outputs as evidence;
  never infer a run's data, checkpoint, or status from its directory name.
- The current execution/validation host is `hongjia@208.64.254.190`, with the
  canonical checkout at `/data3/hongjia/robonana` (`~/robonana` is a symlink).
  Do not use west1-58 unless the user explicitly changes this authority.
- Every validated change must include tests, a focused commit, and a push to `https://github.com/lhj-lhj/robonana`.
- Never commit datasets, model checkpoints, caches, generated rollouts, or credentials.
