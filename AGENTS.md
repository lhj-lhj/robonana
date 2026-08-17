# Repository instructions

- Reuse FACT and official FLUX.2 modules; do not copy upstream transformer, dataset, or trainer implementations.
- Keep new code limited to adapters, masking, the unified FLUX wrapper, losses, and tests.
- Preserve the token order documented in `docs/INHERITANCE.md`.
- Run all executable validation only on `pyromind-west1-58` in `/workspace/hongjia/robonana`.
- Every validated change must include tests, a focused commit, and a push to `https://github.com/lhj-lhj/robonana`.
- Never commit datasets, model checkpoints, caches, generated rollouts, or credentials.

