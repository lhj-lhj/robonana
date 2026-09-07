# Inspect the world prediction for every Q-selected action

Enable this diagnostic on the dynamic-batch RoboTwin evaluator by setting
`ROBONANA_SELECTED_WORLD_ROOT` to a fresh output directory. Run the regular
`action_q_rejection` evaluation with 32 candidates and horizon 48.

The server first samples candidates and selects `argmax Q`. It then calls
the same `sample_mac_world` used by critic training, conditioning on that
exact selected normalized action, current state, image tokens, and language.
It does not rerank candidates based on the diagnostic. World noise uses a
separate deterministic generator and does not advance the policy RNG.

For each task/episode seed and pre-execution control step, the client saves:

- `step_XXXX_predicted_t48.png`: one generated endpoint image at t+48.
- `step_XXXX_cam_*.png`: the three actual input camera images at t.
- `step_XXXX.json`: selected Q/index, all candidate Q values, normalized clean
  action, dispatched action, 48 reward logits/probabilities/expected rewards,
  discounted chunk return, success probability, terminal decision and timings.
- `outcome.json` and `actual_final_*.png`: real episode outcome and final cameras.

Expected reward at step i is `-1 + sigmoid(reward_logit[i])` under the current
reward convention. `chunk_return` sums these with the configured discount.
`success_probability >= 0.5` means predicted successful termination by the
fixed endpoint; timeout failure is not successful termination. Diagnostics
record the corresponding hard bootstrap mask used in imaginary training.
Neither Q nor discounted return is a success probability.

This produces one endpoint image and 48 reward predictions per action chunk,
not 48 predicted images. It adds world sampling and VAE decoding latency.
Use the same checkpoint, action seeds and sampler settings for comparisons;
simulator nondeterminism can still change outcomes.

After the run, build a local, self-contained HTML inspection report with:

```bash
python scripts/report_selected_world_eval.py --root <selected_world_root>
```

The report pairs each prediction with the next replan observation (t+48), or
the absorbing terminal image if the episode ended earlier. It labels a missing
endpoint rather than comparing against an unrelated frame. This is an online
selected-policy diagnostic, not a held-out world-model accuracy benchmark.
