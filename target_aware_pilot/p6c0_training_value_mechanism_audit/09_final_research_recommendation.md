# Final Research Recommendation

H_A is supported by consistently lower E1 M4 loss and tail mass. H_B has structural-proxy support, but learning-signal redundancy is not directly established. H_C is supported as distribution-specific poor transfer: E1 fits its own population, worsens CE on E0, and regresses on Original70 in all replicates; it is not broad negative transfer because several synthetic benchmarks improve.

## One next hypothesis

Test one hardness-exposure hypothesis: Top5 target matching failed because it selected an M4-easy training distribution. The minimal causal experiment can use only the already Concorde-labeled E0∪E1 instances for each replicate: predeclare the same 48-reference quota, a fixed tolerance on E1 Top5 OT cost, and hardness-decile quotas matching E0; then select one 10k arm and train with the identical M4→M5 protocol. Compare against the frozen E1 M5. This changes hardness exposure while keeping target assignment and all training hyperparameters controlled. No new evolution or Concorde labeling is required. Existing observations strongly motivate this test but cannot establish that restoring hardness will cause transfer recovery.

No P6-C1 action was started.
