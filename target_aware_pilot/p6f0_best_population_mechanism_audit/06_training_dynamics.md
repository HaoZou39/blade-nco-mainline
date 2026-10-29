# Training dynamics

All runs use the same initialization and optimizer budget and select fixed E256, not a validation-picked checkpoint. Mean late-32-epoch CE is P5-E1 **0.06926**, B **0.06715**, C **0.07010**. Training-curve differences do not track the P5 seed ordering strongly enough to explain Original/S1 success and S2 failure.

Own-population training CE is not a generalization metric. No historical validation traces or P5 cross-population CE matrix were found, so training dynamics are an insufficient causal explanation rather than evidence that optimization was identical in every stochastic detail.
