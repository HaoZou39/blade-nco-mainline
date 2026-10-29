# Runtime and Compute Audit

- Evolution end-to-end wall (sum of replicate invocations): 8191.8s.
- Physical mutation wall (9 batches, 990k attempts): 2451.9s; conceptual paired budget 1,320,000 attempts.
- New Top5 extraction wall (parallel shard invocation sums): 4839.7s.
- Concorde labeling wall: 2169.2s; new calls 57,457.
- BQ training-loop wall sum: 4835.9s; maximum allocated GPU memory 1.99 GiB.
- Greedy evaluation wall sum: 4007.9s.
- Maximum recorded evolution RSS: 6371.5 MiB.
- Mutation/F40/HiGHS run on CPU; BQ training and greedy evaluation use the declared CUDA devices. All stages are independently resumable.
