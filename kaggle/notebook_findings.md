## Findings

- Every fixed rule wins for some test batches and loses for others. Full-group validation picks strong centring and
  costs 0.03 to 0.04 balanced accuracy when the test groups have two or three items; validating on small cuts costs
  0.03 to 0.06 when they are full.
- Matching the test's cluster-size histogram does not help on crowded small-group tests: clusters merge there, look
  large, and the rule picks strong centring again.
- Clustering the validation items together with the unlabelled test batch is the only rule within 0.014 of the
  oracle on all six test batches, because the validation items then meet the same crowding and merges as the test.
  It needs enough validation groups (regret 0.011 with 30, 0.004 with 300 on the mixed test).
- Without access to the test batch, the mean over several cuts is the safer default (worst regret 0.030, against
  0.037 for full-group validation).
