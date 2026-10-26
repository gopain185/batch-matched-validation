# Validating test-time centring for the test batch

Rules that read the whole test batch, such as centring each item's scores or features inside its content cluster,
have settings (how much to centre, from which cluster size on) that are chosen on validation data. The right setting
depends on the group structure of the test batch: how many items each group has and how crowded the batch is. In the
Name That LLM competition, the setting my prompt-grouped CV preferred was the worst of four on the public LB.

This notebook compares six ways of validating on a known-truth simulation (code embedded below, dataset
`locdunhat/grouped-batches-unknown-group-size`): full groups, small cuts, the mean over cuts, the worst case over
cuts, cuts matched to the test's cluster-size histogram, and validation items clustered together with the
unlabelled test batch ("embedded"). Regret is the test accuracy the chosen configuration leaves on the table
against the best configuration for that test.
