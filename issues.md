





# viz.ipynb

1. We create an average over all episodes over all policies. But we should actually create an average over all episodes for every policy, then create the average across the policy averages. Otherwise, the estimate is incorrect if the number of episodes differs between two policies.

2. A .db file might not be read correctly.


