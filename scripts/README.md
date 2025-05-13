# Description of scripts

The scripts within the `experiments` folder include the following for each instance family:

- `alpine_comparison`: Compares default Alpine, Alpine+SP, and Alpine+ML.
- `mccormick`: Solves the termwise McCormick relaxation.
- `presolve`: Solves the pooling instances to local optimality using KNitro.
- `strong_partitioning`: Uses Strong Partitioning to determine the initial partitioning points.

The scripts within the `ML` folder implement the AdaBoost regression-based approximation of Strong Partitioning.

The scripts within the `plots` folder can be used to generate the plots presented in the paper.

Note: `d=2` and `d=4` correspond to using two and four partitioning points per variable, respectively.