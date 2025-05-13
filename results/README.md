# Description of results


The `bilinear`, `qcqp`, and `pooling` folders contain the results described in Section 6 of the paper. Each instance family includes subfolders with the following contents:

- `alpine_comparison`: Comparison of default Alpine, Alpine+SP, and Alpine+ML for each instance. (Note: The first two runs are warm-up runs to mitigate Julia's initial compilation time overhead.)
- `baron_output`: Output of BARON for each instance.
- `gurobi_output`: Output of Gurobi for each instance.
- `mccormick_output`: Solutions of the termwise McCormick relaxations for each instance.
- `ml_features`: Features used by the AdaBoost regression model for each instance.
- `ml_pred`: Predictions of the Strong Partitioning points made by the AdaBoost regression model for each instance.
- `plots`: Plots used in the paper.
- `strong_partitioning_output`: Output corresponding to Algorithm 3 in the paper.
- `strong_partitioning_points`: Strong Partitioning points for each instance.

Note: `d=2` and `d=4` correspond to using two and four partitioning points per variable, respectively.