# Random families of nonconvex QCQPs
Scripts for generating a family of random non-convex quadratically constrained quadratic program (QCQP) instances, as described in the paper:
```bibtex
@article{kannan2025strong,
  author = {Kannan, Rohit and Nagarajan, Harsha and Deka, Deepjyoti},
  title = {Strong Partitioning and a Machine Learning Approximation for Accelerating the Global Optimization of Nonconvex {QCQPs}},
  journal = {INFORMS Journal on Computing},
  year = {2025},
  doi = {10.1287/ijoc.2023.0424},
}
```

To generate 1000 random bilinear instances with `N` variables (`N = 10, 20, 50`), enter into the `bilinear` folder and run
```
python3 generate_bilinear_instances.py --numVariables N --numInstances 1000
```

To generate 1000 random QCQP instances (including both bilinear and univariate quadratic terms) with `N` variables (`N = 10, 20, 50`), enter into the `qcqp` folder and run
```
python3 generate_qcqp_instances.py --numVariables N --numInstances 1000
```

To generate 1000 random pooling instances, enter into the `pooling` folder and run
```
python3 generate_pooling_instances.py --numInstances 1000
```
Additional data files for generating the pooling instances can be found within the `pooling` folder. These data files were generated using the scripts [here](https://github.com/poolinginstances/poolinginstances).


The `bilinear`, `qcqp`, and `pooling` folders contain the best-found solutions and optimal objective values for these instances.