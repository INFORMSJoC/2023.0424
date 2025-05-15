[![INFORMS Journal on Computing Logo](https://INFORMSJoC.github.io/logos/INFORMS_Journal_on_Computing_Header.jpg)](https://pubsonline.informs.org/journal/ijoc)

# Strong Partitioning and a Machine Learning Approximation for Accelerating the Global Optimization of Nonconvex QCQPs

This archive is distributed in association with the [INFORMS Journal on
Computing](https://pubsonline.informs.org/journal/ijoc) under a [License from Triad National Security, LLC](LICENSE).

The software and data in this repository are a snapshot of the software and data
that were used in the research reported [in the paper](https://doi.org/10.1287/ijoc.2023.0424) by R. Kannan, H. Nagarajan, and D. Deka. 


## Cite

To cite the contents of this repository, please cite both the paper and this repo, using their respective DOIs.

[The journal article](https://doi.org/10.1287/ijoc.2023.0424):

```bibtex
@article{kannan2025strong,
  author = {Kannan, Rohit and Nagarajan, Harsha and Deka, Deepjyoti},
  title = {Strong Partitioning and a Machine Learning Approximation for Accelerating the Global Optimization of Nonconvex {QCQPs}},
  journal = {INFORMS Journal on Computing},
  year = {2025},
  doi = {10.1287/ijoc.2023.0424},
}
```

[This repository](https://doi.org/10.1287/ijoc.2023.0424.cd):

```bibtex
@misc{kannan2025github,
  author =        {Kannan, Rohit and Nagarajan, Harsha and Deka, Deepjyoti},
  publisher =     {INFORMS Journal on Computing},
  title =         {Strong Partitioning and a Machine Learning Approximation for Accelerating the Global Optimization of Nonconvex {QCQPs}},
  year =          {2025},
  doi =           {10.1287/ijoc.2023.0424.cd},
  url =           {https://github.com/INFORMSJoC/2023.0424},
  note =          {Available for download at https://github.com/INFORMSJoC/2023.0424},
}  
```

## Description

This repository provides the following resources for reproducing the results in our paper:

- Codes for generating the test problems used in the study (`data` folder).
- Source code for Strong Partitioning (`src` folder).
- Sample scripts for running experiments (`scripts` folder).
- Raw experimental results (`results` folder).

Each folder includes a `README` file with additional details.

## Setup and Installation

Follow the instructions in the `src` folder to install and configure the code. The implementation depends on the following software versions:

- Julia 1.6.3
- JuMP 1.1.1
- Alpine 0.4.1
- Gurobi 9.1.2 (via Gurobi.jl 0.11.3)
- BARON 23.6.23 (via BARON.jl 0.8.2)
- CPLEX 22.1.0
- Ipopt 3.14.4 (via Ipopt.jl 1.0.3)
- KNitro 12.4.0 (via KNITRO.jl 0.13.0)
- scikit-learn 0.23.2



## Replicating

To replicate the experiments, follow the instructions provided in the `README` files within each subfolder of the `scripts` directory. The output should match the corresponding results in the `results` folder.


## Contact

For questions or feedback, please contact:
- Rohit Kannan: rohitkannan@vt.edu
- Harsha Nagarajan: harsha@lanl.gov
- Deepjyoti Deka: deepj87@mit.edu

Alternatively, please open an
[issue on the original GitHub repository](https://github.com/rohitkannan/qcqp-sp-ml/issues/new) instead of this one.