# Self-supervised optimization of random material microstructures in the small-data regime

# Highlight

- stochastic inversion of the complete process-structure-property chain for high-dimensional process parameters
- optimization of the marginal distribution of thermal and structural properties (as given by Hill's averaging theorem) according to flexible optimality criteria
- process parameters pertain to spectral representation (i.e. SDF) of a Gaussian Process underlying formation of binary microstructures
- active learning strategy drives adaptive enrichment of dataset with acquisition function coupled to optimality criteria
- surrogate of structure-property link based on convolutional neural network
- use of Expectation-Maximization algorithm to drive the high-dimensional stochastic inversion problem

# Content

This repository contains the implementation of a data-driven stochastic optimization procedure aimining at the full inversion of the process-structure-property chain in a high-dimensional setting, i.e. we seek to identify optimal process parameter underlying a stochastic process governing the formation of microscopic material structures, such that the marginal distribution of effective physical properties obeys some notion of optimality (exemplary, specified domain of target properties, or a desired target distribution). For this purpose a data-driven surrogate based on a convolutional neural network is coupled to an active learning approach, in order to mitigate the dependence on data and incrementally refine the predictions of the surrogate conditional on the optimality criteria as well as the current state of the trajectory in the design space (i.e., the data acquisition is wrapped in the outer-loop of the optimization algorithm). The optimization algorithm itself (inner loop) is driven by means of stochastic variational inference (pertaining to the optimal microstructures and their effective physical properties) embedded within an Expectation Maximization algorithm (providing point estimates of the optimal process parameters upon convergence).

# Dependencies


The implementation makes use of PyTorch [PyTorch](https://pytorch.org/) for automatic differentiation, surrogate and probabilistic inference, while [FEniCS](https://fenicsproject.org/) is employed to solve the homogenization problem.

* pytorch 1.7.1
* fenics 2019.1.0
* scipy 1.5.0


# Data

Both microstructures as well as their associated effective physical properties are generated and computed by the code itself. The precompued dataset used for the initial training of the surrogate (as well as determining the baseline performance) can be downloaded [here](https://syncandshare.lrz.de/getlink/fiWMg16Duz5NncYng42r7nfT/hdata.zip) as a zip file (substitute for the empty hdata folder). With the active learning scheme of course, new data points are adaptively generated at runtime.


