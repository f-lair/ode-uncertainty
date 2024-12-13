# Bayesian Filtering for Black Box Simulators


![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![License](https://img.shields.io/github/license/f-lair/ode-uncertainty)

Master thesis in the Methods of Machine Learning group at the University of Tübingen.

## Abstract

Ordinary differential equations (ODEs) are essential for modeling the behavior of complex dynamical systems in a variety of disciplines.
In the absence of an analytical solution, however, the use of numerical integration methods, which are always subject to error, is often unavoidable.
Thus, there is an interest in probabilistic solutions that take the uncertainty in numerical simulations into account.
Another challenging problem is determining the model parameters that best explain experimental measurements of the real system.
In the event of such needs, it might be difficult to replace a simulator that has previously been optimized for a specific practical application.

In this work, methods for both problems are developed that allow the usage of existing ODE solvers as a black box, provided that they are differentiable.
For probabilistic solutions, an estimator for the local error is also required, which is usually available, e.g., in commonly used embedded Runge-Kutta methods.
The two proposed methods then apply Bayesian filtering techniques, primarily the extended Kalman Filter, to perform inference in a probabilistic model over the ODE solution.
While the first method models the uncertainty about the solution through estimates of the local error, the second method, which we call _process noise tempering_, gradually reduces the added noise during gradient-based optimization of the parameters, facilitating convergence to the global optimum.

An experimental evaluation shows that the produced probabilistic ODE solutions capture the structure of the uncertainty on a qualitative level, but are not always calibrated ideally.
The use of overly large step sizes for simulation, however, can lead to catastrophic failure in the form of mode collapses.
Process noise tempering, on the other hand, proves to estimate parameters reliably even for complex Hodgkin-Huxley models with more than ten parameters.



## Installation

To run the scripts in this repository, **Python 3.10** is needed.
Then, simply create a virtual environment and install the required packages via

```bash
pip install -r requirements.txt
```

## Usage

All scripts are located in the subdirectory `scripts/`.

**Important:** Scripts must be run inside that directory.

For the details, invoke the scripts with the flag `-h`/`--help`.
