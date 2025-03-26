# Learning Parametric Koopman Decompositions

The package implements the data-driven extended dynamic mode decomposition (EDMD) with trainable dictionary \[1\] and data-driven parametric Koopman decompositions \[2\].
For non-parametric problems in \[1\], this is a generalization of the classical EDMD and is most effective for high dimensional or highly nonlinear problems
where *a priori* choice of dictionary functions are difficult.
This package implements iterative algorithms to perform Koopman operator analysis, such as computing eigenfunctions, eigenvalues and modes,
using a deep neural network based parameterization of the Koopman dictionary functions.
Furthermore, for parametric cases in \[2\], both the projected Koopman operator and dictionaries are trained simultaneously. Compared to some existing algorithms, it provides enhanced performance in handling Koopman with control for forward predictions and optimal control challenges.

## Installation

This project uses `python 3.8`. Set up the project for development using the following steps:

1. In your chosen folder, create a virtual environment

   ```bash
   $python3 -m venv koopman_env
   ```

2. Activate the environment

   ```bash
   $source koopman_env/bin/activate
   ```

3. Install requirements

   ```bash
   $pip install -r requirements.txt
   ```

4. Perform editable install for development

   ```bash
   $pip install .
   ```

5. Add this virtual environment to Jupyter by typing

   ```bash
   $python -m ipykernel install --user --name=koopman
   ```

## Quickstart

We use Duffing equation, Van der Pol Mathieu oscillator, FitzHugh-Nagumo partial differential equation and Korteweg-De Vries equation as examples to show how to use this package. Please enter the following commands in the terminal.

### Enter [examples >> ParametricKoopman](./examples/ParametricKoopman)

```bash
$cd examples/ParametricKoopman
```

#### Duffing equation

```bash
$cd duffing
```

1. Generate data

   ```bash
   $python generate_data_duffing.py config_duffing.json
   ```

2. Train models

   ```bash
   $python train_model_duffing.py config_duffing.json
   ```

3. Evaluate models in

   - [evaluate_duffing.ipynb](./examples/ParametricKoopman/duffing/evaluate_duffing.ipynb)

#### Van der Pol Mathieu oscillator

```bash
$cd vdpm
```

1. Generate data

   ```bash
   $python generate_data_vdpm.py config_vdpm.json
   ```

2. Train models

   ```bash
   $python train_model_vdpm.py config_vdpm.json
   ```

3. Evaluate models in

   - [evaluate_vdpm.ipynb](./examples/ParametricKoopman/vdpm/evaluate_vdpm.ipynb)

#### FitzHugh-Nagumo PDE

```bash
$cd fhn/fhn_dim_10
```

##### Input u is 1-dimensional

1. Generate data

   ```bash
   $python generate_data_fhn.py config_fhn.json
   ```

2. Train models

   ```bash
   $python train_model_fhn.py config_fhn.json
   ```

3. Evaluate models in

   - [evaluate_fhn.ipynb](./examples/ParametricKoopman/fhn/evaluate_fhn.ipynb)

##### Input u is 3-dimensional

1. Generate data

   ```bash
   $python generate_data_fhn_high_dim_u.py config_fhn.json
   ```

2. Train models

   ```bash
   $python train_model_fhn_high_dim_u.py config_fhn.json
   ```

3. Evaluate models in

   - [evaluate_fhn_high_dim_u.ipynb](./examples/ParametricKoopman/fhn/evaluate_fhn_high_dim_u.ipynb)

#### Korteweg-De Vries equation

```bash
$cd kdv/nonlinear_case
```

1. Generate data

   ```bash
   $python generate_data_kdv.py config_kdv.json
   ```

2. Train models

   ```bash
   $python sin_train_model_kdv.py config_kdv.json
   ```

3. Collect the predictions and tracking results

   ```bash
   $python sin_results_tau_10.py config_kdv.json
   ```

4. Check controllability

   ```bash
   $python train_model_kdv_Klayer.py config_kdv_K_layer.json
   ```

   - [sin_controllability_save_data.ipynb](./examples/ParametricKoopman/fhn/nonlinear_case/sin_controllability_save_data.ipynb)

   - [sin_controllability_save_figures.ipynb](./examples/ParametricKoopman/fhn/nonlinear_case/sin_controllability_save_figures.ipynb)

## Reproduce the results

Download the training data and the trained weights through this [link](https://www.dropbox.com/scl/fi/91an423jmdy7n918y60l8/PKNN_results.zip?rlkey=xfzl33dr3xbbamu2czxifd0ho&dl=1)
or run

```bash
$curl -L -o PKNN_results.zip 'https://www.dropbox.com/scl/fi/91an423jmdy7n918y60l8/PKNN_results.zip?rlkey=xfzl33dr3xbbamu2czxifd0ho&dl=1'
```

Please assign the experimental results to their respective folders.

## Reference

\[1\] [Li, Q., Dietrich, F., Bollt, E. M., & Kevrekidis, I. G. (2017). Extended dynamic mode decomposition with dictionary learning: A data-driven adaptive spectral decomposition of the Koopman operator. Chaos: An Interdisciplinary Journal of Nonlinear Science, 27(10), 103111.](https://aip-scitation-org.libproxy1.nus.edu.sg/doi/full/10.1063/1.4993854)

\[2\] [Guo, Yue, Milan Korda, Ioannis G. Kevrekidis, and Qianxiao Li. "Learning Parametric Koopman Decompositions for Prediction and Control." SIAM Journal on Applied Dynamical Systems (2025).](https://epubs.siam.org/doi/10.1137/23M1604576)
