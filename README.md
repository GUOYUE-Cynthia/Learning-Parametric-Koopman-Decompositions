# Learning Parametric Koopman Decomposition
The code of the experiments of Learning  Parametric Koopman Decomposition

## Installation

This project uses `python 3.8`. Set up the project for development using the following steps:

1. In your chosen folder, create a virtual environment
    ```bash
    $python3 -m venv koopman_gpu
    ```
2. Activate the environment
    ```bash
    $source ~/.virtualenvs/koopman_gpu/bin/activate
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

We use Duffing equation, Van der Pol Mathieu oscillator, FitzHugh-Nagumo partial differential equation and Korteweg-De Vries equation as examples to show how to use this package.

### Enter [examples](./examples)
* Type this in terminal 
    ```bash
    $cd examples
    ```

#### Duffing equation
* Type this in terminal 
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

   - [evaluate_duffing.ipynb](./examples/duffing/evaluate_duffing.ipynb)

#### Van der Pol Mathieu oscillator

* Type this in terminal 
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

   - [evaluate_vdpm.ipynb](./examples/duffing/evaluate_vdpm.ipynb)


#### FitzHugh-Nagumo PDE

* Type this in terminal 
    ```bash
    $cd fhn
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

   - [evaluate_fhn.ipynb](./examples/fhn/evaluate_fhn.ipynb)

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

   - [evaluate_fhn_high_dim_u.ipynb](./examples/fhn/evaluate_fhn_high_dim_u.ipynb)


#### Korteweg-De Vries equation

* Type this in terminal 
    ```bash
    $cd kdv
    ```

1. Generate data
    ```bash
    $python generate_data_kdv.py config_kdv.json
    ```
2. Train models
    ```bash
    $python train_model_kdv.py config_kdv.json
    ```
3. Evaluate models in

   - [evaluate_kdv_mass_momentum_pk_mpc.ipynb.ipynb](./examples/fhn/evaluate_kdv_mass_momentum_pk_mpc.ipynb)