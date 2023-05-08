# Energy Dispatch Experiment

This folder contains the code and data for the energy dispatch experiment presented in the paper _End-to-End Learning for Stochastic Optimization: A Bayesian Perspective_.

## Files and Folders

- `build_folder_structure.py`: Builds the folder structure for the experiments. This creates all the folders to hold the results.
- `e2e-cal.py`: Contains experiments and implementation for constraint-aware layers.
- `e2e-opl.py`: Contains experiments and implementation for optimization problem layers.
- `mle.py`: Contains experiments and implementation for the baseline based on Maximum Likelihood Estimation (MLE).
- `or-baselines.py`: Contains experiments and implementation for the baselines of traditional methods without considering contextual information and the oracle-based baseline (which knows the true wind energy).
- `run.py`: Contains code which runs the experiment for all methods.
- `analyse_results.py`: Contains code for analyzing the experiment results.
- `data`: A folder containing the dataset for the experiment.

## Running the Experiment

To run the experiment, navigate to the 'Energy Dispatch' folder in your terminal and execute the following command:
```
python run.py
```
This will run the energy dispatch experiment using the provided code and dataset.
