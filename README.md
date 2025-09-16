# Hierarchical-MCFR

This repository contains the code to replicate the experiments comparing different neural network architectures for multi-treatment causal effect estimation.

---
### About the Experiments 🤔

This project runs a series of simulated experiments to explore a key question in causal machine learning: How should we design a model when we have multiple, related treatments?

We test several models, including a flexible baseline (`mcfrnet`) and a novel structured model (`hierarchical_mcfr`), on different simulated worlds:

* 🎓 **Education**: An unordered scenario with four distinct teaching methods.
* 💊 **Medication**: A complex, ordered scenario with increasing drug dosages and potential side effects.
* 🌱 **Fertilizer**: A simple, monotonic ordered scenario with increasing fertilizer amounts.

For each world, we vary the degree of confounding (`--kappa`) to see how robust each model is when the data gets more biased. 📉

---
### Key Files for Replication

The main files for reproducing the experiments are located in the `src/` directory:

1.  **`run_all_experiments.sh`**: The master bash script that automates the entire simulation suite.
2.  **`mcfr.py`**: The core Python script that generates a dataset, builds a model, trains it, and evaluates it for a single experimental condition.
3.  **`analysis.ipynb`**: A Jupyter Notebook for aggregating all results and generating the final plots.

---
### How to Replicate

1.  **Run the Full Simulation Suite:**
    From the `src/` directory, execute the main bash script. This will generate all the raw results and save them in timestamped subdirectories within `results/`.
    ```bash
    ./run_all_experiments.sh
    ```

2.  **Analyze and Plot:**
    Open and run the cells in the `analysis.ipynb` notebook. It contains the functions to parse all the generated results and create the summary plots and tables.

---
### `mcfr.py` Key Arguments

To run individual experiments, you can call `mcfr.py` directly. The most important arguments are:

* `--scenario`: The simulation scenario to use.
    * Choices: `education`, `medication`, `fertilizer`.
* `--model_type`: The model to train.
    * Choices: `mcfrnet`, `hierarchical_mcfr`, `causal_forest`.
* `--n_samples`: The number of samples to generate (e.g., `5000`).
* `--kappa`: The treatment assignment bias parameter. A higher value means less overlap (e.g., `0.5`, `2.0`, `5.0`).
* `--seed`: The random seed for the replication run (e.g., `1`, `2`, ...).

---
### Disclaimer

Parts of the simulation, analysis, and plotting scripts were generated with the assistance of an AI.