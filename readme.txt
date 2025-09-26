# Package Instructions

This package contains Python code for training and testing constrained and unconstrained neural ODEs for wastewater ASM1 modelling, performing multiple independent runs, and visualizing the results.

## Prerequisites

Before running the code, ensure the following prerequisites are met:

1.  **Python Environment:**
    * Python version **3.12 or higher** is required.
    * Ensure you have a properly configured Python environment.

2.  **Required Packages:**
    * All necessary Python packages must be installed. These include:
        * `torch`
        * `numpy`
        * `torchdiffeq`
        * `matplotlib`

## Instructions

Follow the steps below to use the package:

make sure you current working directory is the one that the files stored.

**1. Generate simulated ASM1 component concentration trajectory data for training, if you do not have real data. You can jump this step if you already have your own data.

* Run  the script `b3_data_generation.py` to generate sythetic ASM1 component concentration trajectory data. This will save the data file to /data folder.

* Open the `b3_data_generation.py` file and modify the parameters within the script as necessary to adjust the initial condition and time length for the simulation.

**2. Training and Testing Models (a1_ys_norm_outside_constrained_unconstrained.py):**

* Run the script `a1_ys_norm_outside_constrained_unconstrained.py` to train and test both the constrained and unconstrained NODE models.

* Open the `a1_ys_norm_outside_constrained_unconstrained.py` file and modify the parameters within the script as needed to adjust the training and testing configurations (e.g., learning rate, number of epochs, model architecture).

**3. Performing Multiple Independent Runs (a2_ys_norm_outside_constrained_unconstrained_100_runs.py):**

* The script `a2_ys_norm_outside_constrained_unconstrained_100_runs.py` is designed to perform 100 independent runs of training and testing and save the results for further analysis.

* **Important Note on Running Time:** Due to the potentially long execution time required for 100 runs, it is **highly recommended** to break down the execution into smaller batches to prevent system instability and the need for restarts.

* **Suggested Approach:** Instead of running the script for 100 iterations at once, consider running it for **10 iterations at a time, 10 separate times.** This allows for intermediate checks and reduces the risk of losing progress due to unexpected issues.

* The results of each run will be saved by the script. Refer to the script's output to determine where the results are stored.

**4. Generating Visualization Figures (h1_plot_100_run_results.py):**

* Once you have collected the results from the multiple independent runs (as described in step 2), execute the script `h1_plot_100_run_results.py`.

* This script is responsible for generating figures and visualizations based on the saved results from the 100 runs.

* Ensure that the script is configured to correctly locate and load the saved result files from the previous step.

---