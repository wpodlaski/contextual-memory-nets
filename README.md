# contextual-memory-nets

Simulation and analysis code accompanying the paper: 

Podlaski, Agnes, &amp; Vogels 2025. "High capacity and dynamic accessibility in associative memory networks with context-dependent neuronal and synaptic gating", Physical Review X.

All code is written in Python 3. Numerical capacity estimations require pytorch, ideally with GPU/CUDA compatibility for reasonable runtimes.

Breakdown of the code:

- `figure_generation_code`: contains code for generating all figures of the paper
    (dependencies: numpy, matplotlib, seaborn)
- `figures`: folder where figures are stored (empty)
- `simulation_code`: contains code for running all theoretical and numerical results
    - `theory_functions.py` : main functions that compute theory (dependencies: numpy)
    - `run_cxt_mod_theory.py` : runs and stores all theoretical results (dependencies: numpy)
    - `cxt_mod_hopfield.py` : contains the main network class used for all simulations (dependencies: pytorch, numpy)
    - `simulation_functions.py` : contains the main functions for simulating recall and stability numerics (dependencies: pytorch, numpy)
    - `run_rand_nrn_gating_numerics.py` : runs all capacity and stability numerics for random neuronal gating (dependencies: pytorch, sklearn, matplotlib, seaborn, tqdm)
    - ... (each model has an accompanying run file) 
- `data`: folder where theory and numerics data is stored
    - `data/theory/gauss_hopfield.pkl` : stores numerical functions for \alpha_\Delta and m_\Delta (Eqs. 6, 8 in the paper)
    - `data/numerics/overlaps/` : stores all numerical overlap data used to estimate capacity (note that these folders are empty and will only be populated if the numerical capacity scripts are run first)
