## How to run

In order to run the kSZ fisher forecast simply run the run_axion_fisher_forecast.py python script in the folder fisher_analysis. You can edit the fiducial axion abundances, step size prescirptions, and various paramters there. Note that the script requires mpi support, i.e. it can't be run as `python run_axion_fisher_forecast.py` but rather you need something like `srun -n 20 -c 2 python run_axion_fisher_forecast.py`.

## Required packages

numpy

scipy

sympy

dill

mpi4py

pandas

mcfit (for FFTLog intergal transforms)

## Settings to edit

line 14 of fisher_analysis.run_axion_fisher_forecast.py: Edit to include path to this package

auxiliary.config.py: edit to include output paths and paths to axionCAMB installation
