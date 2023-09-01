#!/bin/sh
#SBATCH --job-name=NLTX_und_rshahan
#SBATCH --nodes=1
#SBATCH -t 4-00:00		#time limit: (D-HH:MM)
#SBATCH --cpus-per-task=12
#Numerical Analysis of K in NLTX_un


time /apps/bin/python3 ~/models/SENSEq2_sin/SENSE.py

