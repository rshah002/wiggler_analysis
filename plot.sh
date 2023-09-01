#!/bin/bash


for K in 0.5 1.0 2.0 3.0 4.0 5.0 6.0 
do	
	cd K_$K
	python3 ../undulator_bessel_real_SENSE.py
	cd ..
done
