#!/bin/bash
python get_analytic_soln.py params_slanting.json 
python get_echemamr_soln.py "plt?????"
gnuplot pic0d_cellvolt.gp
gnuplot pic0d_conc.gp
