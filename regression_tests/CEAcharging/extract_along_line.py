import yt
from sys import argv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.optimize import fsolve

#main
ds=yt.load(argv[1])

prob_lo=ds.domain_left_edge.d
prob_hi=ds.domain_right_edge.d
lengths=prob_hi-prob_lo
res3d=ds.domain_dimensions
print(res3d)
res=res3d[0]
eps=1e-10
print(res)
dx=lengths[:]/res3d[:]
lb = yt.LineBuffer(ds, (prob_lo[0], (0.5+eps)*(prob_lo[1]+prob_hi[1]), (0.5+eps)*(prob_lo[2]+prob_hi[2])), \
        (prob_hi[0], (0.5+eps)*(prob_lo[1]+prob_hi[1]), (0.5+eps)*(prob_lo[2]+prob_hi[2])), res)
fld_pot = lb["Potential"]
fld_conc = lb["Concentration"]
fld_lset = lb["levelset"]
fld_anode = lb["Anode"]
fld_cathode = lb["Cathode"]

xarr = np.linspace(prob_lo[0]+0.5*dx[0],prob_hi[0]-0.5*dx[0],res)
print(xarr.size,res)
print(fld_pot.size)
print(fld_conc.size)
print(fld_pot[0])
print(fld_pot[res//2])
print("anode frac:",np.mean(fld_anode))
print("cathode frac:",np.mean(fld_cathode))
np.savetxt("pot_conc.dat",np.transpose(np.vstack((xarr,fld_pot,fld_conc,fld_lset))),delimiter="  ")
