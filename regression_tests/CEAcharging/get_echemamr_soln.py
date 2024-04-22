import yt
from sys import argv
import os
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.optimize import fsolve

#main
fn_pattern= argv[1]
fn_list=[]
try:
    fn_list = sorted(glob.glob(fn_pattern), key=lambda f: int(f.split("plt")[1]))
except:
    if(fn_list==[]):
        print("using file of plotfiles..")
        infile=open(argv[1],'r')
        for line in infile:
            fn_list.append(line.split()[0])
        infile.close()
print(fn_list)


timearr=np.zeros(len(fn_list))
cellvolt=np.zeros(len(fn_list))
conc_c=np.zeros(len(fn_list))
conc_a=np.zeros(len(fn_list))
for i, fn in enumerate(fn_list):
    ds=yt.load(fn)
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
    fld_anode = lb["Anode"]
    fld_cathode = lb["Cathode"]
    fld_conc = lb["Concentration"]
    timearr[i]=ds.current_time
    cellvolt[i]=fld_pot[0]-fld_pot[-1]
    conc_c[i]=np.sum(fld_conc*fld_cathode)/np.sum(fld_cathode)
    conc_a[i]=np.sum(fld_conc*fld_anode)/np.sum(fld_anode)


np.savetxt("zerod_echemamr.dat",np.transpose(np.vstack((timearr,conc_a,conc_c,cellvolt))),delimiter="  ")
