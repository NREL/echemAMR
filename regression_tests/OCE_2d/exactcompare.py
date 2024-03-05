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
font={'family':'Helvetica', 'size':'15'}
mpl.rc('font',**font)
mpl.rc('xtick',labelsize=15)
mpl.rc('ytick',labelsize=15)
root_dirctory = os.path.abspath(os.path.join(argv[1], os.pardir))+"/"
ds=yt.load(argv[1])

prob_lo=ds.domain_left_edge.d
prob_hi=ds.domain_right_edge.d
lengths=prob_hi-prob_lo
res3d=ds.domain_dimensions
print(res3d)
res=res3d[0]
print(res)
lb = yt.LineBuffer(ds, (prob_lo[0], 0.5*(prob_lo[1]+prob_hi[1]), lengths[2]/2), \
        (prob_hi[0], 0.5*(prob_lo[1]+prob_hi[1]), lengths[2]/2), res)
fld_pot = lb["Potential"]
fld_lset = lb["levelset"]
print(lb.keys())
oned_length=lengths[0]
x = np.linspace(0,oned_length,res)

ind1=np.argmax(np.abs(np.diff(fld_lset[0:res//2])))
ind2=np.argmax(np.abs(np.diff(fld_lset[res//2+1:])))
x1=x[ind1+1]
x2=x[ind2+res//2+1+1];
print(x1,x2)

#=======================================
#exact solution
#=======================================
ocp=0.2
#linear solution
phi_exact=np.zeros(res)

for i in range(res):
    if(x[i]<x1):
        phi_exact[i]=0.0
    elif(x[i]>=x1 and x[i]<x2):
        phi_exact[i]=ocp
    else:
        phi_exact[i]=0.0

np.savetxt("exactsoln_"+str(res)+".dat",np.transpose(np.vstack((x,phi_exact))),delimiter="  ")
np.savetxt("pot_"+str(res)+".dat",np.transpose(np.vstack((x,fld_pot))),delimiter="  ")
#=======================================
