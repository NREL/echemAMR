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
#axialdir = int(argv[2])
root_dirctory = os.path.abspath(os.path.join(argv[1], os.pardir))+"/"
ds=yt.load(argv[1])

axialdir=np.argmax(ds.domain_dimensions)
prob_lo=ds.domain_left_edge.d
prob_hi=ds.domain_right_edge.d
lengths=prob_hi-prob_lo
clength   = lengths[axialdir]
cwidth    = lengths[(axialdir+1)%3]
cdepth    = lengths[(axialdir+2)%3]
ar     = (clength/cwidth)

axialdir_char=chr(ord('x')+axialdir)
res=ds.domain_dimensions[axialdir]
slicedepth = cdepth/2
slc = ds.slice((axialdir+2)%3,slicedepth)
frb = slc.to_frb(((clength,'cm'),(cwidth,'cm')),res)
x = np.linspace(0,clength,res)
fld_pot = np.array(frb["Potential"])[res//2,:]

#=======================================
#exact solution
#=======================================
j0=0.3
phi0=0.05
ocp_c=3.5
ocp_a=0.6
sigma_e=1.0
sigma_a=10.0
sigma_c=5.0
x1=0.357*clength
x2=0.5*clength
jin=20.0

#linear solution
a1=-jin/sigma_c
a2=-jin/sigma_e
a3=-jin/sigma_a
k1=np.arcsinh(jin/j0)*phi0
#k1=(jin/j0)*phi0
rhs1=k1+a2*x1-a1*x1+ocp_c
rhs2=ocp_c+ocp_a-a1*x1+a2*(x1+x2)-a3*(x2-clength)
b3=-a3*clength
b2=rhs1-rhs2
b1=rhs1+b2

phi_exact=np.zeros(res)

for i in range(res):
    if(x[i]<x1):
        phi_exact[i]=a1*x[i]+b1
    elif(x[i]>=x1 and x[i]<x2):
        phi_exact[i]=a2*x[i]+b2
    else:
        phi_exact[i]=a3*x[i]+b3

#=======================================
#Plot solutions
#=======================================
fig,ax=plt.subplots(1,1,figsize=(6,6))
ax.plot(x,fld_pot,'k^',label="echemAMR",markersize=2)
ax.plot(x,phi_exact,'r-',label="exact",markersize=6,markevery=1)
ax.legend(loc="best")

dir_char=chr(ord('x')+int(axialdir))
fig.suptitle("Potential solution along "+dir_char+" direction (CEA)")
plt.savefig(root_dirctory+"pot_"+dir_char+"_CEA.png")
plt.show()
#=======================================

