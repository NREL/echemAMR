import yt
from sys import argv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def bvfunction_inv(jbv,j0):
    
    #sinh definition
    finv=-np.arcsinh(jbv/j0)
    #linear case
    #finv=-jbv/j0
    return(finv)

#axialdir = int(argv[2])
ds=yt.load(argv[1])

axialdir=0
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
frb = slc.to_frb((1.0,'cm'),res)
x = np.linspace(0,clength,res)
fld_pot = np.array(frb["potential"])[res//2,:]

#=======================================
#exact solution
#=======================================
j_app=3.0
ocp=0.2
phi_right=0.0
sigma_a=3.0
sigma_e=1.0
j0=3.0
phi0=1.0
x_interface=0.5

phi_exact=np.zeros(res)
finv=bvfunction_inv(j_app,j0)
Ba=-finv*phi0+ocp+j_app/sigma_e-j_app*x_interface*(1/sigma_e-1/sigma_a)

for i in range(res):
    if(x[i]<x_interface):
        phi_exact[i]=-j_app/sigma_a*x[i]+Ba
    else:
        phi_exact[i]=-j_app/sigma_e*(x[i]-clength)


#=======================================
#Plot solutions
#=======================================
fig,ax=plt.subplots(1,1,figsize=(4,4))
ax.plot(x,fld_pot,'k-',label="echemAMR",markersize=2)
ax.plot(x,phi_exact,'r^',label="exact",markersize=2)
ax.legend(loc="best")

dir_char=chr(ord('x')+int(axialdir))
fig.suptitle("potential solution along "+dir_char+" direction ")
plt.savefig("pot_"+dir_char+".png")
plt.show()
#=======================================

