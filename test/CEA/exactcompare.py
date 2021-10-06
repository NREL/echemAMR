import yt
from sys import argv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from scipy.optimize import fsolve

def nonlinsolve_func(a1,data):
    sigma_c=data[0]
    sigma_e=data[1]
    sigma_a=data[2]
    ocp_c=data[3]
    ocp_a=data[4]
    j0=data[5]
    phi0=data[6]
    x1=data[7]
    x2=data[8]
    L=data[9]
    k_c=np.arcsinh(-sigma_c*a1/j0)*phi0
    val=a1*(x1-sigma_c/sigma_e*(x1+x2)-sigma_c/sigma_a*(L-x2)-2*x1*(1-sigma_c/sigma_e))+2*k_c+ocp_c-ocp_a
    return(val)

def bvfunction_inv(jbv,j0):
    
    #sinh definition
    finv=-np.arcsinh(jbv/j0)
    #linear case
    #finv=-jbv/j0
    return(finv)

#main
font={'family':'Helvetica', 'size':'15'}
mpl.rc('font',**font)
mpl.rc('xtick',labelsize=15)
mpl.rc('ytick',labelsize=15)
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
fld_pot = np.array(frb["Potential"])[res//2,:]

#=======================================
#exact solution
#=======================================
j0=3.0
phi0=1.0
ocp_c=1.0
ocp_a=0.2
sigma_e=1.0
sigma_a=5.0
sigma_c=2.0
x1=clength/3
x2=2*x1

#linear solution
data=[sigma_c,sigma_e,sigma_a,ocp_c,ocp_a,j0,phi0,x1,x2,clength]
a1=fsolve(nonlinsolve_func,0.0,args=data)
a2=sigma_c/sigma_e*a1
a3=sigma_c/sigma_a*a1

k_c=np.arcsinh(-sigma_c*a1/j0)*phi0
b1=0.0
b2=a1*x1*(1-sigma_c/sigma_e)-ocp_c-k_c
b3=-a3*clength

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
ax.plot(x,fld_pot,'k-',label="echemAMR",markersize=2)
ax.plot(x,phi_exact,'r^-',label="exact",markersize=6,markevery=1)
ax.legend(loc="best")

dir_char=chr(ord('x')+int(axialdir))
fig.suptitle("Potential solution along "+dir_char+" direction (CEA)")
plt.savefig("pot_"+dir_char+"_CEA.png")
plt.show()
#=======================================

