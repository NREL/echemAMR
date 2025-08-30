import yt
from sys import argv
import os
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
root_dirctory = os.path.abspath(os.path.join(argv[1], os.pardir))+"/"
ds=yt.load(argv[1])

prob_lo=ds.domain_left_edge.d
prob_hi=ds.domain_right_edge.d
lengths=prob_hi-prob_lo
res3d=ds.domain_dimensions
print(res3d)
res=int(np.sqrt(res3d[0]*res3d[0]+res3d[1]*res3d[1]))
print(res)
lb = yt.LineBuffer(ds, (prob_lo[0], prob_lo[1], lengths[2]/2), \
        (prob_hi[0], prob_hi[1], lengths[2]/2), res)
fld_pot = lb["Potential"]
fld_lset = lb["levelset"]
oned_length=np.sqrt(lengths[0]*lengths[0]+lengths[1]*lengths[1])
x = np.linspace(0,oned_length,res)

#=======================================
#exact solution
#=======================================
j0=3.0
phi0=1.0
ocp_c=1.0
ocp_a=0.2
sigma_a=float(argv[2])
sigma_c=float(argv[3])
sigma_e=float(argv[4])

#default
#x1=float(argv[2])*lengths[0]/np.sqrt(2)
#x2=oned_length-float(argv[2])*lengths[0]/np.sqrt(2)
#print(x1,x2)

ind1=np.argmax(np.abs(np.diff(fld_lset[0:res//2])))
ind2=np.argmax(np.abs(np.diff(fld_lset[res//2+1:])))
x1=x[ind1+1]
x2=x[ind2+res//2+1+1];
print(x1,x2)
#x1=oned_length/4
#x2=3*oned_length/4


#linear solution
data=[sigma_c,sigma_e,sigma_a,ocp_c,ocp_a,j0,phi0,x1,x2,oned_length]
a1=fsolve(nonlinsolve_func,0.0,args=data)[0]
a2=sigma_c/sigma_e*a1
a3=sigma_c/sigma_a*a1

k_c=np.arcsinh(-sigma_c*a1/j0)*phi0
b1=0.0
b2=a1*x1*(1-sigma_c/sigma_e)-ocp_c-k_c
b3=-a3*oned_length

phi_exact=np.zeros(res)

for i in range(res):
    if(x[i]<x1):
        phi_exact[i]=a1*x[i]+b1
    elif(x[i]>=x1 and x[i]<x2):
        phi_exact[i]=a2*x[i]+b2
    else:
        phi_exact[i]=a3*x[i]+b3

np.savetxt("exactsoln_"+str(res)+".dat",np.transpose(np.vstack((x,phi_exact))),delimiter="  ")
np.savetxt("pot_"+str(res)+".dat",np.transpose(np.vstack((x,fld_pot))),delimiter="  ")
