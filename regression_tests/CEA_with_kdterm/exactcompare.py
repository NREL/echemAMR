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
frb = slc.to_frb((1.0,'cm'),res)
x = np.linspace(0+0.5*clength/res,clength-0.5*clength/res,res)
fld_pot = np.array(frb["Potential"])[res//2,:]

#=======================================
#python fd solution
#=======================================
exactsoln=np.loadtxt("soln_fd_257.dat")
np.shape(exactsoln)
x_exact=exactsoln[:,0]
phi_exact=exactsoln[:,1]
#=======================================
#Plot solutions
#=======================================
fig,ax=plt.subplots(1,1,figsize=(6,6))
ax.plot(x,fld_pot,'k*-',label="echemAMR",markersize=8,markevery=5)
ax.plot(x_exact,phi_exact,'r^-',label="exact",markersize=6,markevery=13)
ax.legend(loc="best")

dir_char=chr(ord('x')+int(axialdir))
fig.suptitle("Potential solution along "+dir_char+" direction (CEA with kd term)")
plt.savefig(root_dirctory+"pot_"+dir_char+"_CEAkd.png")
plt.show()
#=======================================
np.savetxt("echemsoln_"+axialdir_char+".dat",np.transpose(np.array([x,fld_pot])),delimiter="  ")
