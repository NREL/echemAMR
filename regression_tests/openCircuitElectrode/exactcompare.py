import yt
from sys import argv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
    
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

res=100
slicedepth = cdepth/2
slc = ds.slice((axialdir+2)%3,slicedepth)
frb = slc.to_frb((1.0,'cm'),res)
x = np.linspace(0,clength,res)
fld_pot = np.array(frb["Potential"])[res//2,:]

c=1.0;
d=0.1;
exactsoln=np.zeros(res);
exactsoln[x>=0.25]=0.2
exactsoln[x>0.75]=0.0
#exactsoln[:]-=0.1
#=======================================

#=======================================
#Plot solutions
#=======================================
fig,ax=plt.subplots(1,1,figsize=(4,4))
ax.plot(x,exactsoln,'r-',label="Exact solution")
ax.plot(x,fld_pot,'ko-',label="echemAMR",markersize=2,linewidth=0.2)
ax.legend(loc="best")

dir_char=chr(ord('x')+int(axialdir))
fig.suptitle("potential solution along "+dir_char+" direction (OCE)")
plt.savefig(root_dirctory+"pot_"+dir_char+"_OCE.png")
plt.show()
#=======================================

