import yt
from sys import argv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
    

#axialdir = int(argv[2])
axialdir=0
clength   = 1.0
cwidth    = 0.125
cdepth    = 0.125
ar     = (clength/cwidth)

axialdir_char=chr(ord('x')+axialdir)

ds=yt.load(argv[1])
res=100
slicedepth = cdepth/2
slc = ds.slice((axialdir+2)%3,slicedepth)
frb = slc.to_frb((1.0,'cm'),res)
x = np.linspace(0,clength,res)
fld_pot = np.array(frb["potential"])[res//2,:]

#=======================================
#Plot solutions
#=======================================
fig,ax=plt.subplots(1,1,figsize=(4,4))
ax.plot(x,fld_pot,'k-',label="echemAMR",markersize=2)
ax.legend(loc="best")

dir_char=chr(ord('x')+int(axialdir))
fig.suptitle("potential solution along "+dir_char+" direction ")
plt.savefig("pot_"+dir_char+".png")
plt.show()
#=======================================

