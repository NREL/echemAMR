import yt
from sys import argv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import matplotlib as mpl

#main
font={'family':'Helvetica', 'size':'15'}
mpl.rc('font',**font)
mpl.rc('xtick',labelsize=15)
mpl.rc('ytick',labelsize=15)

fn_pattern = argv[1]
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

ds=yt.load(fn_list[0])

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

for i,fn in enumerate(fn_list):

    ds=yt.load(fn)
    slc = ds.slice((axialdir+2)%3,slicedepth)
    frb = slc.to_frb((1.0,'cm'),res)
    x = np.linspace(0,clength,res)
    fld_conc = np.array(frb["Concentration"])[res//2,:]

    #=======================================
    #Plot solutions
    #=======================================

    fig,ax=plt.subplots(1,1,figsize=(6,6))
    ax.plot(x,fld_conc,'b-',linewidth=3,label="Concentration (Non-dim)")
    ax.set_ylim(0,1700)
    ax.set_xlabel("Non-dim distance")
    ax.set_xlim(0,1)
    ax.legend(loc="best")
    plt.savefig("conc_%4.4d.png"%(i))

