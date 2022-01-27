import yt
from sys import argv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from mpi4py import MPI

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

print(fn_list)

avgconc_anode=np.zeros(len(fn_list))
avgconc_cathode=np.zeros(len(fn_list))
avgconc_elyte=np.zeros(len(fn_list))
avgpot_anode=np.zeros(len(fn_list))
avgpot_cathode=np.zeros(len(fn_list))
avgpot_elyte=np.zeros(len(fn_list))
chargetime=np.zeros(len(fn_list))

for i, fn in enumerate(fn_list):

    ds=yt.load(fn)
    ad=ds.all_data()
    prob_lo=ds.domain_left_edge.d
    prob_hi=ds.domain_right_edge.d
    lengths=prob_hi-prob_lo

    anodevfrac=np.array(ad["Anode"])
    cathodevfrac=np.array(ad["Cathode"])
    elytevfrac=np.array(ad["Electrolyte"])
    conc=np.array(ad["Concentration"])
    pot=np.array(ad["Potential"])
    
    chargetime[i]=ds.current_time
    avgconc_anode[i]=np.mean(anodevfrac*conc)/np.mean(anodevfrac)
    avgconc_cathode[i]=np.mean(cathodevfrac*conc)/np.mean(cathodevfrac)
    avgconc_elyte[i]=np.mean(elytevfrac*conc)/np.mean(elytevfrac)

    avgpot_anode[i]=np.mean(anodevfrac*pot)/np.mean(anodevfrac)
    avgpot_cathode[i]=np.mean(cathodevfrac*pot)/np.mean(cathodevfrac)
    avgpot_elyte[i]=np.mean(elytevfrac*pot)/np.mean(elytevfrac)

outarr=np.array([chargetime,avgconc_anode,avgconc_cathode,avgconc_elyte,
    avgpot_anode,avgpot_cathode,avgpot_elyte])

np.savetxt("avgdata",np.transpose(outarr),delimiter="   ")
