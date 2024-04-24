import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sys import argv
import json

FARADCONST=96485.332
GASCONST=8.3144598

def exchcurr_c(c,cmax):
    ic=c/cmax
    i0=5.0*(ic-ic**2)*4.0
    return(i0)

def ocp_c(c,cmax):
    ic=c/cmax
    ocp_c=5.0*(1.0-ic)
    return(ocp_c)

def exchcurr_a(c,cmax):
    ic=c/cmax
    i0=6.0*(ic-ic**2)*4.0
    #i0=0.3;
    return(i0)

def ocp_a(c,cmax):
    ic=c/cmax
    ocp_a=0.8*(1.0-ic)
    #ocp_a=0.6
    return(ocp_a)


def bvfunction(c,phia,phie,phi0,camax,ccmax,is_anode):
    ocp=0.0
    j0=0.0
    if(is_anode):
        j0=exchcurr_a(c,camax)
        ocp=ocp_a(c,camax)
    else:
        j0=exchcurr_c(c,ccmax)
        ocp=ocp_c(c,ccmax)

    #sinh definition
    jbv=j0*np.sinh((phia-phie-ocp)/phi0)
    return(jbv)

def bvfunc_inv(jbv,j0):
    finv=np.arcsinh(jbv/j0)
    return(finv)


def get_cathode_voltage(c,ccmax,phie,j,phi0):
    j0=exchcurr_c(c,ccmax)
    ocp=ocp_c(c,ccmax)
    finv=bvfunc_inv(j,j0)
    phic=finv*phi0+ocp+phie
    return(phic)

def get_electrolyte_voltage(c,camax,phia,j,phi0):

    #assume anode is grounded
    j0=exchcurr_a(c,camax)
    ocp=ocp_a(c,camax)
    #j=-j0*sinh((phia-phie-ocp)/phi0)
    finv=bvfunc_inv(-j,j0)
    phie=phia-ocp-finv*phi0
    return(phie)

if __name__ == "__main__":
    infile=open(argv[1])
    inpt = json.load(infile)
    finaltime=inpt["finaltime"]
    nsteps=inpt["nsteps"]
    cathodevol=inpt["cathodevolume"]
    anodevol=inpt["anodevolume"]
    elytevol=inpt["electrolytevolume"]
    area=inpt["area"]
    jin=inpt["inputcurrent"]
    camax=inpt["camax"]
    ccmax=inpt["ccmax"]
    ca_init=inpt["ca_init"]
    cc_init=inpt["cc_init"]
    ce_init=inpt["ce_init"]
    temperature=inpt["temperature"]

    phi0=2.0*GASCONST*temperature/FARADCONST
    print("phi0:",phi0)
    
    dt=finaltime/nsteps
    tarr=np.linspace(0,finaltime,nsteps+1)

    ca=np.zeros(nsteps+1)
    ce=np.zeros(nsteps+1)
    cc=np.zeros(nsteps+1)
    phia=np.zeros(nsteps+1)
    phie=np.zeros(nsteps+1)
    phic=np.zeros(nsteps+1)

    #initial condition
    ca[0]=ca_init
    cc[0]=cc_init
    ce[:]=ce_init #doesnt change
    phia[:]=0.0 #grounded
    phie[0]=get_electrolyte_voltage(ca[0],camax,phia[0],jin,phi0)
    phic[0]=get_cathode_voltage(cc[0],ccmax,phie[0],jin,phi0)
    print("cathode and electrolyte voltages:",phic[0],phie[0])

    for i in range(nsteps):
        cc[i+1]=cc[i]-(jin/FARADCONST)*(area/cathodevol)*dt
        ca[i+1]=ca[i]+(jin/FARADCONST)*(area/anodevol)*dt
        phie[i+1]=get_electrolyte_voltage(ca[i+1],camax,phia[i+1],jin,phi0)
        phic[i+1]=get_cathode_voltage(cc[i+1],ccmax,phie[i+1],jin,phi0)


    np.savetxt("zerod_analytic.dat",np.transpose(np.vstack((tarr,ca,cc,phic,phie))),delimiter=" ")
