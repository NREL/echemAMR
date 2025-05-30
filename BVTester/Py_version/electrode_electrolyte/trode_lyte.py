import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from sys import argv

def amean(val1,val2):
    return(0.5*(val1+val2))

def hmean(val1,val2):
    eps=1e-15
    return(2*val1*val2/(val1+val2+eps))

def activefunc(levset_f,levset_grad,levset_tol,gradlevset_tol):

    npts=len(levset_f)
    fL=np.zeros(npts)

    #half activation
    #fL=np.exp(-4.0*(levset_f-0.5)**2)
    #fL=fL-np.min(fL)
    #fL/=np.max(fL)

    #grad activation
    #fL[abs(levset_grad)>gradlevset_tol]=1.0

    #tanh
    #fL=np.tanh(abs(levset_grad))

    #field activation
    fL[levset_f*(1.0-levset_f)>levset_tol]=1.0
    #fL=levset_f*(1.0-levset_f)/0.25

    return(fL)

def bvfunction(jump_phi,ocp,j0,phi0):

    #jump here is electrolyte_pot-electrode_pot

    #sinh definition
    jbv=-j0*np.sinh((jump_phi+ocp)/phi0)

    #linear case
    #jbv=-(j0/phi0)*(jump_phi+ocp)

    return(jbv)

def bvfunction_inv(jbv,j0):

    #sinh definition
    finv=-np.arcsinh(jbv/j0)

    #linear case
    #finv=-jbv/j0

    return(finv)

def bvfunction_der(jump_phi,ocp,j0,phi0):

    #sinh definition
    bvder=-j0*np.cosh((jump_phi+ocp)/phi0)*(1.0/phi0)

    #linear case
    #bvder=-(j0/phi0)

    return(bvder)


def get_exact_soln(sigma_c,sigma_e,j0,ocp,x_interface,L,xdomain):

    #linear solution
    npts=len(xdomain)
    phi_exact=np.zeros(npts)
    finv=bvfunction_inv(j_app,j0)
    Ba=-finv*phi0+ocp+j_app/sigma_e-j_app*x_interface*(1/sigma_e-1/sigma_c)

    for i in range(npts):
        if(xdomain[i]<x_interface):
            phi_exact[i]=-j_app/sigma_c*xdomain[i]+Ba
        else:
            phi_exact[i]=-j_app/sigma_e*(xdomain[i]-L)

    return(phi_exact)

def get_finitediff_soln(maxiter,tol,xmin,xmax,dx,gradc_tol,\
        sigma_field,levset_grad,fL,fcL,ocp_c,j0,phi0,j_app,right_voltage):

    ncells=len(sigma_field)
    residual=np.zeros(ncells)

    phi_fd=np.zeros(ncells)
    fdA=np.zeros((ncells,ncells))
    fdA_expl=np.zeros((ncells,ncells))
    fdrhs=np.zeros(ncells)
    prvs_soln=np.copy(phi_fd)
    xmid=0.5*(xmin+xmax)
    outfile=open("errconv_"+str(ncells+1)+".dat","w")

    for iter in range(maxiter):
    
        fdA[:,:]=0.0
        fdrhs[:]=0.0
        fdA_expl[:,:]=0.0
        
        for i in range(ncells):
        
            ocp=ocp_c
        
            bv_sigma_iphalf = 0.0
            jbv_ip_half_f   = 0.0
            bv_sigma_imhalf = 0.0
            jbv_im_half_f   = 0.0
            linear_expl_iphalf = 0.0
            linear_expl_imhalf = 0.0
       
            if(i==0):
                grad_phi_iphalf = (phi_fd[i+1]-phi_fd[i])/dx
                sigma_iphalf    = hmean(sigma_field[i],sigma_field[i+1])*fcL[i+1]
                sigma_imhalf    = hmean(sigma_field[i],sigma_field[i])*fcL[i]
                grad_phi_imhalf = -j_app/sigma_imhalf
            elif(i==(ncells-1)):
                grad_phi_iphalf = (right_voltage-phi_fd[i])/(0.5*dx)
                grad_phi_imhalf = (phi_fd[i]-phi_fd[i-1])/dx
                sigma_iphalf    = hmean(sigma_field[i],sigma_field[i])*fcL[i+1]
                sigma_imhalf    = hmean(sigma_field[i],sigma_field[i-1])*fcL[i]
            else:
                grad_phi_iphalf = (phi_fd[i+1]-phi_fd[i])/dx
                grad_phi_imhalf = (phi_fd[i]-phi_fd[i-1])/dx
                sigma_iphalf    = hmean(sigma_field[i],sigma_field[i+1])*fcL[i+1]
                sigma_imhalf    = hmean(sigma_field[i],sigma_field[i-1])*fcL[i]
    
            #i+1/2 face
            if(abs(levset_grad[i+1])>gradc_tol):
            
                jump_phi        =  grad_phi_iphalf*levset_grad[i+1]/(levset_grad[i+1]*levset_grad[i+1])
                djbvdphi        =  bvfunction_der(jump_phi,ocp,j0,phi0)
                bv_sigma_iphalf = -djbvdphi/abs(levset_grad[i+1])*fL[i+1]
        
                #explicit terms
                jbv_ip_half_f = bvfunction(jump_phi,ocp,j0,phi0)*fL[i+1]*levset_grad[i+1]/abs(levset_grad[i+1])
        
                #linear explicit term
                linear_expl_iphalf=-djbvdphi*jump_phi*fL[i+1]*levset_grad[i+1]/abs(levset_grad[i+1])
            
            #i-1/2 face
            if(abs(levset_grad[i]) > gradc_tol):
            
                jump_phi=grad_phi_imhalf*levset_grad[i]/(levset_grad[i]*levset_grad[i])
                djbvdphi=bvfunction_der(jump_phi,ocp,j0,phi0)
                bv_sigma_imhalf = -djbvdphi/abs(levset_grad[i])*fL[i]
        
                #explicit terms
                jbv_im_half_f=bvfunction(jump_phi,ocp,j0,phi0)*fL[i]*levset_grad[i]/abs(levset_grad[i])
            
                #linear explicit term
                linear_expl_imhalf=-djbvdphi*jump_phi*fL[i]*levset_grad[i]/abs(levset_grad[i])
        

            if(i==0):
                fdA[i,i+1]=(-sigma_iphalf-bv_sigma_iphalf)/dx
                fdA[i,i]=(sigma_iphalf+bv_sigma_iphalf)/dx
                
                fdA_expl[i,i+1]=(-sigma_iphalf)/dx
                fdA_expl[i,i]=(sigma_iphalf)/dx

            elif(i==(ncells-1)):
                fdA[i,i-1]=(-sigma_imhalf-bv_sigma_imhalf)/dx
                fdA[i,i]=(sigma_iphalf+bv_sigma_iphalf)/(0.5*dx)+(sigma_imhalf+bv_sigma_imhalf)/dx
                
                fdA_expl[i,i-1]=(-sigma_imhalf)/dx
                fdA_expl[i,i]=(sigma_iphalf)/(0.5*dx)+(sigma_imhalf)/dx

            else:
                fdA[i,i]=(sigma_iphalf+bv_sigma_iphalf)/dx+(sigma_imhalf+bv_sigma_imhalf)/dx
                fdA[i,i-1]=(-sigma_imhalf-bv_sigma_imhalf)/dx
                fdA[i,i+1]=(-sigma_iphalf-bv_sigma_iphalf)/dx
                
                fdA_expl[i,i]=(sigma_iphalf)/dx+(sigma_imhalf)/dx
                fdA_expl[i,i-1]=(-sigma_imhalf)/dx
                fdA_expl[i,i+1]=(-sigma_iphalf)/dx

            fdrhs[i] += -(jbv_ip_half_f-jbv_im_half_f)-(linear_expl_iphalf-linear_expl_imhalf)
            residual[i] = -(jbv_ip_half_f-jbv_im_half_f)

            if(i==0):
                fdrhs[i] += j_app
                residual[i] += j_app
            if(i==(ncells-1)):
                fdrhs[i] += right_voltage*(sigma_iphalf+bv_sigma_iphalf)/(0.5*dx)
                residual[i] += right_voltage*(sigma_iphalf)/(0.5*dx)

        #print(np.linalg.det(fdA))
        #print(fdA)
        #print(fdrhs)
        Ax = np.dot(fdA_expl,phi_fd)
        residual = residual - Ax
        phi_fd=np.linalg.solve(fdA,fdrhs)
        #print(phi_fd)
        norm=np.sqrt(np.mean(phi_fd-prvs_soln)**2)
        resnorm=np.sqrt(np.mean(residual**2))
        outfile.write("%d\t%e\n"%(iter+1,resnorm))
        prvs_soln=np.copy(phi_fd)
        #print("done.........\n")
   
    
        if(norm<tol):
            break

    outfile.close()
    return(phi_fd)

def setup(xmin,xmax,nfaces,ctol,sigma_c,sigma_e,x1):

    xdomainface=np.linspace(xmin,xmax,nfaces)
    xdomain=0.5*(xdomainface[0:-1]+xdomainface[1:])
    ncells=len(xdomain)
    dx=xdomain[1]-xdomain[0]
    xmid=0.5*(xmin+xmax)

    sigma_field=np.zeros(ncells)
    levset_field=np.zeros(ncells)
    levset_field[(xdomain>x1)]=1.0
    #sharpness_factor=50
    #levset_field=1.0-((1.0-np.tanh(sharpness_factor*(xdomain-x1)/L))/2+(1.0+np.tanh(sharpness_factor*(xdomain-x2)/L))/2)

    cell1=np.floor((x1-xmin)/dx).astype(int)

    levset_field[cell1]=(1.0-(x1-(xmin+cell1*dx))/dx)
    #print(levset_field[cell1])
    for i in range(cell1+1,ncells):
        levset_field[i]=1.0;

    #print(cell1,cell2)
    #print(xdomain)
    #print(levset_field)

    levset_grad=np.zeros(nfaces)
    levset_grad[1:-1]=np.diff(levset_field)/dx
    levset_grad[0]=levset_grad[1]
    levset_grad[-1]=levset_grad[-2]

    for i in range(ncells):
        sigma_field[i]=sigma_c*(1.0-levset_field[i])+sigma_e*levset_field[i]

    max_levsetgrad=1.0/dx
    gradc_tol=1e-4*max_levsetgrad

    levset_f=np.zeros(nfaces)
    levset_f[1:-1]=0.5*(levset_field[0:-1]+levset_field[1:])
    levset_f[0]=levset_f[1]
    levset_f[-1]=levset_f[-2]

    fL=activefunc(levset_f,levset_grad,ctol,gradc_tol)
    #activation function complement
    fcL=1-fL

    return(xdomain,gradc_tol,sigma_field,levset_field,levset_grad,fL,fcL)

if __name__ == "__main__":

    xmin=0.0
    xmax=1.0
    L=(xmax-xmin)
    nfaces=int(argv[1])
    ctol=float(argv[2])
    x1=float(argv[3])
    showplots=int(argv[4])
    ocp_c=0.2
    sigma_c=3.0
    sigma_e=1.0
    j0=3.0
    phi0=1.0
    maxiter=10
    left_voltage=0.0
    right_voltage=0.0
    tol=1e-10
    j_app=3.0

    (xdomain,gradc_tol,sigma_field,levset_field,levset_grad,fL,fcL)=setup(xmin,xmax,nfaces,ctol,sigma_c,sigma_e,x1)
    dx=xdomain[1]-xdomain[0]
    ncells=len(xdomain)
    phi_fd=get_finitediff_soln(maxiter,tol,xmin,xmax,dx,gradc_tol,\
            sigma_field,levset_grad,fL,fcL,ocp_c,j0,phi0,j_app,right_voltage)
    
    phi_exact=get_exact_soln(sigma_c,sigma_e,j0,ocp_c,x1,L,xdomain)
    

    print(ncells,np.sqrt(np.mean(phi_fd-phi_exact)**2))
    plt.figure()
    plt.title("potential solution")
    plt.plot(xdomain,phi_fd,label="computed")
    plt.plot(xdomain,phi_exact,'r-*',label="exact")
    plt.xlabel("distance (non-dim)")
    plt.ylabel("potential (non-dim)")
    plt.legend()
    plt.figure()
    plt.title("Color field")
    plt.plot(xdomain,levset_field,'k*-')
    np.savetxt("lsetfield_"+argv[1]+".dat",np.transpose(np.vstack((xdomain,levset_field))),delimiter="   ")
    np.savetxt("soln_exact.dat",np.transpose(np.vstack((xdomain,phi_exact))),delimiter="   ")
    np.savetxt("soln_fd_"+argv[1]+".dat",np.transpose(np.vstack((xdomain,phi_fd))),delimiter="   ")
    plt.legend()
    if(showplots):
        plt.show()
