#ifndef _PROB_H_
#define _PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <Chemistry.H>
#include <ProbParm.H>
#include <echemAMR_constants.H>
#include <echemAMR.H>

using namespace amrex;

AMREX_INLINE
void amrex_probinit (ProbParm * h_prob_parm, ProbParm * d_prob_parm)
{
    amrex::ParmParse pp("prob");
    pp.query("interface_location", h_prob_parm->interface_location );
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_prob_parm, 
                     h_prob_parm+1,d_prob_parm);

    pp.query("electrode_conductivity",h_prob_parm->electrode_conductivity);
    pp.query("electrolyte_conductivity",h_prob_parm->electrolyte_conductivity);    
    pp.query("electrode_dcoeff",h_prob_parm->electrode_dcoeff);
    pp.query("electrolyte_dcoeff",h_prob_parm->electrolyte_dcoeff);    
    pp.query("spltcell_x",h_prob_parm->spltcell_x);
    pp.query("spltcell_y",h_prob_parm->spltcell_y);
    pp.query("spltcell_z",h_prob_parm->spltcell_z);
    
    pp.query("jin", h_prob_parm->jin);
    
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_prob_parm, 
                     h_prob_parm+1,d_prob_parm);

    echemAMR::host_global_storage->pot_bc_lo[0]
    =h_prob_parm->jin/h_prob_parm->electrolyte_conductivity;
    
    amrex::Print()<<"lo bc:"<<echemAMR::host_global_storage->pot_bc_lo[0]<<"\n";
}

AMREX_GPU_DEVICE
AMREX_INLINE
void initdomaindata(Box const& bx, Array4<Real> const& phi, GeometryData const& geomdata,
        ProbParm *prob_parm)
{
    const auto lo = lbound(bx);
    const auto hi = ubound(bx);    

    const Real* AMREX_RESTRICT prob_lo = geomdata.ProbLo();
    const Real* AMREX_RESTRICT prob_hi = geomdata.ProbHi();
    const Real* AMREX_RESTRICT dx      = geomdata.CellSize();

    Real int_loc=prob_parm->interface_location;
    int transport_dir=0;
    int spltcell[3]={prob_parm->spltcell_x,prob_parm->spltcell_y,prob_parm->spltcell_z};
    
    const Real Lx = prob_hi[0] - prob_lo[0];
    const Real Ly = prob_hi[1] - prob_lo[1];
    const Real Lz = prob_hi[2] - prob_lo[2];

    Real rad=Ly*0.8;
    Real rad2=rad*rad;
    Real center1[AMREX_SPACEDIM];
    Real center2[AMREX_SPACEDIM];

    center1[0]=prob_lo[0];
    center1[1]=prob_hi[1];
    center1[2]=prob_lo[2]+0.5*Lz;
    
    center2[0]=prob_lo[0];
    center2[1]=prob_lo[1];
    center2[2]=prob_lo[2]+0.5*Lz;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (GPU::notInLaunchRegion)
#endif
    for (int k = lo.z; k <= hi.z; ++k) 
    {
        for (int j = lo.y; j <= hi.y; ++j) 
        {
            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i) 
            {
                phi(i,j,k,LS_ID)=0.0;
                for(int kk=0;kk<spltcell[2];kk++)
                {
                    for(int jj=0;jj<spltcell[1];jj++)
                    {
                        for(int ii=0;ii<spltcell[0];ii++)
                        {
                            Real xx= prob_lo[0]+(i+(ii+0.5)/spltcell[0])*dx[0];
                            Real yy= prob_lo[1]+(j+(jj+0.5)/spltcell[1])*dx[1];
                            Real zz= prob_lo[2]+(j+(kk+0.5)/spltcell[2])*dx[2];

                            Real dist1_2=pow((xx-center1[0]),2.0)+pow((yy-center1[1]),2.0)+pow((zz-center1[2]),2.0);
                            Real dist2_2=pow((xx-center2[0]),2.0)+pow((yy-center2[1]),2.0)+pow((zz-center2[2]),2.0);
                
                            Real localrad_2=std::pow(rad+0.1*rad*sin(2*M_PI*yy*5.0/Ly),2.0);
                
                            if(dist1_2 < localrad_2 || dist2_2 < localrad_2)
                            {
                                phi(i,j,k,LS_ID)+=1;
                            }

                        }
                    }
                }
                phi(i,j,k,LS_ID)=phi(i,j,k,LS_ID)/Real(spltcell[0]*spltcell[1]*spltcell[2]);
                phi(i,j,k,LS_ID)=1.0-phi(i,j,k,LS_ID);

                if(phi(i,j,k,LS_ID)>0.0)
                {
                   phi(i,j,k,LI_ID)=prob_parm->C_e_init;
                }
                else
                {
                    phi(i,j,k,LI_ID)=prob_parm->C_c_init;
                }
                phi(i,j,k,EFX_ID) = 0.0;
                phi(i,j,k,EFY_ID) = 0.0;
                phi(i,j,k,EFZ_ID) = 0.0;
                phi(i,j,k,POT_ID) = 0.0;
            }
        }
    }
}

AMREX_INLINE
void initialconditions(ProbParm & h_prob_parm, ProbParm & d_prob_parm)
{

}

AMREX_GPU_DEVICE
AMREX_INLINE
void initproblemdata(Box const& bx, Array4<Real> const& phi, GeometryData const& geomdata,
                     ProbParm* prob_parm)
{

}

AMREX_INLINE
void print_init_data(ProbParm* prob_parm)
{

}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void externalbc(const amrex::Real x[AMREX_SPACEDIM],
                const amrex::Real s_int[],
                amrex::Real s_ext[],
                const int idir,
                const int sgn,
                const amrex::Real time,
                amrex::GeometryData const& geomdata)
{
    Real dirc_val_S1;

    //default to extrapolation
    for(int c=0;c<NVAR;c++)
    {
        s_ext[c]=s_int[c];
    }
}
#endif
