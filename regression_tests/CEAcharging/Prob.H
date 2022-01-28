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
void amrex_probinit (ProbParm & h_prob_parm, ProbParm & d_prob_parm)
{
    amrex::Vector<amrex::Real> plo;
    amrex::Vector<amrex::Real> phi;

    amrex::Real lenx,leny,lenz,vol;
    amrex::Real phi_c,phi_a,phi_cell,time_charge;

    // read problem specific parmparse parameters here
    amrex::ParmParse pp("prob");
    pp.query("interface1", h_prob_parm.int1 );
    pp.query("interface2", h_prob_parm.int2 );

#ifdef AMREX_USE_GPU
    amrex::Gpu::htod_memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#else
    std::memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#endif

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

    amrex::Real lenx=prob_hi[0]-prob_lo[0];
    
    Real xi1=prob_parm->int1*lenx;
    Real xi2=prob_parm->int2*lenx;
    
    
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (GPU::notInLaunchRegion)
#endif
    for (int k = lo.z; k <= hi.z; ++k) 
    {
        for (int j = lo.y; j <= hi.y; ++j) 
        {
            Real z = prob_lo[2] + (0.5+k) * dx[2];
            Real y = prob_lo[1] + (0.5+j) * dx[1];
            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i) 
            {
                Real x = prob_lo[0] + (0.5+i) * dx[0];
                Real xmid=0.5*(prob_lo[0]+prob_hi[0]);
                
                if(x<xi1)
                {
                    phi(i,j,k,A_ID)  = 0.0;
                    phi(i,j,k,C_ID)  = 1.0;
                    phi(i,j,k,E_ID)  = 0.0;
                    phi(i,j,k,LS_ID) = 0.0;
                    phi(i,j,k,CO_ID) = prob_parm->C_c_init;
                    phi(i,j,k,EFX_ID) = 0.0;
                    phi(i,j,k,EFY_ID) = 0.0;
                    phi(i,j,k,EFZ_ID) = 0.0;
                    phi(i,j,k,POT_ID) = 4.0;
                }
                else if(x>xi1 && x<=xi2)
                {
                    phi(i,j,k,A_ID)  = 0.0;
                    phi(i,j,k,C_ID)  = 0.0;
                    phi(i,j,k,E_ID)  = 1.0;
                    phi(i,j,k,LS_ID) = 1.0;
                    phi(i,j,k,CO_ID) = prob_parm->C_e_init;
                    phi(i,j,k,EFX_ID) = 0.0;
                    phi(i,j,k,EFY_ID) = 0.0;
                    phi(i,j,k,EFZ_ID) = 0.0;
                    phi(i,j,k,POT_ID) = 0.0;
                }
                else 
                {
                    phi(i,j,k,A_ID)  = 1.0;
                    phi(i,j,k,C_ID)  = 0.0;
                    phi(i,j,k,E_ID)  = 0.0;
                    phi(i,j,k,LS_ID) = 0.0;
                    phi(i,j,k,CO_ID) = prob_parm->C_a_init;
                    phi(i,j,k,EFX_ID) = 0.0;
                    phi(i,j,k,EFY_ID) = 0.0;
                    phi(i,j,k,EFZ_ID) = 0.0;
                    phi(i,j,k,POT_ID) = 0.3;
                } 
            }
        }
    }
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
    //default to extrapolation
    for(int c=0;c<NVAR;c++)
    {
        s_ext[c]=s_int[c];
    }
}
#endif
