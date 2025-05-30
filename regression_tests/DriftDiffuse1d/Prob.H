#ifndef _PROB_H_
#define _PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <Chemistry.H>
#include <ProbParm.H>

using namespace amrex;

AMREX_INLINE
void amrex_probinit (ProbParm * h_prob_parm, ProbParm * d_prob_parm)
{
    // read problem specific parmparse parameters here
    amrex::ParmParse pp("prob");
    pp.query("source1", h_prob_parm->r1 );
    pp.query("source2", h_prob_parm->r2 );
    
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_prob_parm, 
                     h_prob_parm+1,d_prob_parm);
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
                phi(i,j,k,S1_ID)  = 0.0;
                phi(i,j,k,S2_ID)  = 0.0;
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
    Real dirc_val_S1,dirc_val_S2;

    //default to extrapolation
    for(int c=0;c<NVAR;c++)
    {
        s_ext[c]=s_int[c];
    }

    if(sgn > 0) //left lo side
    {
        dirc_val_S1=0.0;
        dirc_val_S2=0.0;
        s_ext[POT_ID]=1.0;
    }
    else
    {
        dirc_val_S1=1.0;
        dirc_val_S2=0.0;
        s_ext[POT_ID]=0.0;
    }

    s_ext[S1_ID]=2.0*dirc_val_S1-s_int[S1_ID];
    s_ext[S2_ID]=2.0*dirc_val_S2-s_int[S2_ID];
}
#endif
