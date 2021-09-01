#ifndef _REACTIONS_H_
#define _REACTIONS_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace electrochem_reactions
{
    AMREX_GPU_DEVICE AMREX_INLINE
        void compute_react_source(int i, int j, int k,
                Array4<Real> const& phi,
                Array4<Real> const& reactsource,
                GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                GpuArray<Real, AMREX_SPACEDIM> dx,
                const Real time,
                ProbParm const& prob_parm)
        {
            reactsource(i,j,k,S1_ID) = prob_parm.r1;
            reactsource(i,j,k,S2_ID) = prob_parm.r2;
        }

    AMREX_GPU_DEVICE AMREX_INLINE
        void compute_potential_source(int i, int j, int k,
                Array4<Real> const& phi,
                Array4<Real> const& source,
                GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                GpuArray<Real, AMREX_SPACEDIM> dx,
                const Real time,
                ProbParm const& prob_parm)
        {
            source(i,j,k)=0.0;
        }

    AMREX_GPU_DEVICE AMREX_INLINE void bvcurrent_and_der(int i,int j,int k,int normaldir,
            Real phi_jump,Array4<Real> const& phi,ProbParm const& prob_parm, Real &jbv,Real &jbvder)
    {
        return(0);
    }

    AMREX_GPU_DEVICE AMREX_INLINE
        Real bv_activation_function(Real c, Real gradc, Real gradc_cutoff)
    {
       return(0);
    }
}
#endif
