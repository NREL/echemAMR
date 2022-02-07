#ifndef _REACTIONS_H_
#define _REACTIONS_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace electrochem_reactions {
AMREX_GPU_DEVICE AMREX_INLINE void compute_react_source(
    int i,
    int j,
    int k,
    Array4<Real> const& phi,
    Array4<Real> const& reactsource,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm)
{
    reactsource(i, j, k, S1_ID) = 0.0;
    reactsource(i, j, k, LS_ID) = 0.0;
}

AMREX_GPU_DEVICE AMREX_INLINE void compute_potential_source(
    int i,
    int j,
    int k,
    Array4<Real> const& phi,
    Array4<Real> const& source,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm)
{
    source(i, j, k) = 0.0;
}

AMREX_GPU_DEVICE AMREX_INLINE void bvcurrent_and_der(int i,int j,int k,int normaldir,
        Real phi_jump,Array4<Real> const& phi,ProbParm const& prob_parm,Real &jbv,Real &jbvder)
{
    Real ocp = 0.2;
    Real j0 = 1.0;
    Real phi0 = 1.0;

    // sinh definition
    jbv = -j0 * sinh((phi_jump + ocp) / phi0);

    // linear case
    // jbv=-(j0/phi0)*(phi_jump+ocp);

    // sinh definition
    jbvder = -j0 * cosh((phi_jump + ocp) / phi0) * (1.0 / phi0);

    // linear case
    // jbvder=-(j0/phi0);
}

AMREX_GPU_DEVICE AMREX_INLINE void species_bvflux(int i,int j,int k,int n,int normaldir,
        Real phi_jump,Array4<Real> const& phi,ProbParm const& prob_parm,Real &jbv)
{
    Real ocp = 0.2;
    Real j0 = 1.0;
    Real phi0 = 1.0;

    // sinh definition
    jbv = -j0 * sinh((phi_jump + ocp) / phi0);

    // linear case
    // jbv=-(j0/phi0)*(phi_jump+ocp);
    //
    jbv=jbv/FARADCONST;
}

AMREX_GPU_DEVICE AMREX_INLINE Real bv_activation_function(Real c, Real gradc, Real gradc_cutoff)
{
    Real f;
    Real maxval = 0.25;
    Real tol = 1e-2;

    // f=(gradc > gradc_cutoff)?1.0:0.0;
    f = (c * (1 - c) > tol * maxval) ? 1.0 : 0.0;
    return (f);
}
} // namespace electrochem_reactions
#endif
