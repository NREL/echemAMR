#ifndef _REACTIONS_H_
#define _REACTIONS_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>
#include <ProbParm.H>
#include <AMReX_Tuple.H>
#include <tuple>
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

    for (int n = 0; n < reactsource.nComp(); ++n)
    {
        reactsource(i, j, k, n) = 0.0;
    }

    /* add intercalation reaction */
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        amrex::Real normal = 1.0 / dx[dir];
        reactsource(i, j, k, CO_ID) += electrochem::ic_reaction(i, j, k, dir, +normal, phi, prob_parm) / prob_parm.Faraday_const;
        reactsource(i, j, k, CO_ID) += electrochem::ic_reaction(i, j, k, dir, -normal, phi, prob_parm) / prob_parm.Faraday_const;
    }
}

AMREX_GPU_DEVICE AMREX_INLINE void compute_potential_source(
    int i,
    int j,
    int k,
    Array4<Real> const& phi,
    Array4<Real> const& potsource,
    GpuArray<Real, AMREX_SPACEDIM> prob_lo,
    GpuArray<Real, AMREX_SPACEDIM> prob_hi,
    GpuArray<Real, AMREX_SPACEDIM> dx,
    const Real time,
    ProbParm const& prob_parm)
{

    // FIXME: add Neummann bc's to potential and concentration source terms
    if (electrochem::is_electrode(i, j, k, phi))
    {

        potsource(i, j, k) = 0.0;

    } else
    {

        // concentration
        const Real Ce = phi(i, j, k, CO_ID);
        const Real KD = electrochem::K_D(Ce, prob_parm); // KD is either KD or KDstar as a function of prob_parm.use_KDstar

        // FIXME: should probably average KD over faces but for now use use value at center
        // FIXME: check sign on this
        // FIXME: this should be KDstar // Is KDstar or KD depending on prob_parm.use_KDstar
        // FIXME: need to account for walls
        potsource(i, j, k) = KD * ((phi(i - 1, j, k, CO_ID) - 2.0 * phi(i, j, k, CO_ID) + phi(i + 1, j, k, CO_ID)) / dx[0] / dx[0] +
                                   (phi(i, j - 1, k, CO_ID) - 2.0 * phi(i, j, k, CO_ID) + phi(i, j + 1, k, CO_ID)) / dx[1] / dx[1] +
                                   (phi(i, j, k - 1, CO_ID) - 2.0 * phi(i, j, k, CO_ID) + phi(i, j, k + 1, CO_ID)) / dx[2] / dx[2]);
    }

    /* add intercalation reaction */
    for (int dir = 0; dir < AMREX_SPACEDIM; ++dir)
    {
        amrex::Real normal = 1.0 / dx[dir];
        potsource(i, j, k) += electrochem::ic_reaction(i, j, k, dir, +normal, phi, prob_parm);
        potsource(i, j, k) += electrochem::ic_reaction(i, j, k, dir, -normal, phi, prob_parm);
    }
}

AMREX_GPU_DEVICE AMREX_INLINE Real bvcurrent(Real phi_jump)
{
    Real ocp = 0.2;
    Real j0 = 3.0;
    Real phi0 = 1.0;
    Real jbv;

    // sinh definition
    jbv = -j0 * sinh((phi_jump + ocp) / phi0);

    // linear case
    // jbv=-(j0/phi0)*(phi_jump+ocp);
    return (jbv);
}

AMREX_GPU_DEVICE AMREX_INLINE Real bvcurrent_der(Real phi_jump)
{
    Real ocp = 0.2;
    Real j0 = 3.0;
    Real phi0 = 1.0;
    Real jbvder;

    // sinh definition
    jbvder = -j0 * cosh((phi_jump + ocp) / phi0) * (1.0 / phi0);

    // linear case
    // jbvder=-(j0/phi0);
    return (jbvder);
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
