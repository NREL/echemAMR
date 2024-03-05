#ifndef _TRANSPORT_H_
#define _TRANSPORT_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace electrochem_transport {
    AMREX_GPU_DEVICE AMREX_INLINE
    void potential_mixedbc(int i, int j, int k,
                           int dir, int sgn,
                           Array4<Real> const& phi,
                           Array4<Real> const& robin_a,
                           Array4<Real> const& robin_b,
                           Array4<Real> const& robin_f,
                           GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                           GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                           GpuArray<Real, AMREX_SPACEDIM> dx,
                           const Real time,
                           const Real bclo,
                           const Real bchi)
    {
        const int im1 = (dir == 0) ? i - 1 : i;
        const int jm1 = (dir == 1) ? j - 1 : j;
        const int km1 = (dir == 2) ? k - 1 : k;

        if(sgn == -1)
        {
            //pure dirichlet
            robin_a(i, j, k) = phi(i,j,k,LS_ID);
            robin_b(i, j, k) = 1.0-phi(i,j,k,LS_ID);
            robin_f(i, j, k) = 0.0;
        } 
        else
        { 
            //pure dirichlet
            robin_a(i, j, k) = phi(im1,jm1,km1,LS_ID);
            robin_b(i, j, k) = 1.0-phi(im1,jm1,km1,LS_ID);
            robin_f(i, j, k) = 0.0;
        }

    }
    AMREX_GPU_DEVICE AMREX_INLINE
    void species_linsolve_bc(int i,int j,int k, int dir, int sgn, 
                             int spec_id, Array4<Real> const &phi, Array4<Real> const& bc,
                             GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                             GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                             GpuArray<Real, AMREX_SPACEDIM> dx,
                             const Real time,
                             ProbParm const& prob_parm)
    {

    }
    AMREX_GPU_DEVICE AMREX_INLINE void potential_bc(
        int i,
        int j,
        int k,
        int dir,
        int sgn,
        Array4<Real> const& phi,
        Array4<Real> const& bc,
        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
        GpuArray<Real, AMREX_SPACEDIM> prob_hi,
        GpuArray<Real, AMREX_SPACEDIM> dx,
        const Real time,
        const Real bclo,
        const Real bchi)
    {
        const int im1 = (dir == 0) ? i - 1 : i;
        const int jm1 = (dir == 1) ? j - 1 : j;
        const int km1 = (dir == 2) ? k - 1 : k;

        if (sgn == -1)
        { // lo sides
            bc(im1, jm1, km1) = bclo;
        } else
        { // hi sides
            bc(i, j, k) = bchi;
        }
    }

    AMREX_GPU_DEVICE AMREX_INLINE void compute_dcoeff(
        int i,
        int j,
        int k,
        Array4<Real> const& phi,
        Array4<Real> const& dcoeff,
        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
        GpuArray<Real, AMREX_SPACEDIM> prob_hi,
        GpuArray<Real, AMREX_SPACEDIM> dx,
        const Real time,
        ProbParm const& prob_parm)
    {
        dcoeff(i, j, k, S1_ID) = 0.1;
        dcoeff(i, j, k, LS_ID) = 0.0;
    }

    AMREX_GPU_DEVICE AMREX_INLINE void compute_eps(
        int i,
        int j,
        int k,
        Array4<Real> const& phi,
        Array4<Real> const& ecoeff)
    {

    }

    AMREX_GPU_DEVICE AMREX_INLINE void compute_vel(
        int i,
        int j,
        int k,
        int dir,
        Array4<Real> const& phi,
        Array4<Real> const& vel,
        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
        GpuArray<Real, AMREX_SPACEDIM> prob_hi,
        GpuArray<Real, AMREX_SPACEDIM> dx,
        const Real time,
        ProbParm const& prob_parm)
    {
        vel(i, j, k, S1_ID) = 0.0;
        vel(i, j, k, LS_ID) = 0.0;
    }

    AMREX_GPU_DEVICE AMREX_INLINE void compute_potential_dcoeff(
        int i,
        int j,
        int k,
        Array4<Real> const& phi,
        Array4<Real> const& dcoeff,
        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
        GpuArray<Real, AMREX_SPACEDIM> prob_hi,
        GpuArray<Real, AMREX_SPACEDIM> dx,
        const Real time,
        ProbParm const& prob_parm)
    {
        dcoeff(i, j, k) = 1.0;
    }

    AMREX_GPU_DEVICE AMREX_INLINE
    amrex::Real compute_kdstar_atface(int i, int j, int k, int idim,
                                      Array4<Real> const &phi, 
                                      GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                                      GpuArray<Real, AMREX_SPACEDIM> prob_hi, 
                                      GpuArray<Real, AMREX_SPACEDIM> dx, 
                                      const Real time, 
                                      ProbParm const& prob_parm)
    {
        amrex::Real kdstar=0.0;
        return(kdstar);
    }

} // namespace electrochem_transport
#endif
