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

        if (sgn == -1)
        {
            //inhomogenous Neumann 
            robin_a(im1, jm1, km1) = 0.0;
            robin_b(im1, jm1, km1) = 1.0;
            robin_f(im1, jm1, km1) = (dir==0) ? (1.0-phi(i,j,k,LS_ID))*bclo:0.0;
        }
        if(sgn == 1)
        {
            //truly mixed, homogenous Neumann at
            //electrolyte (phi=1), dirichlet at electrode (phi=0)
            robin_a(i, j, k) = 1.0-phi(im1,jm1,km1,LS_ID);
            robin_b(i, j, k) = phi(im1,jm1,km1,LS_ID);
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
        Real xlen, ylen, zlen, maxlen;
        int axialdir = 0;

        const int im1 = (dir == 0) ? i - 1 : i;
        const int jm1 = (dir == 1) ? j - 1 : j;
        const int km1 = (dir == 2) ? k - 1 : k;
        
        if (sgn == -1)
        { 
            // lo sides
            //inhomogenous Neumann
            //no current input at electrolyte (phi=1)
            bc(im1, jm1, km1) = (1.0-phi(i,j,k,LS_ID))*bclo;
        }
        if(sgn == 1)
        { // hi sides
            bc(i, j, k) = 0.0;
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
        // Diffusion coefficient = sum( diffusion_coefficient domain i * levelset i), with levelset i=1 if x within domain i, 0 otherwise
        dcoeff(i,j,k,CO_ID) = (electrochem::Ds_bulk_anode(phi(i, j, k, CO_ID), prob_parm) * phi(i,j,k,A_ID)
        + electrochem::Ds_bulk_cathode(phi(i, j, k, CO_ID), prob_parm) * phi(i,j,k,C_ID)
        + electrochem::De_bulk(phi(i, j, k, CO_ID), prob_parm) * phi(i,j,k,LS_ID)) / phi(i,j,k,MAC_ID);
        dcoeff(i, j, k, A_ID) = 0.0;
        dcoeff(i, j, k, C_ID) = 0.0;
        dcoeff(i, j, k, E_ID) = 0.0;
        dcoeff(i, j, k, S_ID) = 0.0;
        dcoeff(i, j, k, LS_ID) = 0.0;
    }

    AMREX_GPU_DEVICE AMREX_INLINE void compute_eps(
        int i,
        int j,
        int k,
        Array4<Real> const& phi,
        Array4<Real> const& ecoeff)
    {
        // set the nanoporosity per cell
        ecoeff(i, j, k, CO_ID) = (
                phi(i,j,k,A_ID) * phi(i,j,k,NP_ID)
              + phi(i,j,k,E_ID) * phi(i,j,k,NP_ID)
              + phi(i,j,k,S_ID) * phi(i,j,k,NP_ID)
              + phi(i,j,k,C_ID) * phi(i,j,k,NP_ID)
            );
        ecoeff(i, j, k, A_ID)  = 1.0;
        ecoeff(i, j, k, C_ID)  = 1.0;
        ecoeff(i, j, k, E_ID)  = 1.0;
        ecoeff(i, j, k, S_ID)  = 1.0;
        ecoeff(i, j, k, LS_ID) = 1.0;
    }

    AMREX_GPU_DEVICE AMREX_INLINE void compute_velx(
        int i,
        int j,
        int k,
        Array4<Real> const& phi,
        Array4<Real> const& velx,
        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
        GpuArray<Real, AMREX_SPACEDIM> prob_hi,
        GpuArray<Real, AMREX_SPACEDIM> dx,
        const Real time,
        ProbParm const& prob_parm)
    {
        for (int n = 0; n < phi.nComp(); ++n)
        {
            velx(i, j, k, n) = 0.0;
        }
    }
    AMREX_GPU_DEVICE AMREX_INLINE void compute_vely(
        int i,
        int j,
        int k,
        Array4<Real> const& phi,
        Array4<Real> const& vely,
        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
        GpuArray<Real, AMREX_SPACEDIM> prob_hi,
        GpuArray<Real, AMREX_SPACEDIM> dx,
        const Real time,
        ProbParm const& prob_parm)
    {
        for (int n = 0; n < phi.nComp(); ++n)
        {
            vely(i, j, k, n) = 0.0;
        }
    }

    AMREX_GPU_DEVICE AMREX_INLINE void compute_velz(
        int i,
        int j,
        int k,
        Array4<Real> const& phi,
        Array4<Real> const& velz,
        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
        GpuArray<Real, AMREX_SPACEDIM> prob_hi,
        GpuArray<Real, AMREX_SPACEDIM> dx,
        const Real time,
        ProbParm const& prob_parm)
    {
        for (int n = 0; n < phi.nComp(); ++n)
        {
            velz(i, j, k, n) = 0.0;
        }
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
        //dcoeff(i, j, k) = 1.0;
        // Conduction coefficient = sum( conductivity_coefficient domain i * levelset i), with levelset i=1 if x within domain i, 0 otherwise
        dcoeff(i,j,k) = (electrochem::Ks_bulk_anode(phi(i, j, k, CO_ID), prob_parm) * phi(i,j,k,A_ID)
        + electrochem::Ks_bulk_cathode(phi(i, j, k, CO_ID), prob_parm) * phi(i,j,k,C_ID)
        + electrochem::Ke_bulk(phi(i, j, k, CO_ID), prob_parm) * phi(i,j,k,LS_ID)) / phi(i,j,k,MAC_ID);
    }

}
#endif
