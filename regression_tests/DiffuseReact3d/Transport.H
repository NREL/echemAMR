#ifndef _TRANSPORT_H_
#define _TRANSPORT_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace electrochem_transport
{
    AMREX_GPU_DEVICE AMREX_INLINE
    void potential_bc(int i, int j, int k,
                      int dir, int sgn,
                      Array4<Real> const& phi,
                      Array4<Real> const& bc,
                      GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                      GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                      GpuArray<Real, AMREX_SPACEDIM> dx,
                      const Real time,
                      const Real bclo,
                      const Real bchi)
    {
        //This function is never called in this case
    }
    AMREX_GPU_DEVICE AMREX_INLINE
    void compute_dcoeff(int i, int j, int k,
                        Array4<Real> const& phi,
                        Array4<Real> const& dcoeff,
                        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                        GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                        GpuArray<Real, AMREX_SPACEDIM> dx,
                        const Real time,
                        ProbParm const& prob_parm)
    {
        dcoeff(i,j,k,S1_ID)=prob_parm.d1;
        dcoeff(i,j,k,S2_ID)=prob_parm.d2;
    }

    AMREX_GPU_DEVICE AMREX_INLINE void compute_eps(
        int i,
        int j,
        int k,
        Array4<Real> const& phi,
        Array4<Real> const& ecoeff)
    {

    }

    AMREX_GPU_DEVICE AMREX_INLINE
    void compute_velx(int i, int j, int k,
                      Array4<Real> const& phi,
                      Array4<Real> const& velx,
                      GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                      GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                      GpuArray<Real, AMREX_SPACEDIM> dx,
                      const Real time,
                      ProbParm const& prob_parm)
    {
        velx(i,j,k,S1_ID)=0.0;
        velx(i,j,k,S2_ID)=0.0;
    }
    AMREX_GPU_DEVICE AMREX_INLINE
    void compute_vely(int i, int j, int k,
                      Array4<Real> const& phi,
                      Array4<Real> const& vely,
                      GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                      GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                      GpuArray<Real, AMREX_SPACEDIM> dx,
                      const Real time,
                      ProbParm const& prob_parm)
    {
        vely(i,j,k,S1_ID)=0.0;
        vely(i,j,k,S2_ID)=0.0;
    }

    AMREX_GPU_DEVICE AMREX_INLINE
    void compute_velz(int i, int j, int k,
                      Array4<Real> const& phi,
                      Array4<Real> const& velz,
                      GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                      GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                      GpuArray<Real, AMREX_SPACEDIM> dx,
                      const Real time,
                      ProbParm const& prob_parm)
    {
        velz(i,j,k,S1_ID)=0.0;
        velz(i,j,k,S2_ID)=0.0;
    }


    AMREX_GPU_DEVICE AMREX_INLINE
    void compute_potential_dcoeff(int i, int j, int k,
                                  Array4<Real> const& phi,
                                  Array4<Real> const& dcoeff,
                                  GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                                  GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                                  GpuArray<Real, AMREX_SPACEDIM> dx,
                                  const Real time,
                                  ProbParm const& prob_parm)
    {
        dcoeff(i,j,k)=1.0;
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

}
#endif
