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
        void compute_dcoeff(int i, int j, int k,
                Array4<Real> const& phi,
                Array4<Real> const& dcoeff,
                GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                GpuArray<Real, AMREX_SPACEDIM> dx,
                const Real time)
        {
            dcoeff(i,j,k,electrochem::S1_ID)=1.0;
            dcoeff(i,j,k,electrochem::S2_ID)=0.1;
        }
    
        AMREX_GPU_DEVICE AMREX_INLINE
        void compute_vel(int i, int j, int k,
                Array4<Real> const& phi,
                Array4<Real> const& velx,
                Array4<Real> const& vely,
                Array4<Real> const& velz,
                GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                GpuArray<Real, AMREX_SPACEDIM> dx,
                const Real time)
        {
            velx(i,j,k,electrochem::S1_ID)=0.0;
            velx(i,j,k,electrochem::S2_ID)=0.0;

            vely(i,j,k,electrochem::S1_ID)=0.0;
            vely(i,j,k,electrochem::S2_ID)=0.0;

            velz(i,j,k,electrochem::S1_ID)=0.0;
            velz(i,j,k,electrochem::S2_ID)=0.0;
        }
}
#endif
