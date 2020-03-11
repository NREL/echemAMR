#ifndef _TRANSPORT_H_
#define _TRANSPORT_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;

AMREX_GPU_DEVICE
AMREX_INLINE
void compute_dcoeff(int i, int j, int k,int n,
                         Array4<Real> const& phi,
                         Array4<Real> const& dcoeff,
                         GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                         GpuArray<Real, AMREX_SPACEDIM> dx,
                         const Real time)
{
    if(n==0)
    {
        dcoeff(i,j,k,n)=1.0;
    }
    if(n==1)
    {
        dcoeff(i,j,k,n)=0.1;
    }
}
#endif
