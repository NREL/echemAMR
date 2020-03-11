#ifndef _compute_diff_flux_3d_H_
#define _compute_diff_flux_3d_H_

#include <AMReX_BLFort.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>

using namespace amrex;

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void compute_flux_x(int i, int j, int k,int n,
                    Array4<Real> const& phi, //state variable
                    Array4<Real> const& dcoeff,
                    Array4<Real> const& fx,
                    const GpuArray<Real, AMREX_SPACEDIM>& dx)
{

    fx(i,j,k,n) = -0.5 * (dcoeff(i,j,k,n)+dcoeff(i-1,j,k,n)) * (phi(i,j,k,n) - phi(i-1,j,k,n))/dx[0];
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void compute_flux_y(int i, int j, int k,int n,
                    Array4<Real> const& phi, //state variable
                    Array4<Real> const& dcoeff,
                    Array4<Real> const& fy,
                    const GpuArray<Real, AMREX_SPACEDIM>& dx)
{

    fy(i,j,k,n) = -0.5 * (dcoeff(i,j,k,n)+dcoeff(i,j-1,k,n)) * (phi(i,j,k,n) - phi(i,j-1,k,n))/dx[1];
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void compute_flux_z(int i, int j, int k,int n,
                    Array4<Real> const& phi, //state variable
                    Array4<Real> const& dcoeff,
                    Array4<Real> const& fz,
                    const GpuArray<Real, AMREX_SPACEDIM>& dx)
{

    fz(i,j,k,n) = -0.5 * (dcoeff(i,j,k,n)+dcoeff(i,j,k-1,n)) * (phi(i,j,k,n) - phi(i,j,k-1,n))/dx[2];
}
#endif
