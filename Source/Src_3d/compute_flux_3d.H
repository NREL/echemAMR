#ifndef _compute_flux_3d_H_
#define _compute_flux_3d_H_

#include <AMReX_BLFort.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>

using namespace amrex;

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void compute_flux_x(int i, int j, int k,int n,
                    Array4<Real> const& phi, //state variable
                    Array4<Real> const& vel,
                    Array4<Real> const& dcoeff,
                    Array4<Real> const& fx,
                    const GpuArray<Real, AMREX_SPACEDIM>& dx)
{

    fx(i,j,k,n) = -0.5 * (dcoeff(i,j,k,n)+dcoeff(i-1,j,k,n)) * (phi(i,j,k,n) - phi(i-1,j,k,n))/dx[0];
    
    //find face vel
    Real smallval  = 1e-10;
    Real r_left    = (phi(i-1,j,k,n) - phi(i-2,j,k,n))/(phi(i,j,k,n) - phi(i-1,j,k,n) + smallval);
    Real lim_left  = std::max(0.0,std::min(1.0,r_left));

    Real r_right   = (phi(i,j,k,n)  - phi(i-1,j,k,n)) / (phi(i+1,j,k,n) - phi(i,j,k,n) + smallval);
    Real lim_right = std::max(0.0,std::min(1.0,r_right));
    
    Real phi_L     = phi(i-1,j,k,n) + 0.5 * lim_left  * (phi(i-1,j,k,n) - phi(i-2,j,k,n));
    Real phi_R     = phi(i,j,k,n)   - 0.5 * lim_right * (phi(i+1,j,k,n) - phi(i,j,k,n));

    Real vel_mid   = 0.5*(vel(i-1,j,k,n) + vel(i,j,k,n));
    fx(i,j,k,n)   += phi_L *0.5 *(vel_mid + fabs(vel_mid))
                   + phi_R *0.5 *(vel_mid - fabs(vel_mid));
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void compute_flux_y(int i, int j, int k,int n,
                    Array4<Real> const& phi, //state variable
                    Array4<Real> const& vel,
                    Array4<Real> const& dcoeff,
                    Array4<Real> const& fy,
                    const GpuArray<Real, AMREX_SPACEDIM>& dx)
{

    fy(i,j,k,n) = -0.5 * (dcoeff(i,j,k,n)+dcoeff(i,j-1,k,n)) * (phi(i,j,k,n) - phi(i,j-1,k,n))/dx[1];
    
    //find face vel
    Real smallval=1e-10;
    Real r_left=(phi(i,j-1,k,n) - phi(i,j-2,k,n))/(phi(i,j,k,n) - phi(i,j-1,k,n) + smallval);
    Real lim_left=std::max(0.0,std::min(1.0,r_left));

    Real r_right=(phi(i,j,k,n) - phi(i,j-1,k,n)) /(phi(i,j+1,k,n) - phi(i,j,k,n) + smallval);
    Real lim_right=std::max(0.0,std::min(1.0,r_right));
    
    Real phi_L  = phi(i,j-1,k,n) + 0.5 * lim_left  * (phi(i,j-1,k,n) - phi(i,j-2,k,n));
    Real phi_R  = phi(i,j,k,n)   - 0.5 * lim_right * (phi(i,j+1,k,n) - phi(i,j,k,n));

    Real vel_mid = 0.5*(vel(i,j-1,k,n) + vel(i,j,k,n));
    fy(i,j,k,n)  +=   phi_L * 0.5 *(vel_mid+fabs(vel_mid))
                    + phi_R * 0.5 *(vel_mid-fabs(vel_mid));
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void compute_flux_z(int i, int j, int k,int n,
                    Array4<Real> const& phi, //state variable
                    Array4<Real> const& vel,
                    Array4<Real> const& dcoeff,
                    Array4<Real> const& fz,
                    const GpuArray<Real, AMREX_SPACEDIM>& dx)
{

    fz(i,j,k,n) = -0.5 * (dcoeff(i,j,k,n)+dcoeff(i,j,k-1,n)) * (phi(i,j,k,n) - phi(i,j,k-1,n))/dx[2];
    
    //find face vel
    Real smallval=1e-10;
    Real r_left=(phi(i,j,k-1,n) - phi(i,j,k-2,n))/(phi(i,j,k,n) - phi(i,j,k-1,n) + smallval);
    Real lim_left=std::max(0.0,std::min(1.0,r_left));

    Real r_right=(phi(i,j,k,n) - phi(i,j,k-1,n)) /(phi(i,j,k+1,n) - phi(i,j,k,n) + smallval);
    Real lim_right=std::max(0.0,std::min(1.0,r_right));
    
    Real phi_L  = phi(i,j,k-1,n) + 0.5 * lim_left  * (phi(i,j,k-1,n) - phi(i,j,k-2,n));
    Real phi_R  = phi(i,j,k,n  ) - 0.5 * lim_right * (phi(i,j,k+1,n) - phi(i,j,k,n));

    Real vel_mid = 0.5*(vel(i,j,k-1,n) + vel(i,j,k,n));
    fz(i,j,k,n)  +=   phi_L *0.5 *(vel_mid+fabs(vel_mid))
                    + phi_R *0.5 *(vel_mid-fabs(vel_mid));
}
#endif
