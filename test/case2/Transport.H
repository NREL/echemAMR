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
            dcoeff(i,j,k,electrochem::S1_ID)=0.1;
            dcoeff(i,j,k,electrochem::S2_ID)=1.0;
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
            Real xlen,ylen,zlen,maxlen;
            int dir=0;

            xlen=prob_hi[0]-prob_lo[0];
            ylen=prob_hi[1]-prob_lo[1];
            zlen=prob_hi[2]-prob_lo[2];

            maxlen=std::max(xlen,std::max(ylen,zlen));

            if(maxlen==xlen) dir=0;
            if(maxlen==ylen) dir=1;
            if(maxlen==zlen) dir=2;

            velx(i,j,k,electrochem::S1_ID)=0.0;
            velx(i,j,k,electrochem::S2_ID)=0.0;

            vely(i,j,k,electrochem::S1_ID)=0.0;
            vely(i,j,k,electrochem::S2_ID)=0.0;

            velz(i,j,k,electrochem::S1_ID)=0.0;
            velz(i,j,k,electrochem::S2_ID)=0.0;

            if(dir==0)
            {
                velx(i,j,k,electrochem::S1_ID)=1.0;
            }
            else if(dir==1)
            {
                vely(i,j,k,electrochem::S1_ID)=1.0;
            }
            else
            {
                velz(i,j,k,electrochem::S1_ID)=1.0;
            }
        }
}
#endif
