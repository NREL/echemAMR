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
    
    }
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
        Real xlen,ylen,zlen,maxlen;
        int axialdir=0;

        xlen=prob_hi[0]-prob_lo[0];
        ylen=prob_hi[1]-prob_lo[1];
        zlen=prob_hi[2]-prob_lo[2];

        maxlen=std::max(xlen,std::max(ylen,zlen));

        if(maxlen==xlen) axialdir=0;
        if(maxlen==ylen) axialdir=1;
        if(maxlen==zlen) axialdir=2;

        const int im1 = (dir == 0) ? i-1 : i;
        const int jm1 = (dir == 1) ? j-1 : j;
        const int km1 = (dir == 2) ? k-1 : k;

        if(sgn == -1) 
        { // lo sides
            // amrex::Print() << "axialdir: " << axialdir << ", dir: " << dir <<  std::endl;
            bc(im1,jm1,km1) = (dir==axialdir) ? -1.0 : 0.0;
        } 
        else 
        { // hi sides
          bc(i,j,k) = 0.0;
        }
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
            dcoeff(i,j,k,S1_ID)   = 1.0;
            dcoeff(i,j,k,LS_ID)   = 0.0;
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
            velx(i,j,k,LS_ID)=0.0;
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
            vely(i,j,k,LS_ID)=0.0;
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
            velz(i,j,k,LS_ID)=0.0;
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
            const amrex::Real ls = phi(i,j,k,LS_ID);
            dcoeff(i,j,k) = (1.0-ls)*prob_parm.electrode_dcoeff + prob_parm.electrolyte_dcoeff*ls;
        }

}
#endif
