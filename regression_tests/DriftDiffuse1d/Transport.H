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
        const int im1 = (dir == 0) ? i-1 : i;
        const int jm1 = (dir == 1) ? j-1 : j;
        const int km1 = (dir == 2) ? k-1 : k;

        if(sgn == -1) 
        { // lo sides
            robin_a(im1,jm1,km1) = 1.0;
            robin_b(im1,jm1,km1) = 0.0;
            robin_f(im1,jm1,km1) = bclo;
        }
        else
        {
            robin_a(i,j,k) = 1.0;
            robin_b(i,j,k) = 0.0;
            robin_f(i,j,k) = 0.0;
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
        const int im1 = (dir == 0) ? i-1 : i;
        const int jm1 = (dir == 1) ? j-1 : j;
        const int km1 = (dir == 2) ? k-1 : k;

        if(sgn == -1) 
        { // lo sides
            bc(im1,jm1,km1) = 0.0;
        } 
        else 
        { // hi sides
            if(spec_id==S1_ID)
            {
                bc(i,j,k) = 1.0;
            }
            else
            {
                bc(i,j,k) = 0.0;
            }
        }

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
        const int im1 = (dir == 0) ? i-1 : i;
        const int jm1 = (dir == 1) ? j-1 : j;
        const int km1 = (dir == 2) ? k-1 : k;

        if(sgn == -1) 
        { // lo sides
            bc(im1,jm1,km1) = bclo;
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
        dcoeff(i,j,k,S1_ID)   = 0.1;
        dcoeff(i,j,k,S2_ID)   = 1.0;
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
        Real xlen,ylen,zlen,maxlen;
        int dir=0;

        xlen=prob_hi[0]-prob_lo[0];
        ylen=prob_hi[1]-prob_lo[1];
        zlen=prob_hi[2]-prob_lo[2];

        maxlen=std::max(xlen,std::max(ylen,zlen));

        if(maxlen==xlen) dir=0;
        if(maxlen==ylen) dir=1;
        if(maxlen==zlen) dir=2;

        velx(i,j,k,S1_ID)=0.0;
        velx(i,j,k,S2_ID)=0.0;

        if(dir==0)
        {
            velx(i,j,k,S1_ID)
            =0.5*(phi(i-1,j,k,EFX_ID)+phi(i,j,k,EFX_ID));
        }
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
        Real xlen,ylen,zlen,maxlen;
        int dir=0;

        xlen=prob_hi[0]-prob_lo[0];
        ylen=prob_hi[1]-prob_lo[1];
        zlen=prob_hi[2]-prob_lo[2];

        maxlen=std::max(xlen,std::max(ylen,zlen));

        if(maxlen==xlen) dir=0;
        if(maxlen==ylen) dir=1;
        if(maxlen==zlen) dir=2;

        vely(i,j,k,S1_ID)=0.0;
        vely(i,j,k,S2_ID)=0.0;

        if(dir==1)
        {
            vely(i,j,k,S1_ID)
            =0.5*(phi(i,j-1,k,EFY_ID)+phi(i,j,k,EFY_ID));
        }
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
        Real xlen,ylen,zlen,maxlen;
        int dir=0;

        xlen=prob_hi[0]-prob_lo[0];
        ylen=prob_hi[1]-prob_lo[1];
        zlen=prob_hi[2]-prob_lo[2];

        maxlen=std::max(xlen,std::max(ylen,zlen));

        if(maxlen==xlen) dir=0;
        if(maxlen==ylen) dir=1;
        if(maxlen==zlen) dir=2;

        velz(i,j,k,S1_ID)=0.0;
        velz(i,j,k,S2_ID)=0.0;

        if(dir==2)
        {
            velz(i,j,k,S1_ID)
            =0.5*(phi(i,j,k-1,EFZ_ID)+phi(i,j,k,EFZ_ID));
        }
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
