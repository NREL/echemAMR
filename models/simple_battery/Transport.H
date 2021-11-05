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
        const int im1 = (dir == 0) ? i-1 : i;
        const int jm1 = (dir == 1) ? j-1 : j;
        const int km1 = (dir == 2) ? k-1 : k;

        if(dir == 0) {
            if(sgn == -1)
            { // lo side along x
                bc(im1,jm1,km1) = bclo;
            } else { // hi side along x
              bc(i,j,k) = bchi;
            }
        } else {
            if(sgn == -1)
            { // lo side along x
                bc(im1,jm1,km1) = 0.0;
            } else { // hi side along x
                bc(i,j,k) = 0.0;
            }
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
            //dont diffuse level sets
            dcoeff(i,j,k,A_ID)    = 0.0;
            dcoeff(i,j,k,C_ID)    = 0.0;
            dcoeff(i,j,k,E_ID)    = 0.0;
            dcoeff(i,j,k,S_ID)    = 0.0;
            dcoeff(i,j,k,LS_ID)   = 0.0;

            dcoeff(i,j,k,CO_ID) =  prob_parm.anode_dcoeff       * phi(i,j,k,A_ID)
                                 + prob_parm.cathode_dcoeff     * phi(i,j,k,C_ID)
                                 + prob_parm.electrolyte_dcoeff * phi(i,j,k,E_ID)
                                 + prob_parm.separator_dcoeff   * phi(i,j,k,S_ID);
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
            for (int n = 0; n < phi.nComp(); ++n)
            {
                velx(i, j, k, n) = 0.0;
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
            for (int n = 0; n < phi.nComp(); ++n)
            {
                vely(i, j, k, n) = 0.0;
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
            for (int n = 0; n < phi.nComp(); ++n)
            {
                velz(i, j, k, n) = 0.0;
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
            //dcoeff(i,j,k)=10.0;
            dcoeff(i,j,k) =  prob_parm.anode_conductivity       * phi(i,j,k,A_ID)
                           + prob_parm.cathode_conductivity     * phi(i,j,k,C_ID)
                           + prob_parm.electrolyte_conductivity * phi(i,j,k,E_ID)
                           + prob_parm.separator_conductivity   * phi(i,j,k,S_ID);

        }

}
#endif
