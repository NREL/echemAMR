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
        
        // Concentration
        Real C = phi(i, j, k, CO_ID);
        // Diffusion coefficient = sum( diffusion_coefficient domain i * levelset i), with levelset i=1 if x within domain i, 0 otherwise
        dcoeff(i,j,k,CO_ID) = electrochem::Ds_bulk_anode(C, prob_parm) * phi(i,j,k,A_AM_ID)
        + electrochem::Ds_bulk_cathode(C, prob_parm) * phi(i,j,k,C_AM_ID)
        + electrochem::Ds_bulk_CBD(prob_parm) * (phi(i,j,k,A_CBD_ID)+phi(i,j,k,C_CBD_ID));
        if (electrochem::Cphi_ele_coupling(prob_parm)==1) {
            // Intermediate coefficient
            Real Ke  =  electrochem::Ke_bulk(C, prob_parm) * phi(i,j,k,LS_ID) / phi(i,j,k,MAC_ID);
            Real Ac  = electrochem::A_c(C, prob_parm) * phi(i,j,k,LS_ID);
            Real tplus  = electrochem::t_plus(C, prob_parm) * phi(i,j,k,LS_ID);      
            // Diffusional conductivity
            Real KDe  = electrochem::Kd(Ke, Ac, tplus, prob_parm) * phi(i,j,k,LS_ID);  
            dcoeff(i,j,k,CO_ID) = dcoeff(i,j,k,CO_ID) + (electrochem::De_bulk(C, prob_parm)/phi(i,j,k,MAC_ID) - tplus/FARADCONST*KDe/C) * phi(i,j,k,LS_ID);
        } else {
            dcoeff(i,j,k,CO_ID) = dcoeff(i,j,k,CO_ID) + (electrochem::De_bulk(C, prob_parm)/phi(i,j,k,MAC_ID)) * phi(i,j,k,LS_ID);
        }        
        
        dcoeff(i, j, k, A_AM_ID) = 0.0;
        dcoeff(i, j, k, A_E_ID) = 0.0;
        dcoeff(i, j, k, A_CBD_ID) = 0.0;
        dcoeff(i, j, k, S_ID) = 0.0;
        dcoeff(i, j, k, C_AM_ID) = 0.0;
        dcoeff(i, j, k, C_E_ID) = 0.0;
        dcoeff(i, j, k, C_CBD_ID) = 0.0;        
        dcoeff(i, j, k, NP_ID) = 0.0;
        dcoeff(i, j, k, MAC_ID) = 0.0;  
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
                phi(i, j, k, A_AM_ID)  * phi(i,j,k,NP_ID)
              + phi(i, j, k, A_E_ID)   * phi(i,j,k,NP_ID)
              + phi(i, j, k, A_CBD_ID) * phi(i,j,k,NP_ID)
              + phi(i, j, k, S_ID)     * phi(i,j,k,NP_ID)
              + phi(i, j, k, C_AM_ID)  * phi(i,j,k,NP_ID)
              + phi(i, j, k, C_E_ID)   * phi(i,j,k,NP_ID)
              + phi(i, j, k, C_CBD_ID) * phi(i,j,k,NP_ID)
            );
    }

    AMREX_GPU_DEVICE AMREX_INLINE void compute_vel(
        int i,
        int j,
        int k,
        int dir,
        Array4<Real> const& phi,
        Array4<Real> const& vel,
        GpuArray<Real, AMREX_SPACEDIM> prob_lo,
        GpuArray<Real, AMREX_SPACEDIM> prob_hi,
        GpuArray<Real, AMREX_SPACEDIM> dx,
        const Real time,
        ProbParm const& prob_parm)
    {
        for (int n = 0; n < phi.nComp(); ++n)
        {
            vel(i, j, k, n) = 0.0;
        }
    }
    
    AMREX_GPU_DEVICE AMREX_INLINE void compute_potential_dcoeff( // Use if prob_parm.CBD_transport==0
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
        // Conduction coefficient = sum( conductivity_coefficient domain i * levelset i), with levelset i=1 if x within domain i, 0 otherwise
        dcoeff(i,j,k) = (electrochem::Ks_bulk_anode(phi(i, j, k, CO_ID), prob_parm) * phi(i,j,k,A_AM_ID)
        + electrochem::Ks_bulk_cathode(phi(i, j, k, CO_ID), prob_parm) * phi(i,j,k,C_AM_ID)
        + electrochem::Ks_bulk_CBD(prob_parm) * (phi(i,j,k,A_CBD_ID)+phi(i,j,k,C_CBD_ID))
        + electrochem::Ke_bulk(phi(i, j, k, CO_ID), prob_parm) * phi(i,j,k,LS_ID)) / phi(i,j,k,MAC_ID);
    }    
    
    AMREX_GPU_DEVICE AMREX_INLINE void compute_electrolytepotential_dcoeff( // Use if prob_parm.CBD_transport==1
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
        // Conduction coefficient = sum( conductivity_coefficient domain i * levelset i), with levelset i=1 if x within domain i, 0 otherwise
        dcoeff(i,j,k) = electrochem::Ke_bulk(phi(i, j, k, CO_ID), prob_parm) * phi(i,j,k,LS_ID) / phi(i,j,k,MAC_ID);
    }

    AMREX_GPU_DEVICE AMREX_INLINE void compute_solidpotential_dcoeff( // Use if prob_parm.CBD_transport==1
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
        // Conduction coefficient = sum( conductivity_coefficient domain i * levelset i), with levelset i=1 if x within domain i, 0 otherwise
        dcoeff(i,j,k) = (electrochem::Ks_bulk_anode(phi(i, j, k, CO_ID), prob_parm) * phi(i,j,k,A_AM_ID)
        + electrochem::Ks_bulk_cathode(phi(i, j, k, CO_ID), prob_parm) * phi(i,j,k,C_AM_ID)
        + electrochem::Ks_bulk_CBD(prob_parm) * (phi(i,j,k,A_CBD_ID)+phi(i,j,k,C_CBD_ID))) / phi(i,j,k,MAC_ID);
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
        IntVect left(i, j, k);
        IntVect right(i, j, k);
        left[idim] -= 1;
        amrex::Real kdstar=0.0;
        
        Real Cleft = phi(left, CO_ID);
        Real Cright = phi(right, CO_ID);
        Real ls_face = 0.5*(phi(left,LS_ID) + phi(right,LS_ID));

        //has electrolyte
        if(ls_face > 0.0)
        {
            Real Cface=0.5*(Cleft*phi(left,LS_ID)+Cright*phi(right,LS_ID));

            // Intermediate coefficients
            Real Ke_left     = electrochem::Ke_bulk(Cleft, prob_parm) * phi(left,LS_ID) / phi(left,MAC_ID);
            Real Ac_left     = electrochem::A_c(Cleft, prob_parm) * phi(left,LS_ID);
            Real tplus_left  = electrochem::t_plus(Cleft, prob_parm) * phi(left,LS_ID);
            Real KD_left     = electrochem::Kd(Ke_left, Ac_left, tplus_left, prob_parm) * phi(left,LS_ID);

            Real Ke_right     = electrochem::Ke_bulk(Cright, prob_parm) * phi(right,LS_ID) / phi(right,MAC_ID);
            Real Ac_right     = electrochem::A_c(Cright, prob_parm) * phi(right,LS_ID);
            Real tplus_right  = electrochem::t_plus(Cright, prob_parm) * phi(right,LS_ID);
            Real KD_right     = electrochem::Kd(Ke_right, Ac_right, tplus_right, prob_parm) * phi(right,LS_ID);

            kdstar=0.5*(KD_left+KD_right)/Cface;
        }

        return(kdstar);
    }
}
#endif
