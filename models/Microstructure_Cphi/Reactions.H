#ifndef _REACTIONS_H_
#define _REACTIONS_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace electrochem_reactions {
    AMREX_GPU_DEVICE AMREX_INLINE Real bv_activation_function(Real c, Real gradc, Real gradc_cutoff)
    {
        Real f;
        Real maxval = 0.25;
        Real tol = 1e-1;

        // f=(gradc > gradc_cutoff)?1.0:0.0;
        f = (c * (1 - c) > tol * maxval) ? 1.0 : 0.0;
        return (f);
    }
    AMREX_GPU_DEVICE AMREX_INLINE void compute_react_source(
            int i,
            int j,
            int k,
            Array4<Real> const& phi,
            Array4<Real> const& reactsource,
            GpuArray<Real, AMREX_SPACEDIM> prob_lo,
            GpuArray<Real, AMREX_SPACEDIM> prob_hi,
            GpuArray<Real, AMREX_SPACEDIM> dx,
            const Real time,
            ProbParm const& prob_parm)
    {
        for (int n = 0; n < reactsource.nComp(); ++n)
        {
            reactsource(i, j, k, n) = 0.0;
        }        

        Real C = phi(i, j, k, CO_ID);
        Real gradc_dummy=0.1;
        Real gradc_cutoff_dummy=1e-10;
        Real Ke  =  electrochem::Ke_bulk(C, prob_parm) * phi(i,j,k,LS_ID) / phi(i,j,k,MAC_ID);
        Real tplus  = electrochem::t_plus(C, prob_parm) * phi(i,j,k,LS_ID);
        Real TFKe  = tplus/FARADCONST * Ke;
        
        Real Kl  = 0.5 * (phi(i-1,j,k,LS_ID)+phi(i,j,k,LS_ID)) * TFKe;
        Real Kr  = 0.5 * (phi(i+1,j,k,LS_ID)+phi(i,j,k,LS_ID)) * TFKe;
        
        Real Kb  = 0.5 * (phi(i,j-1,k,LS_ID)+phi(i,j,k,LS_ID)) * TFKe;
        Real Kt  = 0.5 * (phi(i,j+1,k,LS_ID)+phi(i,j,k,LS_ID)) * TFKe;
        
        Real Kk  = 0.5 * (phi(i,j,k-1,LS_ID)+phi(i,j,k,LS_ID)) * TFKe;
        Real Kf  = 0.5 * (phi(i,j,k+1,LS_ID)+phi(i,j,k,LS_ID)) * TFKe;
        
        Real Actl =  1.0-bv_activation_function(0.5*(phi(i-1,j,k,LS_ID)+phi(i,j,k,LS_ID)),gradc_dummy,gradc_cutoff_dummy);
        Real Actr =  1.0-bv_activation_function(0.5*(phi(i+1,j,k,LS_ID)+phi(i,j,k,LS_ID)),gradc_dummy,gradc_cutoff_dummy);
        
        Real Actb =  1.0-bv_activation_function(0.5*(phi(i,j-1,k,LS_ID)+phi(i,j,k,LS_ID)),gradc_dummy,gradc_cutoff_dummy);
        Real Actt =  1.0-bv_activation_function(0.5*(phi(i,j+1,k,LS_ID)+phi(i,j,k,LS_ID)),gradc_dummy,gradc_cutoff_dummy);
        
        Real Actk =  1.0-bv_activation_function(0.5*(phi(i,j,k-1,LS_ID)+phi(i,j,k,LS_ID)),gradc_dummy,gradc_cutoff_dummy);
        Real Actf =  1.0-bv_activation_function(0.5*(phi(i,j,k+1,LS_ID)+phi(i,j,k,LS_ID)),gradc_dummy,gradc_cutoff_dummy);

        //reactsource(i,j,k,CO_ID) +=  (( Actr*Kr*(phi(i+1,j,k,POT_ID)-phi(i,j,k,POT_ID)) - Actl*Kl*(phi(i,j,k,POT_ID)-phi(i-1,j,k,POT_ID)) )/dx[0]/dx[0]
        //                            +( Actt*Kt*(phi(i,j+1,k,POT_ID)-phi(i,j,k,POT_ID)) - Actb*Kb*(phi(i,j,k,POT_ID)-phi(i,j-1,k,POT_ID)) )/dx[1]/dx[1]
        //                            +( Actf*Kf*(phi(i,j,k+1,POT_ID)-phi(i,j,k,POT_ID)) - Actk*Kk*(phi(i,j,k,POT_ID)-phi(i,j,k-1,POT_ID)) )/dx[2]/dx[2]); 

    }

    AMREX_GPU_DEVICE AMREX_INLINE void compute_potential_source(
            int i,
            int j,
            int k,
            Array4<Real> const& phi,
            Array4<Real> const& source,
            GpuArray<Real, AMREX_SPACEDIM> prob_lo,
            GpuArray<Real, AMREX_SPACEDIM> prob_hi,
            GpuArray<Real, AMREX_SPACEDIM> dx,
            const Real time,
            ProbParm const& prob_parm)
    {
       // concentration
        Real C = phi(i, j, k, CO_ID);
        Real gradc_dummy=0.1;
        Real gradc_cutoff_dummy=1e-10;

        // Intermediate coefficients
        Real Ke  =  electrochem::Ke_bulk(C, prob_parm) * phi(i,j,k,LS_ID) / phi(i,j,k,MAC_ID);
        Real Ac  = electrochem::A_c(C, prob_parm) * phi(i,j,k,LS_ID);
        Real tplus  = electrochem::t_plus(C, prob_parm) * phi(i,j,k,LS_ID);
      
        // Diffusional conductivit
        Real KDe  = electrochem::Kd(Ke, Ac, tplus, prob_parm) * phi(i,j,k,LS_ID);
              
        Real KDl  = 0.5 * (phi(i-1,j,k,LS_ID)+phi(i,j,k,LS_ID)) * KDe;
        Real KDr  = 0.5 * (phi(i+1,j,k,LS_ID)+phi(i,j,k,LS_ID)) * KDe;
        
        Real KDb  = 0.5 * (phi(i,j-1,k,LS_ID)+phi(i,j,k,LS_ID)) * KDe;
        Real KDt  = 0.5 * (phi(i,j+1,k,LS_ID)+phi(i,j,k,LS_ID)) * KDe;
        
        Real KDk  = 0.5 * (phi(i,j,k-1,LS_ID)+phi(i,j,k,LS_ID)) * KDe;
        Real KDf  = 0.5 * (phi(i,j,k+1,LS_ID)+phi(i,j,k,LS_ID)) * KDe;
        
        Real Actl =  1.0-bv_activation_function(0.5*(phi(i-1,j,k,LS_ID)+phi(i,j,k,LS_ID)),gradc_dummy,gradc_cutoff_dummy);
        Real Actr =  1.0-bv_activation_function(0.5*(phi(i+1,j,k,LS_ID)+phi(i,j,k,LS_ID)),gradc_dummy,gradc_cutoff_dummy);
        
        Real Actb =  1.0-bv_activation_function(0.5*(phi(i,j-1,k,LS_ID)+phi(i,j,k,LS_ID)),gradc_dummy,gradc_cutoff_dummy);
        Real Actt =  1.0-bv_activation_function(0.5*(phi(i,j+1,k,LS_ID)+phi(i,j,k,LS_ID)),gradc_dummy,gradc_cutoff_dummy);
        
        Real Actk =  1.0-bv_activation_function(0.5*(phi(i,j,k-1,LS_ID)+phi(i,j,k,LS_ID)),gradc_dummy,gradc_cutoff_dummy);
        Real Actf =  1.0-bv_activation_function(0.5*(phi(i,j,k+1,LS_ID)+phi(i,j,k,LS_ID)),gradc_dummy,gradc_cutoff_dummy);
      
        // Enable for KD
        //source(i,j,k) = -(  (  Actr*KDr*std::log(phi(i+1,j,k,CO_ID)/phi(i,j,k,CO_ID)) - Actl*KDl*std::log(phi(i,j,k,CO_ID)/phi(i-1,j,k,CO_ID)) )/dx[0]/dx[0]
        //                +   (  Actt*KDt*std::log(phi(i,j+1,k,CO_ID)/phi(i,j,k,CO_ID)) - Actb*KDb*std::log(phi(i,j,k,CO_ID)/phi(i,j-1,k,CO_ID)) )/dx[1]/dx[1]
        //                +   (  Actf*KDf*std::log(phi(i,j,k+1,CO_ID)/phi(i,j,k,CO_ID)) - Actk*KDk*std::log(phi(i,j,k,CO_ID)/phi(i,j,k-1,CO_ID)) )/dx[2]/dx[2] );

        source(i, j, k) = 0.0; // No Kd

    }

    AMREX_GPU_DEVICE AMREX_INLINE void bvcurrent_and_der(int i,int j,int k,int normaldir,
            Real phi_jump, Array4<Real> const& phi, ProbParm const& prob_parm, Real &jbv, Real &jbvder)
    {
        Real cs,ce;
    
        int trans1dir = (normaldir + 1) % AMREX_SPACEDIM;
        int trans2dir = (normaldir + 2) % AMREX_SPACEDIM;

        IntVect left(i, j, k);
        IntVect right(i, j, k);
        left[normaldir] -= 1;

        Real anodefrac   = 0.5*(phi(left,A_AM_ID) + phi(right,A_AM_ID));
        Real cathodefrac = 0.5*(phi(left,C_AM_ID) + phi(right,C_AM_ID));
        
        Real ls_left  = phi(left,LS_ID);
        Real ls_right = phi(right,LS_ID);
        Real c_left   = phi(left,CO_ID);
        Real c_right  = phi(right,CO_ID);

        cs = (1.0-ls_left)*c_left;
        ce = ls_left*c_left;
        for(int t2=-1; t2<=1; t2++)
        {
            for(int t1=-1; t1<=1; t1++)
            {
                for(int n=-1; n<=0; n++)
                {
                   IntVect curid(i,j,k);
                   Real ls,conc;

                   curid[normaldir] += n;
                   curid[trans1dir] += t1;
                   curid[trans2dir] += t2;

                   ls   = phi(curid, LS_ID);
                   conc = phi(curid, CO_ID);

                   if(ls*conc > ce)
                   {
                        ce=ls*conc;
                   }
                   if((1.0-ls)*conc > cs)
                   {
                        cs=(1.0-ls)*conc;
                   } 
                }
            }
        }

        jbv = 0.0;
        jbvder = 0.0;
        if(anodefrac > 0.0)
        {
            jbv  = electrochem::intercalation_reaction_anode(cs, ce, phi_jump, prob_parm);
            jbvder  = electrochem::intercalation_reaction_anode_derivative(cs, ce, phi_jump, prob_parm);
        }
        if(cathodefrac > 0.0)
        {
            jbv  = electrochem::intercalation_reaction_cathode(cs, ce, phi_jump, prob_parm);
            jbvder  = electrochem::intercalation_reaction_cathode_derivative(cs, ce, phi_jump, prob_parm);
        }
 
    }
    
    AMREX_GPU_DEVICE AMREX_INLINE void species_bvflux(int i,int j,int k,int n,int normaldir,Real phi_jump,
            Array4<Real> const& phi,ProbParm const& prob_parm,Real &jbv)
    {
        Real jbvder=0.0;
        jbv=0.0;
        bvcurrent_and_der(i,j,k,normaldir,phi_jump,phi,prob_parm,jbv,jbvder);
        jbv=jbv/FARADCONST;
    }

}
#endif
