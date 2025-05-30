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
        source(i, j, k) = 0.0;
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

        Real anode_AM_frac   = 0.5*(phi(left,A_AM_ID) + phi(right,A_AM_ID));
        Real cathode_AM_frac = 0.5*(phi(left,C_AM_ID) + phi(right,C_AM_ID));
        Real anode_CBD_frac = 0.5*(phi(left,A_CBD_ID) + phi(right,A_CBD_ID));
        Real cathode_CBD_frac = 0.5*(phi(left,C_CBD_ID) + phi(right,C_CBD_ID));

        Real facecolor=0.5*(phi(left,LS_ID)+phi(right,LS_ID));

        jbv = 0.0;
        jbvder = 0.0;  

        if(facecolor > 0.0)
        {
            if((anode_CBD_frac==0.0) and (cathode_CBD_frac==0.0)) // No reaction between CBD and Electrolyte
            {
                Real ls_left  = phi(left,LS_ID);
                // Real ls_right = phi(right,LS_ID); // Not used
                Real c_left   = phi(left,CO_ID);
                // Real c_right  = phi(right,CO_ID); // Not used

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


                if(anode_AM_frac > 0.0)
                {
                    jbv  = electrochem::intercalation_reaction_anode(cs, ce, phi_jump, prob_parm);
                    jbvder  = electrochem::intercalation_reaction_anode_derivative(cs, ce, phi_jump, prob_parm);
                }
                if(cathode_AM_frac > 0.0)
                {
                    jbv  = electrochem::intercalation_reaction_cathode(cs, ce, phi_jump, prob_parm);
                    jbvder  = electrochem::intercalation_reaction_cathode_derivative(cs, ce, phi_jump, prob_parm);
                }
            }
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
