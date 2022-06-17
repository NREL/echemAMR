#ifndef _REACTIONS_H_
#define _REACTIONS_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace electrochem_reactions {
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
        reactsource(i, j, k, CO_ID) = 0.0;
        reactsource(i, j, k, A_ID) = 0.0;
        reactsource(i, j, k, C_ID) = 0.0;
        reactsource(i, j, k, E_ID) = 0.0;
        reactsource(i, j, k, LS_ID) = 0.0;
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

        Real anodefrac   = 0.5*(phi(left,A_ID) + phi(right,A_ID));
        Real cathodefrac = 0.5*(phi(left,C_ID) + phi(right,C_ID));
        
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
    
    AMREX_GPU_DEVICE AMREX_INLINE void species_bvflux(int i,int j,int k,int n,int normaldir,
            Real phi_jump, Array4<Real> const& phi, ProbParm const& prob_parm, Real &jbv)
    {
        Real ocp,j0,phi0;
        Real jcutoff = 0.0000001;
        Real cs,ce;
    
        int trans1dir = (normaldir + 1) % AMREX_SPACEDIM;
        int trans2dir = (normaldir + 2) % AMREX_SPACEDIM;

        IntVect left(i, j, k);
        IntVect right(i, j, k);
        left[normaldir] -= 1;

        Real anodefrac   = 0.5*(phi(left,A_ID) + phi(right,A_ID));
        Real cathodefrac = 0.5*(phi(left,C_ID) + phi(right,C_ID));

        //phi0 = 2.0*GASCONST*prob_parm.Temp/FARADCONST;
        phi0 = 0.05;
        
        Real ls_left  = phi(left,LS_ID);
        Real ls_right = phi(right,LS_ID);
        Real c_left   = phi(left,CO_ID);
        Real c_right  = phi(right,CO_ID);

        //Real cs = amrex::max( (1.0-ls_left)*c_left, (1.0-ls_right)*c_right);
        //Real ce = amrex::max(ls_left*c_left, ls_right*c_right);
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

        j0=0.0;
        ocp=0.0;
        if(anodefrac > 0.0)
        {
            ocp = OCP_a(cs,prob_parm);
            j0  = Io_a(cs,ce,prob_parm);
        }
        if(cathodefrac > 0.0)
        {
            ocp = OCP_c(cs,prob_parm);
            j0 = Io_c(cs,ce,prob_parm);
        }

        j0=amrex::max(j0,jcutoff);
        //amrex::Print()<<"normaldir,j0,phi0,ocp,phijump,anodefrac,cathodefrac:"
          //  <<"\t"<<normaldir<<"\t"<<j0<<"\t"<<phi0<<"\t"
           // <<ocp<<"\t"<<phi_jump<<"\t"<<anodefrac<<"\t"<<cathodefrac<<"\n";

        // sinh definition
        jbv = -j0*sinh((phi_jump+ocp)/phi0);

        // linear case
        //jbv=-(j0/phi0)*(phi_jump+ocp);
        //
        jbv=jbv/FARADCONST;

    }

    AMREX_GPU_DEVICE AMREX_INLINE Real bv_activation_function(Real c, Real gradc, Real gradc_cutoff)
    {
        Real f;
        Real maxval = 0.25;
        Real tol = 1e-1;

        // f=(gradc > gradc_cutoff)?1.0:0.0;
        f = (c * (1 - c) > tol * maxval) ? 1.0 : 0.0;
        return (f);
    }
}
#endif
