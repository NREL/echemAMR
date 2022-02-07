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
    
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real Io_c(amrex::Real Cs, amrex::Real Ce, const ProbParm& prob_parm)
    {
        amrex::Real x = Cs/prob_parm.Csmax_c;
        amrex::Real ioc = 6.0 * pow(Ce / 1000, 0.5) * pow(x, 0.5) * pow(1 - x, 0.5);

        ioc=0.3;
        return ioc;
    }
    
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real OCP_c(amrex::Real Cs, const ProbParm& prob_parm)
    {
        amrex::Real x = Cs / prob_parm.Csmax_c;
        constexpr amrex::Real a3 = -1.3711668;
        constexpr amrex::Real a2 = 3.9033218;
        constexpr amrex::Real a1 = -4.3438781;
        constexpr amrex::Real a0 = 5.3701447;
        amrex::Real OCPc = ((a3 * x + a2) * x + a1) * x + a0;

        OCPc=3.5;
        return OCPc;
    }
    
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real Io_a(amrex::Real Cs, amrex::Real Ce, const ProbParm& prob_parm)
    {
        amrex::Real x = Cs / prob_parm.Csmax_a;
        amrex::Real ioa = 11 * pow(Ce / 1000, 0.5) * pow(x, 0.5) * pow(1 - x, 0.5);

        ioa=0.3;
        return ioa;
    }
    
    // Anode Open circuit voltage [V]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real OCP_a(amrex::Real Cs, const ProbParm& prob_parm)
    {
        amrex::Real x = Cs / prob_parm.Csmax_a;
        constexpr amrex::Real a7 = -201.8048;
        constexpr amrex::Real a6 = 730.918;
        constexpr amrex::Real a5 = -1074.645;
        constexpr amrex::Real a4 = 825.0173;
        constexpr amrex::Real a3 = -354.3316;
        constexpr amrex::Real a2 = 84.66978;
        constexpr amrex::Real a1 = -10.57909;
        constexpr amrex::Real a0 = 0.6848288;
        amrex::Real OCPa = ((((((a7 * x + a6) * x + a5) * x + a4) * x + a3) * x + a2) * x + a1) * x + a0;

        OCPa=0.6;
        return OCPa;
    }

    AMREX_GPU_DEVICE AMREX_INLINE void bvcurrent_and_der(int i,int j,int k,int normaldir,
            Real phi_jump, Array4<Real> const& phi, ProbParm const& prob_parm, Real &jbv, Real &jbvder)
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

        // sinh definition
        jbvder = -j0 * cosh((phi_jump + ocp) / phi0) * (1.0 / phi0);

        // linear case
        //jbvder=-(j0/phi0);
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
} // namespace electrochem_reactions
#endif
