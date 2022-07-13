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
        for(int n=0;n<NUM_SPECIES;n++)
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
        Real ocp[NUM_SURF_REACTIONS];
        Real j0[NUM_SURF_REACTIONS],phi0[NUM_SURF_REACTIONS];
        Real bvcur[NUM_SURF_REACTIONS];
        Real bvcur_der[NUM_SURF_REACTIONS];

        //in volts
        ocp[0] = 0.11;
        ocp[1] = 0.02;

        //in A/m2
        j0[0] = -0.02;
        j0[1] = -0.02;

        //in volts
        //phi0[0]=0.025;
        //phi0[1]=0.025;
        
        phi0[0]=1.0;
        phi0[1]=1.0;
        
        //Reactions
        //CO2 + H2O + 2e-  --> CO + 2OH-
        //CO2 + H2O + 2e-  --> HCOO- + OH-
    
        // sinh definition
        bvcur[0] = -j0[0]*sinh((phi_jump+ocp[0])/phi0[0]);
        bvcur[1] = -j0[1]*sinh((phi_jump+ocp[1])/phi0[1]);
        
        // sinh definition
        bvcur_der[0] = -j0[0] * cosh((phi_jump + ocp[0]) / phi0[0]) 
                              * (1.0 / phi0[0]);
        bvcur_der[1] = -j0[1] * cosh((phi_jump + ocp[1]) / phi0[1]) 
                              * (1.0 / phi0[1]);

        //- for -ve ions, first 2 terms for oh- and second for hcoo-
        jbv    = -(2.0*bvcur[0] + bvcur[1] + bvcur[1]);

        jbvder = -(2.0*bvcur_der[0] + bvcur_der[1] + bvcur_der[1]);

        /*bvcur[0] = -j0[0] * sinh((phi_jump+ocp[0])/phi0[0]);
        bvcur_der[0] = -j0[0] * cosh((phi_jump + ocp[0]) / phi0[0]) 
                              * (1.0 / phi0[0]);
        
        jbv=2.0*bvcur[0];
        jbvder=2.0*bvcur_der[0];*/

    }
    
    AMREX_GPU_DEVICE AMREX_INLINE void species_bvflux(int i,int j,int k,int n,int normaldir,
            Real phi_jump, Array4<Real> const& phi, ProbParm const& prob_parm, Real &jbv)
    {
        Real ocp[NUM_SURF_REACTIONS];
        Real j0[NUM_SURF_REACTIONS],phi0[NUM_SURF_REACTIONS];
        Real bvcur[NUM_SURF_REACTIONS];
        Real cs,ce;

        //in volts
        ocp[0] = 0.11;
        ocp[1] = 0.02;

        //in A/m2
        j0[0] = -0.02;
        j0[1] = -0.02;

        //phi0[0]=0.025;
        //phi0[1]=0.025;
        
        phi0[0]=1.0;
        phi0[1]=1.0;
    
        int trans1dir = (normaldir + 1) % AMREX_SPACEDIM;
        int trans2dir = (normaldir + 2) % AMREX_SPACEDIM;

        // sinh definition
        bvcur[0] = -j0[0]*sinh((phi_jump+ocp[0])/phi0[0]);
        bvcur[1] = -j0[1]*sinh((phi_jump+ocp[1])/phi0[1]);
        
        amrex::Print()<<"bvcur:"<<bvcur[0]<<"\n";

        //Reactions
        //CO2 + H2O + 2e-  --> CO + 2OH-
        //CO2 + H2O + 2e-  --> HCOO- + OH-

        if(n==CO2_ID)
        {
           jbv = (-bvcur[0]-bvcur[1])/FARADCONST;
           amrex::Print()<<"jbv co2:"<<jbv<<"\t"<<bvcur[0]<<"\t"<<bvcur[1]<<"\n";
        }
        if(n==CO_ID)
        {
           jbv = bvcur[0]/FARADCONST;
        }
        if(n==OHM_ID)
        {
           jbv = (2.0*bvcur[0]+bvcur[1])/FARADCONST;
        }
        if(n==HCOOM_ID)
        {
           jbv = bvcur[1]/FARADCONST;
        }
        
        /*bvcur[0] = -j0[0]*sinh((phi_jump+ocp[0])/phi0[0]);

        if(n==CO2_ID)
        {
           jbv = -bvcur[0]/FARADCONST;
        }
        if(n==CO_ID)
        {
           jbv = bvcur[0]/FARADCONST;
        }
        if(n==OHM_ID)
        {
           jbv = 2.0*bvcur[0]/FARADCONST;
        }
        if(n==HCOOM_ID)
        {
           jbv=0.0;
        }*/
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
