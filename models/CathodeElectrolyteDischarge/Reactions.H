#ifndef _REACTIONS_H_
#define _REACTIONS_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace electrochem_reactions
{
    AMREX_GPU_DEVICE AMREX_INLINE
    void compute_react_source(int i, int j, int k,
                              Array4<Real> const& phi,
                              Array4<Real> const& reactsource,
                              GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                              GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                              GpuArray<Real, AMREX_SPACEDIM> dx,
                              const Real time,
                              ProbParm const& prob_parm)
    {
        for(int c=0;c<NVAR;c++)
        {
            reactsource(i,j,k,c)=0.0;
        }
    }

    AMREX_GPU_DEVICE AMREX_INLINE
    void compute_potential_source(int i, int j, int k,
                                  Array4<Real> const& phi,
                                  Array4<Real> const& source,
                                  GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                                  GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                                  GpuArray<Real, AMREX_SPACEDIM> dx,
                                  const Real time,
                                  ProbParm const& prob_parm)
    {
        source(i,j,k)=0.0;
    }


    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real Io_c(amrex::Real Cs, 
                                                              amrex::Real Ce, 
                                                              const ProbParm& prob_parm)
    {
        amrex::Real ic=Cs/prob_parm.Csmax_c;
        amrex::Real ioa=6.0*(ic-ic*ic)*4.0;
        //ioa=0.3;
        return ioa;
    }

    // Anode Open circuit voltage [V]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real OCP_c(amrex::Real Cs, const ProbParm& prob_parm)
    {
        amrex::Real ic=Cs/prob_parm.Csmax_c;
        amrex::Real OCPa=0.8*(1.0-ic);
        //OCPa=0.6;
        return OCPa;
    }

    AMREX_GPU_DEVICE AMREX_INLINE void bvcurrent_and_der(int i,int j,int k,int normaldir,
                                                         Real phi_jump, Array4<Real> const& phi, 
                                                         ProbParm const& prob_parm, Real &jbv, Real &jbvder)
    {
        Real ocp,j0,phi0;
        Real jcutoff = 0.0000001;
        Real cs,ce;
        Real temp=1450.563269; //K

        int trans1dir = (normaldir + 1) % AMREX_SPACEDIM;
        int trans2dir = (normaldir + 2) % AMREX_SPACEDIM;

        IntVect left(i, j, k);
        IntVect right(i, j, k);
        left[normaldir] -= 1;

        phi0 = 2.0*GASCONST*temp/FARADCONST; //becomes 0.25

        Real ls_left  = phi(left,LS_ID);
        Real ls_right = phi(right,LS_ID);
        Real c_left   = phi(left,LI_ID);
        Real c_right  = phi(right,LI_ID);
        Real electrodefrac = 0.5*((1.0-ls_left)+(1.0-ls_right));

        cs = c_left;
        ce = c_right;
        Real electrolyte_frac=0.0;
        Real solid_frac=0.0;
        int found_solid=0;
        for(int t2=-1; t2<=1; t2++)
        {
            for(int t1=-1; t1<=1; t1++)
            {
                for(int n=-2; n<=1; n++)
                {
                    IntVect curid(i,j,k);
                    Real ls,conc;

                    curid[normaldir] += n;
                    curid[trans1dir] += t1;
                    curid[trans2dir] += t2;

                    ls   = phi(curid, LS_ID);
                    conc = phi(curid, LI_ID);

                    if(ls > electrolyte_frac)
                    {
                        electrolyte_frac=ls;
                        ce=conc;
                    }
                    if((1.0-ls) > solid_frac)
                    {
                        solid_frac=1.0-ls;
                        cs=conc;
                    } 
                    if(ls==0)
                    {
                        found_solid=1;
                    }
                }
            }
        }

        if(!found_solid)
        {
            for(int t2=-1; t2<=1; t2++)
            {
                for(int t1=-1; t1<=1; t1++)
                {
                    for(int n=-2; n<=1; n++)
                    {
                        IntVect curid(i,j,k);
                        Real ls,conc;

                        curid[normaldir] += n;
                        curid[trans1dir] += t1;
                        curid[trans2dir] += t2;

                        ls   = phi(curid, LS_ID);
                        conc = phi(curid, LI_ID);

                        //amrex::Print()<<"solid not found curid,ls,conc,normaldir: "<<curid<<"\t"
                        //<<ls<<"\t"<<conc<<"\t"<<normaldir<<"\t"<<i<<"\t"<<j<<"\t"<<k<<"\n";
                    }
                }
            }
            //amrex::Print()<<"\n===================\n";
        }

        j0=0.0;
        ocp=0.0;
        ocp = OCP_c(cs,prob_parm);
        j0  = Io_c(cs,ce,prob_parm);

        j0=amrex::max(j0,jcutoff);
        //amrex::Print()<<"normaldir,j0,phi0,ocp,phijump,anodefrac,cathodefrac,i,j,k,ls_l,ls_r:"
        //    <<"\t"<<normaldir<<"\t"<<j0<<"\t"<<phi0<<"\t"
        //    <<ocp<<"\t"<<phi_jump<<"\t"<<anodefrac<<"\t"<<cathodefrac<<"\t"<<
        //    i<<"\t"<<j<<"\t"<<k<<"\t"<<phi(left,LS_ID)<<"\t"<<phi(right,LS_ID)<<"\n";

        // sinh definition
        jbv = -j0*sinh((phi_jump+ocp)/phi0);

        // linear case
        //jbv=-(j0/phi0)*(phi_jump+ocp);

        // sinh definition
        jbvder = -j0 * cosh((phi_jump + ocp) / phi0) * (1.0 / phi0);

        // linear case
        //jbvder=-(j0/phi0);
    }

    AMREX_GPU_DEVICE AMREX_INLINE void species_bvflux(int i,int j,int k,int n,int normaldir,Real phi_jump,
                                                      Array4<Real> const& phi,ProbParm const& prob_parm,Real &jbv)
    {
        Real jbvder=0.0;
        jbv=0.0;
        bvcurrent_and_der(i,j,k,normaldir,phi_jump,phi,prob_parm,jbv,jbvder);
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
