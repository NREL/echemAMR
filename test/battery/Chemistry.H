#ifndef _CHEMISTRY_H_
#define _CHEMISTRY_H_

#include<AMReX_REAL.H>
#include<AMReX.H>
#include<string>
#include<AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Box.H>
#include <ChemistryProbParm.H>

#define NUM_SPECIES 6
#define CO_ID 0
#define PO_ID 1
#define A_ID 2
#define C_ID 3
#define E_ID 4
#define S_ID 5
#define POT_ID 6
#define NVAR NUM_SPECIES+1

#define myeps 1.0e-12

namespace electrochem
{

    enum component {anode = 0, cathode, electrolyte, separator, last_comp = separator};

    extern amrex::Vector<std::string> specnames;
    void init();
    void close();
    int find_id(std::string specname);

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    int level_set_to_component(int i, int j, int k, amrex::Array4<amrex::Real> const& phase){

        int comp = -1;
        int ncomp = 0;

        if(phase(i,j,k,A_ID) > 0.0) {comp = anode; ++ncomp;}
        if(phase(i,j,k,C_ID) > 0.0) {comp = cathode; ++ncomp;}
        if(phase(i,j,k,E_ID) > 0.0) {comp = electrolyte; ++ncomp;}
        if(phase(i,j,k,S_ID) > 0.0) {comp = separator; ++ncomp;}

        // FIXME: need to make sure we only do this for real cells and not ghost cells
//        if(comp == -1) {
//            amrex::Print() << i << ' ' << j << ' ' << k << std::endl;
//            amrex::Abort("error component not found");
//        }
//
//        if(ncomp > 1) {
//            amrex::Print() << i << ' ' << j << ' ' << k << std::endl;
//            amrex::Abort("too many components found");
//        }

        return comp;

    }

     AMREX_GPU_DEVICE AMREX_FORCE_INLINE
     bool is_electrode(int i, int j, int k, amrex::Array4<amrex::Real> const& phase){

         auto comp = level_set_to_component(i,j,k,phase);

         if(comp == anode || comp == cathode)
             return true;
         else
             return false;
     }


    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real KeC(amrex::Real con)
    {
        return amrex::max(-0.00000025*con*(con-4000.0), myeps);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real DeC(amrex::Real con)
    {
        return amrex::max(3.5e-10 - 3.5e-10/4000.0*con, myeps);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real tplus(amrex::Real con)
    {
        return amrex::max(0.4 + 0.1/4000.0*con, myeps);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real AC(amrex::Real con)
    {
        return amrex::max(1.0 + 25.0/4000.0*con, myeps);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real KD(amrex::Real con, const ProbParm& prob_parm)
    {
        const amrex::Real RTF = prob_parm.R_gas_const*prob_parm.T0/prob_parm.Faraday_const;
        return amrex::min((1.0-tplus(con))*(1.0+AC(con))*2.0*KeC(con)*RTF, -myeps);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real KDstar(amrex::Real con, const ProbParm& prob_parm)
    {
        return 0.0;
    }




}
#endif
