#ifndef _CHEMISTRY_H_
#define _CHEMISTRY_H_

#include<AMReX_REAL.H>
#include<AMReX.H>
#include<string>
#include<AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Box.H>

#define NUM_SPECIES 6
#define CO_ID 0
#define PO_ID 1
#define A_ID 2
#define C_ID 3
#define E_ID 4
#define S_ID 5
#define POT_ID 6
#define NVAR NUM_SPECIES+1

namespace electrochem
{

    enum component {anode = 0, cathode, electrolyte, separator, last_comp = separator};

    const amrex::Real K_a = 1.0e-3; // conductivity anode
    const amrex::Real K_c = 2.0e-3; // conductivity cathode
    const amrex::Real D_a = 3.0e-3; // diffusivity anode
    const amrex::Real D_c = 4.0e-3; // diffusivity cathode
    const amrex::Real Faraday_const = 96487.0;
    const amrex::Real R_gas_const = 8.31446261815324;
    const amrex::Real T0 = 300.0; // temperature Kelvin

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

}
#endif
