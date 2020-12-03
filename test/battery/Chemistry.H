#ifndef _CHEMISTRY_H_
#define _CHEMISTRY_H_

#include<AMReX_REAL.H>
#include<AMReX.H>
#include<string>
#include<AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Box.H>

namespace electrochem
{
    const int nspecies=4;

    enum species {CO_ID = 0, PO_ID, AC_ID, ES_ID, last_ID=ES_ID};
    enum component {cathode = 0, anode, electrolyte, electrolyte_separator, last_comp=electrolyte_separator};

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

        if(phase(i,j,k,AC_ID) > 0.0) // electrode
            return phase(i,j,k,ES_ID) > 0.0 ? cathode : anode;
        else
            return phase(i,j,k,ES_ID) > 0.0 ? electrolyte : electrolyte_separator;
    }

     AMREX_GPU_DEVICE AMREX_FORCE_INLINE
     bool is_electrode(int i, int j, int k, amrex::Array4<amrex::Real> const& phase){
         return (phase(i,j,k,AC_ID) > 0.0) ? true : false;
     }

}
#endif
