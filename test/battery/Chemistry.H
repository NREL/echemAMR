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

    const int CO_ID=0;
    const int PO_ID=1;
    const int AC_ID=2;
    const int ES_ID=3;

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

}
#endif
