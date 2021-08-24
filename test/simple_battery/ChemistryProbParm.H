#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real electrolyte_dcoeff       = 1.0e-10;  //m2/s
    amrex::Real separator_dcoeff         = 1.0e-10;  //m2/s
    amrex::Real anode_dcoeff             = 2.25e-14; //m2/s
    amrex::Real cathode_dcoeff           = 1.0e-14;  //m2/s
    amrex::Real anode_conductivity       = 100.0;    //S/m
    amrex::Real cathode_conductivity     = 10.0;     //S/m
    amrex::Real electrolyte_conductivity = 1.0;      //S/m
    amrex::Real separator_conductivity   = 1.0;      //S/m
    amrex::Real Temperature              = 300.0;    //K
};
struct GlobalStorage
{
    // potential boundary conditions
    amrex::Vector<amrex::Real> pot_bc_lo{3.0, 0.0, 0.0};
    amrex::Vector<amrex::Real> pot_bc_hi{0.0,  0.0, 0.0};
};
#endif
