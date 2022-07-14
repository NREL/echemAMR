#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
    //factor of the domain that is solid copper
    amrex::Real int1 = 0.5; // interface 1

    //units of m2/s
    amrex::Real electrolyte_dcoeff       = 1.0e-9;
    amrex::Real cathode_dcoeff           = 1.0e-16;  

    //units of S/m
    amrex::Real cathode_conductivity     = 10.0; 
    amrex::Real electrolyte_conductivity = 1.0;  
};
struct GlobalStorage
{
    // potential boundary conditions
    amrex::Vector<amrex::Real> pot_bc_lo{-1.0, 0.0, 0.0};
    amrex::Vector<amrex::Real> pot_bc_hi{ 0.0, 0.0, 0.0};
};
#endif
