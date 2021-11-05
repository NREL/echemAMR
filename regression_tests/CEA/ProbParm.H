#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real int1 = 0.3333333; // interface 1
    amrex::Real int2 = 0.6666666; // interface 2
    amrex::Real sharpness_factor=1000.0;
    amrex::Real electrolyte_dcoeff       = 1.0e-6;
    amrex::Real anode_dcoeff             = 1.0e-6; 
    amrex::Real cathode_dcoeff           = 1.0e-6;  
    amrex::Real anode_conductivity       = 5.0;   
    amrex::Real cathode_conductivity     = 2.0; 
    amrex::Real electrolyte_conductivity = 1.0; 
    int direction = 0;   
};
struct GlobalStorage
{
    // potential boundary conditions
    amrex::Vector<amrex::Real> pot_bc_lo{0.0, 0.0, 0.0};
    amrex::Vector<amrex::Real> pot_bc_hi{0.0, 0.0, 0.0};
};
#endif
