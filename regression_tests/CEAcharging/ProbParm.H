#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real int1 = 0.357; // interface 1
    amrex::Real int2 = 0.5;   // interface 2
    amrex::Real electrolyte_dcoeff       = 1.0e-10;
    amrex::Real anode_dcoeff             = 1.0e-12; 
    amrex::Real cathode_dcoeff           = 1.0e-12;  
    amrex::Real anode_conductivity       = 10.0;   
    amrex::Real cathode_conductivity     = 5.0; 
    amrex::Real electrolyte_conductivity = 1.0;  

    amrex::Real crate=1.0;
    amrex::Real Temp=300.0;

    amrex::Real Csmax_a = 28000.0;
    amrex::Real SOC_max_a = 0.999; 
    amrex::Real SOC_min_a = 0.001; 
    
    amrex::Real Csmax_c = 49600.0;
    amrex::Real SOC_max_c = 0.999; 
    amrex::Real SOC_min_c = 0.4;

    amrex::Real C_c_init = 49550.0; 
    amrex::Real C_a_init = 28.0; 
    amrex::Real C_e_init = 1200.0; 

};
struct GlobalStorage
{
    // potential boundary conditions
    amrex::Vector<amrex::Real> pot_bc_lo{-4.0, 0.0, 0.0};
    amrex::Vector<amrex::Real> pot_bc_hi{ 0.0, 0.0, 0.0};
};
#endif
