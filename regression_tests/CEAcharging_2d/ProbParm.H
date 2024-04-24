#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real angle = 45; // interface 1
    amrex::Real electrolyte_dcoeff       = 1.0e-6;
    amrex::Real anode_dcoeff             = 1.0e-6; 
    amrex::Real cathode_dcoeff           = 1.0e-6;  
    amrex::Real anode_conductivity       = 50.0;   
    amrex::Real cathode_conductivity     = 30.0; 
    amrex::Real electrolyte_conductivity = 20.0; 
    int spltcell_x=1;
    int spltcell_y=1;
    int spltcell_z=1;
    amrex::Real f0=0.25;
    
    amrex::Real Csmax_a = 20000.0;
    amrex::Real Csmax_c = 40000.0;
    amrex::Real C_c_init = 30000.0; 
    amrex::Real C_a_init = 4000.0; 
    amrex::Real C_e_init = 1000.0; 
    amrex::Real jin = 20.0;
};
struct GlobalStorage
{
    // potential boundary conditions
    amrex::Vector<amrex::Real> pot_bc_lo{0.0, 0.0, 0.0};
    amrex::Vector<amrex::Real> pot_bc_hi{0.0, 0.0, 0.0};
};
#endif
