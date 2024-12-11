#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real interface_location=0.5;
    amrex::Real current_intput=3.0;
    amrex::Real electrode_dcoeff=1e-8;
    amrex::Real electrolyte_dcoeff=1e-8;
    amrex::Real electrode_conductivity   = 50.0;   
    amrex::Real electrolyte_conductivity = 20.0; 
    amrex::Real Csmax_c = 20000.0;
    amrex::Real jin=3.0;
    int spltcell_x=1;
    int spltcell_y=1;
    int spltcell_z=1;
    
    amrex::Real C_c_init = 400.0; 
    amrex::Real C_e_init = 4000.0; 
};
struct GlobalStorage
{
    // potential boundary conditions
    amrex::Vector<amrex::Real> pot_bc_lo{0.0, 0.0, 0.0};
    amrex::Vector<amrex::Real> pot_bc_hi{0.0, 0.0, 0.0};
};
#endif
