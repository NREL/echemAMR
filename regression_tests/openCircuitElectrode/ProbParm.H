#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real int1 = 0.25; // interface 1
    amrex::Real int2 = 0.75; // interface 2
    amrex::Real sharpness_factor=100.0;
    int direction = 0;   
};
struct GlobalStorage
{
    // potential boundary conditions
    amrex::Vector<amrex::Real> pot_bc_lo{0.0, 0.0, 0.0};
    amrex::Vector<amrex::Real> pot_bc_hi{0.0, 0.0, 0.0};
};
#endif
