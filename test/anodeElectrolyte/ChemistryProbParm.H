#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real interface_location=0.5;
    amrex::Real sharpness_factor=1000.0;
    amrex::Real current_intput=3.0;
    amrex::Real electrode_dcoeff=3.0;
    amrex::Real electrolyte_dcoeff=1.0;
};
struct GlobalStorage
{
    // potential boundary conditions
    amrex::Vector<amrex::Real> pot_bc_lo{0.0, 0.0, 0.0};
    amrex::Vector<amrex::Real> pot_bc_hi{0.0, 0.0, 0.0};
};
#endif
