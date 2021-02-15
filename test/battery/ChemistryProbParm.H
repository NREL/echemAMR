#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real K_a = 1.0e-3; // conductivity anode
    amrex::Real K_c = 2.0e-3; // conductivity cathode
    amrex::Real D_a = 3.0e-3; // diffusivity anode
    amrex::Real D_c = 4.0e-3; // diffusivity cathode
    amrex::Real Faraday_const = 96487.0;
    amrex::Real R_gas_const = 8.31446261815324;
    amrex::Real T0 = 300.0_rt; // temperature Kelvin
};

#endif
