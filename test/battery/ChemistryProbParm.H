#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real K_a = 100.0; // conductivity anode
    amrex::Real K_c = 10.0; // conductivity cathode
    amrex::Real D_a = 3.0e-14; // diffusivity anode
    amrex::Real D_c = 8.39694e-15; // diffusivity cathode
    amrex::Real Faraday_const = 96487.0;
    amrex::Real R_gas_const = 8.31446261815324;
    amrex::Real T0 = 300.0_rt; // temperature Kelvin

    // potential boundary conditions
    amrex::Vector<amrex::Real> pot_bc_lo{0.0, 0.0, 0.0};
    amrex::Vector<amrex::Real> pot_bc_hi{0.0, 0.0, 0.0};
};

#endif
