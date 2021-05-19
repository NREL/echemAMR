#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{

    // ------------------
    // PHYSICAL-CONSTANTS
    // ------------------
    amrex::Real Faraday_const = 96487.0; // [C.mol-1]
    amrex::Real R_gas_const = 8.31446261815324; // [J.mol-1.K-1]

    // --------------
    // USER-PARAMETER
    // --------------

    // INITIAL CONDITIONS
    amrex::Real ce_atrest = 1200; // [mol.m-3]
    amrex::Real soc_anode_t0 = 0.04; // [] Should be equal to anode_soc_min (see Applied loading)
    amrex::Real soc_cathode_t0 = 0.92; // [] Should be equal to cathode_soc_max (see Applied loading)
    amrex::Real phi_reference = 5.0; // [V] Arbitrary value (avoid 0.0 for numerical convergence)

    // APPLIED LOADING
    amrex::Real T0 = 300.0_rt; // temperature Kelvin
    amrex::Real Crate = 1; // [] Typically ranges from 1/20 to 6 (6 and above are considered fast-charge)
    // Pratical state of charge range is more a material property (should be in Chemistry.H).
    // Although as it informs the initial state of charge and that I prefer not to hard-code an equality (to let the user free to initialize not the typical value) the values are there.
    amrex::Real cathode_soc_min = 0.567; // [] All the theoritical state of charge range is typically not used
    amrex::Real cathode_soc_max = 0.92; // []
    amrex::Real anode_soc_min = 0.04; // [] All the theoritical state of charge range is typically not used
    amrex::Real anode_soc_max = 0.77056; // []


    // MATERIAL SELECTION
    // Electrolyte
    //    0=Generation 2 with constant parameters
    //    1=Generation 2 with actual parameters
    int mater_electrolyte = 1;
    bool use_KDstar = true; // Modify equation to use modified expression of diffusionial conducitivty KD
    // Cathode
    //    0=NMC532 with constant diffusion and conductivity, cubic polynomial OCP and symmetric io
    //    1=NMC532 with constant conductivity, actual diffusion, OCP and io
    int mater_cathode = 1; 
    // Anode
    //    0=graphite SLC1506T2 with constant diffusion and conductivity, 8-th order polynomial OCP, and symmetric io
    //    1=graphite SLC1506T2 with constant diffusion and conductivity, actual OCP, and symmetric io
    int mater_anode = 1; 

    // Lower scale parameters
    // Electrolyte: lower scale solid phase being not meshed: additive phase (CBD) for the cathode and the anode domains, and separator solid for the separator domain
    // To run the model without impact of this parameter, set porosity=1.0
    // Porosiy value: real [0,1], Bruggeman exponent: real >=1.5
    amrex::Real lowerscale_vf_pore_anode = 1.0; // In the anode, each mesh cell has a porosity of
    amrex::Real lowerscale_vf_pore_separator = 0.4; // In the separator, each mesh cell has a porosity of
    amrex::Real lowerscale_vf_pore_cathode = 1.0; // In the cathode, each mesh cell has a porosity of
    amrex::Real lowerscale_p_pore_anode = 3.0; // In the anode, each mesh cell has a Bruggeman coefficient of
    amrex::Real lowerscale_p_pore_separator = 2.4; // In the separator, each mesh cell has a Bruggeman coefficient of
    amrex::Real lowerscale_p_pore_cathode = 3.0; // In the cathode, each mesh cell has a Bruggeman coefficient of

    // potential boundary conditions
    amrex::Vector<amrex::Real> pot_bc_lo{0.0, 0.0, 0.0};
    amrex::Vector<amrex::Real> pot_bc_hi{0.0, 0.0, 0.0};

    // Volumes
    amrex::Real anode_volume = 0.0;
    amrex::Real ele_sep_volume = 0.0;
    amrex::Real cathode_volume = 0.0;

    // Surface Areas
    amrex::Real anode_SA = 0.0;
    amrex::Real electrolyte_SA = 0.0;
    amrex::Real sepatator_SA = 0.0;
    amrex::Real cathode_SA = 0.0;
    amrex::Real anode_ele_SA = 0.0;
    amrex::Real cathode_ele_SA = 0.0;
    amrex::Real anode_cc_SA = 0.0;
    amrex::Real cathode_cc_SA = 0.0;

};

#endif
