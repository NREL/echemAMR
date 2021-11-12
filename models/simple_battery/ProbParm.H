#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
    //amrex::Real electrolyte_dcoeff       = 1.0e-10;  //m2/s
    //amrex::Real separator_dcoeff         = 1.0e-10;  //m2/s
    //amrex::Real anode_dcoeff             = 2.25e-14; //m2/s
    //amrex::Real cathode_dcoeff           = 1.0e-14;  //m2/s
    //amrex::Real anode_conductivity       = 100.0;    //S/m
    //amrex::Real cathode_conductivity     = 10.0;     //S/m
    //amrex::Real electrolyte_conductivity = 1.0;      //S/m
    //amrex::Real separator_conductivity   = 1.0;      //S/m
    //amrex::Real Temperature              = 300.0;    //K

    // --------------
    // GEOMETRY
    // --------------
    std::string geometry_filename = "Geometry_cube.txt";
    //std::string geometry_filename = "Geometry_Cube_noSeparator.txt";
    //std::string geometry_filename = "Geometry_Concave_noSeparator.txt";
    //std::string geometry_filename = "Geometry_NMC_noseparator.txt";


    // --------------
    // MATERIAL SELECTION
    // --------------
    // Cathode
    //    0=NMC532 with constant diffusion and conductivity, cubic polynomial OCP and symmetric io
    //    1=NMC532 with constant conductivity, actual diffusion, OCP and io
    int mater_cathode = 1; 
    // Anode
    //    0=graphite SLC1506T2 with constant diffusion and conductivity, 8-th order polynomial OCP, and symmetric io
    //    1=graphite SLC1506T2 with constant diffusion and conductivity, actual OCP, and symmetric io
    int mater_anode = 1; 
    // Electrolyte
    //    0=Gen2 with constand coefficient
    //....1=Gen2 with concentration dependent coefficients
    int mater_electrolyte = 0;

    // --------------
    // INITIAL CONDITIONS and APPLIED LOADING
    // --------------
    amrex::Real ce_atrest = 1200; // Electrolyte initial concentration in both separator and electrodes [mol.m-3]
    amrex::Real soc_anode_t0 = 0.5; // [] Initial state of charge for the solid anode (ratio of cs,max). Must be between anode_soc_min and anode_soc_max.
    amrex::Real soc_cathode_t0 = 0.92; // [] Initial state of charge for the solid cathode (ratio of cs,max). Must be between cathode_soc_min and cathode_soc_max.
    amrex::Real phi_reference = 0.0; // [V] Arbitrary value

    // Electrode materials (de)lithiation range does not go from SOC 0 to 1 due to crystallographic instability.
    // Therefore, we define a min-max range for SOC, which corresponds to the pratical range of the electrode.
    // Currently, there is no additional physics in the model that occurs beyond this range, therefore nothing prevents us to go above or beyond.
    // Although, in such a case, model results are no more representative of real applicaiton.
    // Loading (charge and discharge rate) are set up so that:
    // - nC theoritical means fully lithiate or delithiate the electrode with the lower maximum mol of Li in 1/n hours (theoritical C-rate is not used).
    // - nC pratical means lithiate (delithiate) the electrode with the lower (maximum mol of Li * (soc_max-soc_min)) from its soc_min (soc_max) to its soc_max (soc_min) in 1/n hours 
    //   Crate parameter in the model corresponds to the pratical Crate.
    amrex::Real cathode_soc_min = 0.567; // [] All the theoritical state of charge range is typically not used
    amrex::Real cathode_soc_max = 0.92; // []
    amrex::Real anode_soc_min = 0.04; // [] All the theoritical state of charge range is typically not used
    amrex::Real anode_soc_max = 0.77056; // []

    // APPLIED LOADING
    amrex::Real Temperature = 300.0; // [K]. Assumed uniform and constant.
    // Temperature increases by a few degrees at fast charge due to exothermal reactions. Constant over time assumption is not ideal.
    // However, at microstructrue scale, the uniform assumption is relevant.
    amrex::Real Crate = 1.0; // [] Pratical C-rate. Typically ranges from 1/20 to 6
    // - 1/20: used during "formation" of the cell (first cycle performed at the begining of life to make sure electrolyte wett all the pores).
    // -       Also used to determine the cell capacity as at such low rate, there is no transport and kinetic limitation
    // - 0.5, 1C: slow charge rate
    // - up to 4C: intermediate charge rate
    // - >= 6C: fast charge (DOE's target is to charge in 10 minutes or less, that's our target).


};
struct GlobalStorage
{
    // potential boundary conditions
    amrex::Vector<amrex::Real> pot_bc_lo{3.0, 0.0, 0.0};
    amrex::Vector<amrex::Real> pot_bc_hi{0.0,  0.0, 0.0};

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

    // Applied current
    amrex::Real Applied_current = 0.0; // [A] Initialization
    amrex::Real charging = 1; // 1 Charging, -1 Discharging
    amrex::Vector<amrex::Real> Mass_error {0,0,0,0,0,0,0}; // time [s] / anode current [mol] / electrolyte current [mol] / cathode current [mol] / anode target [mol] / electrolyte target [mol] / cathode target [mol]

};
#endif
