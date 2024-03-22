#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>
#include <AMReX_Vector.H>

using namespace amrex::literals;

struct ProbParm
{
    // NOTE: All values are overwritten by inputs
    // --------------
    // MATERIAL SELECTION
    // --------------
    // Mater_* = 0: test expression
    //         = 1: constant (except for OCP and io: variable with a simple expression)
    //         = 2: variable, real (i.e., complicated) expression  
    
    // Cathode
    //    0=constant diffusion, conductivity, OCP, and io
    //    1=NMC532 with constant diffusion and conductivity, cubic polynomial OCP and symmetric io
    //    2=NMC532 with constant conductivity, actual diffusion, OCP and io
    int mater_cathode = 2; 
    int mater_mech_cathode = 0;
    // Specific surface area correction
    amrex::Real Sp_corr_cathode = 1.0;    
    
    // Anode
    //    0=constant diffusion, conductivity, OCP, and io
    //    1=graphite SLC1506T2 with constant diffusion and conductivity, 8-th order polynomial OCP, and symmetric io
    //    2=graphite SLC1506T2 with constant diffusion and conductivity, actual OCP, and symmetric io
    int mater_anode = 2; 
    int mater_mech_anode = 0;
    // Specific surface area correction
    amrex::Real Sp_corr_anode = 1.0;    
    
    // Electrolyte (diffusivity, conductivity)
    //    0=constant coefficient
    //    1=Gen2 with constant coefficient
    //....2=Gen2 with concentration dependent coefficients
    int mater_electrolyte = 2;
    int mater_mech_electrolyte = 0;
    int mater_mech_separator = 0;

    // Electrolyte transference number, Activity coefficient
    //    0=constant coefficient
    //    1=Gen2 with constant coefficient
    //....2=Gen2 with concentration dependent coefficients
    int mater_electrolyte_tpAc = 1;

    // Carbon black binder
    int mater_CBD = 1;
    int mater_mech_CBD = 0;

    // Nanoscale
    // = 0: nanoporosity = 1 and MacMullin = 1 (overwritte nanoporosity_filename and macmullin_filename)
    // = 1: use txt files
    int Nanoscale_coefficients_txtimport_CBD = 1;
    int Nanoscale_coefficients_txtimport_separator = 1;
    // CBD_transport = 0: only transport e- (solid potential), i.e., pure solid phase.
    //                 BC: same as active material but only for applied loading
    //                = 1: transport e- (solid potential) and Li+ (electrolyte concentration), i.e., a mixed phase with nanoporosity
    //                 BC: same as active material
    int CBD_transport = 1;

    // Replace electrolyte with CBD
    int ELEisCBD = 0;
    // Possible combinations:
    // 1) no CBD mesh + CBD_transport=0 + Nanoscale_coefficients_txtimport_CBD=0 + ELEisCBD=0: no nanporosity (oversimplication)
    // 2) no CBD mesh + CBD_transport=0 + Nanoscale_coefficients_txtimport_CBD=1 + ELEisCBD=0: nanoporosity impact on ionic diffusion
    // 3) no CBD mesh + CBD_transport=1 + Nanoscale_coefficients_txtimport_CBD=1 + ELEisCBD=1: nanoporosity, mixed CBD
    // 4) CBD mesh + CBD_transport=0 + Nanoscale_coefficients_txtimport_CBD=0 + ELEisCBD=0: no nanoporosity, solid CBD
    // 5) CBD mesh + CBD_transport=1 + Nanoscale_coefficients_txtimport_CBD=1 + ELEisCBD=0: nanoporosity, mixed CBD


    // Special case: 0D model comparison [0=false, 1=true]
    // If true, diffusivity and conductivity are set to un-realistic high values
    int zeroDmodel = 0;

    // Full cell (=0), or half cell, anode is Li-reference (=1) or cathode is Li-reference (=2)
    // If 1 or 2, overwritte coefficients from prob.mater_*
    int halfcell_case = 0;

    // PHYSICS
    // Electrolyte concentration-potential coupling
    //    0=no
    //    1=yes (transference number, activity coefficient)
    int Cphi_electrolyte_coupling = 0;    

    // --------------
    // GEOMETRY
    // --------------
    int dimension_geometry[3];
    int* geo1D = nullptr;
    amrex::Real* Nanoporosity1D = nullptr;
    amrex::Real* MacMullin1D = nullptr;

    // std::vector<int> geo1D;
    // amrex::Vector<int> geo1D;
    // amrex::Gpu::PinnedVector<int> geo1D;
    // amrex::Gpu::ManagedVector<int> geo1D{0};
    // --------------
    // INITIAL CONDITIONS and APPLIED LOADING
    // --------------
    amrex::Real ce_atrest = 1200; // Electrolyte initial concentration in both separator and electrodes [mol.m-3]
    amrex::Real soc_anode_t0 = 0.001; // [] Initial state of charge for the solid anode (ratio of cs,max). Must be between anode_soc_min and anode_soc_max.
    amrex::Real soc_cathode_t0 = 0.999; // [] Initial state of charge for the solid cathode (ratio of cs,max). Must be between cathode_soc_min and cathode_soc_max.
    amrex::Real phi_reference = 0.0; // [V] Arbitrary value
    
    // Electrode materials (de)lithiation range does not go from SOC 0 to 1 due to crystallographic instability.
    // Therefore, we define a min-max range for SOC, which corresponds to the pratical range of the electrode.
    // Currently, there is no additional physics in the model that occurs beyond this range, therefore nothing prevents us to go above or beyond.
    // Although, in such a case, model results are no more representative of real applicaiton.
    // Loading (charge and discharge rate) are set up so that:
    // - nC theoritical means fully lithiate or delithiate the electrode with the lower maximum mol of Li in 1/n hours (theoritical C-rate is not used).
    // - nC pratical means lithiate (delithiate) the electrode with the lower (maximum mol of Li * (soc_max-soc_min)) from its soc_min (soc_max) to its soc_max (soc_min) in 1/n hours 
    //   Crate parameter in the model corresponds to the pratical Crate.
    // Coefficients below should be within Chemistry.H as they are material-dependent, however, it's more simple to put them here since initial SOC must be within their bounds.
    amrex::Real cathode_soc_min = 0.4; // [] All the theoritical state of charge range is typically not used
    amrex::Real cathode_soc_max = 0.999; // []
    amrex::Real anode_soc_min = 0.001; // [] All the theoritical state of charge range is typically not used
    amrex::Real anode_soc_max = 0.999; // []

    // APPLIED LOADING
    amrex::Real Temperature = 300.0; // [K]. Assumed uniform and constant.
    // Temperature increases by a few degrees at fast charge due to exothermal reactions. Constant over time assumption is not ideal.
    // However, at microstructrue scale, the uniform assumption is relevant.
    
    // Crate is overwritte in input file
    amrex::Real Crate = 1.0; // [] Pratical C-rate. Typically ranges from 1/20 to 6
    // - 1/20: used during "formation" of the cell (first cycle performed at the begining of life to make sure electrolyte wett all the pores).
    // -       Also used to determine the cell capacity as at such low rate, there is no transport and kinetic limitation
    // - 0.5, 1C: slow charge rate
    // - up to 4C: intermediate charge rate
    // - >= 6C: fast charge (DOE's target is to charge in 10 minutes or less, that's our target).


    amrex::Real phis_a_to;
    amrex::Real phie_to;
    amrex::Real phis_c_to;
    amrex::Real cs_a_to;
    amrex::Real cs_c_to;
    amrex::Real ce_to;


};
struct GlobalStorage
{
    // potential boundary conditions
    amrex::Vector<amrex::Real> pot_bc_lo{-1.0, 0.0, 0.0};
    amrex::Vector<amrex::Real> pot_bc_hi{ 0.0, 0.0, 0.0};

    // Volumes for capacity calculation
    amrex::Real anode_AM_volume = 0.0;
    amrex::Real cathode_AM_volume = 0.0;

    // Surface Areas
    // Current collector - solid e- conductor (Applied loading)
    amrex::Real anode_ccAM_SA = 0.0;
    amrex::Real cathode_ccAM_SA = 0.0;
    amrex::Real anode_ccCBDs_SA = 0.0;
    amrex::Real cathode_ccCBDs_SA = 0.0;

    // Electrolyte - active material (Intercalation reaction)
    amrex::Real anode_eleAM_SA = 0.0;
    amrex::Real cathode_eleAM_SA = 0.0;
    amrex::Real anode_seppAM_SA = 0.0;
    amrex::Real cathode_seppAM_SA = 0.0;
    amrex::Real anode_CBDpAM_SA = 0.0;
    amrex::Real cathode_CBDpAM_SA = 0.0;

    // Carbon binder and separator representation
    amrex::Real CBD_transport = -1;
    amrex::Real Nanoscale_coefficients_txtimport_CBD = -1;
    amrex::Real Nanoscale_coefficients_txtimport_separator = -1;

    // Applied current
    amrex::Real Applied_current = 0.0; // [A] Initialization
    amrex::Real charging = 1; // 1 Charging, -1 Discharging
    amrex::Vector<amrex::Real> Mass_error {0,0,0,0,0,0,0}; // time [s] / anode current [mol] / electrolyte current [mol] / cathode current [mol] / anode target [mol] / electrolyte target [mol] / cathode target [mol]

};
#endif
