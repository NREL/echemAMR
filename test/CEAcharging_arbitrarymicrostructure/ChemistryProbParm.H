#ifndef CNS_PROB_PARM_H_
#define CNS_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real int1 = 0.357; // interface 1
    amrex::Real int2 = 0.5;   // interface 2

    // Real order of magnitude
    amrex::Real electrolyte_dcoeff       = 1.0e-10; // m2.s-1
    amrex::Real anode_dcoeff             = 3.0e-14; // m2.s-1 
    amrex::Real cathode_dcoeff           = 1.0e-15; // m2.s-1 
    amrex::Real anode_conductivity       = 100.0; // S.m-1  
    amrex::Real cathode_conductivity     = 10.0; // S.m-1 
    amrex::Real electrolyte_conductivity = 1.0; // S.m-1  

    // For 0D case, need near infinite diffusivity and conductivity
    // amrex::Real electrolyte_dcoeff       = 1.0e-8; // m2.s-1
    // amrex::Real anode_dcoeff             = 3.0e-8; // m2.s-1 
    // amrex::Real cathode_dcoeff           = 1.0e-8; // m2.s-1 
    // amrex::Real anode_conductivity       = 100.0; // S.m-1  
    // amrex::Real cathode_conductivity     = 100.0; // S.m-1 
    // amrex::Real electrolyte_conductivity = 100.0; // S.m-1  	

    amrex::Real crate=1.0;
    amrex::Real Temp=300.0;

    amrex::Real Csmax_a = 28000.0;
    amrex::Real SOC_max_a = 0.999; 
    amrex::Real SOC_min_a = 0.001; 
    
    amrex::Real Csmax_c = 49600.0;
    amrex::Real SOC_max_c = 0.999; 
    amrex::Real SOC_min_c = 0.4;

    amrex::Real C_c_init = 49550.0;  // 49550.0
    amrex::Real C_a_init = 28.0;  // 28.0
    amrex::Real C_e_init = 1200.0; 

    // --------------
    // GEOMETRY
    // --------------
    //std::string geometry_filename = "Geometry_cube.txt";
    std::string geometry_filename = "Geometry_Cube_noSeparator.txt";
    //std::string geometry_filename = "Geometry_Concave_noSeparator.txt";

};
struct GlobalStorage
{
    // potential boundary conditions
    amrex::Vector<amrex::Real> pot_bc_lo{-1.0, 0.0, 0.0};
    amrex::Vector<amrex::Real> pot_bc_hi{ 0.0, 0.0, 0.0};
};
#endif
