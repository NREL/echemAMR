#ifndef _MECHANICS_H_
#define _MECHANICS_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace electrochem_mechanics {
    
    AMREX_GPU_DEVICE AMREX_INLINE void compute_shear_modulus(
    int i,
    int j,
    int k,
    Array4<Real> const& phi,
    Array4<Real> const& out_arr)
    {
        // Define domain specific Young's modulus
        const amrex::Real anode_E       = 32.0e9; // Pa
        const amrex::Real cathode_E     = 120.0e9; // Pa // 120.0e9 Peter document (cathode cracking literature review)
        const amrex::Real separator_E = 500.0e6; // Pa
        const amrex::Real electrolyte_E = 10.0e3; // Pa
        const amrex::Real CBD_E = 10.0e3; // Pa

        const amrex::Real E = anode_E * phi(i,j,k,A_AM_ID) 
                            + cathode_E * phi(i,j,k,C_AM_ID) 
                            + separator_E * phi(i,j,k,S_ID)
                            + CBD_E * (phi(i,j,k,A_CBD_ID) + phi(i,j,k,C_CBD_ID))
                            + electrolyte_E * (phi(i,j,k,A_E_ID) + phi(i,j,k,C_E_ID));

        // Define domain specific Poisson ratio []
        const amrex::Real anode_nu       = 0.3;
        const amrex::Real cathode_nu     = 0.3;
        const amrex::Real electrolyte_nu = 0.3;
        const amrex::Real nu = anode_nu       * (phi(i,j,k,A_AM_ID) + phi(i,j,k,A_CBD_ID))
                            + cathode_nu     * (phi(i,j,k,C_AM_ID) + phi(i,j,k,C_CBD_ID))
                            + electrolyte_nu * (phi(i,j,k,A_E_ID) + phi(i,j,k,S_ID) + phi(i,j,k,C_E_ID));

        // 2nd Lame coefficient (Shear modulus) (Pa)
        out_arr(i, j, k) = 0.5 * E / (1.0 + nu);
    }   

    AMREX_GPU_DEVICE AMREX_INLINE void compute_bulk_modulus(
    int i,
    int j,
    int k,
    Array4<Real> const& phi,
    Array4<Real> const& eta,
    Array4<Real> const& out_arr)
    {
        // Define domain specific Young's modulus
        const amrex::Real anode_E       = 32.0e9; // Pa
        const amrex::Real cathode_E     = 120.0e9; // Pa // 120.0e9 Peter document (cathode cracking literature review)
        const amrex::Real separator_E = 500.0e6; // Pa
        const amrex::Real electrolyte_E = 10.0e3; // Pa
        const amrex::Real CBD_E = 10.0e3; // Pa

        const amrex::Real E = anode_E * phi(i,j,k,A_AM_ID) 
                            + cathode_E * phi(i,j,k,C_AM_ID) 
                            + separator_E * phi(i,j,k,S_ID)
                            + CBD_E * (phi(i,j,k,A_CBD_ID) + phi(i,j,k,C_CBD_ID))
                            + electrolyte_E * (phi(i,j,k,A_E_ID) + phi(i,j,k,C_E_ID));

        // Define domain specific Poisson ratio
        const amrex::Real anode_nu       = 0.3;
        const amrex::Real cathode_nu     = 0.3;
        const amrex::Real electrolyte_nu = 0.3;
        const amrex::Real nu = anode_nu       * (phi(i,j,k,A_AM_ID) + phi(i,j,k,A_CBD_ID))
                            + cathode_nu     * (phi(i,j,k,C_AM_ID) + phi(i,j,k,C_CBD_ID))
                            + electrolyte_nu * (phi(i,j,k,A_E_ID) + phi(i,j,k,S_ID) + phi(i,j,k,C_E_ID));        
        
        
        // 1st Lame coefficient (Pa)
        const amrex::Real lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);

        // Bulk modulus (Pa)
        out_arr(i, j, k) = lambda + 2.0/3.0*eta(i,j,k);
    }  

    AMREX_GPU_DEVICE AMREX_INLINE void compute_lamG_deltaT(
    int i,
    int j,
    int k,
    Array4<Real> const& phi,
    Array4<Real> const& out_arr)
    {
        // Define domain specific Young's modulus
        const amrex::Real anode_E       = 32.0e9; // Pa
        const amrex::Real cathode_E     = 120.0e9; // Pa // 120.0e9 Peter document (cathode cracking literature review)
        const amrex::Real separator_E = 500.0e6; // Pa
        const amrex::Real electrolyte_E = 10.0e3; // Pa
        const amrex::Real CBD_E = 10.0e3; // Pa

        const amrex::Real E = anode_E * phi(i,j,k,A_AM_ID) 
                            + cathode_E * phi(i,j,k,C_AM_ID) 
                            + separator_E * phi(i,j,k,S_ID)
                            + CBD_E * (phi(i,j,k,A_CBD_ID) + phi(i,j,k,C_CBD_ID))
                            + electrolyte_E * (phi(i,j,k,A_E_ID) + phi(i,j,k,C_E_ID));

        // Define domain specific Poisson ratio
        const amrex::Real anode_nu       = 0.3;
        const amrex::Real cathode_nu     = 0.3;
        const amrex::Real electrolyte_nu = 0.3;
        const amrex::Real nu = anode_nu       * (phi(i,j,k,A_AM_ID) + phi(i,j,k,A_CBD_ID))
                            + cathode_nu     * (phi(i,j,k,C_AM_ID) + phi(i,j,k,C_CBD_ID))
                            + electrolyte_nu * (phi(i,j,k,A_E_ID) + phi(i,j,k,S_ID) + phi(i,j,k,C_E_ID));                             

        // Define domain specific chemical expansion
	// equation: https://orcid.org/0000-0002-3924-3769 : (volvar/100) / ((x_max-x_min)*csmax) with volvar=13.2%
	// graphite: https://doi.org/10.1021/acs.jpcc.8b01873 : graphite
        const amrex::Real anode_CCE       = 4.714e-6; // with volvar=13.2% and soc range = 1.0
        const amrex::Real cathode_CCE     = 1.344e-6; // //  1.344e-6 with volvar=4% and soc range = 0.6
        const amrex::Real CBD_CCE = 0.0;        
        const amrex::Real electrolyte_CCE = 0.0;
        const amrex::Real CCE = anode_CCE * phi(i,j,k,A_AM_ID)
                            + cathode_CCE * phi(i,j,k,C_AM_ID)
                            + CBD_CCE * (phi(i,j,k,A_CBD_ID) + phi(i,j,k,C_CBD_ID)) 
                            + electrolyte_CCE * (phi(i,j,k,A_E_ID) + phi(i,j,k,S_ID) + phi(i,j,k,C_E_ID));

        // Define change in concentration
        const amrex::Real anode_init       = 0.1*28000.0; // Stress-free concentration
        const amrex::Real cathode_init     = 0.9*49600.0; // Stress-free concentration
        const amrex::Real electrolyte_init = 1200.0; // Stress-free concentration
        const amrex::Real C_Init = anode_init       * (phi(i,j,k,A_AM_ID) + phi(i,j,k,A_CBD_ID))
                                 + cathode_init     * (phi(i,j,k,C_AM_ID) + phi(i,j,k,C_CBD_ID))
                                 + electrolyte_init * (phi(i,j,k,A_E_ID) + phi(i,j,k,S_ID) + phi(i,j,k,C_E_ID));                            
        const amrex::Real deltaC = phi(i,j,k,CO_ID)-C_Init;

        // Define remaining constants
        const amrex::Real lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu); // 1st Lame coefficient (Pa)
        const amrex::Real G = 0.5 * E / (1.0 + nu); // 2nd Lame coefficient (Shear modulus) (Pa)

        out_arr(i, j, k) = -(3.0*lambda + 2.0*G)*CCE*deltaC;
    }


}
#endif