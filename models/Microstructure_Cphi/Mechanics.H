#ifndef _MECHANICS_H_
#define _MECHANICS_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace electrochem_mechanics {
    
    AMREX_GPU_DEVICE AMREX_INLINE void compute_eta(
    int i,
    int j,
    int k,
    Array4<Real> const& phi,
    Array4<Real> const& out_arr)
    {
        // Define domain specific Young's modulus
        const amrex::Real anode_E       = 10.0e9;
        const amrex::Real cathode_E     = 10.0e9;
        const amrex::Real electrolyte_E = 10.0e9;
        const amrex::Real E = anode_E       * (phi(i,j,k,A_AM_ID) + phi(i,j,k,A_CBD_ID))
                            + cathode_E     * (phi(i,j,k,C_AM_ID) + phi(i,j,k,C_CBD_ID))
                            + electrolyte_E * (phi(i,j,k,A_E_ID) + phi(i,j,k,S_ID) + phi(i,j,k,C_E_ID));

        // Define remaining constants
        // const amrex::Real E = 10.0e9;
        const amrex::Real nu = 0.3;

        out_arr(i, j, k) = 0.5 * E / (1.0 + nu);
    }   

    AMREX_GPU_DEVICE AMREX_INLINE void compute_kappa(
    int i,
    int j,
    int k,
    Array4<Real> const& phi,
    Array4<Real> const& eta,
    Array4<Real> const& out_arr)
    {
        // Define domain specific Young's modulus
        const amrex::Real anode_E       = 10.0e9;
        const amrex::Real cathode_E     = 10.0e9;
        const amrex::Real electrolyte_E = 10.0e9;
        const amrex::Real E = anode_E       * (phi(i,j,k,A_AM_ID) + phi(i,j,k,A_CBD_ID))
                            + cathode_E     * (phi(i,j,k,C_AM_ID) + phi(i,j,k,C_CBD_ID))
                            + electrolyte_E * (phi(i,j,k,A_E_ID) + phi(i,j,k,S_ID) + phi(i,j,k,C_E_ID));

        // Define remaining constants
        // const amrex::Real E = 10.0e9;
        const amrex::Real nu = 0.3;
        const amrex::Real lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);

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
        const amrex::Real anode_E       = 10.0e9;
        const amrex::Real cathode_E     = 10.0e9;
        const amrex::Real electrolyte_E = 10.0e9;
        const amrex::Real E = anode_E       * (phi(i,j,k,A_AM_ID) + phi(i,j,k,A_CBD_ID))
                            + cathode_E     * (phi(i,j,k,C_AM_ID) + phi(i,j,k,C_CBD_ID))
                            + electrolyte_E * (phi(i,j,k,A_E_ID) + phi(i,j,k,S_ID) + phi(i,j,k,C_E_ID));

        // Define change in concentration
        const amrex::Real anode_init       = 0.1*28000.0;
        const amrex::Real cathode_init     = 0.9*49600.0;
        const amrex::Real electrolyte_init = 1200.0;
        const amrex::Real C_Init = anode_init       * (phi(i,j,k,A_AM_ID) + phi(i,j,k,A_CBD_ID))
                                 + cathode_init     * (phi(i,j,k,C_AM_ID) + phi(i,j,k,C_CBD_ID))
                                 + electrolyte_init * (phi(i,j,k,A_E_ID) + phi(i,j,k,S_ID) + phi(i,j,k,C_E_ID));                            
        const amrex::Real deltaT = phi(i,j,k,CO_ID)-C_Init;

        // Define remaining constants
        // const amrex::Real E = 10.0e9;
        // const amrex::Real deltaT = 10.0;
        const amrex::Real beta = 10.0e-6;
        const amrex::Real nu = 0.3;
        const amrex::Real lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
        const amrex::Real G = 0.5 * E / (1.0 + nu);

        out_arr(i, j, k) = -(3.0*lambda + 2.0*G)*beta*deltaT;
    }


}
#endif