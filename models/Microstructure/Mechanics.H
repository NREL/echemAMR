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
    Array4<Real> const& out_arr,
    ProbParm const& prob_parm)
    {
        // Get current concentration value
        Real C = phi(i, j, k, CO_ID);

        // Define domain specific Young's modulus
        const amrex::Real anode_E       = electrochem::E_anode(C, prob_parm);
        const amrex::Real cathode_E     = electrochem::E_cathode(C, prob_parm);
        const amrex::Real separator_E   = electrochem::E_separator(C, prob_parm);
        const amrex::Real electrolyte_E = electrochem::E_electrolyte(C, prob_parm);
        const amrex::Real CBD_E         = electrochem::E_CBD(C, prob_parm);

        const amrex::Real E = anode_E * phi(i,j,k,A_AM_ID) 
                            + cathode_E * phi(i,j,k,C_AM_ID) 
                            + separator_E * phi(i,j,k,S_ID)
                            + electrolyte_E * (phi(i,j,k,A_E_ID) + phi(i,j,k,C_E_ID))
                            + CBD_E * (phi(i,j,k,A_CBD_ID) + phi(i,j,k,C_CBD_ID));

        // Define domain specific Poisson ratio []
        const amrex::Real anode_nu       = electrochem::nu_anode(C, prob_parm);
        const amrex::Real cathode_nu     = electrochem::nu_cathode(C, prob_parm);
        const amrex::Real separator_nu   = electrochem::nu_separator(C, prob_parm);
        const amrex::Real electrolyte_nu = electrochem::nu_electrolyte(C, prob_parm);
        const amrex::Real CBD_nu         = electrochem::nu_CBD(C, prob_parm);

        const amrex::Real nu = anode_nu * phi(i,j,k,A_AM_ID) 
                            + cathode_nu * phi(i,j,k,C_AM_ID) 
                            + separator_nu * phi(i,j,k,S_ID)
                            + electrolyte_nu * (phi(i,j,k,A_E_ID) + phi(i,j,k,C_E_ID))
                            + CBD_nu * (phi(i,j,k,A_CBD_ID) + phi(i,j,k,C_CBD_ID));

        // 2nd Lame coefficient (Shear modulus) (Pa)
        out_arr(i, j, k) = 0.5 * E / (1.0 + nu);
    }   

    AMREX_GPU_DEVICE AMREX_INLINE void compute_bulk_modulus(
    int i,
    int j,
    int k,
    Array4<Real> const& phi,
    Array4<Real> const& eta,
    Array4<Real> const& out_arr,
    ProbParm const& prob_parm)
    {
        // Get current concentration value
        Real C = phi(i, j, k, CO_ID);

        // Define domain specific Young's modulus
        const amrex::Real anode_E       = electrochem::E_anode(C, prob_parm);
        const amrex::Real cathode_E     = electrochem::E_cathode(C, prob_parm);
        const amrex::Real separator_E   = electrochem::E_separator(C, prob_parm);
        const amrex::Real electrolyte_E = electrochem::E_electrolyte(C, prob_parm);
        const amrex::Real CBD_E         = electrochem::E_CBD(C, prob_parm);

        const amrex::Real E = anode_E * phi(i,j,k,A_AM_ID) 
                            + cathode_E * phi(i,j,k,C_AM_ID) 
                            + separator_E * phi(i,j,k,S_ID)
                            + electrolyte_E * (phi(i,j,k,A_E_ID) + phi(i,j,k,C_E_ID))
                            + CBD_E * (phi(i,j,k,A_CBD_ID) + phi(i,j,k,C_CBD_ID));

        // Define domain specific Poisson ratio
        const amrex::Real anode_nu       = electrochem::nu_anode(C, prob_parm);
        const amrex::Real cathode_nu     = electrochem::nu_cathode(C, prob_parm);
        const amrex::Real separator_nu   = electrochem::nu_separator(C, prob_parm);
        const amrex::Real electrolyte_nu = electrochem::nu_electrolyte(C, prob_parm);
        const amrex::Real CBD_nu         = electrochem::nu_CBD(C, prob_parm);
        
        const amrex::Real nu = anode_nu * phi(i,j,k,A_AM_ID) 
                            + cathode_nu * phi(i,j,k,C_AM_ID) 
                            + separator_nu * phi(i,j,k,S_ID)
                            + electrolyte_nu * (phi(i,j,k,A_E_ID) + phi(i,j,k,C_E_ID))      
                            + CBD_nu * (phi(i,j,k,A_CBD_ID) + phi(i,j,k,C_CBD_ID));
        
        
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
    Array4<Real> const& out_arr,
    ProbParm const& prob_parm)
    {
        // Get current concentration value
        Real C = phi(i, j, k, CO_ID);

        // Define domain specific Young's modulus
        const amrex::Real anode_E       = electrochem::E_anode(C, prob_parm);
        const amrex::Real cathode_E     = electrochem::E_cathode(C, prob_parm);
        const amrex::Real separator_E   = electrochem::E_separator(C, prob_parm);
        const amrex::Real electrolyte_E = electrochem::E_electrolyte(C, prob_parm);
        const amrex::Real CBD_E         = electrochem::E_CBD(C, prob_parm);

        const amrex::Real E = anode_E * phi(i,j,k,A_AM_ID) 
                            + cathode_E * phi(i,j,k,C_AM_ID) 
                            + separator_E * phi(i,j,k,S_ID)
                            + electrolyte_E * (phi(i,j,k,A_E_ID) + phi(i,j,k,C_E_ID))
                            + CBD_E * (phi(i,j,k,A_CBD_ID) + phi(i,j,k,C_CBD_ID));

        // Define domain specific Poisson ratio
        const amrex::Real anode_nu       = electrochem::nu_anode(C, prob_parm);
        const amrex::Real cathode_nu     = electrochem::nu_cathode(C, prob_parm);
        const amrex::Real separator_nu   = electrochem::nu_separator(C, prob_parm);
        const amrex::Real electrolyte_nu = electrochem::nu_electrolyte(C, prob_parm);
        const amrex::Real CBD_nu         = electrochem::nu_CBD(C, prob_parm);
        
        const amrex::Real nu = anode_nu * phi(i,j,k,A_AM_ID) 
                            + cathode_nu * phi(i,j,k,C_AM_ID) 
                            + separator_nu * phi(i,j,k,S_ID)
                            + electrolyte_nu * (phi(i,j,k,A_E_ID) + phi(i,j,k,C_E_ID))                          
                            + CBD_nu * (phi(i,j,k,A_CBD_ID) + phi(i,j,k,C_CBD_ID));

        // Define domain specific chemical expansion
        const amrex::Real anode_CCE       = electrochem::CCE_anode(C, prob_parm);
        const amrex::Real cathode_CCE     = electrochem::CCE_cathode(C, prob_parm);
        const amrex::Real separator_CCE   = electrochem::CCE_separator(C, prob_parm);
        const amrex::Real electrolyte_CCE = electrochem::CCE_electrolyte(C, prob_parm);
        const amrex::Real CBD_CCE         = electrochem::CCE_CBD(C, prob_parm);

        const amrex::Real CCE = anode_CCE * phi(i,j,k,A_AM_ID)
                              + cathode_CCE * phi(i,j,k,C_AM_ID)
                              + separator_CCE * phi(i,j,k,S_ID)
                              + electrolyte_CCE * (phi(i,j,k,A_E_ID) + phi(i,j,k,C_E_ID))
                              + CBD_CCE * (phi(i,j,k,A_CBD_ID) + phi(i,j,k,C_CBD_ID));

        // Define change in concentration
        const amrex::Real anode_init       = prob_parm.soc_anode_t0*electrochem::Cs_max_anode(prob_parm); // Stress-free concentration
        const amrex::Real cathode_init     = prob_parm.soc_cathode_t0*electrochem::Cs_max_cathode(prob_parm); // Stress-free concentration
        const amrex::Real electrolyte_init = prob_parm.ce_atrest; // Stress-free concentration

        const amrex::Real C_Init = anode_init       * (phi(i,j,k,A_AM_ID) + phi(i,j,k,A_CBD_ID))
                                 + cathode_init     * (phi(i,j,k,C_AM_ID) + phi(i,j,k,C_CBD_ID))
                                 + electrolyte_init * (phi(i,j,k,A_E_ID) + phi(i,j,k,S_ID) + phi(i,j,k,C_E_ID));                            

        const amrex::Real deltaC = phi(i,j,k,CO_ID)-C_Init;

        // Define remaining constants
        const amrex::Real lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu); // 1st Lame coefficient (Pa)
        const amrex::Real G = 0.5 * E / (1.0 + nu); // 2nd Lame coefficient (Shear modulus) (Pa)

        out_arr(i, j, k) = -(3.0*lambda + 2.0*G)*CCE*deltaC;
    }

    AMREX_INLINE std::array<int, 8>  get_solution_ids()
    {
        std::array<int, 8> sln_list = {DIS_U_ID, VON_M_ID, Sigma11_ID, Sigma22_ID, Sigma33_ID, Sigma12_ID, Sigma13_ID, Sigma23_ID};
        return sln_list;
    }



}
#endif