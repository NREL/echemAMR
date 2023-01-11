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

    }  

    AMREX_GPU_DEVICE AMREX_INLINE void compute_lamG_deltaT(
    int i,
    int j,
    int k,
    Array4<Real> const& phi,
    Array4<Real> const& out_arr,
    ProbParm const& prob_parm)
    {

    }

    AMREX_GPU_DEVICE AMREX_INLINE std::array<int, 8>  get_solution_ids()
    {
        std::array<int, 8> sln_list = {0, 0, 0, 0, 0, 0, 0, 0};
        return sln_list;
    }


}
#endif