#ifndef _CHEMISTRY_H_
#define _CHEMISTRY_H_

#include<AMReX_REAL.H>
#include<AMReX.H>
#include<string>
#include<AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Box.H>
#include <ChemistryProbParm.H>

#define NUM_SPECIES 6
#define CO_ID 0
#define PO_ID 1
#define A_ID 2
#define C_ID 3
#define E_ID 4
#define S_ID 5
#define POT_ID 6
#define NVAR NUM_SPECIES+1

#define myeps 1.0e-12

namespace electrochem
{

    enum component {anode = 0, cathode, electrolyte, separator, last_comp = separator};

    extern amrex::Vector<std::string> specnames;
    void init();
    void close();
    int find_id(std::string specname);

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    int level_set_to_component(int i, int j, int k, amrex::Array4<amrex::Real> const& phase){

        int comp = -1;
        int ncomp = 0;

        if(phase(i,j,k,A_ID) > 0.0) {comp = anode; ++ncomp;}
        if(phase(i,j,k,C_ID) > 0.0) {comp = cathode; ++ncomp;}
        if(phase(i,j,k,E_ID) > 0.0) {comp = electrolyte; ++ncomp;}
        if(phase(i,j,k,S_ID) > 0.0) {comp = separator; ++ncomp;}

        // FIXME: need to make sure we only do this for real cells and not ghost cells
//        if(comp == -1) {
//            amrex::Print() << i << ' ' << j << ' ' << k << std::endl;
//            amrex::Abort("error component not found");
//        }
//
//        if(ncomp > 1) {
//            amrex::Print() << i << ' ' << j << ' ' << k << std::endl;
//            amrex::Abort("too many components found");
//        }

        return comp;

    }

     AMREX_GPU_DEVICE AMREX_FORCE_INLINE
     bool is_electrode(int i, int j, int k, amrex::Array4<amrex::Real> const& phase){

         auto comp = level_set_to_component(i,j,k,phase);

         if(comp == anode || comp == cathode)
             return true;
         else
             return false;
     }


    // Electrolyte conductivity
    // Ce in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real KeC(amrex::Real Ce)
    {
        constexpr amrex::Real a4 = -3.334591539362114e-14;
        constexpr amrex::Real a3 = 3.867473118129116e-10;
        constexpr amrex::Real a2 = -1.559193894834466e-06;
        constexpr amrex::Real a1 = 0.002182354774294;
        constexpr amrex::Real a0 = 0.0;
        return amrex::max((((a4 * Ce + a3) * Ce + a2) * Ce + a1) * Ce + a0, myeps);
    }

    // Electrolyte diffusion coefficient
    // Ce in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real DeC(amrex::Real Ce)
    {
        constexpr amrex::Real a4 = 7.414554029503555e-25;
        constexpr amrex::Real a3 = -1.159860784909586e-20;
        constexpr amrex::Real a2 = 7.492287453960106e-17;
        constexpr amrex::Real a1 = -2.451314899670142e-13;
        constexpr amrex::Real a0 = 3.396397365975209e-10;
        return amrex::max((((a4 * Ce + a3) * Ce + a2) * Ce + a1) * Ce + a0, myeps); // [m2.s-1]
    }

    // Transference number
    // Ce in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real t_plus(amrex::Real Ce)
    {
        constexpr amrex::Real a2 = -2.266955710225041e-09;
        constexpr amrex::Real a1 = 2.125258249150030e-05;
        constexpr amrex::Real a0 = 0.440686353438825;
        return amrex::max( (a2 * Ce + a1) * Ce + a0, myeps);
    }

    // Electrolyte activity
    // Ce in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real Activity(amrex::Real Ce)
    {
        // Valid for temperature = 30 + 273.15 K
        constexpr amrex::Real a2 = 1.598531153290851e-06;
        constexpr amrex::Real a1 = -1.997682922513610e-04;
        constexpr amrex::Real a0 = 0.806611708428853;
        return amrex::max((a2 * Ce + a1) * Ce + a0, myeps);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real KD(amrex::Real con, const ProbParm& prob_parm)
    {
        const amrex::Real RTF = prob_parm.R_gas_const*prob_parm.T0/prob_parm.Faraday_const;
        // FIXME: there is a sign mismatch... double check this tplus-1 vs 1-tplus
        return amrex::min((1.0 - t_plus(con)) * (1.0 + Activity(con)) * 2.0 * KeC(con) * RTF, -myeps);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real KDstar(amrex::Real con, const ProbParm& prob_parm)
    {
        return 0.0;
    }


    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real intercalation_function(amrex::Real Ca, amrex::Real Ce)
    {
        return Ca * Ce;
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real intercalation_reaction_anode(amrex::Real Ca, amrex::Real phi_a, amrex::Real Ce, amrex::Real phi_e, const ProbParm& prob_parm)
    {
        const amrex::Real ia = intercalation_function(Ca, Ce);
        const amrex::Real eta_alpha = phi_a - phi_e;// FIXME - OCP?
        const amrex::Real FRT = prob_parm.Faraday_const/(prob_parm.R_gas_const * prob_parm.T0);
        return 2.0 * ia * std::sinh(0.5 * FRT * eta_alpha);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real intercalation_reaction_cathode(amrex::Real Cc, amrex::Real phi_c, amrex::Real Ce, amrex::Real phi_e, const ProbParm& prob_parm)
    {
        // these are the same for now
        return intercalation_reaction_anode(Cc, phi_c, Ce, phi_e, prob_parm);
    }


    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real ic_reaction(const int i, const int j, const int k, const int dir, const amrex::Real normal, amrex::Array4<amrex::Real> const& phi, const ProbParm& prob_parm)
    {

        // components
        const int c0 = level_set_to_component(i,j,k,phi);
        int cm1;
        int cp1;

        // shifted indices
        int im1 = i;
        int ip1 = i;
        int jm1 = j;
        int jp1 = j;
        int km1 = k;
        int kp1 = k;

        switch (dir) {
            case 0:
                cm1 = level_set_to_component(i-1,j,k,phi);
                cp1 = level_set_to_component(i+1,j,k,phi);
                im1 = i-1;
                ip1 = i+1;
                break;
            case 1:
                cm1 = level_set_to_component(i,j-1,k,phi);
                cp1 = level_set_to_component(i,j+1,k,phi);
                jm1 = j-1;
                jp1 = j+1;
                break;
            case 2:
                cm1 = level_set_to_component(i,j,k-1,phi);
                cp1 = level_set_to_component(i,j,k+1,phi);
                km1 = k-1;
                kp1 = k+1;
                break;
            default:
                amrex::Abort("bad direction");
                break;
        }

        amrex::Real reaction = 0.0;

        if(normal > 0.0){

            if(c0 == cathode && cp1 == electrolyte ){
                const amrex::Real Cc = phi(i,j,k,CO_ID);
                const amrex::Real Pc = phi(i,j,k,PO_ID);
                const amrex::Real Ce = phi(ip1,jp1,kp1,CO_ID);
                const amrex::Real Pe = phi(ip1,jp1,kp1,PO_ID);
                reaction = -normal*intercalation_reaction_cathode(Cc, Pc, Ce, Pe, prob_parm);
            }

            if(c0 == electrolyte && cp1 == anode){
                const amrex::Real Ce = phi(i,j,k,CO_ID);
                const amrex::Real Pe = phi(i,j,k,PO_ID);
                const amrex::Real Ca = phi(ip1,jp1,kp1,CO_ID);
                const amrex::Real Pa = phi(ip1,jp1,kp1,PO_ID);
                reaction = -normal*intercalation_reaction_anode(Ca, Pa, Ce, Pe, prob_parm);
            }

            if(c0 == electrolyte && cp1 == cathode){
                const amrex::Real Ce = phi(i,j,k,CO_ID);
                const amrex::Real Pe = phi(i,j,k,PO_ID);
                const amrex::Real Cc = phi(ip1,jp1,kp1,CO_ID);
                const amrex::Real Pc = phi(ip1,jp1,kp1,PO_ID);
                reaction = normal*intercalation_reaction_cathode(Cc, Pc, Ce, Pe, prob_parm);
            }

            if(c0 == anode && cp1 == electrolyte){
                const amrex::Real Ca = phi(i,j,k,CO_ID);
                const amrex::Real Pa = phi(i,j,k,PO_ID);
                const amrex::Real Ce = phi(ip1,jp1,kp1,CO_ID);
                const amrex::Real Pe = phi(ip1,jp1,kp1,PO_ID);
                reaction = normal*intercalation_reaction_anode(Ca, Pa, Ce, Pe, prob_parm);
            }


        } else {

            if(cm1 == cathode && c0 == electrolyte){
                const amrex::Real Cc = phi(im1,jm1,km1,CO_ID);
                const amrex::Real Pc = phi(im1,jm1,km1,PO_ID);
                const amrex::Real Ce = phi(i,j,k,CO_ID);
                const amrex::Real Pe = phi(i,j,k,PO_ID);
                reaction = -normal*intercalation_reaction_cathode(Cc, Pc, Ce, Pe, prob_parm);
            }

            if(cm1 == electrolyte && c0 == anode){
                const amrex::Real Ce = phi(im1,jm1,km1,CO_ID);
                const amrex::Real Pe = phi(im1,jm1,km1,PO_ID);
                const amrex::Real Ca = phi(i,j,k,CO_ID);
                const amrex::Real Pa = phi(i,j,k,PO_ID);
                reaction = -normal*intercalation_reaction_anode(Ca, Pa, Ce, Pe, prob_parm);
            }

            if(cm1 == electrolyte && c0 == cathode){
                const amrex::Real Ce = phi(im1,jm1,km1,CO_ID);
                const amrex::Real Pe = phi(im1,jm1,km1,PO_ID);
                const amrex::Real Cc = phi(i,j,k,CO_ID);
                const amrex::Real Pc = phi(i,j,k,PO_ID);
                reaction = normal*intercalation_reaction_cathode(Cc, Pc, Ce, Pe, prob_parm);
            }

            if(cm1 == anode && c0 == electrolyte){
                const amrex::Real Ca = phi(im1,jm1,km1,CO_ID);
                const amrex::Real Pa = phi(im1,jm1,km1,PO_ID);
                const amrex::Real Ce = phi(i,j,k,CO_ID);
                const amrex::Real Pe = phi(i,j,k,PO_ID);
                reaction = normal*intercalation_reaction_anode(Ca, Pa, Ce, Pe, prob_parm);
            }
        }

        return reaction;

    }


}
#endif
