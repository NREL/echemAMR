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

#define NUM_SPECIES 5
#define CO_ID 0
#define A_ID 1
#define C_ID 2
#define E_ID 3
#define S_ID 4
#define PO_ID 5
#define NVAR NUM_SPECIES+1

#define myeps 1.0e-30

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
        if(comp == -1) {
            amrex::Print() << i << ' ' << j << ' ' << k << std::endl;
            amrex::Abort("error component not found");
        }

        if(ncomp > 1) {
            amrex::Print() << i << ' ' << j << ' ' << k << std::endl;
            amrex::Abort("too many components found");
        }

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
    amrex::Real KeC(amrex::Real Ce, const ProbParm& prob_parm)
    {

        if(prob_parm.constant_coeff) {
            return prob_parm.Kebulk;
        } else {
            constexpr amrex::Real a4 = -3.334591539362114e-14;
            constexpr amrex::Real a3 = 3.867473118129116e-10;
            constexpr amrex::Real a2 = -1.559193894834466e-06;
            constexpr amrex::Real a1 = 0.002182354774294;
            constexpr amrex::Real a0 = 0.0;
            return amrex::max((((a4 * Ce + a3) * Ce + a2) * Ce + a1) * Ce + a0, myeps);
        }
    }

    // Electrolyte diffusion coefficient
    // Ce in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real DeC(amrex::Real Ce, const ProbParm& prob_parm)
    {

        if(prob_parm.constant_coeff){
            return prob_parm.Debulk;
        } else {
            constexpr amrex::Real a4 = 7.414554029503555e-25;
            constexpr amrex::Real a3 = -1.159860784909586e-20;
            constexpr amrex::Real a2 = 7.492287453960106e-17;
            constexpr amrex::Real a1 = -2.451314899670142e-13;
            constexpr amrex::Real a0 = 3.396397365975209e-10;
            return amrex::max((((a4 * Ce + a3) * Ce + a2) * Ce + a1) * Ce + a0, myeps); // [m2.s-1]
        }
    }

    // Transference number
    // Ce in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real t_plus(amrex::Real Ce, const ProbParm& prob_parm)
    {
        if(prob_parm.constant_coeff) {
            return prob_parm.tplus;
        } else {
            constexpr amrex::Real a2 = -2.266955710225041e-09;
            constexpr amrex::Real a1 = 2.125258249150030e-05;
            constexpr amrex::Real a0 = 0.440686353438825;
            return amrex::max( (a2 * Ce + a1) * Ce + a0, myeps);
        }
    }

    // Electrolyte activity
    // Ce in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real Activity(amrex::Real Ce, const ProbParm& prob_parm)
    {
        if(prob_parm.constant_coeff) {
            return prob_parm.Activityel;
        } else {
            // Valid for temperature = 30 + 273.15 K
            constexpr amrex::Real a2 = 1.598531153290851e-06;
            constexpr amrex::Real a1 = -1.997682922513610e-04;
            constexpr amrex::Real a0 = 0.806611708428853;
            return amrex::max((a2 * Ce + a1) * Ce + a0, myeps);
        }
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real KD(amrex::Real Ce, const ProbParm& prob_parm)
    {
        const amrex::Real RTF = prob_parm.R_gas_const*prob_parm.T0/prob_parm.Faraday_const;
        // FIXME: there is a sign mismatch... double check this tplus-1 vs 1-tplus
        return amrex::min((1.0 - t_plus(Ce, prob_parm)) * (1.0 + Activity(Ce, prob_parm)) * 2.0 * KeC(Ce, prob_parm) * RTF, -myeps);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real kestar(amrex::Real Ce, const ProbParm& prob_parm)
    {
        // FIXME: does this have a constant coefficient form?
        constexpr amrex::Real a1 = 1.0;
        constexpr amrex::Real a2 = 0.0;
        constexpr amrex::Real a3 = 0.0;
        constexpr amrex::Real a4 = 0.0;

        return a1 + a2*Ce + a3*Ce*Ce + a4*Ce*Ce*Ce;
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real KDstar(amrex::Real Ce, const ProbParm& prob_parm)
    {
        return amrex::min(kestar(Ce, prob_parm) * (1.0 - t_plus(Ce, prob_parm)) * (1.0 + Activity(Ce, prob_parm)), -myeps);
    }


    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real intercalation_function(amrex::Real Ca, amrex::Real Ce)
    {
        // FIXME: not sure what should be here yet
        return std::sqrt(Ca*Ce);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real intercalation_reaction_anode(amrex::Real Ca, amrex::Real phi_a, amrex::Real Ce, amrex::Real phi_e, const ProbParm& prob_parm)
    {
        const amrex::Real ia = intercalation_function(Ca, Ce);
        const amrex::Real eta_alpha = phi_a - phi_e;// FIXME: - OCP?
        const amrex::Real FRT = prob_parm.Faraday_const/(prob_parm.R_gas_const * prob_parm.T0);
        return 2.0 * ia * std::sinh(0.5 * FRT * eta_alpha); //FIXME: gpu safe?
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
        int cm1,cp1;

        // shifted indices
        int ileft=i;
        int iright=i;
        int jleft=j;
        int jright=j;
        int kleft=k;
        int kright=k;

        switch (dir) {
            case 0:
                cm1 = level_set_to_component(i-1,j,k,phi);
                cp1 = level_set_to_component(i+1,j,k,phi);
                ileft = i-1;
                iright = i+1;
                break;
            case 1:
                cm1 = level_set_to_component(i,j-1,k,phi);
                cp1 = level_set_to_component(i,j+1,k,phi);
                jleft = j-1;
                jright = j+1;
                break;
            case 2:
                cm1 = level_set_to_component(i,j,k-1,phi);
                cp1 = level_set_to_component(i,j,k+1,phi);
                kleft = k-1;
                kright = k+1;
                break;
            default:
                amrex::Abort("bad direction");
                break;
        }

        int cl,cr;

        if(normal > 0.0) {
            cl = c0;
            cr = cp1;
            ileft = i;
            jleft = j;
            kleft = k;
        } else {
            cl = cm1;
            cr = c0;
            iright = i;
            jright = j;
            kright = k;
        }

        amrex::Real reaction = 0.0;

        if(cl == cathode && cr == electrolyte ){
            const amrex::Real Cc = phi(ileft,jleft,kleft,CO_ID);
            const amrex::Real Pc = phi(ileft,jleft,kleft,PO_ID);
            const amrex::Real Ce = phi(iright,jright,kright,CO_ID);
            const amrex::Real Pe = phi(iright,jright,kright,PO_ID);
            reaction = -normal*intercalation_reaction_cathode(Cc, Pc, Ce, Pe, prob_parm);
        }

        if(cl == electrolyte && cr == cathode){
            const amrex::Real Ce = phi(ileft,jleft,kleft,CO_ID);
            const amrex::Real Pe = phi(ileft,jleft,kleft,PO_ID);
            const amrex::Real Cc = phi(iright,jright,kright,CO_ID);
            const amrex::Real Pc = phi(iright,jright,kright,PO_ID);
            reaction = normal*intercalation_reaction_cathode(Cc, Pc, Ce, Pe, prob_parm);
        }

        if(cl == electrolyte && cr == anode){
            const amrex::Real Ce = phi(ileft,jleft,kleft,CO_ID);
            const amrex::Real Pe = phi(ileft,jleft,kleft,PO_ID);
            const amrex::Real Ca = phi(iright,jright,kright,CO_ID);
            const amrex::Real Pa = phi(iright,jright,kright,PO_ID);
            reaction = -normal*intercalation_reaction_anode(Ca, Pa, Ce, Pe, prob_parm);
        }

        if(cl == anode && cr == electrolyte){
            const amrex::Real Ca = phi(ileft,jleft,kleft,CO_ID);
            const amrex::Real Pa = phi(ileft,jleft,kleft,PO_ID);
            const amrex::Real Ce = phi(iright,jright,kright,CO_ID);
            const amrex::Real Pe = phi(iright,jright,kright,PO_ID);
            reaction = normal*intercalation_reaction_anode(Ca, Pa, Ce, Pe, prob_parm);
        }


        return reaction;

    }


}
#endif
