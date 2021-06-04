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
#include <math.h>  // signbit, exp
#include <cmath>   // pow

#define NUM_SPECIES 5
#define CO_ID 0
#define A_ID 1
#define C_ID 2
#define E_ID 3
#define S_ID 4
#define EFX_ID 5
#define EFY_ID 6
#define EFZ_ID 7
#define PO_ID 8
#define NVAR NUM_SPECIES+4

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
#ifndef AMREX_USE_GPU
        if(comp == -1) {
            amrex::Print() << i << ' ' << j << ' ' << k << std::endl;
            amrex::Abort("error component not found");
        }

        if(ncomp > 1) {
            amrex::Print() << i << ' ' << j << ' ' << k << std::endl;
            amrex::Abort("too many components found");
        }
#endif

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


    // ---------------------
    // MATERIAL COEFFICIENTS
    // ---------------------
    // Parameter selection is available in ChemistryProbParm.H

    // ELECTROLYTE MATERIAL COEFFICIENTS
    
    // Bulk to effective  (lower scale to model scale) corrective factor
    // Modificator to be applied on effective medium: electrolyte and separator
    // FIX ME: Xeff_Xbulk should be different for anode, separator and cathode (lowerscale_vf_pore_*, and lowerscale_p_pore_*), currently all set with anode parameter
    //         use const int comp = electrochem::level_set_to_component(i,j,k,phi);
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real XeffXbulk(const ProbParm& prob_parm)
    {
        amrex::Real Xeff_Xbulk = pow(prob_parm.lowerscale_vf_pore_anode,prob_parm.lowerscale_p_pore_anode);
        return Xeff_Xbulk;
    }

    // Electrolyte conductivity [S.m-1]. Bulk value, not be used directly in transport equations
    // Ce in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real Ke_Bulk(amrex::Real Ce, const ProbParm& prob_parm)
    {

        if(prob_parm.mater_electrolyte==0) {
            amrex::Real Kebulk = 0.97274;
            return Kebulk; 
        } else if (prob_parm.mater_electrolyte==1) {
            constexpr amrex::Real a4 = -3.334591539362114e-14;
            constexpr amrex::Real a3 = 3.867473118129116e-10;
            constexpr amrex::Real a2 = -1.559193894834466e-06;
            constexpr amrex::Real a1 = 0.002182354774294;
            constexpr amrex::Real a0 = 0.0; // MUST BE ZERO FOR MODIFED EXPRESSION OF DIFFUSIONAL CONDUCTIVITY TO BE CORRECT
            return amrex::max((((a4 * Ce + a3) * Ce + a2) * Ce + a1) * Ce + a0, myeps);
        } else {
            amrex::Abort("Incorrect user-parameter mater_electrolyte in ChemistryProbParm.H");
            return 0; 
        }
    }

    // Electrolyte conductivity [S.m-1]. Actual value to be used in transport equations
    // Ce in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real KeC(amrex::Real Ce, const ProbParm& prob_parm)
    {
        amrex::Real Xeff_Xbulk = XeffXbulk(prob_parm);
        amrex::Real Xbulk = Ke_Bulk(Ce, prob_parm);
        amrex::Real Xeff = Xbulk*Xeff_Xbulk;
        return Xeff;
    }

    // Modified electrolyte conductivity (Ke bulk / Ce) for modified expression of the diffusional conductivity
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real Ke_Bulk_star(amrex::Real Ce, const ProbParm& prob_parm)
    {
        // Coefficients must be identical with Electrolyte conductivity Kebulk
        constexpr amrex::Real a4 = -3.334591539362114e-14;
        constexpr amrex::Real a3 = 3.867473118129116e-10;
        constexpr amrex::Real a2 = -1.559193894834466e-06;
        constexpr amrex::Real a1 = 0.002182354774294;
        constexpr amrex::Real a0 = 0.0; // MUST BE ZERO FOR MODIFED EXPRESSION OF DIFFUSIONAL CONDUCTIVITY TO BE CORRECT
        return amrex::max((((a4 * Ce + a3) * Ce + a2) * Ce + a1), myeps);
    }

    // Modified electrolyte conductivity (Ke / Ce) for modified expression of the diffusional conductivity
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real KeC_star(amrex::Real Ce, const ProbParm& prob_parm)
    {
        amrex::Real Xeff_Xbulk = XeffXbulk(prob_parm);
        amrex::Real Xbulk = Ke_Bulk_star(Ce, prob_parm);
        amrex::Real Xeff = Xbulk*Xeff_Xbulk;
        return Xeff;
    }

    // Electrolyte diffusion coefficient [m2.s-1]. Bulk value, not be used directly in transport equations
    // Ce in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real De_Bulk(amrex::Real Ce, const ProbParm& prob_parm)
    {

        if(prob_parm.mater_electrolyte==0) {
            amrex::Real Debulk = 1.34866e-10; 
            return Debulk;
        } else if (prob_parm.mater_electrolyte==1) {
            constexpr amrex::Real a4 = 7.414554029503555e-25;
            constexpr amrex::Real a3 = -1.159860784909586e-20;
            constexpr amrex::Real a2 = 7.492287453960106e-17;
            constexpr amrex::Real a1 = -2.451314899670142e-13;
            constexpr amrex::Real a0 = 3.396397365975209e-10;
            return amrex::max((((a4 * Ce + a3) * Ce + a2) * Ce + a1) * Ce + a0, myeps); // [m2.s-1]
        } else {
            amrex::Abort("Incorrect user-parameter mater_electrolyte in ChemistryProbParm.H");
            return 0; 
        }
    }

    // Electrolyte diffusion coefficient [m2.s-1]. Actual value to be used in transport equations
    // Ce in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real De_C(amrex::Real Ce, const ProbParm& prob_parm)
    {
        amrex::Real Xeff_Xbulk = XeffXbulk(prob_parm);
        amrex::Real Xbulk = De_Bulk(Ce, prob_parm);
        amrex::Real Xeff = Xbulk*Xeff_Xbulk;
        return Xeff;
    }

    // Transference number [-]
    // Ce in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real t_plus(amrex::Real Ce, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_electrolyte==0) {
            amrex::Real tplus = 0.462925;
            return tplus;
        } else if (prob_parm.mater_electrolyte==1) {
            constexpr amrex::Real a2 = -2.266955710225041e-09;
            constexpr amrex::Real a1 = 2.125258249150030e-05;
            constexpr amrex::Real a0 = 0.440686353438825;
            return amrex::max( (a2 * Ce + a1) * Ce + a0, myeps);
        } else {
            amrex::Abort("Incorrect user-parameter mater_electrolyte in ChemistryProbParm.H");
            return 0; 
        }
    }

    // Electrolyte activity [-]
    // Ce in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real Activity_el(amrex::Real Ce, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_electrolyte==0) {
            amrex::Real Activityel = 2.86877;
            return Activityel;
        } else if (prob_parm.mater_electrolyte==1) {
            // Valid for temperature = 30 + 273.15 K
            constexpr amrex::Real a2 = 1.598531153290851e-06;
            constexpr amrex::Real a1 = -1.997682922513610e-04;
            constexpr amrex::Real a0 = 0.806611708428853;
            return amrex::max((a2 * Ce + a1) * Ce + a0, myeps);
        } else {
            amrex::Abort("Incorrect user-parameter mater_electrolyte in ChemistryProbParm.H");
            return 0; 
        }
    }

    // Electrolyte diffusional conductivity [C.m-1.s-1]
    // Function of transference number, activity coefficient, conductivity, and temperature
    // Kd is negative, unlike the other electrolyte coefficients.
    // - Moddifed expression of the diffusional conductivity Kd star (J. Allen et al., https://doi.org/10.1007/s10915-021-01410-5, eq. 36)
    //   Kd star is only relevant when Ce tends to 0 (i.e., for thick electrodes and/or high applied current).
    //   For other cases, the baseline expression works. (to test/debug the model, Kd start is not required, except if testing is for fast charge and/or thick electrodes)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real K_D(amrex::Real Ce, const ProbParm& prob_parm)
    {
        // FIXME: hard coding this for now
        return -0.1;

//        const amrex::Real RTF = prob_parm.R_gas_const*prob_parm.T0/prob_parm.Faraday_const;
//        if(prob_parm.use_KDstar)  { 
//            return amrex::min(-(1.0 - t_plus(Ce, prob_parm)) * (1.0 + Activity_el(Ce, prob_parm)) * 2.0 * KeC_star(Ce, prob_parm) * RTF, -myeps);
//        } else {
//            return amrex::min(-(1.0 - t_plus(Ce, prob_parm)) * (1.0 + Activity_el(Ce, prob_parm)) * 2.0 * KeC(Ce, prob_parm) * RTF, -myeps);
//        }
    }

    // ANODE MATERIAL COEFFICIENTS

    // Anode maximum concentration [mol.m-3]
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real Cs_max_anode(const ProbParm& prob_parm)
    {
        if(prob_parm.mater_anode==0) {
            const amrex::Real csmax_anode = 28000.0;
            return csmax_anode; 
        } else if (prob_parm.mater_anode==1) {
            const amrex::Real csmax_anode = 28000.0;
            return csmax_anode; 
        } else {
            amrex::Abort("Incorrect user-parameter mater_anode in ChemistryProbParm.H");
            return 0; 
        }
    }

    // Anode conductivity [S.m-1]
    // Cs in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real Ka_bulk(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_anode==0) {
            amrex::Real Kabulk = 100;
            return Kabulk; 
        } else if (prob_parm.mater_anode==1) {
            amrex::Real Kabulk = 100;
            return Kabulk; 
        } else {
            amrex::Abort("Incorrect user-parameter mater_anode in ChemistryProbParm.H");
            return 0; 
        }
    }

    // Anode diffusion coefficient [m2.s-1]
    // Cs in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real Da_bulk(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_anode==0) {
            amrex::Real Dabulk = 3e-14;
            return Dabulk; 
        } else if (prob_parm.mater_anode==1) {
            amrex::Real Dabulk = 3e-14;
            return Dabulk; 
        } else {
            amrex::Abort("Incorrect user-parameter mater_anode in ChemistryProbParm.H");
            return 0; 
        }
    }

    // Anode exchange current density [A.m-2]
    // Cs in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real Io_a(amrex::Real Cs, amrex::Real Ce, const ProbParm& prob_parm)
    {
        amrex::Real x = Cs/Cs_max_anode(prob_parm);
        if(prob_parm.mater_anode==0) {
            amrex::Real ioa = 11 * pow(Ce/1000,0.5) * pow(x,0.5) * pow(1-x,0.5);
            return ioa; 
        } else if (prob_parm.mater_anode==1) {
            amrex::Real ioa = 11 * pow(Ce/1000,0.5) * pow(x,0.5) * pow(1-x,0.5);
            return ioa; 
        } else {
            amrex::Abort("Incorrect user-parameter mater_anode in ChemistryProbParm.H");
            return 0; 
        }
    }

    // Anode Open circuit voltage [V]
    // Cs in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real OCP_a(amrex::Real Cs, const ProbParm& prob_parm)
    {
        amrex::Real x = Cs/Cs_max_anode(prob_parm);
        if(prob_parm.mater_anode==0) {
            constexpr amrex::Real a7 = -201.8048;
            constexpr amrex::Real a6 = 730.918;
            constexpr amrex::Real a5 = -1074.645;
            constexpr amrex::Real a4 = 825.0173;
            constexpr amrex::Real a3 = -354.3316;
            constexpr amrex::Real a2 = 84.66978;
            constexpr amrex::Real a1 = -10.57909;
            constexpr amrex::Real a0 = 0.6848288;
            amrex::Real OCPa = ((((((a7*x+a6)*x+a5)*x+a4)*x+a3)*x+a2)*x+a1)*x+a0;
            return OCPa;
        } else if (prob_parm.mater_anode==1) {
            if (x>=0.98)
                {
                    constexpr amrex::Real a = (1e-4 - 0.014763)/(1.1-0.98);
                    constexpr amrex::Real b = 0.014763-a*0.98;
                    amrex::Real OCPa = a*x+b;
                    return OCPa; 
                }
            else
                {
                    constexpr amrex::Real a1 = -1.059423355572770E-02;
                    constexpr amrex::Real a2 = 2.443615203087110E-02;
                    constexpr amrex::Real a3 = -1.637520788053810E-02;
                    constexpr amrex::Real a4 = -6.542365622896410E-02;
                    constexpr amrex::Real a5 = -4.173226059293490E-02;
                    constexpr amrex::Real a6 = - 4.792178163846890E-01;
                    constexpr amrex::Real a7 = - 4.364293924074990E-02;
                    constexpr amrex::Real a8 = - 8.241166396760410E-02;

                    constexpr amrex::Real b1 = 1.453708425609560E-02;
                    constexpr amrex::Real b2 = 5.464261369950400E-01;
                    constexpr amrex::Real b3 = 5.639025014475490E-01;
                    constexpr amrex::Real b4 = 5.960370524233590E-01;
                    constexpr amrex::Real b5 = 1.787670587868640E-01;
                    constexpr amrex::Real b6 = - 3.845707852011820E-03;
                    constexpr amrex::Real b7 = 9.449231893318330E-02;
                    constexpr amrex::Real b8 = 7.746685789572230E-02;

                    constexpr amrex::Real c1 = 9.089868397988610E-05;
                    constexpr amrex::Real c2 = 6.270508166379020E-01;
                    constexpr amrex::Real c3 = 7.053886409518520E-02;
                    constexpr amrex::Real c4 = 1.409966536648620E+00;
                    constexpr amrex::Real c5 = 7.693844911793470E-02;
                    constexpr amrex::Real c6 = 4.112633446959460E-02;
                    constexpr amrex::Real c7 = -2.046776012570780E-02;
                    constexpr amrex::Real c8 = 3.593817905677970E-02;

                    amrex::Real U1 = a1*tanh((x - b1)/c1) + a2*tanh((x - b2)/c2) + a3*tanh((x - b3)/c3) + a4*tanh((x - b4)/c4) + a5*tanh((x - b5)/c5) + a6*tanh((x - b6)/c6) + a7*tanh((x - b7)/c7) + a8*tanh((x - b8)/c8) + 6.594735004847470e-1;

                    constexpr amrex::Real d0 = - 5.037944982759270E+01;
                    constexpr amrex::Real d1 = - 1.228217254296760E+01;
                    constexpr amrex::Real d2 = - 6.906367679257650E+01;
                    constexpr amrex::Real d3 = + 3.437968012320620E+00;
                    constexpr amrex::Real d4 = + 3.322960033709470E+01;
                    constexpr amrex::Real d5 = + 5.913206621637760E+01;
                    constexpr amrex::Real d6 = 1.233160814852810E+02;
                    constexpr amrex::Real d7 = + 8.252008712749000E+01;
                    constexpr amrex::Real d8 = -1.731504647676420E+02;

                    amrex::Real U2 = (((((((d8*x+d7)*x+d6)*x+d5)*x+d4)*x+d3)*x+d2)*x+d1)*x+d0;

                    amrex::Real OCPa = U1 + (U2 - U1) / (1.0 + exp(-1.0e2*(x - 1.02956203215198)));
                    return OCPa; 
                }
        } else {
            amrex::Abort("Incorrect user-parameter mater_anode in ChemistryProbParm.H");
            return 0; 
        }
    }


    // CATHODE MATERIAL COEFFICIENTS

    // Cathode maximum concentration [mol.m-3]
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real Cs_max_cathode(const ProbParm& prob_parm)
    {

        if(prob_parm.mater_cathode==0) {
            const amrex::Real csmax_cathode = 49600.0;
            return csmax_cathode; 
        } else if (prob_parm.mater_cathode==1) {
            const amrex::Real csmax_cathode = 49600.0;
            return csmax_cathode; 
        } else {
            amrex::Abort("Incorrect user-parameter mater_cathode in ChemistryProbParm.H");
            return 0; 
        }
    }

    // Cathode conductivity [S.m-1]
    // Cs in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real Kc_bulk(amrex::Real Cs, const ProbParm& prob_parm)
    {

        if(prob_parm.mater_cathode==0) {
            amrex::Real Kcbulk = 10;
            return Kcbulk; 
        } else if (prob_parm.mater_cathode==1) {
            amrex::Real Kcbulk = 10;
            return Kcbulk; 
        } else {
            amrex::Abort("Incorrect user-parameter mater_cathode in ChemistryProbParm.H");
            return 0; 
        }
    }

    // Cathode diffusion coefficient [m2.s-1]
    // Cs in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real Dc_bulk(amrex::Real Cs, const ProbParm& prob_parm)
    {
        amrex::Real x = Cs/Cs_max_cathode(prob_parm);
        if(prob_parm.mater_cathode==0) {
            amrex::Real Dcbulk = 8.39694e-15;
            return Dcbulk; 
        } else if (prob_parm.mater_cathode==1) {
            constexpr amrex::Real tolerance = 1e-9;
            constexpr amrex::Real x_threshold_1 = -0.10000000000000000555;
            constexpr amrex::Real x_threshold_2 = 0.00000000000000000000;
            constexpr amrex::Real x_threshold_3 = 0.10000000000000000555;
            constexpr amrex::Real x_threshold_4 = 0.40000000000000002220;
            constexpr amrex::Real x_threshold_5 = 0.59999999999999997780;
            constexpr amrex::Real x_threshold_6 = 0.80000000000000004441;

            constexpr amrex::Real a1_3 = 0.00000000000000000000;
            constexpr amrex::Real a1_2 = 0.00000000000000000000;
            constexpr amrex::Real a1_1 = 0.00000000000000000000;
            constexpr amrex::Real a1_0 = 0.00000000000000205222;
            amrex::Real u1 = ((a1_3 *(x-x_threshold_1) +a1_2)*(x-x_threshold_1)+a1_1)*(x-x_threshold_1)+a1_0;

            constexpr amrex::Real a2_3 = 0.00000000000000000000;
            constexpr amrex::Real a2_2 = 0.00000000000000000000;
            constexpr amrex::Real a2_1 = 0.00000000000000000000;
            constexpr amrex::Real a2_0 = 0.00000000000000205222;
            amrex::Real u2 = ((a2_3 *(x-x_threshold_2) +a2_2)*(x-x_threshold_2)+a2_1)*(x-x_threshold_2)+a2_0;

            constexpr amrex::Real a3_3 = -0.00000000000000001127;
            constexpr amrex::Real a3_2 = -0.00000000000000007010;
            constexpr amrex::Real a3_1 = 0.00000000000000000000;
            constexpr amrex::Real a3_0 = 0.00000000000000205222;
            amrex::Real u3 = ((a3_3 *(x-x_threshold_3) +a3_2)*(x-x_threshold_3)+a3_1)*(x-x_threshold_3)+a3_0;

            constexpr amrex::Real a4_3 = 0.00000000000000146457;
            constexpr amrex::Real a4_2 = -0.00000000000000272898;
            constexpr amrex::Real a4_1 = -0.00000000000000004510;
            constexpr amrex::Real a4_0 = 0.00000000000000204560;
            amrex::Real u4 = ((a4_3 *(x-x_threshold_4) +a4_2)*(x-x_threshold_4)+a4_1)*(x-x_threshold_4)+a4_0;

            constexpr amrex::Real a5_3 = 0.00000000000020947640;
            constexpr amrex::Real a5_2 = -0.00000000000006175755;
            constexpr amrex::Real a5_1 = -0.00000000000000096095;
            constexpr amrex::Real a5_0 = 0.00000000000000193914;
            amrex::Real u5 = ((a5_3 *(x-x_threshold_5) +a5_2)*(x-x_threshold_5)+a5_1)*(x-x_threshold_5)+a5_0;

            constexpr amrex::Real a6_3 = -0.00000000000000006031;
            constexpr amrex::Real a6_2 = 0.00000000000000090514;
            constexpr amrex::Real a6_1 = -0.00000000000000052680;
            constexpr amrex::Real a6_0 = 0.00000000000000095246;
            amrex::Real u6 = ((a6_3 *(x-x_threshold_6) +a6_2)*(x-x_threshold_6)+a6_1)*(x-x_threshold_6)+a6_0;

            amrex::Real sign1a = signbit(-( x_threshold_2-x-tolerance));
            amrex::Real sign1b = signbit(-(x-x_threshold_2+tolerance));
            amrex::Real sign2a = signbit(-(x_threshold_3-x-tolerance));
            amrex::Real sign2b = signbit(-(x-x_threshold_3+tolerance));
            amrex::Real sign3a = signbit(-(x_threshold_4-x-tolerance));
            amrex::Real sign3b = signbit(-(x-x_threshold_4+tolerance));
            amrex::Real sign4a = signbit(-(x_threshold_5-x-tolerance));
            amrex::Real sign4b = signbit(-(x-x_threshold_5+tolerance));
            amrex::Real sign5a = signbit(-(x_threshold_6-x-tolerance));
            amrex::Real sign5b = signbit(-(x-x_threshold_6+tolerance));

            //amrex::Real validity_1 = (sign1a + amrex::abs(sign1a))/2;
            //amrex::Real validity_2 = ((sign1b + amrex::abs(sign1b))/2 + (sign2a + amrex::abs(sign2a))/2)-1;
            //amrex::Real validity_3 = ((sign2b + amrex::abs(sign2b))/2 + (sign3a + amrex::abs(sign3a))/2)-1;
            //amrex::Real validity_4 = ((sign3b + amrex::abs(sign3b))/2 + (sign4a + amrex::abs(sign4a))/2)-1;
            //amrex::Real validity_5 = ((sign4b + amrex::abs(sign4b))/2 + (sign5a + amrex::abs(sign5a))/2)-1;
            //amrex::Real validity_6 = (sign5b + amrex::abs(sign5b))/2;

            amrex::Real validity_1 = (sign1a + std::abs(sign1a))/2;
            amrex::Real validity_2 = ((sign1b + std::abs(sign1b))/2 + (sign2a + std::abs(sign2a))/2)-1;
            amrex::Real validity_3 = ((sign2b + std::abs(sign2b))/2 + (sign3a + std::abs(sign3a))/2)-1;
            amrex::Real validity_4 = ((sign3b + std::abs(sign3b))/2 + (sign4a + std::abs(sign4a))/2)-1;
            amrex::Real validity_5 = ((sign4b + std::abs(sign4b))/2 + (sign5a + std::abs(sign5a))/2)-1;
            amrex::Real validity_6 = (sign5b + std::abs(sign5b))/2;

            amrex::Real D0 = validity_1*u1 + validity_2*u2 + validity_3*u3 + validity_4*u4 + validity_5*u5 + validity_6*u6;

            // Radius correction
            constexpr amrex::Real radius_fit = 1.8e-6; // m
            constexpr amrex::Real radius_cathodeparticle = 5e-6; // m
            amrex::Real Dcbulk = D0*1.5/sqrt( pow(radius_fit, 2) / pow(radius_cathodeparticle, 2) );
            return Dcbulk; 
        } else {
            amrex::Abort("Incorrect user-parameter mater_cathode in ChemistryProbParm.H");
            return 0; 
        }
    }

    // Cathode exchange current density [A.m-2]
    // Cs in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real Io_c(amrex::Real Cs, amrex::Real Ce, const ProbParm& prob_parm)
    {
        amrex::Real x = Cs/Cs_max_cathode(prob_parm);
        if(prob_parm.mater_cathode==0) {
            amrex::Real ioc = 6 * pow(Ce/1000,0.5) * pow(x,0.5) * pow(1-x,0.5);
            return ioc; 
        } else if (prob_parm.mater_cathode==1) {
            constexpr amrex::Real a0 = 0.303490440978371;
            constexpr amrex::Real a1 = 1.271944700013477;
            constexpr amrex::Real a2 = 4.420894220185683e+02;
            constexpr amrex::Real a3 = -5.783762746199664e+03;
            constexpr amrex::Real a4 = 3.822682327855755e+04;
            constexpr amrex::Real a5 = -1.416477460103355e+05;
            constexpr amrex::Real a6 = 3.113802647858406e+05;
            constexpr amrex::Real a7 = -4.169011915077865e+05;
            constexpr amrex::Real a8 = 3.347705415406199e+05;
            constexpr amrex::Real a9 = -1.485221335897379e+05;
            constexpr amrex::Real a10 = 2.803425068966447e+04;
            amrex::Real P = (((((((((a10*x+a9)*x+a8)*x+a7)*x+a6)*x+a5)*x+a4)*x+a3)*x+a2)*x+a1)*x+a0;
            amrex::Real ioc = P * sqrt(Ce/prob_parm.ce_atrest);
            return ioc; 
        } else {
            amrex::Abort("Incorrect user-parameter mater_cathode in ChemistryProbParm.H");
            return 0; 
        }
    }

    // Cathode Open circuit voltage [V]
    // Cs in mol.m-3
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real OCP_c(amrex::Real Cs, const ProbParm& prob_parm)
    {
        amrex::Real x = Cs/Cs_max_cathode(prob_parm);
        if(prob_parm.mater_cathode==0) {
            constexpr amrex::Real a3 = -1.3711668;
            constexpr amrex::Real a2 = 3.9033218;
            constexpr amrex::Real a1 = -4.3438781;
            constexpr amrex::Real a0 = 5.3701447;
            amrex::Real OCPc = ((a3*x+a2)*x+a1)*x+a0;
            return OCPc;
        } else if (prob_parm.mater_cathode==1) {
            constexpr amrex::Real a14=-3.640118e3;
            constexpr amrex::Real a13=1.317658e4;
            constexpr amrex::Real a12=- 1.455742e4;
            constexpr amrex::Real a11=- 1.571094e3;
            constexpr amrex::Real a10=1.265631e4;
            constexpr amrex::Real a9=- 2.057809e3;
            constexpr amrex::Real a8=- 1.074374e4;
            constexpr amrex::Real a7=8.698113e3;
            constexpr amrex::Real a6=- 8.297905e2;
            constexpr amrex::Real a5=- 2.073765e3;
            constexpr amrex::Real a4=1.190223e3;
            constexpr amrex::Real a3=- 2.724852e2;
            constexpr amrex::Real a2=2.723409e1;
            constexpr amrex::Real a1=- 4.158277;
            constexpr amrex::Real a0=5.314736;
            amrex::Real A = (((((((((((((a14*x+a13)*x+a12)*x+a11)*x+a10)*x+a9)*x+a8)*x+a7)*x+a6)*x+a5)*x+a4)*x+a3)*x+a2)*x+a1)*x+a0;
            amrex::Real OCPc = A - 5.573191e-4 * exp( 6.560241 * pow(x,41.482093 ));
            return OCPc; 
        } else {
            amrex::Abort("Incorrect user-parameter mater_cathode in ChemistryProbParm.H");
            return 0; 
        }
    }


    // ------------------------------------------------------
    // ACTIVE MATERIAL - ELECTROLYTE ELECTROCHEMICAL REACTION
    // ------------------------------------------------------

/*
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real intercalation_function(amrex::Real Ca, amrex::Real Ce)
    {
        // FIXME: not sure what should be here yet
        // Fixed: this guy is the exchange current density io
        return std::sqrt(Ca*Ce);
    }
*/

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real intercalation_reaction_anode(amrex::Real Ca, amrex::Real phi_a, amrex::Real Ce, amrex::Real phi_e, const ProbParm& prob_parm)
    {
        const amrex::Real ioa = Io_a(Ca, Ce, prob_parm);
        const amrex::Real OCPa = OCP_a(Ca, prob_parm);
        const amrex::Real eta = phi_a - phi_e - OCPa;
        const amrex::Real FRT = prob_parm.Faraday_const/(prob_parm.R_gas_const * prob_parm.T0);
        bool use_sinh = false;
        if(use_sinh) {
            return 2.0 * ioa * std::sinh(0.5 * FRT * eta); //FIXME: gpu safe?
        } else {
            amrex::Real A = 0.5 * FRT * eta;
            return ioa * ( exp(A) - exp(-A) ); // equivalent expression
        }
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real intercalation_reaction_cathode(amrex::Real Cc, amrex::Real phi_c, amrex::Real Ce, amrex::Real phi_e, const ProbParm& prob_parm)
    {
        const amrex::Real ioc = Io_c(Cc, Ce, prob_parm);
        const amrex::Real OCPc = OCP_c(Cc, prob_parm);
        const amrex::Real eta = phi_c - phi_e - OCPc;
        const amrex::Real FRT = prob_parm.Faraday_const/(prob_parm.R_gas_const * prob_parm.T0);
        bool use_sinh = false;
        if(use_sinh) {
            return 2.0 * ioc * std::sinh(0.5 * FRT * eta); //FIXME: gpu safe?
        } else {
            amrex::Real A = 0.5 * FRT * eta;
            return ioc * ( exp(A) - exp(-A) ); // equivalent expression
        }
    }


    // ------------------
    // INITIAL POTENTIALS
    // ------------------
    // Consists in solving the set of equation for an equivalent 0D model (i.e., in each domain potential and concentration are uniform)

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real get_phie_initial_from_anode(amrex::Real j, amrex::Real phi_s, const ProbParm& prob_parm)
    {
        const amrex::Real Ca = prob_parm.soc_anode_t0 * Cs_max_anode(prob_parm);
        const amrex::Real io = Io_a(Ca, prob_parm.ce_atrest, prob_parm);
        const amrex::Real ocp = OCP_a(Ca, prob_parm);
        amrex::Real A = asinh(j/(2*io)); // asinh safe?
        amrex::Real eta = A*prob_parm.R_gas_const*prob_parm.T0/(0.5*prob_parm.Faraday_const);
        return phi_s-eta-ocp;
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real get_phic_initial_from_electrolyte(amrex::Real j, amrex::Real phi_e, const ProbParm& prob_parm)
    {
        const amrex::Real Cc = prob_parm.soc_cathode_t0 * Cs_max_cathode(prob_parm);
        const amrex::Real io = Io_c(Cc, prob_parm.ce_atrest, prob_parm);
        const amrex::Real ocp = OCP_c(Cc, prob_parm);
        amrex::Real A = asinh(j/(2*io)); // asinh safe?
        amrex::Real eta = A*prob_parm.R_gas_const*prob_parm.T0/(0.5*prob_parm.Faraday_const);
        return eta+phi_e+ocp;
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
        } else if(cl == electrolyte && cr == cathode){
            const amrex::Real Ce = phi(ileft,jleft,kleft,CO_ID);
            const amrex::Real Pe = phi(ileft,jleft,kleft,PO_ID);
            const amrex::Real Cc = phi(iright,jright,kright,CO_ID);
            const amrex::Real Pc = phi(iright,jright,kright,PO_ID);
            reaction = normal*intercalation_reaction_cathode(Cc, Pc, Ce, Pe, prob_parm);
        } else if(cl == electrolyte && cr == anode){
            const amrex::Real Ce = phi(ileft,jleft,kleft,CO_ID);
            const amrex::Real Pe = phi(ileft,jleft,kleft,PO_ID);
            const amrex::Real Ca = phi(iright,jright,kright,CO_ID);
            const amrex::Real Pa = phi(iright,jright,kright,PO_ID);
            reaction = -normal*intercalation_reaction_anode(Ca, Pa, Ce, Pe, prob_parm);
        } else if(cl == anode && cr == electrolyte){
            const amrex::Real Ca = phi(ileft,jleft,kleft,CO_ID);
            const amrex::Real Pa = phi(ileft,jleft,kleft,PO_ID);
            const amrex::Real Ce = phi(iright,jright,kright,CO_ID);
            const amrex::Real Pe = phi(iright,jright,kright,PO_ID);
            reaction = normal*intercalation_reaction_anode(Ca, Pa, Ce, Pe, prob_parm);
        }

        if(isnan(reaction)) amrex::Abort("NaN in ic reaction");
        if(isinf(reaction)) amrex::Abort("Infinity in ic reaction");

        // FIXME: added min and max to prevent nans... not sure if we need this
        if(reaction > 0.0)
            return amrex::min(reaction,0.1*std::abs(normal));
        else
            return amrex::max(reaction,-0.1*std::abs(normal));

//        return reaction;

    }


}
#endif
