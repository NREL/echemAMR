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

#define NUM_SPECIES 6
#define CO_ID   0 //conc
#define A_ID    1 //anode
#define C_ID    2 //cathode
#define E_ID    3 //electrolyte
#define S_ID    4 //separator
#define LS_ID   5
#define EFX_ID  6
#define EFY_ID  7
#define EFZ_ID  8 
#define POT_ID  9
#define NVAR NUM_SPECIES+4 


namespace electrochem
{
    extern amrex::Vector<std::string> specnames;
    void init();
    void close();
    int find_id(std::string specname);

    // ---------------------
    // MATERIAL COEFFICIENTS
    // ---------------------

    // ANODE MATERIAL COEFFICIENTS

    // Anode maximum concentration [mol.m-3]
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
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

    // Anode exchange current density [A.m-2]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
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
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
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
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
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

    // Cathode exchange current density [A.m-2]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
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
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
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


    // ------------------
    // INITIAL POTENTIALS
    // ------------------
    // Consists in solving the set of equation for an equivalent 0D model (i.e., in each domain potential and concentration are uniform)

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real get_phie_initial_from_anode(amrex::Real j, amrex::Real phi_s, const ProbParm& prob_parm)
    {
        const amrex::Real Ca = prob_parm.soc_anode_t0 * Cs_max_anode(prob_parm);
        const amrex::Real io = Io_a(Ca, prob_parm.ce_atrest, prob_parm);
        const amrex::Real ocp = OCP_a(Ca, prob_parm);
        amrex::Real A = asinh(j/(2*io));
        amrex::Real eta = A*prob_parm.R_gas_const*prob_parm.Temperature/(0.5*prob_parm.Faraday_const);
        return phi_s-eta-ocp;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real get_phic_initial_from_electrolyte(amrex::Real j, amrex::Real phi_e, const ProbParm& prob_parm)
    {
        const amrex::Real Cc = prob_parm.soc_cathode_t0 * Cs_max_cathode(prob_parm);
        const amrex::Real io = Io_c(Cc, prob_parm.ce_atrest, prob_parm);
        const amrex::Real ocp = OCP_c(Cc, prob_parm);
        amrex::Real A = asinh(j/(2*io));
        amrex::Real eta = A*prob_parm.R_gas_const*prob_parm.Temperature/(0.5*prob_parm.Faraday_const);
        return eta+phi_e+ocp;
    }



    // ------------------------------------------------------
    // ACTIVE MATERIAL - ELECTROLYTE ELECTROCHEMICAL REACTION
    // ------------------------------------------------------

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real intercalation_reaction_anode(amrex::Real Ca, amrex::Real Ce, amrex::Real Phi_jump, const ProbParm& prob_parm)
    {
        const amrex::Real ioa = Io_a(Ca, Ce, prob_parm);
        const amrex::Real OCPa = OCP_a(Ca, prob_parm);
        const amrex::Real eta = Phi_jump - OCPa; // Phi_jump = phi_a - phi_e
        const amrex::Real FRT = prob_parm.Faraday_const/(prob_parm.R_gas_const * prob_parm.Temperature);
        bool use_sinh = false;
        if(use_sinh) {
            return 2.0 * ioa * std::sinh(0.5 * FRT * eta); //FIXME: gpu safe?
        } else {
            amrex::Real A = 0.5 * FRT * eta;
            return ioa * ( exp(A) - exp(-A) ); // equivalent expression
        }
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real intercalation_reaction_cathode(amrex::Real Cc, amrex::Real Ce, amrex::Real Phi_jump, const ProbParm& prob_parm)
    {
        const amrex::Real ioc = Io_c(Cc, Ce, prob_parm);
        const amrex::Real OCPc = OCP_c(Cc, prob_parm);
        const amrex::Real eta = Phi_jump - OCPc; // Phi_jump = phi_c - phi_e
        const amrex::Real FRT = prob_parm.Faraday_const/(prob_parm.R_gas_const * prob_parm.Temperature);
        bool use_sinh = false;
        if(use_sinh) {
            return 2.0 * ioc * std::sinh(0.5 * FRT * eta); //FIXME: gpu safe?
        } else {
            amrex::Real A = 0.5 * FRT * eta;
            return ioc * ( exp(A) - exp(-A) ); // equivalent expression
        }
    }

/*
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
            const amrex::Real Pc = phi(ileft,jleft,kleft,POT_ID);
            const amrex::Real Ce = phi(iright,jright,kright,CO_ID);
            const amrex::Real Pe = phi(iright,jright,kright,POT_ID);
            reaction = -normal*intercalation_reaction_cathode(Cc, Pc, Ce, Pe, prob_parm);
        } else if(cl == electrolyte && cr == cathode){
            const amrex::Real Ce = phi(ileft,jleft,kleft,CO_ID);
            const amrex::Real Pe = phi(ileft,jleft,kleft,POT_ID);
            const amrex::Real Cc = phi(iright,jright,kright,CO_ID);
            const amrex::Real Pc = phi(iright,jright,kright,POT_ID);
            reaction = normal*intercalation_reaction_cathode(Cc, Pc, Ce, Pe, prob_parm);
        } else if(cl == electrolyte && cr == anode){
            const amrex::Real Ce = phi(ileft,jleft,kleft,CO_ID);
            const amrex::Real Pe = phi(ileft,jleft,kleft,POT_ID);
            const amrex::Real Ca = phi(iright,jright,kright,CO_ID);
            const amrex::Real Pa = phi(iright,jright,kright,POT_ID);
            reaction = normal*intercalation_reaction_anode(Ca, Pa, Ce, Pe, prob_parm);
        } else if(cl == anode && cr == electrolyte){
            const amrex::Real Ca = phi(ileft,jleft,kleft,CO_ID);
            const amrex::Real Pa = phi(ileft,jleft,kleft,POT_ID);
            const amrex::Real Ce = phi(iright,jright,kright,CO_ID);
            const amrex::Real Pe = phi(iright,jright,kright,POT_ID);
            reaction = -normal*intercalation_reaction_anode(Ca, Pa, Ce, Pe, prob_parm);
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
*/

}
#endif
