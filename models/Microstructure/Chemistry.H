#ifndef _CHEMISTRY_H_
#define _CHEMISTRY_H_

#include<AMReX_REAL.H>
#include<AMReX.H>
#include<string>
#include<AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Box.H>
#include <ProbParm.H>
#include <echemAMR_constants.H>
#include <math.h>  // signbit, exp
#include <cmath>   // pow

#define NUM_SPECIES 12
#define CO_ID    0 //concentration
#define POTs_ID  1 // solid potential (Used only if CBD mixed domain (prob.CBD_transport = 1)
#define A_AM_ID  2 //anode active material
#define A_E_ID   3 // anode electrolyte
#define A_CBD_ID 4 // anode carbon black binder
#define S_ID     5 //separator
#define C_AM_ID  6 // cathode active material
#define C_E_ID   7 // cathode electrolyte 
#define C_CBD_ID 8 // cathode carbon black binder
#define NP_ID    9 //Nanoporosity
#define MAC_ID   10 //MacMullin_number
#define LS_ID    11 //levelset
#define EFX_ID   12
#define EFY_ID   13
#define EFZ_ID   14
#define POT_ID   15 // potential
#define DIS_U_ID 16 // displacement in the x
#define DIS_V_ID 17 // displacement in the y
#define DIS_W_ID 18 // displacement in the z
#define VON_M_ID 19 // storage for the von mises
#define Sigma11_ID 20 // storage for Sigma11 stress
#define Sigma22_ID 21 // storage for Sigma22 stress
#define Sigma33_ID 22 // storage for Sigma33 stress
#define Sigma12_ID 23 // storage for Sigma12 stress
#define Sigma13_ID 24 // storage for Sigma13 stress
#define Sigma23_ID 25 // storage for Sigma23 stress
#define NVAR NUM_SPECIES+14 

namespace electrochem
{
    extern amrex::Vector<std::string> specnames;
    void init();
    void close();
    int find_id(std::string specname);

    // ---------------------
    // MATERIAL COEFFICIENTS
    // ---------------------

    // Mater_* = 0: test expression
    //         = 1: constant (except for OCP and io: variable with a simple expression)
    //         = 2: variable, real (i.e., complicated) expression    

    // ANODE MATERIAL COEFFICIENTS

    // Anode maximum concentration [mol.m-3]
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Cs_max_anode(const ProbParm& prob_parm)
    {
        if(prob_parm.halfcell_case==1) {      
            const amrex::Real csmax_anode = 1e6; // Very high
            return csmax_anode;
        } else {               
            if(prob_parm.mater_anode==0) {
                const amrex::Real csmax_anode = 28000.0;
                return csmax_anode; 
            } else if (prob_parm.mater_anode==1) {
                const amrex::Real csmax_anode = 28000.0;
                return csmax_anode; 
            } else if (prob_parm.mater_anode==2) {
                const amrex::Real csmax_anode = 28000.0;
                return csmax_anode;           
            } else {
                amrex::Abort("Incorrect user-parameter mater_anode in ProbParm.H");
                return 0; 
            }
        }
    }

    // Anode diffusion coefficient [m2.s-1]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Ds_bulk_anode(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.zeroDmodel==1) {
            amrex::Real Dsbulkanode = 1.0e-10; // Very high
            //amrex::Real Dsbulkanode = 1.0e-11; // Very high
            return Dsbulkanode;
        } else if (prob_parm.halfcell_case==1) {    
            amrex::Real Dsbulkanode = 1.0e-13; // High
            return Dsbulkanode;
        } else {
            if(prob_parm.mater_anode==0) {
                amrex::Real Dsbulkanode = 1.0e-14;
                return Dsbulkanode; 
            } else if (prob_parm.mater_anode==1) {
                amrex::Real Dsbulkanode = 3.0e-14;
                return Dsbulkanode; 
            } else if (prob_parm.mater_anode==2) {
                amrex::Real Dsbulkanode = 3.0e-14;
                return Dsbulkanode;               
            } else {
                amrex::Abort("Incorrect user-parameter mater_anode in ProbParm.H");
                return 0; 
            }
        }
    }

    // Anode conductivity coefficient [S.m-1]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Ks_bulk_anode(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.zeroDmodel==1) {
            amrex::Real Ksbulkanode = 100; // Very high
            return Ksbulkanode;
        } else {          
            if(prob_parm.mater_anode==0) {
                amrex::Real Ksbulkanode = 10;
                return Ksbulkanode; 
            } else if (prob_parm.mater_anode==1) {
                amrex::Real Ksbulkanode = 100;
                return Ksbulkanode; 
            } else if (prob_parm.mater_anode==2) {
                amrex::Real Ksbulkanode = 100;
                return Ksbulkanode;             
            } else {
                amrex::Abort("Incorrect user-parameter mater_anode in ProbParm.H");
                return 0; 
            }
        }
    }

    // Anode exchange current density [A.m-2]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Io_a(amrex::Real Cs, amrex::Real Ce, const ProbParm& prob_parm)
    {
        amrex::Real x = Cs/Cs_max_anode(prob_parm);
        amrex::Real io_cutoff = 1e-2;
        if ((Ce>0) && (x>0) && (x<1)) {
            if (prob_parm.halfcell_case==1) {    
                amrex::Real ioa = 10.0; // High
                return ioa;
            } else {
                if (prob_parm.mater_anode==0) {
                    amrex::Real ioa = 1.0;
                    return ioa; 
                } else if (prob_parm.mater_anode==1) {  
                    amrex::Real ioa = 11 * pow(Ce/1000,0.5) * pow(x,0.5) * pow(1-x,0.5);
                    return ioa;    
                } else if (prob_parm.mater_anode==2) {  
                    amrex::Real ioa = 11 * pow(Ce/1000,0.5) * pow(x,0.5) * pow(1-x,0.5);
                    ioa=amrex::max(ioa,io_cutoff);
                    return ioa;    
                } else {
                    amrex::Abort("Incorrect user-parameter mater_anode in ProbParm.H");
                    return 0; 
                }
            }                                         
        } else {
            amrex::Real ioa = io_cutoff;
            return ioa;            
        }
    }       

    // Anode surface area correction
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Spcorr_anode(const ProbParm& prob_parm)
    {      
        const amrex::Real Spcorranode = prob_parm.Sp_corr_anode;
        return Spcorranode; 
    }   

    // Anode Open circuit voltage [V]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real OCP_a(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if (prob_parm.halfcell_case==1) {    
            amrex::Real OCPa = 0.0; // By definition
            return OCPa;
        } else {
            amrex::Real x = Cs/Cs_max_anode(prob_parm);
            if(prob_parm.mater_anode==0) {
                amrex::Real OCPa = 0.6;
                return OCPa;

            } else if (prob_parm.mater_anode==1) {
                
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
            } else if (prob_parm.mater_anode==2) {
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
            } else {
                amrex::Abort("Incorrect user-parameter mater_anode in ProbParm.H");
                return 0; 
            }
        }
    }


    // CATHODE MATERIAL COEFFICIENTS

    // Cathode maximum concentration [mol.m-3]
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Cs_max_cathode(const ProbParm& prob_parm)
    {
        if(prob_parm.halfcell_case==2) {      
            const amrex::Real csmax_cathode = 1e6; // Very high
            return csmax_cathode;
        } else {               
            if(prob_parm.mater_cathode==0) {
                const amrex::Real csmax_cathode = 49600.0;
                return csmax_cathode; 
            } else if (prob_parm.mater_cathode==1) {
                const amrex::Real csmax_cathode = 49600.0;
                return csmax_cathode; 
            } else if (prob_parm.mater_cathode==2) {
                const amrex::Real csmax_cathode = 49600.0;
                return csmax_cathode;                
            } else {
                amrex::Abort("Incorrect user-parameter mater_cathode in ProbParm.H");
                return 0; 
            }
        }
    }

    // Cathode diffusion coefficient [m2.s-1]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Ds_bulk_cathode(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.zeroDmodel==1) {
            amrex::Real Dsbulkcathode = 1.0e-10; // Very high
            //amrex::Real Dsbulkcathode = 1.0e-11; // Very high
            return Dsbulkcathode;
        } else if (prob_parm.halfcell_case==2) {    
                amrex::Real Dsbulkcathode = 1.0e-13; // High
                return Dsbulkcathode;
        } else {          
            if(prob_parm.mater_cathode==0) {
                amrex::Real Dsbulkcathode = 1.0e-12;
                return Dsbulkcathode; 
            } else if (prob_parm.mater_cathode==1) {
                amrex::Real Dsbulkcathode = 1e-15;
                return Dsbulkcathode; 
            } else if (prob_parm.mater_cathode==2) {

                amrex::Real x = Cs/Cs_max_cathode(prob_parm); // So i can copy directly code from FEniCS

                amrex::Real tolerance = 1e-9;

                amrex::Real x_threshold_1 = -0.10000000000000000555;
                amrex::Real x_threshold_2 = 0.00000000000000000000;
                amrex::Real x_threshold_3 = 0.10000000000000000555;
                amrex::Real x_threshold_4 = 0.40000000000000002220;
                amrex::Real x_threshold_5 = 0.59999999999999997780;
                amrex::Real x_threshold_6 = 0.80000000000000004441;
                amrex::Real a1_3 = 0.00000000000000000000;
                amrex::Real a1_2 = 0.00000000000000000000;
                amrex::Real a1_1 = 0.00000000000000000000;
                amrex::Real a1_0 = 0.00000000000000205222;
                amrex::Real u1 = ((a1_3 *(x-x_threshold_1) +a1_2)*(x-x_threshold_1)+a1_1)*(x-x_threshold_1)+a1_0;
                amrex::Real a2_3 = 0.00000000000000000000;
                amrex::Real a2_2 = 0.00000000000000000000;
                amrex::Real a2_1 = 0.00000000000000000000;
                amrex::Real a2_0 = 0.00000000000000205222;
                amrex::Real u2 = ((a2_3 *(x-x_threshold_2) +a2_2)*(x-x_threshold_2)+a2_1)*(x-x_threshold_2)+a2_0;
                amrex::Real a3_3 = -0.00000000000000001127;
                amrex::Real a3_2 = -0.00000000000000007010;
                amrex::Real a3_1 = 0.00000000000000000000;
                amrex::Real a3_0 = 0.00000000000000205222;
                amrex::Real u3 = ((a3_3 *(x-x_threshold_3) +a3_2)*(x-x_threshold_3)+a3_1)*(x-x_threshold_3)+a3_0;
                amrex::Real a4_3 = 0.00000000000000146457;
                amrex::Real a4_2 = -0.00000000000000272898;
                amrex::Real a4_1 = -0.00000000000000004510;
                amrex::Real a4_0 = 0.00000000000000204560;
                amrex::Real u4 = ((a4_3 *(x-x_threshold_4) +a4_2)*(x-x_threshold_4)+a4_1)*(x-x_threshold_4)+a4_0;
                amrex::Real a5_3 = 0.00000000000020947640;
                amrex::Real a5_2 = -0.00000000000006175755;
                amrex::Real a5_1 = -0.00000000000000096095;
                amrex::Real a5_0 = 0.00000000000000193914;
                amrex::Real u5 = ((a5_3 *(x-x_threshold_5) +a5_2)*(x-x_threshold_5)+a5_1)*(x-x_threshold_5)+a5_0;
                amrex::Real a6_3 = -0.00000000000000006031;
                amrex::Real a6_2 = 0.00000000000000090514;
                amrex::Real a6_1 = -0.00000000000000052680;
                amrex::Real a6_0 = 0.00000000000000095246;
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
                amrex::Real validity_1 = (sign1a + std::abs(sign1a))/2;
                amrex::Real validity_2 = ((sign1b + std::abs(sign1b))/2 + (sign2a + std::abs(sign2a))/2)-1;
                amrex::Real validity_3 = ((sign2b + std::abs(sign2b))/2 + (sign3a + std::abs(sign3a))/2)-1;
                amrex::Real validity_4 = ((sign3b + std::abs(sign3b))/2 + (sign4a + std::abs(sign4a))/2)-1;
                amrex::Real validity_5 = ((sign4b + std::abs(sign4b))/2 + (sign5a + std::abs(sign5a))/2)-1;
                amrex::Real validity_6 = (sign5b + std::abs(sign5b))/2;
                amrex::Real D = validity_1*u1 + validity_2*u2 + validity_3*u3 + validity_4*u4 + validity_5*u5 + validity_6*u6;

                amrex::Real Dsbulkcathode = D*1.5; // D*1.5; 

                return Dsbulkcathode; 
            } else {
                amrex::Abort("Incorrect user-parameter mater_cathode in ProbParm.H");
                return 0; 
            }
        }
    }

    // Cathode conductivity coefficient [S.m-1]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Ks_bulk_cathode(amrex::Real Cs, const ProbParm& prob_parm)
    {

        if(prob_parm.zeroDmodel==1) {
            amrex::Real Ksbulkcathode = 100;
            return Ksbulkcathode;
        } else {          
            if(prob_parm.mater_cathode==0) {
                amrex::Real Ksbulkcathode = 5.0;
                return Ksbulkcathode; 
            } else if (prob_parm.mater_cathode==1) {
                amrex::Real Ksbulkcathode = 10;
                return Ksbulkcathode; 
            } else if (prob_parm.mater_cathode==2) {
                amrex::Real Ksbulkcathode = 10;
                return Ksbulkcathode;              
            } else {
                amrex::Abort("Incorrect user-parameter mater_cathode in ProbParm.H");
                return 0; 
            }
        }
    }

    // Cathode exchange current density [A.m-2]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Io_c(amrex::Real Cs, amrex::Real Ce, const ProbParm& prob_parm)
    {
        amrex::Real x = Cs/Cs_max_cathode(prob_parm);
        amrex::Real io_cutoff = 1e-2;
        if ((Ce>0) && (x>0) && (x<1)) {
            if (prob_parm.halfcell_case==2) {    
                amrex::Real ioc = 10.0; // High
                return ioc;
            } else {
                if (prob_parm.mater_cathode==0) {
                    amrex::Real ioc = 3.0;
                    return ioc; 
                } else if (prob_parm.mater_cathode==1) {  
                    amrex::Real ioc = 7 * pow(Ce/1000,0.5) * pow(x,0.5) * pow(1-x,0.5);
                    ioc=amrex::max(ioc,io_cutoff);
                    return ioc;    
                } else if (prob_parm.mater_cathode==2) {  
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
                    ioc=amrex::max(ioc,io_cutoff);
                    return ioc;     
                } else {
                    amrex::Abort("Incorrect user-parameter mater_cathode in ProbParm.H");
                    return 0; 
                }
            }                                         
        } else {
            amrex::Real ioc = io_cutoff;
            return ioc;            
        }     
    }   

    // Cathode surface area correction
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Spcorr_cathode(const ProbParm& prob_parm)
    {      
        const amrex::Real Spcorrcathode = prob_parm.Sp_corr_cathode;
        return Spcorrcathode; 
    }     


    // Cathode Open circuit voltage [V]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real OCP_c(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if (prob_parm.halfcell_case==2) {    
            amrex::Real OCPc = 0.0; // By definition
            return OCPc;
        } else {
            amrex::Real x = Cs/Cs_max_cathode(prob_parm);
            if(prob_parm.mater_cathode==0) {        
                amrex::Real OCPc = 3.5;
                return OCPc;
            } else if (prob_parm.mater_cathode==1) {
                constexpr amrex::Real a3 = -1.3711668;
                constexpr amrex::Real a2 = 3.9033218;
                constexpr amrex::Real a1 = -4.3438781;
                constexpr amrex::Real a0 = 5.3701447;
                amrex::Real OCPc = ((a3*x+a2)*x+a1)*x+a0;
                return OCPc;                
            } else if (prob_parm.mater_cathode==2) {
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
                amrex::Abort("Incorrect user-parameter mater_cathode in ProbParm.H");
                return 0; 
            }
        }
    }

    // ELECTROLYTE MATERIAL COEFFICIENTS

    // Electrolyte dense diffusion coefficient [m2.s-1]
    // Ce in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real De_bulk(amrex::Real Ce, const ProbParm& prob_parm)
    {
        amrex::Real Debulk_cutoff = 1e-12;
        if(prob_parm.zeroDmodel==1) {
            amrex::Real Debulk = 1.0e-9; // Very high
            //amrex::Real Debulk = 1.0e-11; // Very high
            return Debulk;
        } else {          
            if(prob_parm.mater_electrolyte==0) {
                amrex::Real Debulk = 1.0e-10;
                return Debulk; 
            } else if (prob_parm.mater_electrolyte==1) {
                amrex::Real Debulk = 1.0e-10;
                return Debulk;           
            } else if (prob_parm.mater_electrolyte==2) {
                constexpr amrex::Real a0 = 3.409857e-10;
                constexpr amrex::Real a1 = -2.553199e-13;
                constexpr amrex::Real a2 = 9.229720e-17;
                constexpr amrex::Real a3 = -2.249695e-20;
                constexpr amrex::Real a4 = 3.510337e-24;
                constexpr amrex::Real a5 = -2.369022e-28;
                amrex::Real Debulk = ((((a5*Ce+a4)*Ce+a3)*Ce+a2)*Ce+a1)*Ce+a0;
                Debulk=amrex::max(Debulk,Debulk_cutoff);
                return Debulk; 
            } else {
                amrex::Abort("Incorrect user-parameter mater_electrolyte in ProbParm.H");
                return 0; 
            }
        }
    }

    // Electrolyte dense conductivity [S.m-1]
    // Ce in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Ke_bulk(amrex::Real Ce, const ProbParm& prob_parm)
    {
        amrex::Real Kebulk_cutoff = 0.01;
        if(prob_parm.zeroDmodel==1) {
            amrex::Real Kebulk = 1.0; // Very high
            return Kebulk;
        } else {          
            if(prob_parm.mater_electrolyte==0) {
                amrex::Real Kebulk = 1.0;
                return Kebulk; 
            } else if (prob_parm.mater_electrolyte==1) {
                amrex::Real Kebulk = 1.0;
                return Kebulk; 
            } else if (prob_parm.mater_electrolyte==2) {
                constexpr amrex::Real a0 = 0.0;
                constexpr amrex::Real a1 = 0.002184643837703;
                constexpr amrex::Real a2 = -1.577170931848559e-06;
                constexpr amrex::Real a3 = 4.120782790092705e-10;
                constexpr amrex::Real a4 = -4.443882945301638e-14;
                constexpr amrex::Real a5 = 1.558277114134419e-18;
                amrex::Real Kebulk = ((((a5*Ce+a4)*Ce+a3)*Ce+a2)*Ce+a1)*Ce+a0;
                Kebulk=amrex::max(Kebulk,Kebulk_cutoff);
                return Kebulk;                
            } else {
                amrex::Abort("Incorrect user-parameter mater_electrolyte in ProbParm.H");
                return 0; 
            }
        } 
    }

    // Electrolyte coupling
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Cphi_ele_coupling(const ProbParm& prob_parm)
    {
        amrex::Real Cphielectrolytecoupling = prob_parm.Cphi_electrolyte_coupling;
        return Cphielectrolytecoupling;         
    }    

    // Electrolyte transference number [.]
    // Ce in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real t_plus(amrex::Real Ce, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_electrolyte_tpAc==0) {
            amrex::Real tplus = 0.5;
            return tplus; 
        } else if (prob_parm.mater_electrolyte_tpAc==1) {
            amrex::Real tplus = 0.46;
            return tplus; 
        } else if (prob_parm.mater_electrolyte_tpAc==2) {
            constexpr amrex::Real a0 = 0.440683;
            constexpr amrex::Real a1 = 2.125497e-5;
            constexpr amrex::Real a2 = -2.2667919e-09;
            amrex::Real tplus = (a2*Ce+a1)*Ce+a0;
            return tplus;            

        } else {
            amrex::Abort("Incorrect user-parameter mater_electrolyte in ProbParm.H");
            return 0; 
        }
    }

    // Electrolyte activity coefficient [.]
    // Ce in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real A_c(amrex::Real Ce, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_electrolyte_tpAc==0) {
            amrex::Real Ac = 2.87;
            return Ac; 
        } else if (prob_parm.mater_electrolyte_tpAc==1) {
            amrex::Real Ac = 2.87;
            return Ac; 
        } else if (prob_parm.mater_electrolyte_tpAc==2) {
            constexpr amrex::Real a0 = 0.806612;
            constexpr amrex::Real a1 = -1.997683e-04;
            constexpr amrex::Real a2 = 1.5985312e-06;
            amrex::Real Ac = (a2*Ce+a1)*Ce+a0;
            return Ac;
        } else {
            amrex::Abort("Incorrect user-parameter mater_electrolyte in ProbParm.H");
            return 0; 
        }
    }

    // Electrolyte Diffusional conductivity [C.m-1.s-1]
    /*
    // Ce in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Kd(amrex::Real Ce, const ProbParm& prob_parm)
    {
        amrex::Real Kd = (2.0 * Ke_bulk(Ce,prob_parm) * GASCONST * prob_parm.Temperature/FARADCONST)*(1.0-t_plus(Ce,prob_parm))*(1+A_c(Ce,prob_parm)) ;
        return Kd; 
    }
    */ // Alternative 
    // Ke in S.m-1, Ac in [], tplus in []
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Kd(amrex::Real Ke, amrex::Real Ac, amrex::Real tplus, const ProbParm& prob_parm)
    {
        amrex::Real Kd = (2.0 * Ke * GASCONST * prob_parm.Temperature/FARADCONST)*(1.0-tplus)*(1+Ac) ; // Kd is positive
        return Kd; 
    }

    // Carbon-black binder (CBD) MATERIAL COEFFICIENTS

    // CBD diffusion coefficient [m2.s-1]
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Ds_bulk_CBD(const ProbParm& prob_parm)
    {
        if(prob_parm.zeroDmodel==1) {
            amrex::Real DsbulkCBD = 1.0e-16;
            return DsbulkCBD;
        } else {          
            if(prob_parm.mater_CBD==0) {
                amrex::Real DsbulkCBD = 1.0e-16;
                return DsbulkCBD; 
            } else if (prob_parm.mater_CBD==1) {
                amrex::Real DsbulkCBD = 3.0e-14;
                return DsbulkCBD; 
            } else {
                amrex::Abort("Incorrect user-parameter mater_CBD in ProbParm.H");
                return 0; 
            }
        }
    }

    // CBD conductivity coefficient [S.m-1]
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real Ks_bulk_CBD(const ProbParm& prob_parm)
    {
        if(prob_parm.zeroDmodel==1) {
            amrex::Real KsbulkCBD = 100;
            return KsbulkCBD;
        } else {          
            if(prob_parm.mater_CBD==0) {
                amrex::Real KsbulkCBD = 100;
                return KsbulkCBD; 
            } else if (prob_parm.mater_CBD==1) {
                amrex::Real KsbulkCBD = 100;
                return KsbulkCBD; 
            } else {
                amrex::Abort("Incorrect user-parameter mater_CBD in ProbParm.H");
                return 0; 
            }
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
        const amrex::Real Sp = Spcorr_anode(prob_parm);
        const amrex::Real ocp = OCP_a(Ca, prob_parm);
        amrex::Real A = asinh(j/(2*Sp*io));
        amrex::Real eta = A*GASCONST*prob_parm.Temperature/(0.5*FARADCONST);
        return phi_s-eta-ocp;
    }

    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real get_phic_initial_from_electrolyte(amrex::Real j, amrex::Real phi_e, const ProbParm& prob_parm)
    {
        const amrex::Real Cc = prob_parm.soc_cathode_t0 * Cs_max_cathode(prob_parm);
        const amrex::Real io = Io_c(Cc, prob_parm.ce_atrest, prob_parm);
        const amrex::Real Sp = Spcorr_cathode(prob_parm);
        const amrex::Real ocp = OCP_c(Cc, prob_parm);
        amrex::Real A = asinh(j/(2*Sp*io));
        amrex::Real eta = A*GASCONST*prob_parm.Temperature/(0.5*FARADCONST);
        return eta+phi_e+ocp;
    }

    // ------------------------------------------------------
    // ACTIVE MATERIAL - ELECTROLYTE ELECTROCHEMICAL REACTION
    // ------------------------------------------------------

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real intercalation_reaction_anode(amrex::Real Ca, amrex::Real Ce, amrex::Real Phi_jump, const ProbParm& prob_parm)
    {
        //Phi_jump = -Phi_jump; // In  source Phi_jump is noted phi e - phi s
        amrex::Real jumphi=-Phi_jump;
        amrex::Real limfactor=100.0;
        const amrex::Real ioa = Io_a(Ca, Ce, prob_parm);
        const amrex::Real Spa = Spcorr_anode(prob_parm);
        const amrex::Real OCPa = OCP_a(Ca, prob_parm);
        const amrex::Real eta = jumphi - OCPa; // Phi_jump = phi_a - phi_e
        const amrex::Real FRT = FARADCONST/(GASCONST * prob_parm.Temperature);
        bool use_sinh = true;
        amrex::Real jbv;
        if(use_sinh) {
            //amrex::Print() << "- Anode. ioa=" << ioa << " OCPa=" << OCPa << " Phi_jump=" << jumphi << " eta=" << eta << " FRT=" << FRT << " jbv=" << 2.0 * ioa * std::sinh(0.5 * FRT * eta) << std::endl;
            jbv= 2.0 * Spa * ioa * std::sinh(0.5 * FRT * eta);
        } else {
            amrex::Real A = 0.5 * FRT * eta;
            jbv = Spa * ioa * ( exp(A) - exp(-A) ); // equivalent expression
        }
        //limiter
        if(amrex::Math::abs(jbv)>amrex::Math::abs(limfactor*ioa))
        {
            jbv=limfactor*std::copysign(1.0,jbv)*amrex::Math::abs(ioa);
        }
        return(jbv);

    }

    // Linearization of jbv is used to solve the nonlinear potential equation, (~Newton method)
    // It is also needed at the interface where the conductivity term goes away. The linearization will add a new conductivity term there to stabilize the linear system
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real intercalation_reaction_anode_derivative(amrex::Real Ca, amrex::Real Ce, amrex::Real Phi_jump, const ProbParm& prob_parm)
    {
        //Phi_jump = -Phi_jump; // In  source Phi_jump is noted phi e - phi s
        amrex::Real limfactor=100.0;
        amrex::Real jumphi=-Phi_jump;
        const amrex::Real ioa = Io_a(Ca, Ce, prob_parm);
        const amrex::Real Spa = Spcorr_anode(prob_parm);
        const amrex::Real OCPa = OCP_a(Ca, prob_parm);
        const amrex::Real eta = jumphi - OCPa; // Phi_jump = phi_a - phi_e
        const amrex::Real FRT = FARADCONST/(GASCONST * prob_parm.Temperature);
        amrex::Real jbvder;
        jbvder=-2.0 * Spa * ioa * std::cosh(0.5 * FRT * eta) * 0.5*FRT; // chain rule of A*sinh(B*x) = A*cosh(Bx)*B
        if(amrex::Math::abs(jbvder)>amrex::Math::abs(limfactor*ioa*0.5*FRT))
        {
            jbvder=std::copysign(1.0,jbvder)*amrex::Math::abs(limfactor*ioa*0.5*FRT);
        }
        return(jbvder);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real intercalation_reaction_cathode(amrex::Real Cc, amrex::Real Ce, amrex::Real Phi_jump, const ProbParm& prob_parm)
    {
        //Phi_jump = -Phi_jump; // In  source Phi_jump is noted phi e - phi s
        amrex::Real limfactor=100.0;
        amrex::Real jumphi=-Phi_jump;
        const amrex::Real ioc = Io_c(Cc, Ce, prob_parm);
        const amrex::Real Spc = Spcorr_anode(prob_parm);
        const amrex::Real OCPc = OCP_c(Cc, prob_parm);
        const amrex::Real eta = jumphi - OCPc; // Phi_jump = phi_c - phi_e
        const amrex::Real FRT = FARADCONST/(GASCONST * prob_parm.Temperature);
        bool use_sinh = true;
        amrex::Real jbv;
        if(use_sinh) {
            //amrex::Print() << "- Cathode. ioc=" << ioc << " OCPc=" << OCPc << " Phi_jump=" << jumphi << " eta=" << eta << " FRT=" << FRT << " jbv=" << 2.0 * ioc * std::sinh(0.5 * FRT * eta) << std::endl;
            jbv = 2.0 * Spc * ioc * std::sinh(0.5 * FRT * eta);
        } else {
            amrex::Real A = 0.5 * FRT * eta;
            jbv = Spc * ioc * ( exp(A) - exp(-A) ); // equivalent expression
        }
        //limiter
        if(amrex::Math::abs(jbv)>amrex::Math::abs(limfactor*ioc))
        {
            jbv=limfactor*std::copysign(1.0,jbv)*amrex::Math::abs(ioc);
        }
        return(jbv);
    }

    AMREX_GPU_DEVICE AMREX_FORCE_INLINE
    amrex::Real intercalation_reaction_cathode_derivative(amrex::Real Ca, amrex::Real Ce, amrex::Real Phi_jump, const ProbParm& prob_parm)
    {
        //Phi_jump = -Phi_jump; // In  source Phi_jump is noted phi e - phi s
        amrex::Real limfactor=100.0;
        amrex::Real jumphi=-Phi_jump;
        const amrex::Real ioc = Io_c(Ca, Ce, prob_parm);
        const amrex::Real Spc = Spcorr_anode(prob_parm);
        const amrex::Real OCPc = OCP_c(Ca, prob_parm);
        const amrex::Real eta = jumphi - OCPc; // Phi_jump = phi_a - phi_e
        const amrex::Real FRT = FARADCONST/(GASCONST * prob_parm.Temperature);
        amrex::Real jbvder;
        jbvder=-2.0 * Spc * ioc * std::cosh(0.5 * FRT * eta) * 0.5*FRT; // chain rule of A*sinh(B*x) = A*cosh(Bx)*B
        if(amrex::Math::abs(jbvder)>amrex::Math::abs(limfactor*ioc*0.5*FRT))
        {
            jbvder=limfactor*std::copysign(1.0,jbvder)*amrex::Math::abs(ioc*0.5*FRT);
        }
        return(jbvder);
    }

    // ------------------------------------------------------
    //                 MECHANICAL PROPERTIES
    // ------------------------------------------------------

    // ANODE MATERIAL COEFFICIENTS

    // Anode Young's Modulus [Pa]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real E_anode(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_mech_anode==0) {
            amrex::Real out = 32.0e9;
            return out; 
        } else if (prob_parm.mater_mech_anode==1) {
            amrex::Real out = 32.0e9;
            return out; 
        } else if (prob_parm.mater_mech_anode==2) {
            amrex::Real out = 32.0e9;
            return out;              
        } else {
            amrex::Abort("Incorrect user-parameter mater_mech_anode in ProbParm.H");
            return 0; 
        }
    }

    // Anode Poisson Ratio [-]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real nu_anode(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_mech_anode==0) {
            amrex::Real out = 0.3;
            return out; 
        } else if (prob_parm.mater_mech_anode==1) {
            amrex::Real out = 0.3;
            return out; 
        } else if (prob_parm.mater_mech_anode==2) {
            amrex::Real out = 0.3;
            return out;              
        } else {
            amrex::Abort("Incorrect user-parameter mater_mech_anode in ProbParm.H");
            return 0; 
        }
    }

    // Anode Chemical Expansion Coeff [-]
    // Cs in mol.m-3
    // equation: https://orcid.org/0000-0002-3924-3769 : (volvar/100) / ((x_max-x_min)*csmax) with volvar=13.2%
    // graphite: https://doi.org/10.1021/acs.jpcc.8b01873 : graphite
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real CCE_anode(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_mech_anode==0) {
            amrex::Real out = 4.714e-6; // with volvar=13.2% and soc range = 1.0
            return out; 
        } else if (prob_parm.mater_mech_anode==1) {
            amrex::Real out = 4.714e-6; // with volvar=13.2% and soc range = 1.0
            return out; 
        } else if (prob_parm.mater_mech_anode==2) {
            amrex::Real out = 4.714e-6; // with volvar=13.2% and soc range = 1.0
            return out;              
        } else {
            amrex::Abort("Incorrect user-parameter mater_mech_anode in ProbParm.H");
            return 0; 
        }
    }




    // CATHODE MATERIAL COEFFICIENTS

    // Cathode Young's Modulus [Pa]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real E_cathode(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_mech_cathode==0) {
            amrex::Real out = 120.0e9; // Pa // 120.0e9 Peter document (cathode cracking literature review)
            return out; 
        } else if (prob_parm.mater_mech_cathode==1) {
            amrex::Real out = 120.0e9; // Pa // 120.0e9 Peter document (cathode cracking literature review)
            return out; 
        } else if (prob_parm.mater_mech_cathode==2) {
            amrex::Real out = 120.0e9; // Pa // 120.0e9 Peter document (cathode cracking literature review)
            return out;              
        } else {
            amrex::Abort("Incorrect user-parameter mater_mech_cathode in ProbParm.H");
            return 0; 
        }
    }

    // Cathode Poisson Ratio [-]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real nu_cathode(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_mech_cathode==0) {
            amrex::Real out = 0.3;
            return out; 
        } else if (prob_parm.mater_mech_cathode==1) {
            amrex::Real out = 0.3;
            return out; 
        } else if (prob_parm.mater_mech_cathode==2) {
            amrex::Real out = 0.3;
            return out;              
        } else {
            amrex::Abort("Incorrect user-parameter mater_mech_cathode in ProbParm.H");
            return 0; 
        }
    }

    // Cathode Chemical Expansion Coeff [-]
    // Cs in mol.m-3
    // equation: https://orcid.org/0000-0002-3924-3769 : (volvar/100) / ((x_max-x_min)*csmax) with volvar=13.2%
    // graphite: https://doi.org/10.1021/acs.jpcc.8b01873 : graphite
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real CCE_cathode(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_mech_cathode==0) {
            amrex::Real out = 1.344e-6; // //  1.344e-6 with volvar=4% and soc range = 0.6
            return out; 
        } else if (prob_parm.mater_mech_cathode==1) {
            amrex::Real out = 1.344e-6; // //  1.344e-6 with volvar=4% and soc range = 0.6
            return out; 
        } else if (prob_parm.mater_mech_cathode==2) {
            amrex::Real out = 1.344e-6; // //  1.344e-6 with volvar=4% and soc range = 0.6
            return out;              
        } else {
            amrex::Abort("Incorrect user-parameter mater_mech_cathode in ProbParm.H");
            return 0; 
        }
    }




    // ELECTROLYTE MATERIAL COEFFICIENTS

    // Electrolyte Young's Modulus [Pa]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real E_electrolyte(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_mech_electrolyte==0) {
            amrex::Real out = 10.0e3;
            return out; 
        } else if (prob_parm.mater_mech_electrolyte==1) {
            amrex::Real out = 10.0e3;
            return out; 
        } else if (prob_parm.mater_mech_electrolyte==2) {
            amrex::Real out = 10.0e3;
            return out;              
        } else {
            amrex::Abort("Incorrect user-parameter mater_mech_electrolyte in ProbParm.H");
            return 0; 
        }
    }

    // Electrolyte Poisson Ratio [-]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real nu_electrolyte(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_mech_electrolyte==0) {
            amrex::Real out = 0.3;
            return out; 
        } else if (prob_parm.mater_mech_electrolyte==1) {
            amrex::Real out = 0.3;
            return out; 
        } else if (prob_parm.mater_mech_electrolyte==2) {
            amrex::Real out = 0.3;
            return out;              
        } else {
            amrex::Abort("Incorrect user-parameter mater_mech_electrolyte in ProbParm.H");
            return 0; 
        }
    }

    // Electrolyte Chemical Expansion Coeff [-]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real CCE_electrolyte(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_mech_electrolyte==0) {
            amrex::Real out = 0.0; 
            return out; 
        } else if (prob_parm.mater_mech_electrolyte==1) {
            amrex::Real out = 0.0; 
            return out; 
        } else if (prob_parm.mater_mech_electrolyte==2) {
            amrex::Real out = 0.0; 
            return out;              
        } else {
            amrex::Abort("Incorrect user-parameter mater_mech_electrolyte in ProbParm.H");
            return 0; 
        }
    }







    // SEPARATOR MATERIAL COEFFICIENTS

    // Separator Young's Modulus [Pa]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real E_separator(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_mech_separator==0) {
            amrex::Real out = 500.0e6;
            return out; 
        } else if (prob_parm.mater_mech_separator==1) {
            amrex::Real out = 500.0e6;
            return out; 
        } else if (prob_parm.mater_mech_separator==2) {
            amrex::Real out = 500.0e6;
            return out;              
        } else {
            amrex::Abort("Incorrect user-parameter mater_mech_separator in ProbParm.H");
            return 0; 
        }
    }

    // Separator Poisson Ratio [-]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real nu_separator(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_mech_separator==0) {
            amrex::Real out = 0.3;
            return out; 
        } else if (prob_parm.mater_mech_separator==1) {
            amrex::Real out = 0.3;
            return out; 
        } else if (prob_parm.mater_mech_separator==2) {
            amrex::Real out = 0.3;
            return out;              
        } else {
            amrex::Abort("Incorrect user-parameter mater_mech_separator in ProbParm.H");
            return 0; 
        }
    }

    // Separator Chemical Expansion Coeff [-]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real CCE_separator(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_mech_separator==0) {
            amrex::Real out = 0.0; 
            return out; 
        } else if (prob_parm.mater_mech_separator==1) {
            amrex::Real out = 0.0; 
            return out; 
        } else if (prob_parm.mater_mech_separator==2) {
            amrex::Real out = 0.0; 
            return out;              
        } else {
            amrex::Abort("Incorrect user-parameter mater_mech_separator in ProbParm.H");
            return 0; 
        }
    }








    // CARBON BINDER MATERIAL COEFFICIENTS

    // CBD Young's Modulus [Pa]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real E_CBD(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_mech_CBD==0) {
            amrex::Real out = 10.0e3;
            return out; 
        } else if (prob_parm.mater_mech_CBD==1) {
            amrex::Real out = 10.0e3;
            return out; 
        } else if (prob_parm.mater_mech_CBD==2) {
            amrex::Real out = 10.0e3;
            return out;              
        } else {
            amrex::Abort("Incorrect user-parameter mater_mech_CBD in ProbParm.H");
            return 0; 
        }
    }

    // CBD Poisson Ratio [-]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real nu_CBD(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_mech_CBD==0) {
            amrex::Real out = 0.3;
            return out; 
        } else if (prob_parm.mater_mech_CBD==1) {
            amrex::Real out = 0.3;
            return out; 
        } else if (prob_parm.mater_mech_CBD==2) {
            amrex::Real out = 0.3;
            return out;              
        } else {
            amrex::Abort("Incorrect user-parameter mater_mech_CBD in ProbParm.H");
            return 0; 
        }
    }

    // CBD Chemical Expansion Coeff [-]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    amrex::Real CCE_CBD(amrex::Real Cs, const ProbParm& prob_parm)
    {
        if(prob_parm.mater_mech_CBD==0) {
            amrex::Real out = 0.0; 
            return out; 
        } else if (prob_parm.mater_mech_CBD==1) {
            amrex::Real out = 0.0; 
            return out; 
        } else if (prob_parm.mater_mech_CBD==2) {
            amrex::Real out = 0.0; 
            return out;              
        } else {
            amrex::Abort("Incorrect user-parameter mater_mech_CBD in ProbParm.H");
            return 0; 
        }
    }




















}
#endif
