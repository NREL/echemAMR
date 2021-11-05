#ifndef _REACTIONS_H_
#define _REACTIONS_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace electrochem_reactions {
    AMREX_GPU_DEVICE AMREX_INLINE void compute_react_source(
            int i,
            int j,
            int k,
            Array4<Real> const& phi,
            Array4<Real> const& reactsource,
            GpuArray<Real, AMREX_SPACEDIM> prob_lo,
            GpuArray<Real, AMREX_SPACEDIM> prob_hi,
            GpuArray<Real, AMREX_SPACEDIM> dx,
            const Real time,
            ProbParm const& prob_parm)
    {
        reactsource(i, j, k, CO_ID) = 0.0;
        reactsource(i, j, k, A_ID) = 0.0;
        reactsource(i, j, k, C_ID) = 0.0;
        reactsource(i, j, k, E_ID) = 0.0;
        reactsource(i, j, k, LS_ID) = 0.0;
    }

    AMREX_GPU_DEVICE AMREX_INLINE void compute_potential_source(
            int i,
            int j,
            int k,
            Array4<Real> const& phi,
            Array4<Real> const& source,
            GpuArray<Real, AMREX_SPACEDIM> prob_lo,
            GpuArray<Real, AMREX_SPACEDIM> prob_hi,
            GpuArray<Real, AMREX_SPACEDIM> dx,
            const Real time,
            ProbParm const& prob_parm)
    {
        source(i, j, k) = 0.0;
    }
    
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real Io_c(amrex::Real Cs, amrex::Real Ce, const ProbParm& prob_parm)
    {
        amrex::Real x = Cs/prob_parm.Csmax_c;

        amrex::Real a0 = 0.303490440978371;
        amrex::Real a1 = 1.271944700013477;
        amrex::Real a2 = 4.420894220185683e+02;
        amrex::Real a3 = -5.783762746199664e+03;
        amrex::Real a4 = 3.822682327855755e+04;
        amrex::Real a5 = -1.416477460103355e+05;
        amrex::Real a6 = 3.113802647858406e+05;
        amrex::Real a7 = -4.169011915077865e+05;
        amrex::Real a8 = 3.347705415406199e+05;
        amrex::Real a9 = -1.485221335897379e+05;
        amrex::Real a10 = 2.803425068966447e+04;
        amrex::Real Ce_atrest = 1200; // This should be an input. That's the initial electrolyte concentration [mol.m-3] C_e_init
        amrex::Real P = (((((((((a10*x+a9)*x+a8)*x+a7)*x+a6)*x+a5)*x+a4)*x+a3)*x+a2)*x+a1)*x+a0;
        amrex::Real ioc = P * pow(Ce/Ce_atrest,0.5);

        return ioc;
    }

    
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real OCP_c(amrex::Real Cs, const ProbParm& prob_parm)
    {
        amrex::Real x = Cs / prob_parm.Csmax_c;
        amrex::Real a14 = -3.640118e3;
        amrex::Real a13 = 1.317658e4;
        amrex::Real a12 = - 1.455742e4;
        amrex::Real a11 = - 1.571094e3;
        amrex::Real a10 = 1.265631e4;
        amrex::Real a9 = - 2.057809e3;
        amrex::Real a8 = - 1.074374e4;
        amrex::Real a7 = 8.698113e3;
        amrex::Real a6 = - 8.297905e2;
        amrex::Real a5 = - 2.073765e3;
        amrex::Real a4 = 1.190223e3;
        amrex::Real a3 = - 2.724852e2;
        amrex::Real a2 = 2.723409e1;
        amrex::Real a1 = - 4.158277;
        amrex::Real a0 = 5.314736;
        amrex::Real A = (((((((((((((a14*x+a13)*x+a12)*x+a11)*x+a10)*x+a9)*x+a8)*x+a7)*x+a6)*x+a5)*x+a4)*x+a3)*x+a2)*x+a1)*x+a0;
        amrex::Real OCPc = A - 5.573191e-4 * exp( 6.560241 * pow(x,41.482093));
        return OCPc;
    }
    
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real Io_a(amrex::Real Cs, amrex::Real Ce, const ProbParm& prob_parm)
    {
        amrex::Real x = Cs / prob_parm.Csmax_a;
        amrex::Real ioa = 11 * pow(Ce / 1000, 0.5) * pow(x, 0.5) * pow(1 - x, 0.5);
        return ioa;
    }
    
    // Anode Open circuit voltage [V]
    // Cs in mol.m-3
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real OCP_a(amrex::Real Cs, const ProbParm& prob_parm)
    {
        amrex::Real x = Cs / prob_parm.Csmax_a;

        amrex::Real a1 = -1.059423355572770E-02;
        amrex::Real a2 = 2.443615203087110E-02;
        amrex::Real a3 = -1.637520788053810E-02;
        amrex::Real a4 = -6.542365622896410E-02;
        amrex::Real a5 = -4.173226059293490E-02;
        amrex::Real a6 = - 4.792178163846890E-01;
        amrex::Real a7 = - 4.364293924074990E-02;
        amrex::Real a8 = - 8.241166396760410E-02;

        amrex::Real b1 = 1.453708425609560E-02;
        amrex::Real b2 = 5.464261369950400E-01;
        amrex::Real b3 = 5.639025014475490E-01;
        amrex::Real b4 = 5.960370524233590E-01;
        amrex::Real b5 = 1.787670587868640E-01;
        amrex::Real b6 = - 3.845707852011820E-03;
        amrex::Real b7 = 9.449231893318330E-02;
        amrex::Real b8 = 7.746685789572230E-02;

        amrex::Real c1 = 9.089868397988610E-05;
        amrex::Real c2 = 6.270508166379020E-01;
        amrex::Real c3 = 7.053886409518520E-02;
        amrex::Real c4 = 1.409966536648620E+00;
        amrex::Real c5 = 7.693844911793470E-02;
        amrex::Real c6 = 4.112633446959460E-02;
        amrex::Real c7 = -2.046776012570780E-02;
        amrex::Real c8 = 3.593817905677970E-02;

        amrex::Real U1 = a1*tanh((x - b1)/c1) + a2*tanh((x - b2)/c2) + a3*tanh((x - b3)/c3) + a4*tanh((x - b4)/c4) + a5*tanh((x - b5)/c5) + a6*tanh((x - b6)/c6) + a7*tanh((x - b7)/c7) + a8*tanh((x - b8)/c8) + 6.594735004847470e-1;

        amrex::Real d0 = - 5.037944982759270E+01;
        amrex::Real d1 = - 1.228217254296760E+01;
        amrex::Real d2 = - 6.906367679257650E+01;
        amrex::Real d3 = + 3.437968012320620E+00;
        amrex::Real d4 = + 3.322960033709470E+01;
        amrex::Real d5 = + 5.913206621637760E+01;
        amrex::Real d6 = 1.233160814852810E+02;
        amrex::Real d7 = + 8.252008712749000E+01;
        amrex::Real d8 = -1.731504647676420E+02;

        amrex::Real U2 = (((((((d8*x+d7)*x+d6)*x+d5)*x+d4)*x+d3)*x+d2)*x+d1)*x+d0;

        amrex::Real OCPa = U1 + (U2 - U1) / (1.0 + exp(-1.0e2*(x - 1.02956203215198)));

        return OCPa;
    }

    AMREX_GPU_DEVICE AMREX_INLINE void bvcurrent_and_der(int i,int j,int k,int normaldir,
            Real phi_jump, Array4<Real> const& phi, ProbParm const& prob_parm, Real &jbv, Real &jbvder)
    {
        Real ocp,j0,phi0;
        Real jcutoff = 0.01;
        Real cs,ce;
    
        int trans1dir = (normaldir + 1) % AMREX_SPACEDIM;
        int trans2dir = (normaldir + 2) % AMREX_SPACEDIM;

        IntVect left(i, j, k);
        IntVect right(i, j, k);
        left[normaldir] -= 1;

        Real anodefrac   = 0.5*(phi(left,A_ID) + phi(right,A_ID));
        Real cathodefrac = 0.5*(phi(left,C_ID) + phi(right,C_ID));

        phi0 = 2.0*GASCONST*prob_parm.Temp/FARADCONST;
        
        Real ls_left  = phi(left,LS_ID);
        Real ls_right = phi(right,LS_ID);
        Real c_left   = phi(left,CO_ID);
        Real c_right  = phi(right,CO_ID);

        //Real cs = amrex::max( (1.0-ls_left)*c_left, (1.0-ls_right)*c_right);
        //Real ce = amrex::max(ls_left*c_left, ls_right*c_right);
        cs = (1.0-ls_left)*c_left;
        ce = ls_left*c_left;
        for(int t2=-1; t2<=1; t2++)
        {
            for(int t1=-1; t1<=1; t1++)
            {
                for(int n=-1; n<=0; n++)
                {
                   IntVect curid(i,j,k);
                   Real ls,conc;

                   curid[normaldir] += n;
                   curid[trans1dir] += t1;
                   curid[trans2dir] += t2;

                   ls   = phi(curid, LS_ID);
                   conc = phi(curid, CO_ID);

                   if(ls*conc > ce)
                   {
                        ce=ls*conc;
                   }
                   if((1.0-ls)*conc > cs)
                   {
                        cs=(1.0-ls)*conc;
                   } 
                }
            }
        }

        j0=0.0;
        ocp=0.0;
        if(anodefrac > 0.0)
        {
            ocp = OCP_a(cs,prob_parm);
            j0  = Io_a(cs,ce,prob_parm);
        }
        if(cathodefrac > 0.0)
        {
            ocp = OCP_c(cs,prob_parm);
            j0 = Io_c(cs,ce,prob_parm);
        }

        /*amrex::Print()<<"normaldir,j0,phi0,ocp,phijump,anodefrac,cathodefrac:"
            <<"\t"<<normaldir<<"\t"<<j0<<"\t"<<phi0<<"\t"
            <<ocp<<"\t"<<phi_jump<<"\t"<<anodefrac<<"\t"<<cathodefrac<<"\n";*/
        j0=amrex::max(j0,jcutoff);

        // sinh definition
        jbv = -j0*sinh((phi_jump+ocp)/phi0);

        // linear case
        // jbv=-(j0/phi0)*(phi_jump+ocp);

        // sinh definition
        jbvder = -j0 * cosh((phi_jump + ocp) / phi0) * (1.0 / phi0);

        // linear case
        // jbvder=-(j0/phi0);
    }

    AMREX_GPU_DEVICE AMREX_INLINE Real bv_activation_function(Real c, Real gradc, Real gradc_cutoff)
    {
        Real f;
        Real maxval = 0.25;
        Real tol = 1e-1;

        // f=(gradc > gradc_cutoff)?1.0:0.0;
        f = (c * (1 - c) > tol * maxval) ? 1.0 : 0.0;
        return (f);
    }
} // namespace electrochem_reactions
#endif
