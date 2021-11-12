#ifndef _REACTIONS_H_
#define _REACTIONS_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace electrochem_reactions
{
    AMREX_GPU_DEVICE AMREX_INLINE
        void compute_react_source(int i, int j, int k,
                Array4<Real> const& phi,
                Array4<Real> const& reactsource,
                GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                GpuArray<Real, AMREX_SPACEDIM> dx,
                const Real time,
                ProbParm const& prob_parm)
        {
            for (int n = 0; n < reactsource.nComp(); ++n)
            {
                reactsource(i, j, k, n) = 0.0;
            }
        }

    AMREX_GPU_DEVICE AMREX_INLINE
        void compute_potential_source(int i, int j, int k,
                Array4<Real> const& phi,
                Array4<Real> const& source,
                GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                GpuArray<Real, AMREX_SPACEDIM> dx,
                const Real time,
                ProbParm const& prob_parm)
        {
            // concentration
            Real C = phi(i, j, k, CO_ID);
            Real Ke = electrochem::Ke_bulk(C, prob_parm);
            Real Ac = electrochem::A_c(C, prob_parm);
            Real tplus = electrochem::t_plus(C, prob_parm);
            Real KDe = electrochem::Kd(Ke, Ac, tplus, prob_parm);

            Real KDl  = 0.5 * (phi(i-1,j,k,E_ID)+phi(i,j,k,E_ID)) * KDe;
            Real KDr  = 0.5 * (phi(i+1,j,k,E_ID)+phi(i,j,k,E_ID)) * KDe;
            
            Real KDb  = 0.5 * (phi(i,j-1,k,E_ID)+phi(i,j,k,E_ID)) * KDe;
            Real KDt  = 0.5 * (phi(i,j+1,k,E_ID)+phi(i,j,k,E_ID)) * KDe;
            
            Real KDk  = 0.5 * (phi(i,j,k-1,E_ID)+phi(i,j,k,E_ID)) * KDe;
            Real KDf  = 0.5 * (phi(i,j,k+1,E_ID)+phi(i,j,k,E_ID)) * KDe;
           
            Real LSX = min(phi(i-1,j,k,LS_ID),min(phi(i,j,k,LS_ID),phi(i+1,j,k,LS_ID)));
            Real LSY = min(phi(i,j-1,k,LS_ID),min(phi(i,j,k,LS_ID),phi(i,j+1,k,LS_ID)));
            Real LSZ = min(phi(i,j,k-1,LS_ID),min(phi(i,j,k,LS_ID),phi(i,j,k+1,LS_ID)));

            source(i,j,k) = -(1.0/C)*(LSX*(KDr*(phi(i+1,j,k,CO_ID)-phi(i,j,k,CO_ID)) - KDl*(phi(i,j,k,CO_ID)-phi(i-1,j,k,CO_ID)))/dx[0]/dx[0]
                                   +  LSY*(KDt*(phi(i,j+1,k,CO_ID)-phi(i,j,k,CO_ID)) - KDb*(phi(i,j,k,CO_ID)-phi(i,j-1,k,CO_ID)))/dx[1]/dx[1]
                                   +  LSZ*(KDf*(phi(i,j,k+1,CO_ID)-phi(i,j,k,CO_ID)) - KDk*(phi(i,j,k,CO_ID)-phi(i,j,k-1,CO_ID)))/dx[2]/dx[2]);
        }

    AMREX_GPU_DEVICE AMREX_INLINE
        void bvcurrent_and_der(int i,int j,int k,int normaldir,
                Real phi_jump, Array4<Real> const& phi, ProbParm const& prob_parm,Real &jbv,Real &jbvder)
        {
            IntVect left(i, j, k);
            IntVect right(i, j, k);
            left[normaldir] -= 1;

            Real Cs= phi(left,CO_ID)*(phi(left,A_ID)+phi(left,C_ID)) + phi(right,CO_ID)*(phi(right,A_ID)+phi(right,C_ID));
            Real Ce= phi(left,CO_ID)*(phi(left,E_ID)+phi(left,S_ID)) + phi(right,CO_ID)*(phi(right,E_ID)+phi(right,S_ID));
            Real Io = electrochem::Io_a(Cs, Ce, prob_parm)*(phi(left,A_ID)+phi(right,A_ID)) + electrochem::Io_c(Cs, Ce, prob_parm)*(phi(left,C_ID)+phi(right,C_ID));
            Real OCP = electrochem::OCP_a(Cs, prob_parm)*(phi(left,A_ID)+phi(right,A_ID)) + electrochem::OCP_c(Cs, prob_parm)*(phi(left,C_ID)+phi(right,C_ID));
            Real eta = phi_jump - OCP;
            Real FRT = FARADCONST/(GASCONST * prob_parm.Temperature);
            jbv = 2.0 * Io * std::sinh(0.5 * FRT * eta);
            jbvder = 1.0;

            /*
            Real ocp=0.2; //FIXME:should be function of phi(i,j,k,CO_ID)
            Real j0=3.0; 
            Real phi0=(2.0*GASCONST*prob_parm.Temperature)/FARADCONST;
            Real csmax=60000;
            Real jbv0;
            Real jbv_cutoff=0.1;
            


            Real x=max(0.5*(phi(left,CO_ID)+phi(right,CO_ID))/csmax,0.0);

            jbv0=max(4.0*j0*x*(1.0-x),jbv_cutoff);

            //sinh definition
            //jbv=-j0*sinh((phi_jump+ocp)/phi0);

            //linear case
            jbv=-(jbv0/phi0)*(phi_jump+ocp);

            //sinh definition
            //jbvder=-j0*cosh((phi_jump+ocp)/phi0)*(1.0/phi0);

            //linear case
            jbvder=-(jbv0/phi0);
            */
        }

    AMREX_GPU_DEVICE AMREX_INLINE
        Real bv_activation_function(Real c, Real gradc, Real gradc_cutoff)
        {
            Real f;
            Real maxval=0.25;
            Real tol=1e-1;

            //f=(gradc > gradc_cutoff)?1.0:0.0;
            f=(c*(1-c) > tol*maxval)?1.0:0.0;
            return(f);
        }
}
#endif
