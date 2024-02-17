#ifndef _compute_flux_3d_H_
#define _compute_flux_3d_H_

#include <AMReX_BLFort.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <echemAMR_constants.H>
#include <Reactions.H>
#include <bv_utils.H>

using namespace amrex;

#define heps 1.0e-30

AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
void compute_flux(
        int i,
        int j,
        int k,
        int n,
        int sweepdir,
        Array4<Real> const& phi, // state variable
        Array4<Real> const& vel,
        Array4<Real> const& dcoeff,
        Array4<Real> const& fx,
        const GpuArray<Real, AMREX_SPACEDIM>& dx,
        ProbParm const& prob_parm,int implicit_diffusion,
        int bvflux,int lset_id,const int bvspec[NVAR],int lsgrad_tol)
{

    //|--left1--|--left--|--right--|--right1--|
    //|--(i-2)--|--(i-1)--|--i--|--(i+1)--|
    //remember: face i separates (i-1) and i cells

    IntVect left(i, j, k);
    IntVect left1(i, j, k);
    IntVect right(i, j, k);
    IntVect right1(i, j, k);

    left[sweepdir] -= 1;
    left1[sweepdir] -= 2;
    right1[sweepdir] += 1;

    // find face vel
    Real smallval = 1e-10;
    Real dtr = phi(right, n) - phi(left, n);
    if (!(std::abs(dtr) > 0.0))
    {
        dtr = dtr + smallval;
    }
    Real r_left = (phi(left, n) - phi(left1, n)) / dtr;
    Real lim_left = std::max(0.0, std::min(1.0, r_left));

    dtr = phi(right1, n) - phi(right, n);
    if (!(std::abs(dtr) > 0.0))
    {
        dtr = dtr + smallval;
    }
    Real r_right = (phi(right, n) - phi(left, n)) / dtr;
    Real lim_right = std::max(0.0, std::min(1.0, r_right));

    Real flx_L = phi(left, n)*vel(left, n)  + 0.5 * lim_left * (phi(left, n)*vel(left, n) - phi(left1, n)*vel(left1, n));
    Real flx_R = phi(right, n)*vel(right,n) - 0.5 * lim_right * (phi(right1, n)*vel(right1,n) - phi(right, n)*vel(right,n));
    
    
    Real vel_sgn = std::copysign(1.0, 0.5*(vel(left,n)+vel(right,n)) );

    //fx(i, j, k, n) = phi_L * 0.5 * (vel_mid + fabs(vel_mid)) + phi_R * 0.5 * (vel_mid - fabs(vel_mid));
    fx(i, j, k, n) = flx_L * 0.5 * (1.0 + vel_sgn) + flx_R * 0.5 * (1.0 - vel_sgn);

    if(!implicit_diffusion)
    {
        const amrex::Real d1 = dcoeff(right, n);
        const amrex::Real d2 = dcoeff(left, n);

        fx(i, j, k, n) += -2.0 * d1 * d2 / (d1 + d2 + heps) * (phi(right, n) - phi(left, n)) / dx[sweepdir];

        //arithmetic mean case
        //fx(i, j, k, n) += -0.5*(d1 + d2) * (phi(right, n) - phi(left, n)) / dx[sweepdir];
    }
    
    if(bvflux)
    {
        if(bvspec[n]==1)
        {
            Real mod_gradc=0.0;
            Real facecolor=0.0;
            Real potjump=0.0;
            Real gradc_cutoff=0.0;
            Real n_ls[AMREX_SPACEDIM];
            Real dphidn=0.0;
            Real dphidt1=0.0;
            Real dphidt2=0.0;

            bv_get_grads_and_jumps(i, j, k, sweepdir, lset_id, dx, phi, lsgrad_tol, 
                    mod_gradc, gradc_cutoff, facecolor, potjump, dphidn, dphidt1, dphidt2, n_ls);

            if(mod_gradc > gradc_cutoff)
            {
                Real activ_func = electrochem_reactions::bv_activation_function(facecolor, mod_gradc, gradc_cutoff);
                fx(i,j,k,n)=fx(i,j,k,n)*(1.0-activ_func);
                Real j_bv;

                electrochem_reactions::species_bvflux(i,j,k,n,sweepdir,potjump,phi,prob_parm,j_bv);

                //note n_ls[0] is along the sweep direction
                fx(i,j,k,n)=fx(i,j,k,n)+j_bv*activ_func*n_ls[0];

                //debug print
                /*amrex::Print()<<"bvflux:"<<j_bv/FARADCONST*activ_func*n_ls[sweepdir]<<"\t"<<
                  phi(left,lset_id)<<"\t"<<phi(right,lset_id)<<"\t"<<
                  phi(left,n)<<"\t"<<phi(right,n)<<"\t"<<
                  potjump<<"\t"<<n_ls[sweepdir]<<"\n";*/
            }
        }
    }

}
#endif
