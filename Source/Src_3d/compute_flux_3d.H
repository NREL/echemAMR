#ifndef _compute_flux_3d_H_
#define _compute_flux_3d_H_

#include <AMReX_BLFort.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <echemAMR_constants.H>
#include <Reactions.H>

using namespace amrex;

//FIXME:need to remove these separate functions for 
//flux along x,y and z.

#define heps 1.0e-30

AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
void bv_get_grads_and_jumps(int i,int j,int k,int normaldir,int lset_id,
        const GpuArray<Real, AMREX_SPACEDIM>& dx,
        Array4<Real> const& phi_arr,
        Real& mod_gradc, Real &gradc_cutoff,
        Real &facecolor, Real &potjump,
        Real n_ls[AMREX_SPACEDIM])
{
    Real min_dx = amrex::min(dx[0], amrex::min(dx[1], dx[2]));
    IntVect left(i, j, k);
    IntVect right(i, j, k);

    IntVect top_left(i, j, k);
    IntVect bottom_left(i, j, k);
    IntVect top_right(i, j, k);
    IntVect bottom_right(i, j, k);

    IntVect front_left(i, j, k);
    IntVect back_left(i, j, k);
    IntVect front_right(i, j, k);
    IntVect back_right(i, j, k);

    int trans1dir = (normaldir + 1) % AMREX_SPACEDIM;
    int trans2dir = (normaldir + 2) % AMREX_SPACEDIM;

    left[normaldir]         -= 1;
    top_left[normaldir]     -= 1;
    bottom_left[normaldir]  -= 1;
    front_left[normaldir]   -= 1;
    back_left[normaldir]    -= 1;

    top_left[trans1dir]     += 1;
    top_right[trans1dir]    += 1;
    bottom_left[trans1dir]  -= 1;
    bottom_right[trans1dir] -= 1;

    front_left[trans2dir]   += 1;
    front_right[trans2dir]  += 1;
    back_left[trans2dir]    -= 1;
    back_right[trans2dir]   -= 1;

    // levelset color
    Real c_left = phi_arr(left, lset_id);
    Real c_right = phi_arr(right, lset_id);

    Real c_top = 0.5 * (phi_arr(top_left, lset_id) + phi_arr(top_right, lset_id));
    Real c_bot = 0.5 * (phi_arr(bottom_left, lset_id) + phi_arr(bottom_right, lset_id));

    Real c_frnt = 0.5 * (phi_arr(front_left, lset_id) + phi_arr(front_right, lset_id));
    Real c_back = 0.5 * (phi_arr(back_left, lset_id) + phi_arr(back_right, lset_id));

    // phi
    Real pot_left = phi_arr(left, POT_ID);
    Real pot_right = phi_arr(right, POT_ID);

    Real pot_top = 0.5 * (phi_arr(top_left, POT_ID) + phi_arr(top_right, POT_ID));
    Real pot_bot = 0.5 * (phi_arr(bottom_left, POT_ID) + phi_arr(bottom_right, POT_ID));

    Real pot_frnt = 0.5 * (phi_arr(front_left, POT_ID) + phi_arr(front_right, POT_ID));
    Real pot_back = 0.5 * (phi_arr(back_left, POT_ID) + phi_arr(back_right, POT_ID));

    // x,y or z
    Real dcdn = (c_right - c_left) / dx[normaldir];
    Real dcdt1 = (c_top - c_bot) / (2.0 * dx[trans1dir]);
    Real dcdt2 = (c_frnt - c_back) / (2.0 * dx[trans2dir]);

    // grad of potential
    Real dphidn = (pot_right - pot_left) / dx[normaldir];
    Real dphidt1 = (pot_top - pot_bot) / (2.0 * dx[trans1dir]);
    Real dphidt2 = (pot_frnt - pot_back) / (2.0 * dx[trans2dir]);

    Real gradc_tolfac = 1e-4;
    Real gradc_max = 1.0 / min_dx; // maximum gradient possible on the current grid
    
    gradc_cutoff = gradc_tolfac * gradc_max;
    mod_gradc = sqrt(dcdn * dcdn + dcdt1 * dcdt1 + dcdt2 * dcdt2);
    facecolor = 0.5*(c_left+c_right);

    //some default values
    potjump   = 0.0;
    n_ls[0]   = 1.0;
    n_ls[1]   = 0.0;
    n_ls[2]   = 0.0;

    if (mod_gradc > gradc_cutoff)
    {
        n_ls[0] = dcdn / mod_gradc;
        n_ls[1] = dcdt1 / mod_gradc;
        n_ls[2] = dcdt2 / mod_gradc;
        // jump along the level set normal (phi_electrolyte-phi_electrode)
        potjump = (dphidn * n_ls[0] + dphidt1 * n_ls[1] + dphidt2 * n_ls[2]) / mod_gradc;
        /*Print()<<"dphidn,dphidt1,dphidt2:"<<dphidn<<"\t"<<dphidt1<<"\t"<<dphidt2<<"\n";
        Print()<<"n_ls:"<<n_ls[0]<<"\t"<<n_ls[1]<<"\t"<<n_ls[2]<<"\n";*/
    }
}

AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
void compute_flux_x(
        int i,
        int j,
        int k,
        int n,
        Array4<Real> const& phi, // state variable
        Array4<Real> const& vel,
        Array4<Real> const& dcoeff,
        Array4<Real> const& fx,
        const GpuArray<Real, AMREX_SPACEDIM>& dx,
        ProbParm const& prob_parm,
        int bvflux=0,int lset_id=0,int bvspec=1)
{

    const amrex::Real d1 = dcoeff(i, j, k, n);
    const amrex::Real d2 = dcoeff(i - 1, j, k, n);

    fx(i, j, k, n) = -2.0 * d1 * d2 / (d1 + d2 + heps) * (phi(i, j, k, n) - phi(i - 1, j, k, n)) / dx[0];
    //    fx(i,j,k,n) = -0.5 * (dcoeff(i,j,k,n)+dcoeff(i-1,j,k,n)) * (phi(i,j,k,n) - phi(i-1,j,k,n))/dx[0];

    // find face vel
    Real smallval = 1e-10;
    Real dtr = phi(i, j, k, n) - phi(i - 1, j, k, n);
    if (!(std::abs(dtr) > 0.0))
    {
        dtr = dtr + smallval;
    }
    Real r_left = (phi(i - 1, j, k, n) - phi(i - 2, j, k, n)) / dtr;
    Real lim_left = std::max(0.0, std::min(1.0, r_left));

    dtr = phi(i + 1, j, k, n) - phi(i, j, k, n);
    if (!(std::abs(dtr) > 0.0))
    {
        dtr = dtr + smallval;
    }
    Real r_right = (phi(i, j, k, n) - phi(i - 1, j, k, n)) / dtr;
    Real lim_right = std::max(0.0, std::min(1.0, r_right));

    Real phi_L = phi(i - 1, j, k, n) + 0.5 * lim_left * (phi(i - 1, j, k, n) - phi(i - 2, j, k, n));
    Real phi_R = phi(i, j, k, n) - 0.5 * lim_right * (phi(i + 1, j, k, n) - phi(i, j, k, n));

    Real vel_mid = vel(i, j, k, n);
    fx(i, j, k, n) += phi_L * 0.5 * (vel_mid + fabs(vel_mid)) + phi_R * 0.5 * (vel_mid - fabs(vel_mid));

    if(bvflux)
    {

        if(n==bvspec)
        {
            Real mod_gradc=0.0;
            Real facecolor=0.0;
            Real potjump=0.0;
            Real gradc_cutoff=0.0;
            Real n_ls[AMREX_SPACEDIM];
            int dir=0;

            bv_get_grads_and_jumps(i,j,k,dir,lset_id,dx,phi,
                    mod_gradc,gradc_cutoff,facecolor,potjump,n_ls);

            if(mod_gradc > gradc_cutoff)
            {
                Real activ_func = electrochem_reactions::bv_activation_function(facecolor, mod_gradc, gradc_cutoff);
                fx(i,j,k,n)=fx(i,j,k,n)*(1.0-activ_func);
                Real j_bv = electrochem_reactions::bvcurrent(i,j,k,dir,potjump,phi,prob_parm);
                fx(i,j,k,n)=fx(i,j,k,n)+j_bv/FARADCONST*activ_func*n_ls[dir];
                /*amrex::Print()<<"bvflux:"<<j_bv/FARADCONST*activ_func*n_ls[dir]<<"\t"<<
                    phi(i-1,j,k,LS_ID)<<"\t"<<phi(i,j,k,LS_ID)<<"\t"<<
                    phi(i-1,j,k,E_ID)<<"\t"<<phi(i,j,k,E_ID)<<"\t"<<
                    phi(i-1,j,k,C_ID)<<"\t"<<phi(i,j,k,C_ID)<<"\t"<<
                    potjump<<"\t"<<n_ls[dir]<<"\n";*/
            }
        }
    }

}


AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
void compute_flux_y(
        int i,
        int j,
        int k,
        int n,
        Array4<Real> const& phi, // state variable
        Array4<Real> const& vel,
        Array4<Real> const& dcoeff,
        Array4<Real> const& fy,
        const GpuArray<Real, AMREX_SPACEDIM>& dx,
        ProbParm const& prob_parm,
        int bvflux=0,int lset_id=0,int bvspec=1)
{

    const amrex::Real d1 = dcoeff(i, j, k, n);
    const amrex::Real d2 = dcoeff(i, j - 1, k, n);
    fy(i, j, k, n) = -2.0 * d1 * d2 / (d1 + d2 + heps) * (phi(i, j, k, n) - phi(i, j - 1, k, n)) / dx[1];
    //    fy(i,j,k,n) = -0.5 * (dcoeff(i,j,k,n)+dcoeff(i,j-1,k,n)) * (phi(i,j,k,n) - phi(i,j-1,k,n))/dx[1];

    // find face vel
    Real smallval = 1e-10;
    Real dtr = phi(i, j, k, n) - phi(i, j - 1, k, n);
    if (!(std::abs(dtr) > 0.0))
    {
        dtr = dtr + smallval;
    }
    Real r_left = (phi(i, j - 1, k, n) - phi(i, j - 2, k, n)) / dtr;
    Real lim_left = std::max(0.0, std::min(1.0, r_left));

    dtr = phi(i, j + 1, k, n) - phi(i, j, k, n);
    if (!(std::abs(dtr) > 0.0))
    {
        dtr = dtr + smallval;
    }
    Real r_right = (phi(i, j, k, n) - phi(i, j - 1, k, n)) / dtr;
    Real lim_right = std::max(0.0, std::min(1.0, r_right));

    Real phi_L = phi(i, j - 1, k, n) + 0.5 * lim_left * (phi(i, j - 1, k, n) - phi(i, j - 2, k, n));
    Real phi_R = phi(i, j, k, n) - 0.5 * lim_right * (phi(i, j + 1, k, n) - phi(i, j, k, n));

    Real vel_mid = vel(i, j, k, n);
    fy(i, j, k, n) += phi_L * 0.5 * (vel_mid + fabs(vel_mid)) + phi_R * 0.5 * (vel_mid - fabs(vel_mid));

    if(bvflux)
    {

        if(n==bvspec)
        {
            Real mod_gradc=0.0;
            Real facecolor=0.0;
            Real potjump=0.0;
            Real gradc_cutoff=0.0;
            Real n_ls[AMREX_SPACEDIM];
            int dir=1;

            bv_get_grads_and_jumps(i,j,k,dir,lset_id,dx,phi,
                    mod_gradc,gradc_cutoff,facecolor,potjump,n_ls);

            if(mod_gradc > gradc_cutoff)
            {
                Real activ_func = electrochem_reactions::bv_activation_function(facecolor, mod_gradc, gradc_cutoff);
                fy(i,j,k,n)=fy(i,j,k,n)*(1.0-activ_func);
                Real j_bv = electrochem_reactions::bvcurrent(i,j,k,dir,potjump,phi,prob_parm);
                fy(i,j,k,n)=fy(i,j,k,n)+j_bv/FARADCONST*activ_func*n_ls[dir];
            }
        }
    }
}

AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
void compute_flux_z(
        int i,
        int j,
        int k,
        int n,
        Array4<Real> const& phi, // state variable
        Array4<Real> const& vel,
        Array4<Real> const& dcoeff,
        Array4<Real> const& fz,
        const GpuArray<Real, AMREX_SPACEDIM>& dx,
        ProbParm const& prob_parm,
        int bvflux=0,int lset_id=0,int bvspec=1)

{

    const amrex::Real d1 = dcoeff(i, j, k, n);
    const amrex::Real d2 = dcoeff(i, j, k - 1, n);
    fz(i, j, k, n) = -2.0 * d1 * d2 / (d1 + d2 + heps) * (phi(i, j, k, n) - phi(i, j, k - 1, n)) / dx[2];
    //    fz(i,j,k,n) = -0.5 * (dcoeff(i,j,k,n)+dcoeff(i,j,k-1,n)) * (phi(i,j,k,n) - phi(i,j,k-1,n))/dx[2];

    // find face vel
    Real smallval = 1e-10;

    Real dtr = phi(i, j, k, n) - phi(i, j, k - 1, n);
    if (!(std::abs(dtr) > 0.0))
    {
        dtr = dtr + smallval;
    }
    Real r_left = (phi(i, j, k - 1, n) - phi(i, j, k - 2, n)) / dtr;
    Real lim_left = std::max(0.0, std::min(1.0, r_left));

    dtr = phi(i, j, k + 1, n) - phi(i, j, k, n);
    if (!(std::abs(dtr) > 0.0))
    {
        dtr = dtr + smallval;
    }
    Real r_right = (phi(i, j, k, n) - phi(i, j, k - 1, n)) / dtr;
    Real lim_right = std::max(0.0, std::min(1.0, r_right));

    Real phi_L = phi(i, j, k - 1, n) + 0.5 * lim_left * (phi(i, j, k - 1, n) - phi(i, j, k - 2, n));
    Real phi_R = phi(i, j, k, n) - 0.5 * lim_right * (phi(i, j, k + 1, n) - phi(i, j, k, n));

    Real vel_mid = vel(i, j, k, n);
    fz(i, j, k, n) += phi_L * 0.5 * (vel_mid + fabs(vel_mid)) + phi_R * 0.5 * (vel_mid - fabs(vel_mid));

    if(bvflux)
    {

        if(n==bvspec)
        {
            Real mod_gradc=0.0;
            Real facecolor=0.0;
            Real potjump=0.0;
            Real gradc_cutoff=0.0;
            Real n_ls[AMREX_SPACEDIM];
            int dir=2;

            bv_get_grads_and_jumps(i,j,k,dir, lset_id,dx,phi,
                    mod_gradc,gradc_cutoff,facecolor,potjump,n_ls);

            if(mod_gradc > gradc_cutoff)
            {
                Real activ_func = electrochem_reactions::bv_activation_function(facecolor, mod_gradc, gradc_cutoff);
                fz(i,j,k,n)=fz(i,j,k,n)*(1.0-activ_func);
                Real j_bv = electrochem_reactions::bvcurrent(i,j,k,dir,potjump,phi,prob_parm);
                fz(i,j,k,n)=fz(i,j,k,n)+j_bv/FARADCONST*activ_func*n_ls[dir];
            }
        }
    }
}
#endif
