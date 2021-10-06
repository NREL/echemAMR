#ifndef _bv_utils_H_
#define _bv_utils_H_

#include <AMReX_BLFort.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <echemAMR_constants.H>
#include <Reactions.H>

using namespace amrex;

AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
void bv_get_grads_and_jumps(int i,int j,int k,int normaldir,int lset_id,
        const GpuArray<Real, AMREX_SPACEDIM>& dx,
        Array4<Real> const& phi_arr,
        Real& mod_gradc, Real &gradc_cutoff,
        Real &facecolor, Real &potjump, Real &dphidn, Real &dphidt1, Real &dphidt2,
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

    // grad of potential (are arguments)
    dphidn = (pot_right - pot_left) / dx[normaldir];
    dphidt1 = (pot_top - pot_bot) / (2.0 * dx[trans1dir]);
    dphidt2 = (pot_frnt - pot_back) / (2.0 * dx[trans2dir]);

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
#endif
