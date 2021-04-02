#ifndef _PROB_H_
#define _PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <Chemistry.H>
#include <ChemistryProbParm.H>

#include<echemAMR.H>

using namespace amrex;

AMREX_INLINE
void amrex_probinit (ProbParm & h_prob_parm, ProbParm & d_prob_parm)
{
    // read problem specific parmparse parameters here
    amrex::ParmParse pp("prob");

    pp.query("faraday_constant", h_prob_parm.Faraday_const );
    pp.query("R_gas_constant", h_prob_parm.R_gas_const );

    pp.query("material_electrolyte", h_prob_parm.mater_electrolyte);
    pp.query("use_KDstar", h_prob_parm.use_KDstar);
    pp.query("material_cathode", h_prob_parm.mater_cathode);
    pp.query("material_anode", h_prob_parm.mater_anode);

    pp.query("lowerscale_vf_pore_anode", h_prob_parm.lowerscale_vf_pore_anode);
    pp.query("lowerscale_vf_pore_separator", h_prob_parm.lowerscale_vf_pore_separator);
    pp.query("lowerscale_vf_pore_cathode", h_prob_parm.lowerscale_vf_pore_cathode);
    pp.query("lowerscale_p_pore_anode", h_prob_parm.lowerscale_p_pore_anode);
    pp.query("lowerscale_p_pore_separator", h_prob_parm.lowerscale_p_pore_separator);
    pp.query("lowerscale_p_pore_cathode", h_prob_parm.lowerscale_p_pore_cathode);

    pp.query("ce_atrest", h_prob_parm.ce_atrest);
    pp.query("soc_anode_t0", h_prob_parm.soc_anode_t0);
    pp.query("soc_cathode_t0", h_prob_parm.soc_cathode_t0);
    pp.query("phi_reference", h_prob_parm.phi_reference);

    pp.queryarr("pot_bc_lo", h_prob_parm.pot_bc_lo, 0, AMREX_SPACEDIM );
    pp.queryarr("pot_bc_hi", h_prob_parm.pot_bc_hi, 0, AMREX_SPACEDIM );

#ifdef AMREX_USE_GPU
    amrex::Gpu::htod_memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#else
    std::memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#endif

}


AMREX_GPU_DEVICE
AMREX_INLINE
void initdata(Box const& bx, Array4<Real> const& phi, GeometryData const& geomdata)
{
    const auto lo = lbound(bx);
    const auto hi = ubound(bx);    

    const Real* AMREX_RESTRICT prob_lo = geomdata.ProbLo();
    const Real* AMREX_RESTRICT prob_hi = geomdata.ProbHi();
    const Real* AMREX_RESTRICT dx      = geomdata.CellSize();

    const int ncomp = phi.nComp();

    const Real Lx = prob_hi[0] - prob_lo[0];
    const Real Ly = prob_hi[1] - prob_lo[1];
    const Real Lz = prob_hi[2] - prob_lo[2];

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (GPU::notInLaunchRegion)
#endif

    // Active material volumes
    // echemAMR::post_timestep(0, 0);

    // Applied loading values (BC)
    // g=f(active material volume)

    // Active interface surface
    // will determine s_anode and s_cathode 

    //Mean Faraday current at the active interfaces
    // this will determine j_anode=s_anode/g and j_cathode=s_cathode/g used in below functions

    // Initial potential
    // const amrex::Real phie_to = get_phie_initial_from_anode(amrex::Real j_anode, const ProbParm& prob_parm);
    // const amrex::Real phi_c_to = get_phic_initial_from_electrolyte(amrex::Real j_cathode, amrex::Real phie_to, const ProbParm& prob_parm);
    const amrex::Real phie_to = -0.1;
    const amrex::Real phi_c_to = 4.0;

    //ProbParm h_prob_parm; // Create a new instance of the structure ProbParm


    //ProbParm* hh_prob_parm = nullptr;
    //ProbParm* dd_prob_parm = nullptr;
    //hh_prob_parm = new ProbParm{};
    //dd_prob_parm = (ProbParm*)The_Arena()->alloc(sizeof(ProbParm));
    //amrex_probinit(*hh_prob_parm, *dd_prob_parm);

    for (int k = lo.z; k <= hi.z; ++k) 
    {
        const Real z = prob_lo[2] + (0.5+k) * dx[2];
        for (int j = lo.y; j <= hi.y; ++j)
        {
            const Real y = prob_lo[1] + (0.5+j) * dx[1];
            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i) 
            {
                const Real x = prob_lo[0] + (0.5+i) * dx[0];

                for(int n = 0; n < ncomp; ++n){
                    phi(i,j,k,n) = -1.0;
                }

                // FIXME: pass prob parm or more inputs so that we can have control over initial condition parameters
                // Separator
                const amrex::Real s_xlo = 0.45*Lx;
                const amrex::Real s_xhi = 0.55*Lx;
                // anode
                const amrex::Real a_yhi = 0.8*Ly;
                const amrex::Real a_ylo = 0.2*Ly;
                const amrex::Real a_x = 0.25*Lx;
                // cathode
                const amrex::Real c_yhi = 0.8*Ly;
                const amrex::Real c_ylo = 0.2*Ly;
                const amrex::Real c_x = 0.75*Lx;

                bool comp_found = false;
                for(int jj = 0; jj < 3; ++jj){
                    Real y0 = jj*Ly/2.0;
                    for(int kk = 0; kk < 3; ++kk){
                        Real z0 = kk*Lz/2.0;
                        // cathode
                        for(int ii = 0; ii < 5; ++ii){
                            Real x0 = ii*.35*Lx/4.0;
                            Real r = std::sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0));
                            if(r < Lx/20.0){
                                phi(i,j,k,C_ID) = 1.0;
                                const amrex::Real soc_init = echemAMR::h_prob_parm->soc_cathode_t0;
                                amrex::Print() << "soc init cathode:       " << soc_init << std::endl;
                                const amrex::Real csmax = electrochem::Cs_max_cathode(*echemAMR::h_prob_parm);
                                phi(i,j,k,CO_ID) = soc_init * csmax;
                                phi(i,j,k,PO_ID) = phi_c_to;
                                comp_found = true;
                            }
                        }
                        // anode
                        for(int ii = 0; ii < 5; ++ii){
                            Real x0 = 0.65*Lx + ii*.35*Lx/4.0;
                            Real r = std::sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0));
                            if(r < Lx/20.0){
                                phi(i,j,k,A_ID) = 1.0;
                                const amrex::Real soc_init = echemAMR::h_prob_parm->soc_anode_t0;
                                amrex::Print() << "soc init anode:       " << soc_init << std::endl;
                                const amrex::Real csmax = electrochem::Cs_max_anode(*echemAMR::h_prob_parm);
                                phi(i,j,k,CO_ID) = soc_init * csmax;
                                phi(i,j,k,PO_ID) = echemAMR::h_prob_parm->phi_reference;
                                comp_found = true;
                            }
                        }
                    }
                }

                if(!comp_found){
                    if(x > s_xlo && x < s_xhi) {
                        // separator
                        phi(i,j,k,S_ID) = 1.0;
                        phi(i,j,k,CO_ID) = echemAMR::h_prob_parm->ce_atrest;
                        phi(i,j,k,PO_ID) = phie_to;
                    } else {
                        // electrolyte
                        phi(i,j,k,E_ID) = 1.0;
                        phi(i,j,k,CO_ID) = echemAMR::h_prob_parm->ce_atrest;
                        phi(i,j,k,PO_ID) = phie_to;
                    }
                }
                phi(i,j,k,EFX_ID) = 0.0;
                phi(i,j,k,EFY_ID) = 0.0;
                phi(i,j,k,EFZ_ID) = 0.0;
                //phi(i,j,k,PO_ID)  = 0.0;

                // this has a check in it to make sure we don't have
                // multiple fields specified in the same cell
                // FIXME: disable for now
//                electrochem::level_set_to_component(i,j,k,phi);
            }
        }
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void externalbc(const amrex::Real x[AMREX_SPACEDIM],
        const amrex::Real s_int[],
        amrex::Real s_ext[],
        const int idir,
        const int sgn,
        const amrex::Real time,
        amrex::GeometryData const& geomdata)
{
    //default to extrapolation
    for(int c=0;c<NVAR;c++)
    {
        s_ext[c]=s_int[c];
    }
    
    if(sgn==1) //left lo side
    {
        s_ext[CO_ID]=0.0;
        s_ext[PO_ID]=1.0;
    }
    else
    {
        s_ext[CO_ID]=0.0;
        s_ext[PO_ID]=0.0;
    }
}

#endif
