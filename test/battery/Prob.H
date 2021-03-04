#ifndef _PROB_H_
#define _PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <Chemistry.H>
#include <ChemistryProbParm.H>

using namespace amrex;

AMREX_INLINE
void amrex_probinit (ProbParm & h_prob_parm, ProbParm & d_prob_parm)
{
    // read problem specific parmparse parameters here
    amrex::ParmParse pp("prob");

    pp.query("anode_conductivity", h_prob_parm.K_a );
    pp.query("cathode_conductivity", h_prob_parm.K_c );

    pp.query("anode_diffusivity", h_prob_parm.D_a );
    pp.query("cathode_diffusivity", h_prob_parm.D_c );

    pp.query("Electrolyte_diffusivity", h_prob_parm.Debulk);
    pp.query("Electrolyte_Transference", h_prob_parm.tplus);
    pp.query("Electrolyte_conductivity", h_prob_parm.Kebulk);
    pp.query("Electrolyte_activity",h_prob_parm.Activityel);
    
    pp.query("faraday_constant", h_prob_parm.Faraday_const );
    pp.query("R_gas_constant", h_prob_parm.R_gas_const );

    pp.query("constant_coefficient", h_prob_parm.constant_coeff);
    
    pp.queryarr("pot_bc_lo", h_prob_parm.pot_bc_lo, 0, AMREX_SPACEDIM );
    pp.queryarr("pot_bc_hi", h_prob_parm.pot_bc_hi, 0, AMREX_SPACEDIM );

#ifdef AMREX_USE_GPU
    amrex::Gpu::htod_memcpy(d_prob_parm, h_prob_parm, sizeof(ProbParm));
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
                                phi(i,j,k,CO_ID) = 10000.0;
                                comp_found = true;
                            }
                        }
                        // anode
                        for(int ii = 0; ii < 5; ++ii){
                            Real x0 = 0.65*Lx + ii*.35*Lx/4.0;
                            Real r = std::sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0));
                            if(r < Lx/20.0){
                                phi(i,j,k,A_ID) = 1.0;
                                phi(i,j,k,CO_ID) = 1000.0;
                                comp_found = true;
                            }
                        }
                    }
                }

                if(!comp_found){
                    if(x > s_xlo && x < s_xhi) {
                        // separator
                        phi(i,j,k,S_ID) = 1.0;
                        phi(i,j,k,CO_ID) = 1200.0;
                    } else {
                        // electrolyte
                        phi(i,j,k,E_ID) = 1.0;
                        phi(i,j,k,CO_ID) = 1200.0;
                    }
                }
                phi(i,j,k,EFX_ID) = 0.0;
                phi(i,j,k,EFY_ID) = 0.0;
                phi(i,j,k,EFZ_ID) = 0.0;
                phi(i,j,k,PO_ID)  = 0.0;

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
