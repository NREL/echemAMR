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
#ifdef AMREX_USE_GPU
    amrex::Gpu::htod_memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#else
    std::memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#endif

}

AMREX_GPU_DEVICE
AMREX_INLINE
void initdomaindata(Box const& bx, Array4<Real> const& phi, GeometryData const& geomdata,
        ProbParm *prob_parm)
{
    const auto lo = lbound(bx);
    const auto hi = ubound(bx);    

    const Real* AMREX_RESTRICT prob_lo = geomdata.ProbLo();
    const Real* AMREX_RESTRICT prob_hi = geomdata.ProbHi();
    const Real* AMREX_RESTRICT dx      = geomdata.CellSize();
    
    const Real Lx = prob_hi[0] - prob_lo[0];
    const Real Ly = prob_hi[1] - prob_lo[1];
    const Real Lz = prob_hi[2] - prob_lo[2];
    
    const int ncomp = phi.nComp();

    //FIXME: need real values from soc and csmax
    Real c_electrolyte = 1200.0;  //mol/m3
    Real c_cathode     = 40000.0; //mol/m3
    Real c_anode       = 4000.0;  //mol/m3
    Real c_separator   = 1200.0;  //mol/m3

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (GPU::notInLaunchRegion)
#endif
    for (int k = lo.z; k <= hi.z; ++k) 
    {
        const Real z = prob_lo[2] + (k+0.5)*dx[2];
        for (int j = lo.y; j <= hi.y; ++j) 
        {
            const Real y = prob_lo[1] + (j+0.5)*dx[1];

            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i) 
            {
                const Real x = prob_lo[0] + (i+0.5)*dx[0];
                
                for (int n = 0; n < ncomp; ++n)
                {
                    phi(i, j, k, n) = 0.0;
                }
                
                //by default electrolyte
                phi(i,j,k,LS_ID)=1.0;
                
                // Separator
                const amrex::Real s_xlo = 0.45*Lx;
                const amrex::Real s_xhi = 0.55*Lx;

                bool comp_found = false;
                for(int jj = 0; jj < 3; ++jj)
                {
                    Real y0 = jj*Ly/2.0;
                    for(int kk = 0; kk < 3; ++kk)
                    {
                        Real z0 = kk*Lz/2.0;
                        // cathode
                        for(int ii = 0; ii < 5; ++ii)
                        {
                            Real x0 = ii*.35*Lx/4.0;
                            Real r = std::sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0));
                            if(r < Lx/20.0)
                            {
                                phi(i,j,k,A_ID)  = 0.0;
                                phi(i,j,k,C_ID)  = 1.0;
                                phi(i,j,k,S_ID)  = 0.0;
                                phi(i,j,k,E_ID)  = 0.0;
                                phi(i,j,k,LS_ID) = 0.0;
                                phi(i,j,k,CO_ID) = c_cathode; 
                                comp_found = true;
                            }
                        }
                        // anode
                        for(int ii = 0; ii < 5; ++ii)
                        {
                            Real x0 = 0.65*Lx + ii*.35*Lx/4.0;
                            Real r = std::sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0));
                            if(r < Lx/20.0)
                            {
                                phi(i,j,k,A_ID)  = 1.0;
                                phi(i,j,k,C_ID)  = 0.0;
                                phi(i,j,k,S_ID)  = 0.0;
                                phi(i,j,k,E_ID)  = 0.0;
                                phi(i,j,k,LS_ID) = 0.0;
                                phi(i,j,k,CO_ID) = c_anode; 
                                comp_found = true;
                            }
                        }
                    }
                }

                if(!comp_found)
                {
                    if(x > s_xlo && x < s_xhi) 
                    {
                        // separator
                        phi(i,j,k,A_ID)  = 0.0;
                        phi(i,j,k,C_ID)  = 0.0;
                        phi(i,j,k,S_ID)  = 1.0;
                        phi(i,j,k,E_ID)  = 0.0;
                        phi(i,j,k,LS_ID) = 1.0;
                        phi(i,j,k,CO_ID) = c_separator; 
                    } 
                    else 
                    {
                        // electrolyte
                        phi(i,j,k,A_ID)  = 0.0;
                        phi(i,j,k,C_ID)  = 0.0;
                        phi(i,j,k,S_ID)  = 0.0;
                        phi(i,j,k,E_ID)  = 1.0;
                        phi(i,j,k,LS_ID) = 1.0;
                        phi(i,j,k,CO_ID) = c_electrolyte; 
                    }
                }

                phi(i,j,k,EFX_ID) = 0.0;
                phi(i,j,k,EFY_ID) = 0.0;
                phi(i,j,k,EFZ_ID) = 0.0;
                phi(i,j,k,POT_ID) = 0.0;
            }
        }
    }
}

AMREX_GPU_DEVICE
    AMREX_INLINE
void initproblemdata(Box const& bx, Array4<Real> const& phi, GeometryData const& geomdata,
        ProbParm* prob_parm)
{

}

    AMREX_INLINE
void print_init_data(ProbParm* prob_parm)
{

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
}
#endif
