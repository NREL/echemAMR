#ifndef _PROB_H_
#define _PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <Chemistry.H>

using namespace amrex;

AMREX_GPU_DEVICE
AMREX_INLINE
void initdata(Box const& bx, Array4<Real> const& phi, GeometryData const& geomdata)
{
    const auto lo = lbound(bx);
    const auto hi = ubound(bx);    

    const Real* AMREX_RESTRICT prob_lo = geomdata.ProbLo();
    const Real* AMREX_RESTRICT dx      = geomdata.CellSize();

    const int ncomp = phi.nComp();

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

                // initial conditions
                // Concentration
                phi(i,j,k,CO_ID) = 3000.0 + 10.0*std::sin(3.14*x);
                // Potential
                phi(i,j,k,PO_ID) = -3.0 + 0.1*x;


                // Separator
                constexpr amrex::Real s_xlo = 0.45;
                constexpr amrex::Real s_xhi = 0.55;
                // anode
                constexpr amrex::Real a_yhi = 0.4;
                constexpr amrex::Real a_ylo = 0.1;
                constexpr amrex::Real a_x = 0.2;
                // cathode
                constexpr amrex::Real c_yhi = 0.4;
                constexpr amrex::Real c_ylo = 0.1;
                constexpr amrex::Real c_x = 0.8;

                if(x > s_xlo && x < s_xhi) {
                    phi(i,j,k,S_ID) = 1.0;
                } else if(x < a_x && y < a_yhi && y > a_ylo) {
                    phi(i,j,k,A_ID) = 1.0;
                } else if(x > c_x && y < c_yhi && y > c_ylo) {
                    phi(i,j,k,C_ID) = 1.0;
                } else {
                    phi(i,j,k,E_ID) = 1.0;
                }

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
