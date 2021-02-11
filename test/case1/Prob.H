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

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (GPU::notInLaunchRegion)
#endif
    for (int k = lo.z; k <= hi.z; ++k) 
    {
        for (int j = lo.y; j <= hi.y; ++j) 
        {
            Real z = prob_lo[2] + (0.5+k) * dx[2];
            Real y = prob_lo[1] + (0.5+j) * dx[1];
            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i) 
            {
                Real x = prob_lo[0] + (0.5+i) * dx[0]; 
                Real r2 = (pow(x-0.5, 2) + pow((y-0.5),2) + pow((z-0.5),2)) / 0.005;
                phi(i,j,k,S1_ID) = 0.0;
                phi(i,j,k,S2_ID) = 0.0;
                phi(i,j,k,POT_ID) = 0.0;
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
        s_ext[S1_ID]=0.0;
        s_ext[S2_ID]=1.0;
        s_ext[POT_ID]=0.0;
    }
    else
    {
        s_ext[S1_ID]=0.0;
        s_ext[S2_ID]=0.0;
        s_ext[POT_ID]=0.0;
    }
}

#endif
