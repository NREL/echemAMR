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
    pp.query("interface_location", h_prob_parm.interface_location);
    pp.query("interface_sharpness", h_prob_parm.sharpness_factor);

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

    Real int_loc=prob_parm->interface_location;
    int transport_dir=0;
    Real x[AMREX_SPACEDIM];

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (GPU::notInLaunchRegion)
#endif
    for (int k = lo.z; k <= hi.z; ++k) 
    {
        x[2] = prob_lo[2] + (k+0.5)*dx[2];
        for (int j = lo.y; j <= hi.y; ++j) 
        {
            Real z = prob_lo[2] + (0.5+k) * dx[2];
            x[1] = prob_lo[1] + (j+0.5)*dx[1];

            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i) 
            {
                x[0] = prob_lo[0] + (i+0.5)*dx[0];

                phi(i,j,k,S1_ID) = 0.5 * (1.0 + tanh( prob_parm->sharpness_factor * (x[transport_dir]-int_loc)));

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
    Real dirc_val_S1;

    //default to extrapolation
    for(int c=0;c<NVAR;c++)
    {
        s_ext[c]=s_int[c];
    }

    if(sgn > 0) //left lo side
    {
        dirc_val_S1=0.0;
        s_ext[POT_ID]=1.0;
    }
    else
    {
        dirc_val_S1=0.0;
        s_ext[POT_ID]=1.0;
    }

    s_ext[S1_ID]=2.0*dirc_val_S1-s_int[S1_ID];
}
#endif
