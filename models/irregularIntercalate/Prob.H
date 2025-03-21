#ifndef _PROB_H_
#define _PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <Chemistry.H>
#include <ProbParm.H>

using namespace amrex;

AMREX_INLINE
void amrex_probinit (ProbParm * h_prob_parm, ProbParm * d_prob_parm)
{
    amrex::ParmParse pp("prob");
    pp.query("currenthalfdecaytime", h_prob_parm->currenthalfdecaytime );
    amrex::Gpu::copy(amrex::Gpu::hostToDevice, h_prob_parm, 
                     h_prob_parm+1,d_prob_parm);
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
    
    const Real Lx = prob_hi[0] - prob_lo[0];
    const Real Ly = prob_hi[1] - prob_lo[1];
    const Real Lz = prob_hi[2] - prob_lo[2];

    Real rad=Ly*0.8;
    Real rad2=rad*rad;
    Real center1[AMREX_SPACEDIM];
    Real center2[AMREX_SPACEDIM];

    center1[0]=prob_lo[0];
    center1[1]=prob_hi[1];
    center1[2]=prob_lo[2]+0.5*Lz;
    
    center2[0]=prob_lo[0];
    center2[1]=prob_lo[1];
    center2[2]=prob_lo[2]+0.5*Lz;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (GPU::notInLaunchRegion)
#endif
    for (int k = lo.z; k <= hi.z; ++k) 
    {
        x[2] = prob_lo[2] + (k+0.5)*dx[2];
        for (int j = lo.y; j <= hi.y; ++j) 
        {
            x[1] = prob_lo[1] + (j+0.5)*dx[1];
            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i) 
            {
                x[0] = prob_lo[0] + (i+0.5)*dx[0];

                Real dist1_2=pow((x[0]-center1[0]),2.0)+pow((x[1]-center1[1]),2.0)+pow((x[2]-center1[2]),2.0);
                Real dist2_2=pow((x[0]-center2[0]),2.0)+pow((x[1]-center2[1]),2.0)+pow((x[2]-center2[2]),2.0);

                phi(i,j,k,LS_ID)=1.0;
                phi(i,j,k,S1_ID)=0.1;

                Real localrad_2=std::pow(rad+0.1*rad*sin(2*M_PI*x[1]*5.0/Ly),2.0);

                if(dist1_2 < localrad_2 || dist2_2 < localrad_2)
                {
                    phi(i,j,k,LS_ID)=0.0; 
                    phi(i,j,k,S1_ID)=10.0;
                }

                phi(i,j,k,EFX_ID) = 0.0;
                phi(i,j,k,EFY_ID) = 0.0;
                phi(i,j,k,EFZ_ID) = 0.0;
                phi(i,j,k,POT_ID) = 0.0;
            }
        }
    }
}

AMREX_INLINE
void initialconditions(ProbParm & h_prob_parm, ProbParm & d_prob_parm)
{

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
}
#endif
