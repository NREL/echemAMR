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
    // read problem specific parmparse parameters here
    amrex::ParmParse pp("prob");
    pp.query("interface1", h_prob_parm->int1 );
    pp.query("interface2", h_prob_parm->int2 );
    pp.query("interface_sharpness", h_prob_parm->sharpness_factor);
    pp.query("direction", h_prob_parm->direction);
    
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

    int axis = prob_parm->direction;

    Real Lmid=0.5*(prob_lo[axis]+prob_hi[axis]);
    Real length = prob_hi[axis] - prob_lo[axis];
    Real L1=prob_parm->int1*length;
    Real L2=prob_parm->int2*length;

    
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (GPU::notInLaunchRegion)
#endif
    for (int k = lo.z; k <= hi.z; ++k) 
    {
        Real z = prob_lo[2] + (0.5+k) * dx[2]; //is moving this safe
        for (int j = lo.y; j <= hi.y; ++j) 
        {
            Real y = prob_lo[1] + (0.5+j) * dx[1];
            AMREX_PRAGMA_SIMD //is this useful because the original assumption is that x is the "long" axis
            for (int i = lo.x; i <= hi.x; ++i) 
            {
                Real x = prob_lo[0] + (0.5+i) * dx[0];

                Real L = x;
                if (axis==1){L = y;}//is there a better way of selecting L
                if (axis==2){L = z;}

                phi(i,j,k,C_ID) = 0.0;
                phi(i,j,k,A_ID) = 0.0;
                phi(i,j,k,E_ID) = 0.0;

                if(L<=L1)
                {
                   phi(i,j,k,CO_ID)=0.2;
                   phi(i,j,k,LS_ID)=0.0;
                   phi(i,j,k,C_ID)=1.0;
                }
                else if(L>L1 && L<L2)
                {
                    phi(i,j,k,CO_ID)=0.1+0.2*(L-L1)/(L2-L1);
                    phi(i,j,k,LS_ID)=1.0;
                    phi(i,j,k,E_ID)=1.0;
                }
                else
                {
                   phi(i,j,k,CO_ID)=1.0;
                   phi(i,j,k,LS_ID)=0.0;
                   phi(i,j,k,A_ID)=1.0;
                }
                phi(i,j,k,EFX_ID) = 0.0;
                phi(i,j,k,EFY_ID) = 0.0;
                phi(i,j,k,EFZ_ID) = 0.0;
                phi(i,j,k,POT_ID) = -0.5;
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
    //default to extrapolation
    for(int c=0;c<NVAR;c++)
    {
        s_ext[c]=s_int[c];
    }
}
#endif
