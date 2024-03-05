#ifndef _PROB_H_
#define _PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <Chemistry.H>
#include <ProbParm.H>

using namespace amrex;

AMREX_INLINE
void amrex_probinit (ProbParm & h_prob_parm, ProbParm & d_prob_parm)
{
    // read problem specific parmparse parameters here
    amrex::ParmParse pp("prob");
    pp.query("angle", h_prob_parm.angle );
    pp.query("spltcell_x",h_prob_parm.spltcell_x);
    pp.query("spltcell_y",h_prob_parm.spltcell_y);
    pp.query("spltcell_z",h_prob_parm.spltcell_z);
    pp.query("f0",h_prob_parm.f0);

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

    amrex::Real x0=prob_lo[0]+prob_parm->f0*(prob_hi[0]-prob_lo[0]);
    amrex::Real y0=prob_lo[1];
    
    amrex::Real x1=prob_lo[0]+(1.0-prob_parm->f0)*(prob_hi[0]-prob_lo[0]);
    amrex::Real y1=prob_hi[1];

    amrex::Real tan_theta=std::tan(prob_parm->angle*3.14159265359/180.0);
    amrex::Real plo_sgn=std::copysign(1.0,(prob_lo[1]-y0)+tan_theta*(prob_lo[0]-x0));
    amrex::Real phi_sgn=std::copysign(1.0,(prob_hi[1]-y1)+tan_theta*(prob_hi[0]-x1));
    int spltcell[3]={prob_parm->spltcell_x,prob_parm->spltcell_y,prob_parm->spltcell_z};


#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (GPU::notInLaunchRegion)
#endif
    for (int k = lo.z; k <= hi.z; ++k) 
    {
        Real z = prob_lo[2] + (0.5+k) * dx[2];
        for (int j = lo.y; j <= hi.y; ++j) 
        {
            Real y = prob_lo[1] + (0.5+j) * dx[1];
            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i) 
            {
                Real x = prob_lo[0] + (0.5+i) * dx[0];
                phi(i,j,k,LS_ID)=0.0;
                
                for(int kk=0;kk<spltcell[2];kk++)
                {
                    for(int jj=0;jj<spltcell[1];jj++)
                    {
                        for(int ii=0;ii<spltcell[0];ii++)
                        {
                            Real xx= prob_lo[0]+(i+(ii+0.5)/spltcell[0])*dx[0];
                            Real yy= prob_lo[1]+(j+(jj+0.5)/spltcell[1])*dx[1];

                            if( (std::copysign(1.0,(yy-y0)+tan_theta*(xx-x0))*plo_sgn > 0.0) 
                             || (std::copysign(1.0,(yy-y1)+tan_theta*(xx-x1))*phi_sgn > 0.0) )
                            {
                                phi(i,j,k,LS_ID)+=1.0;
                            }
                        }
                    }
                }
                
                phi(i,j,k,LS_ID)=phi(i,j,k,LS_ID)/Real(spltcell[0]*spltcell[1]*spltcell[2]);

                phi(i,j,k,S1_ID)   = phi(i,j,k,LS_ID);
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
