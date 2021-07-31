#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <global_defines.H>

using namespace amrex;

AMREX_GPU_DEVICE AMREX_INLINE Real bvcurrent(Real phi_jump)
{
    Real ocp=0.2;
    Real j0=1.0;
    Real phi0=1.0;
    Real jbv;

    //sinh definition
    jbv=-j0*sinh((phi_jump+ocp)/phi0);

    //linear case
    //jbv=-(j0/phi0)*(phi_jump+ocp);
    return(jbv);
}

AMREX_GPU_DEVICE AMREX_INLINE Real bvcurrent_der(Real phi_jump)
{
    Real ocp=0.2;
    Real j0=1.0;
    Real phi0=1.0;
    Real jbvder;

    //sinh definition
    jbvder=-j0*cosh((phi_jump+ocp)/phi0)*(1.0/phi0);

    //linear case
    //jbvder=-(j0/phi0);
    return(jbvder);
}

AMREX_GPU_DEVICE AMREX_INLINE Real bv_activation_function(Real c, Real gradc, Real gradc_cutoff)
{
    Real f;
    Real maxval=0.25;
    Real tol=1e-1;

    //f=(gradc > gradc_cutoff)?1.0:0.0;
    f=(c*(1.0-c) > tol*maxval)?1.0:0.0;
    return(f);
}

void initlevset(MultiFab &levset,Geometry geom)
{
        Real sphere_rad=0.3;
        Real sharpness_factor=100.0;

        ParmParse pp;
        pp.query("sphere_rad",sphere_rad);
        pp.query("sharpness",sharpness_factor);

        for (MFIter mfi(levset); mfi.isValid(); ++mfi)
        {
            const auto dx = geom.CellSizeArray();
            const Box& bx = mfi.tilebox();
            const Box& gbx = amrex::grow(bx,1);
            auto prblo = geom.ProbLoArray();
            auto prbhi = geom.ProbHiArray();

            Array4<Real> ls_arr = levset.array(mfi);

            amrex::ParallelFor(gbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                    Real x[AMREX_SPACEDIM],dist;
                    x[0] = prblo[0] + (i+0.5)*dx[0];    
                    x[1] = prblo[1] + (j+0.5)*dx[1];    
                    x[2] = prblo[2] + (k+0.5)*dx[2];

                    Real center[AMREX_SPACEDIM];
                    center[0]=0.5*(prblo[0]+prbhi[0]);
                    center[1]=0.5*(prblo[1]+prbhi[1]);
                    center[2]=0.5*(prblo[2]+prbhi[2]);

                    dist=sqrt( (x[0]-center[0])*(x[0]-center[0]) +
                               (x[1]-center[1])*(x[1]-center[1]) +
                               (x[2]-center[2])*(x[2]-center[2]) );

                    ls_arr(i,j,k) = 0.5*(1.0 + tanh(sharpness_factor*(dist-sphere_rad)));

                    });

        }
}

void setbc(MultiFab &phi_bc,MultiFab &phi, Geometry geom,
        amrex::Vector<int> bc_lo,amrex::Vector<int> bc_hi)
{
    Real leftdircval;
    Real rightdircval;

    ParmParse pp;
    amrex::MultiFab::Copy(phi_bc,phi,0,0,1,0);

    const int* domloarr = geom.Domain().loVect();
    const int* domhiarr = geom.Domain().hiVect();

    GpuArray<int,AMREX_SPACEDIM> domlo={domloarr[0],domloarr[1],domloarr[2]};
    GpuArray<int,AMREX_SPACEDIM> domhi={domhiarr[0],domhiarr[1],domhiarr[2]};

    for (MFIter mfi(phi); mfi.isValid(); ++mfi)
    {
        const auto dx = geom.CellSizeArray();
        const Box& bx = mfi.tilebox();
        const Box& gbx = amrex::grow(bx,1);
        auto prblo = geom.ProbLoArray();
        auto prbhi = geom.ProbHiArray();
        const Box& domain = geom.Domain();

        Array4<Real> phibc_arr=phi_bc.array(mfi);

        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i,int j,int k)
        {
            IntVect cellid(i,j,k);
            IntVect gcellid(i,j,k);

            if(cellid[XDIR]==domlo[XDIR])
            {
                gcellid[XDIR] -=1;
                if(bc_lo[XDIR]==DIRICHLET_ID)
                {
                    phibc_arr(gcellid) = 0.0;
                }
            }
            if(cellid[XDIR]==domhi[XDIR])
            {
                gcellid[XDIR] += 1;
                if(bc_hi[XDIR]==DIRICHLET_ID)
                {
                    phibc_arr(gcellid) = 0.0;
                }
            }

            if(cellid[YDIR]==domlo[YDIR])
            {
                gcellid[YDIR] -=1;
                if(bc_lo[YDIR]==DIRICHLET_ID)
                {
                    phibc_arr(gcellid) = 0.0;
                }
            }
            if(cellid[YDIR]==domhi[YDIR])
            {
                gcellid[YDIR] += 1;
                if(bc_hi[YDIR]==DIRICHLET_ID)
                {
                    phibc_arr(gcellid) = 0.0;
                }
            }
            
            if(cellid[ZDIR]==domlo[ZDIR])
            {
                gcellid[ZDIR] -=1;
                if(bc_lo[ZDIR]==DIRICHLET_ID)
                {
                    phibc_arr(gcellid) = 0.0;
                }
            }
            if(cellid[ZDIR]==domhi[ZDIR])
            {
                gcellid[ZDIR] += 1;
                if(bc_hi[ZDIR]==DIRICHLET_ID)
                {
                    phibc_arr(gcellid) = 0.0;
                }
            }
        });
    }
}
