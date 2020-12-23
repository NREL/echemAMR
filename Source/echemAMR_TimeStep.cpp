#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>
#include <Kernels_3d.H>

#ifdef AMREX_MEM_PROFILING
#include <AMReX_MemProfiler.H>
#endif

#include <echemAMR.H>
#include<Chemistry.H>
#include<Transport.H>

// a wrapper for EstTimeStep
void echemAMR::ComputeDt ()
{
    Vector<Real> dt_tmp(finest_level+1);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
	dt_tmp[lev] = EstTimeStep(lev, true);
    }
    ParallelDescriptor::ReduceRealMin(&dt_tmp[0], dt_tmp.size());

    constexpr Real change_max = 1.1;
    Real dt_0 = dt_tmp[0];
    int n_factor = 1;
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
	dt_tmp[lev] = std::min(dt_tmp[lev], change_max*dt[lev]);
	n_factor *= nsubsteps[lev];
	dt_0 = std::min(dt_0, n_factor*dt_tmp[lev]);
    }

    // Limit dt's by the value of stop_time.
    const Real eps = 1.e-3*dt_0;
    if (t_new[0] + dt_0 > stop_time - eps) 
    {
	dt_0 = stop_time - t_new[0];
    }

    dt[0] = dt_0;
    for (int lev = 1; lev <= finest_level; ++lev) 
    {
	dt[lev] = dt[lev-1] / nsubsteps[lev];
    }
}

// compute dt from CFL considerations
Real echemAMR::EstTimeStep (int lev, bool local)
{
    BL_PROFILE("echemAMR::EstTimeStep()");

    Real dt_est = std::numeric_limits<Real>::max();
    auto prob_lo = geom[lev].ProbLoArray();
    auto prob_hi = geom[lev].ProbLoArray();
    const auto dx = geom[lev].CellSizeArray();
    
    const Real cur_time = t_new[lev];
    MultiFab& S_new = phi_new[lev];

    MultiFab dcoeff(S_new.boxArray(), S_new.DistributionMap(), S_new.nComp(), 0);
    MultiFab vel(S_new.boxArray(), S_new.DistributionMap(), S_new.nComp(), 0);
    int ncomp=S_new.nComp();
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) 
        {
            const Box& bx = mfi.tilebox();
            Array4<Real> statearray = S_new.array(mfi);
            Array4<Real> dcoeffarray = dcoeff.array(mfi);

            FArrayBox velx_fab(bx,ncomp);
            FArrayBox vely_fab(bx,ncomp);
            FArrayBox velz_fab(bx,ncomp);
            Array4<Real> velxarray = velx_fab.array();
            Array4<Real> velyarray = vely_fab.array();
            Array4<Real> velzarray = velz_fab.array();
            Array4<Real> velarray  = vel.array(mfi);

            amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                electrochem_transport::compute_dcoeff(i, j, k, statearray, dcoeffarray, 
                        prob_lo, prob_hi,
                        dx, cur_time);
            });

            amrex::ParallelFor(bx,
                  [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                electrochem_transport::compute_vel(i, j, k, statearray, velxarray, velyarray, velzarray,
                        prob_lo, prob_hi, dx, cur_time);

                for(int comp=0;comp<ncomp;comp++)
                {
                    velarray(i,j,k,comp)=sqrt(velxarray(i,j,k,comp)*velxarray(i,j,k,comp) +
                            velyarray(i,j,k,comp)*velyarray(i,j,k,comp) +
                            velzarray(i,j,k,comp)*velzarray(i,j,k,comp));

                }
            });

        }
    }


    Real maxdcoeff=dcoeff.norm0(0,0,true);
    for(int comp=0;comp<ncomp;comp++)
    {
        Real diffcomp=dcoeff.norm0(comp,0,true);
        if(diffcomp > maxdcoeff)
        {
           maxdcoeff=diffcomp;
        }
    }
    for(int i=0;i<AMREX_SPACEDIM;i++)
    {
        dt_est = std::min(dt_est, (0.5/AMREX_SPACEDIM)*(dx[i]*dx[i])/maxdcoeff);
    }
    
    Real maxvel=vel.norm0(0,0,true);
    for(int comp=0;comp<ncomp;comp++)
    {
        Real velcomp=vel.norm0(comp,0,true);
        if(velcomp > maxvel)
        {
           maxvel=velcomp;
        }
    }
    if(maxvel > 0.0)
    {
        for(int i=0;i<AMREX_SPACEDIM;i++)
        {
            dt_est = std::min(dt_est, (dx[i]/maxvel));
        }
    }

    // Currently, this never happens (function called with local = true).
    // Reduction occurs outside this function.
    if (!local) 
    {
        ParallelDescriptor::ReduceRealMin(dt_est);
    }

    dt_est *= cfl;
    return dt_est;
}
