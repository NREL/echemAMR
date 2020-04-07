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
    for (int lev = 0; lev <= finest_level; ++lev) {
	dt_tmp[lev] = std::min(dt_tmp[lev], change_max*dt[lev]);
	n_factor *= nsubsteps[lev];
	dt_0 = std::min(dt_0, n_factor*dt_tmp[lev]);
    }

    // Limit dt's by the value of stop_time.
    const Real eps = 1.e-3*dt_0;
    if (t_new[0] + dt_0 > stop_time - eps) {
	dt_0 = stop_time - t_new[0];
    }

    dt[0] = dt_0;
    for (int lev = 1; lev <= finest_level; ++lev) {
	dt[lev] = dt[lev-1] / nsubsteps[lev];
    }
}

// compute dt from CFL considerations
Real echemAMR::EstTimeStep (int lev, bool local)
{
    BL_PROFILE("echemAMR::EstTimeStep()");

    Real dt_est = std::numeric_limits<Real>::max();

    const auto dx = geom[lev].CellSizeArray();
//    const Real* prob_lo = geom[lev].ProbLo();
    const Real cur_time = t_new[lev];
    MultiFab& S_new = phi_new[lev];

    MultiFab dcoeff(S_new.boxArray(), S_new.DistributionMap(), S_new.nComp(), 0);
#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) 
        {
            const Box& bx = mfi.tilebox();
            Array4<Real> statearray = S_new.array(mfi);
            Array4<Real> dcoeffarray = dcoeff.array(mfi);
            auto prob_lo = geom[lev].ProbLoArray();

            amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                    electrochem_transport::compute_dcoeff(i, j, k, statearray, dcoeffarray, prob_lo, 
                            dx, cur_time);
                    });

        }
    }


    Real maxdcoeff=dcoeff.norm0(0,0,true);
    for(int i=0;i<AMREX_SPACEDIM;i++)
    {
        dt_est = std::min(dt_est, (0.5/AMREX_SPACEDIM)*(dx[i]*dx[i])/maxdcoeff);
    }

    // Currently, this never happens (function called with local = true).
    // Reduction occurs outside this function.
    if (!local) {
        ParallelDescriptor::ReduceRealMin(dt_est);
    }

    dt_est *= cfl;
    return dt_est;
}
