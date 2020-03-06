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
Real echemAMR::EstTimeStep (int lev, bool local) const
{
    BL_PROFILE("echemAMR::EstTimeStep()");

    Real dt_est = std::numeric_limits<Real>::max();

    const Real* dx = geom[lev].CellSize();
//    const Real* prob_lo = geom[lev].ProbLo();
    const Real cur_time = t_new[lev];
    const MultiFab& S_new = phi_new[lev];

    Array<MultiFab,AMREX_SPACEDIM> facevel;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        facevel[idim].define(amrex::convert(S_new.boxArray(), IntVect::TheDimensionVector(idim)),
                             S_new.DistributionMap(), 1, 0);
    }

#ifdef _OPENMP
#pragma omp parallel reduction(min:dt_est) if (Gpu::notInLaunchRegion())
#endif
    {
        // Calculate face velocities.
	for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
	{
            AMREX_D_TERM(const Box& nbxx = mfi.nodaltilebox(0);,
                         const Box& nbxy = mfi.nodaltilebox(1);,
                         const Box& nbxz = mfi.nodaltilebox(2););

            GpuArray<Array4<Real>, AMREX_SPACEDIM> vel { AMREX_D_DECL(facevel[0].array(mfi),
                                                                      facevel[1].array(mfi),
                                                                      facevel[2].array(mfi)) };

            const Box& psibox = Box(IntVect(AMREX_D_DECL(std::min(nbxx.smallEnd(0)-1, nbxy.smallEnd(0)-1),
                                                         std::min(nbxx.smallEnd(1)-1, nbxy.smallEnd(0)-1),
                                                         0)),
                                    IntVect(AMREX_D_DECL(std::max(nbxx.bigEnd(0),     nbxy.bigEnd(0)+1),
                                                         std::max(nbxx.bigEnd(1)+1,   nbxy.bigEnd(1)),
                                                         0)));

            FArrayBox psifab(psibox, 1);
            Elixir psieli = psifab.elixir();
            Array4<Real> psi = psifab.array();
            GeometryData geomdata = geom[lev].data();
            auto prob_lo = geom[lev].ProbLoArray();
            auto dx = geom[lev].CellSizeArray();

            amrex::launch(psibox, 
            [=] AMREX_GPU_DEVICE (Box const& tbx)
            {
                get_face_velocity_psi(tbx, cur_time, psi, geomdata); 
            });

            AMREX_D_TERM(
                         amrex::ParallelFor(nbxx,
                         [=] AMREX_GPU_DEVICE (int i, int j, int k)
                         {
                             get_face_velocity_x(i, j, k, vel[0], psi, prob_lo, dx); 
                         });,

                         amrex::ParallelFor(nbxy,
                         [=] AMREX_GPU_DEVICE (int i, int j, int k)
                         {
                             get_face_velocity_y(i, j, k, vel[1], psi, prob_lo, dx);
                         });,

                         amrex::ParallelFor(nbxz,
                         [=] AMREX_GPU_DEVICE (int i, int j, int k)
                         {
                             get_face_velocity_z(i, j, k, vel[2], psi, prob_lo, dx);
                         });
                        );

	}
    }

    for (int i=0; i<BL_SPACEDIM; ++i)
    {
        Real est = facevel[i].norm0(0,0,true);
        dt_est = std::min(dt_est, dx[i]/est);
    }

    // Currently, this never happens (function called with local = true).
    // Reduction occurs outside this function.
    if (!local) {
	ParallelDescriptor::ReduceRealMin(dt_est);
    }

    dt_est *= cfl;

    return dt_est;
}
