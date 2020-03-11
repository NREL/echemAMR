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

// advance solution to final time
void echemAMR::Evolve ()
{
    Real cur_time = t_new[0];
    int last_plot_file_step = 0;

    for (int step = istep[0]; step < max_step && cur_time < stop_time; ++step)
    {
        amrex::Print() << "\nCoarse STEP " << step+1 << " starts ..." << std::endl;

	ComputeDt();

	int lev = 0;
	int iteration = 1;
	timeStep(lev, cur_time, iteration);

	cur_time += dt[0];

        amrex::Print() << "Coarse STEP " << step+1 << " ends." << " TIME = " << cur_time
                       << " DT = " << dt[0]  << std::endl;

	// sync up time
	for (lev = 0; lev <= finest_level; ++lev) {
	    t_new[lev] = cur_time;
	}

	if (plot_int > 0 && (step+1) % plot_int == 0) {
	    last_plot_file_step = step+1;
	    WritePlotFile();
	}

        if (chk_int > 0 && (step+1) % chk_int == 0) {
            WriteCheckpointFile();
        }

#ifdef AMREX_MEM_PROFILING
        {
            std::ostringstream ss;
            ss << "[STEP " << step+1 << "]";
            MemProfiler::report(ss.str());
        }
#endif

	if (cur_time >= stop_time - 1.e-6*dt[0]) break;
    }

    if (plot_int > 0 && istep[0] > last_plot_file_step) {
	WritePlotFile();
    }

}

// advance a level by dt
// includes a recursive call for finer levels
void echemAMR::timeStep (int lev, Real time, int iteration)
{
    if (regrid_int > 0)  // We may need to regrid
    {

        // help keep track of whether a level was already regridded
        // from a coarser level call to regrid
        static Vector<int> last_regrid_step(max_level+1, 0);

        // regrid changes level "lev+1" so we don't regrid on max_level
        // also make sure we don't regrid fine levels again if 
        // it was taken care of during a coarser regrid
        if (lev < max_level && istep[lev] > last_regrid_step[lev]) 
        {
            if (istep[lev] % regrid_int == 0)
            {
                // regrid could add newly refine levels (if finest_level < max_level)
                // so we save the previous finest level index
		int old_finest = finest_level; 
		regrid(lev, time);

                // mark that we have regridded this level already
		for (int k = lev; k <= finest_level; ++k) {
		    last_regrid_step[k] = istep[k];
		}

                // if there are newly created levels, set the time step
		for (int k = old_finest+1; k <= finest_level; ++k) {
		    dt[k] = dt[k-1] / MaxRefRatio(k-1);
		}
	    }
	}
    }

    if (Verbose()) {
	amrex::Print() << "[Level " << lev << " step " << istep[lev]+1 << "] ";
	amrex::Print() << "ADVANCE with time = " << t_new[lev] 
                       << " dt = " << dt[lev] << std::endl;
    }

    // advance a single level for a single time step, updates flux registers
    Advance(lev, time, dt[lev], iteration, nsubsteps[lev]);

    ++istep[lev];

    if (Verbose())
    {
	amrex::Print() << "[Level " << lev << " step " << istep[lev] << "] ";
        amrex::Print() << "Advanced " << CountCells(lev) << " cells" << std::endl;
    }

    if (lev < finest_level)
    {
        // recursive call for next-finer level
	for (int i = 1; i <= nsubsteps[lev+1]; ++i)
	{
	    timeStep(lev+1, time+(i-1)*dt[lev+1], i);
	}

	if (do_reflux)
	{
            // update lev based on coarse-fine flux mismatch
	    flux_reg[lev+1]->Reflux(phi_new[lev], 1.0, 0, 0, phi_new[lev].nComp(), geom[lev]);
	}

	AverageDownTo(lev); // average lev+1 down to lev
    }
    
}

// advance a single level for a single time step, updates flux registers
void echemAMR::Advance (int lev, Real time, Real dt_lev, int iteration, int ncycle)
{
    constexpr int num_grow = 1; 

    std::swap(phi_old[lev], phi_new[lev]);
    t_old[lev] = t_new[lev];
    t_new[lev] += dt_lev;

    MultiFab& S_new = phi_new[lev];

    const Real old_time = t_old[lev];
    const Real new_time = t_new[lev];
    const Real ctr_time = 0.5*(old_time+new_time);

    const auto dx = geom[lev].CellSizeArray();
    GpuArray<Real, AMREX_SPACEDIM> dtdx;
    for (int i=0; i<AMREX_SPACEDIM; ++i)
    {
        dtdx[i] = dt_lev/(dx[i]);
    }

    const Real* prob_lo = geom[lev].ProbLo();

    MultiFab fluxes[BL_SPACEDIM];
    if (do_reflux)
    {
	for (int i = 0; i < BL_SPACEDIM; ++i)
	{
	    BoxArray ba = grids[lev];
	    ba.surroundingNodes(i);
	    fluxes[i].define(ba, dmap[lev], S_new.nComp(), 0);
	}
    }

    // State with ghost cells
    MultiFab Sborder(grids[lev], dmap[lev], S_new.nComp(), num_grow);
    FillPatch(lev, time, Sborder, 0, Sborder.nComp());


/*
    // Allocate fabs for fluxes and Godunov velocities. (Kept for reference).
    for (int i = 0; i < BL_SPACEDIM ; i++) {
	const Box& bxtmp = amrex::surroundingNodes(bx,i);
	flux[i].resize(bxtmp,S_new.nComp());
	uface[i].resize(amrex::grow(bxtmp,1),1);
    }
*/
    int ncomp=S_new.nComp();

    // Build temporary multiFabs to work on.
    Array<MultiFab, AMREX_SPACEDIM> fluxcalc;
    MultiFab dcoeff(grids[lev], dmap[lev], S_new.nComp(), num_grow);

    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        BoxArray ba = amrex::convert(S_new.boxArray(), IntVect::TheDimensionVector(idim));

        fluxcalc[idim].define (ba,S_new.DistributionMap(), S_new.nComp(), 0);
    }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
	for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
	{
	    const Box& bx = mfi.tilebox();
            const Box& gbx = amrex::grow(bx,num_grow);

            GpuArray<Box, AMREX_SPACEDIM> nbx;
            AMREX_D_TERM(nbx[0] = mfi.nodaltilebox(0);,
                         nbx[1] = mfi.nodaltilebox(1);,
                         nbx[2] = mfi.nodaltilebox(2););

            Array4<Real> statein  = Sborder.array(mfi);
            Array4<Real> stateout = S_new.array(mfi);
            Array4<Real> diffcoeff = dcoeff.array(mfi);

            GpuArray<Array4<Real>, AMREX_SPACEDIM> flux{ AMREX_D_DECL(fluxcalc[0].array(mfi),
                                                                      fluxcalc[1].array(mfi),
                                                                      fluxcalc[2].array(mfi)) };
            auto prob_lo = geom[lev].ProbLoArray();

            
            amrex::ParallelFor(gbx,ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                compute_dcoeff(i, j, k, n, statein, diffcoeff, prob_lo, dx, new_time);
            });


            amrex::ParallelFor(amrex::growHi(bx,0,1),ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                compute_flux_x(i, j, k, n, statein, diffcoeff, flux[0], dx);
            });

            amrex::ParallelFor(amrex::growHi(bx,1,1),ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                compute_flux_y(i, j, k, n, statein, diffcoeff, flux[1], dx);
            });

            amrex::ParallelFor(amrex::growHi(bx,2,1),ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                compute_flux_z(i, j, k, n, statein, diffcoeff, flux[2], dx);
            });

            // compute new state (stateout) and scale fluxes based on face area.
            // ===========================

            // Do a conservative update 
            amrex::ParallelFor(bx,ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k,int n)
            {
                conservative(i, j, k, n,
                             statein, stateout,
                             AMREX_D_DECL(flux[0], flux[1], flux[2]),
                             dtdx);
            });

            amrex::ParallelFor(amrex::growHi(bx, 0, 1),ncomp,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k,int n)
                    {
                        flux_scale_x(i, j, k, n, flux[0], dt_lev, dx);
                    });

            amrex::ParallelFor(amrex::growHi(bx, 1, 1),ncomp,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k,int n)
                    {
                        flux_scale_y(i, j, k, n, flux[1], dt_lev, dx);
                    });

            amrex::ParallelFor(amrex::growHi(bx, 2, 1),ncomp,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k,int n)
                    {
                        flux_scale_z(i, j, k, n, flux[2], dt_lev, dx);
                    });

            GpuArray<Array4<Real>, AMREX_SPACEDIM> fluxout{ AMREX_D_DECL(fluxes[0].array(mfi),
                    fluxes[1].array(mfi),
                    fluxes[2].array(mfi)) };

            if (do_reflux) 
            {
                for (int idim = 0; idim < BL_SPACEDIM; ++idim) 
                {
                    amrex::ParallelFor(nbx[idim],ncomp,
                            [=] AMREX_GPU_DEVICE (int i, int j, int k,int n)
                    {
                        fluxout[idim](i,j,k,n) = flux[idim](i,j,k,n);
                    });
                }
            }
        }
    }

    // ======== END OF GPU EDIT, (FOR NOW) =========

    // increment or decrement the flux registers by area and time-weighted fluxes
    // Note that the fluxes have already been scaled by dt and area
    // In this example we are solving phi_t = -div(+F)
    // The fluxes contain, e.g., F_{i+1/2,j} = (phi*u)_{i+1/2,j}
    // Keep this in mind when considering the different sign convention for updating
    // the flux registers from the coarse or fine grid perspective
    // NOTE: the flux register associated with flux_reg[lev] is associated
    // with the lev/lev-1 interface (and has grid spacing associated with lev-1)
    if (do_reflux) { 
        if (flux_reg[lev+1]) {
            for (int i = 0; i < BL_SPACEDIM; ++i) {
                // update the lev+1/lev flux register (index lev+1)   
                flux_reg[lev+1]->CrseInit(fluxes[i],i,0,0,fluxes[i].nComp(), -1.0);
            }	    
        }
        if (flux_reg[lev]) {
            for (int i = 0; i < BL_SPACEDIM; ++i) {
                // update the lev/lev-1 flux register (index lev) 
                flux_reg[lev]->FineAdd(fluxes[i],i,0,0,fluxes[i].nComp(), 1.0);
            }
        }
    }
}
