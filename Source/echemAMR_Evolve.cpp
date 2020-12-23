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
#include<Reactions.H>

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
	for (lev = 0; lev <= finest_level; ++lev) 
        {
	    t_new[lev] = cur_time;
	}

	if (plot_int > 0 && (step+1) % plot_int == 0) 
        {
	    last_plot_file_step = step+1;
	    WritePlotFile();
	}

        if (chk_int > 0 && (step+1) % chk_int == 0) 
        {
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

    if (plot_int > 0 && istep[0] > last_plot_file_step) 
    {
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
		for (int k = lev; k <= finest_level; ++k) 
                {
		    last_regrid_step[k] = istep[k];
		}

                // if there are newly created levels, set the time step
		for (int k = old_finest+1; k <= finest_level; ++k) 
                {
		    dt[k] = dt[k-1] / MaxRefRatio(k-1);
		}
	    }
	}
    }

    if (Verbose()) 
    {
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
void echemAMR::Advance(int lev, Real time, Real dt_lev,int iteration,int ncycle)
{
    constexpr int num_grow = 2; 
    std::swap(phi_old[lev], phi_new[lev]); //old becomes new and new becomes old
    t_old[lev] = t_new[lev];  //old time is now current time (time)
    t_new[lev] += dt_lev;     //new time is ahead
    MultiFab& S_new = phi_new[lev];  //this is the old value, beware!
    MultiFab& S_old = phi_old[lev]; //current value

    // State with ghost cells
    MultiFab Sborder(grids[lev], dmap[lev], S_new.nComp(), num_grow);
    //source term 
    MultiFab dsdt(grids[lev],dmap[lev],S_new.nComp(),0);

    //stage 1
    
    //time is current time which is t_old
    FillPatch(lev, time, Sborder, 0, Sborder.nComp()); 
    //compute dsdt for 1/2 timestep
    compute_dsdt(lev, num_grow, Sborder, dsdt, time, 0.5*dt_lev, false); 
    //S_new=S_old+0.5*dt*dsdt //sold is the current value
    MultiFab::LinComb(S_new, 1.0, Sborder, 0, 0.5*dt_lev, dsdt, 0, 0, S_new.nComp(), 0);

    //stage 2
    
    //time+dt_lev lets me pick S_new for sborder
    FillPatch(lev, time+dt_lev, Sborder, 0, Sborder.nComp()); 
    //dsdt for full time-step
    compute_dsdt(lev, num_grow, Sborder, dsdt, time+0.5*dt_lev, dt_lev, true); 
    //S_new=S_old+dt*dsdt
    MultiFab::LinComb(S_new, 1.0, S_old, 0, dt_lev, dsdt, 0, 0, S_new.nComp(), 0);


}

// advance a single level for a single time step, updates flux registers
void echemAMR::compute_dsdt (int lev, const int num_grow, 
        MultiFab &Sborder, MultiFab &dsdt, Real time, Real dt, bool reflux_this_stage)
{

    const auto dx = geom[lev].CellSizeArray();
    auto prob_lo = geom[lev].ProbLoArray();
    auto prob_hi = geom[lev].ProbLoArray();

    int ncomp=Sborder.nComp();

    // Build temporary multiFabs to work on.
    Array<MultiFab, AMREX_SPACEDIM> flux;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) 
    {
        BoxArray ba = amrex::convert(dsdt.boxArray(), IntVect::TheDimensionVector(idim));
        flux[idim].define (ba,dsdt.DistributionMap(), ncomp, 0);
    }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
	for (MFIter mfi(dsdt,TilingIfNotGPU()); mfi.isValid(); ++mfi)
	{
	    const Box& bx = mfi.tilebox();
            const Box& gbx = amrex::grow(bx,num_grow);

            FArrayBox dcoeff_fab(gbx,ncomp);
            FArrayBox velx_fab(gbx,ncomp);
            FArrayBox vely_fab(gbx,ncomp);
            FArrayBox velz_fab(gbx,ncomp);
            FArrayBox reactsource_fab(bx,ncomp);
            
            Elixir dcoeff_fab_eli=dcoeff_fab.elixir();
            Elixir velx_fab_eli=velx_fab.elixir();
            Elixir vely_fab_eli=vely_fab.elixir();
            Elixir velz_fab_eli=velz_fab.elixir();
            Elixir reactsource_fab_eli=reactsource_fab.elixir();

            GpuArray<Box, AMREX_SPACEDIM> nbx;
            AMREX_D_TERM(nbx[0] = mfi.nodaltilebox(0);,
                         nbx[1] = mfi.nodaltilebox(1);,
                         nbx[2] = mfi.nodaltilebox(2););

            Array4<Real> sborder_arr  = Sborder.array(mfi);
            Array4<Real> dcoeff_arr   = dcoeff_fab.array();
            Array4<Real> dsdt_arr     = dsdt.array(mfi);
            Array4<Real> reactsource_arr = reactsource_fab.array();
            Array4<Real> velx_arr      = velx_fab.array();
            Array4<Real> vely_arr      = vely_fab.array();
            Array4<Real> velz_arr      = velz_fab.array();

            GpuArray<Array4<Real>, AMREX_SPACEDIM> flux_arr{ AMREX_D_DECL(flux[0].array(mfi),
                                                                          flux[1].array(mfi),
                                                                          flux[2].array(mfi)) };
            
            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                electrochem_transport::compute_vel(i, j, k, sborder_arr, velx_arr, 
                        vely_arr, velz_arr, prob_lo, prob_hi, dx, time);
            });

            
            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                electrochem_transport::compute_dcoeff(i, j, k, sborder_arr, 
                        dcoeff_arr, prob_lo, prob_hi, dx, time);
            });
            
            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                electrochem_reactions::compute_react_source(i, j, k, sborder_arr, reactsource_arr, 
                        prob_lo, prob_hi, dx, time);
            });

            amrex::ParallelFor(amrex::growHi(bx,0,1),ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                compute_flux_x(i, j, k, n, sborder_arr, velx_arr, dcoeff_arr, flux_arr[0], dx);
            });

            amrex::ParallelFor(amrex::growHi(bx,1,1),ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                compute_flux_y(i, j, k, n, sborder_arr, vely_arr, dcoeff_arr, flux_arr[1], dx);
            });

            amrex::ParallelFor(amrex::growHi(bx,2,1),ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                compute_flux_z(i, j, k, n, sborder_arr, velz_arr, dcoeff_arr, flux_arr[2], dx);
            });

            // update residual
            amrex::ParallelFor(bx,ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k,int n)
            {
                update_residual(i, j, k, n, dsdt_arr,reactsource_arr,
                        AMREX_D_DECL(flux_arr[0], flux_arr[1], flux_arr[2]),
                             dx);
            });
        }
    }

    if (do_reflux and reflux_this_stage) 
    { 
        if (flux_reg[lev+1]) 
        {
            for (int i = 0; i < BL_SPACEDIM; ++i) 
            {
                // update the lev+1/lev flux register (index lev+1)   
                const Real dA = (i == 0) ? dx[1]*dx[2] 
                    : ((i == 1) ? dx[0]*dx[2] : dx[0]*dx[1]);
                const Real scale = -dt*dA;
                flux_reg[lev+1]->CrseInit(flux[i], i, 0, 0, ncomp, scale);
            }	    
        }
        if (flux_reg[lev]) 
        {
            for (int i = 0; i < BL_SPACEDIM; ++i) 
            {
                // update the lev/lev-1 flux register (index lev) 
                const Real dA = (i == 0) ? dx[1]*dx[2] 
                    : ((i == 1) ? dx[0]*dx[2] : dx[0]*dx[1]);
                const Real scale = dt*dA;
                flux_reg[lev]->FineAdd(flux[i], i, 0, 0, ncomp, scale);
            }
        }
    }
}
