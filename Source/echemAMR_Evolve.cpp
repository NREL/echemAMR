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
    constexpr int num_grow = 3; 

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

    // Build temporary multiFabs to work on.
    Array<MultiFab, AMREX_SPACEDIM> fluxcalc;
    Array<MultiFab, AMREX_SPACEDIM> facevel;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        BoxArray ba = amrex::convert(S_new.boxArray(), IntVect::TheDimensionVector(idim));

        fluxcalc[idim].define (ba,         S_new.DistributionMap(), S_new.nComp(), 0);
        facevel [idim].define (ba.grow(1), S_new.DistributionMap(),             1, 0);
    }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
	for (MFIter mfi(S_new,TilingIfNotGPU()); mfi.isValid(); ++mfi)
	{

        // ======== GET FACE VELOCITY =========
            GpuArray<Box, AMREX_SPACEDIM> nbx;
            AMREX_D_TERM(nbx[0] = mfi.nodaltilebox(0);,
                         nbx[1] = mfi.nodaltilebox(1);,
                         nbx[2] = mfi.nodaltilebox(2););

            AMREX_D_TERM(const Box& ngbxx = amrex::grow(mfi.nodaltilebox(0),1);,
                         const Box& ngbxy = amrex::grow(mfi.nodaltilebox(1),1);,
                         const Box& ngbxz = amrex::grow(mfi.nodaltilebox(2),1););

            GpuArray<Array4<Real>, AMREX_SPACEDIM> vel{ AMREX_D_DECL( facevel[0].array(mfi),
                                                                      facevel[1].array(mfi),
                                                                      facevel[2].array(mfi)) };

            const Box& psibox = Box(IntVect(AMREX_D_DECL(std::min(ngbxx.smallEnd(0)-1, ngbxy.smallEnd(0)-1),
                                                         std::min(ngbxx.smallEnd(1)-1, ngbxy.smallEnd(0)-1),
                                                         0)),
                                    IntVect(AMREX_D_DECL(std::max(ngbxx.bigEnd(0),   ngbxy.bigEnd(0)+1),
                                                         std::max(ngbxx.bigEnd(1)+1, ngbxy.bigEnd(1)),
                                                         0)));

            FArrayBox psifab(psibox, 1);
            Elixir psieli = psifab.elixir();
            Array4<Real> psi = psifab.array();
            GeometryData geomdata = geom[lev].data();
            auto prob_lo = geom[lev].ProbLoArray();
            auto dx = geom[lev].CellSizeArray();

            amrex::launch(psibox,
            [=] AMREX_GPU_DEVICE (const Box& tbx)
            {
                get_face_velocity_psi(tbx, ctr_time,
                                      psi, geomdata); 
            });

            AMREX_D_TERM(
                         amrex::ParallelFor(ngbxx,
                         [=] AMREX_GPU_DEVICE (int i, int j, int k)
                         {
                             get_face_velocity_x(i, j, k, vel[0], psi, prob_lo, dx); 
                         });,

                         amrex::ParallelFor(ngbxy,
                         [=] AMREX_GPU_DEVICE (int i, int j, int k)
                         {
                             get_face_velocity_y(i, j, k, vel[1], psi, prob_lo, dx);
                         });,

                         amrex::ParallelFor(ngbxz,
                         [=] AMREX_GPU_DEVICE (int i, int j, int k)
                         {
                             get_face_velocity_z(i, j, k, vel[2], psi, prob_lo, dx);
                         });
                        );

        // ======== FLUX CALC AND UPDATE =========

	    const Box& bx = mfi.tilebox();
            const Box& gbx = amrex::grow(bx, 1);

            Array4<Real> statein  = Sborder.array(mfi);
            Array4<Real> stateout = S_new.array(mfi);

            GpuArray<Array4<Real>, AMREX_SPACEDIM> flux{ AMREX_D_DECL(fluxcalc[0].array(mfi),
                                                                      fluxcalc[1].array(mfi),
                                                                      fluxcalc[2].array(mfi)) };

            AMREX_D_TERM(const Box& dqbxx = amrex::grow(bx, IntVect{2, 1, 1});,
                         const Box& dqbxy = amrex::grow(bx, IntVect{1, 2, 1});,
                         const Box& dqbxz = amrex::grow(bx, IntVect{1, 1, 2}););

            FArrayBox slope2fab (amrex::grow(bx, 2), 1);
            Elixir slope2eli = slope2fab.elixir();
            Array4<Real> slope2 = slope2fab.array();
            FArrayBox slope4fab (amrex::grow(bx, 1), 1);
            Elixir slope4eli = slope4fab.elixir();
            Array4<Real> slope4 = slope4fab.array();

            // compute longitudinal fluxes
            // ===========================

            // x -------------------------
            FArrayBox phixfab (gbx, 1);
            Elixir phixeli = phixfab.elixir();
            Array4<Real> phix = phixfab.array();

            amrex::launch(dqbxx,
            [=] AMREX_GPU_DEVICE (const Box& tbx)
            {
                slopex2(tbx, statein, slope2);
            });

            amrex::launch(gbx,
            [=] AMREX_GPU_DEVICE (const Box& tbx)
            {
                slopex4(tbx, statein, slope2, slope4);
            });

            amrex::ParallelFor(amrex::growLo(gbx, 0, -1),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                flux_x(i, j, k, statein, vel[0], phix, slope4, dtdx); 
            });

            // y -------------------------
            FArrayBox phiyfab (gbx, 1);
            Elixir phiyeli = phiyfab.elixir();
            Array4<Real> phiy = phiyfab.array();

            amrex::launch(dqbxy,
            [=] AMREX_GPU_DEVICE (const Box& tbx)
            {
                slopey2(tbx, statein, slope2);
            });

            amrex::launch(gbx,
            [=] AMREX_GPU_DEVICE (const Box& tbx)
            {
                slopey4(tbx, statein, slope2, slope4);
            });

            amrex::ParallelFor(amrex::growLo(gbx, 1, -1),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                flux_y(i, j, k, statein, vel[1], phiy, slope4, dtdx); 
            });

            // z -------------------------
            FArrayBox phizfab (gbx, 1);
            Elixir phizeli = phizfab.elixir();
            Array4<Real> phiz = phizfab.array();

            amrex::launch(dqbxz,
            [=] AMREX_GPU_DEVICE (const Box& tbx)
            {
                slopez2(tbx, statein, slope2);
            });

            amrex::launch(gbx,
            [=] AMREX_GPU_DEVICE (const Box& tbx)
            {
                slopez4(tbx, statein, slope2, slope4);
            });

            amrex::ParallelFor(amrex::growLo(gbx, 2, -1),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                flux_z(i, j, k, statein, vel[2], phiz, slope4, dtdx); 
            });

            // compute transverse fluxes
            // ===========================

            AMREX_D_TERM(const Box& gbxx = amrex::grow(bx, 0, 1);,
                         const Box& gbxy = amrex::grow(bx, 1, 1);,
                         const Box& gbxz = amrex::grow(bx, 2, 1););

            // xy & xz --------------------
            FArrayBox phix_yfab (gbx, 1);
            FArrayBox phix_zfab (gbx, 1);
            Elixir phix_yeli = phix_yfab.elixir();
            Elixir phix_zeli = phix_zfab.elixir();
            Array4<Real> phix_y = phix_yfab.array();
            Array4<Real> phix_z = phix_zfab.array();

            amrex::ParallelFor(amrex::growHi(gbxz, 0, 1),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                flux_xy(i, j, k, 
                        AMREX_D_DECL(vel[0], vel[1], vel[2]),
                        AMREX_D_DECL(phix, phiy, phiz),
                        phix_y, dtdx);
            }); 

            amrex::ParallelFor(amrex::growHi(gbxy, 0, 1),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                flux_xz(i, j, k,
                        AMREX_D_DECL(vel[0], vel[1], vel[2]),
                        AMREX_D_DECL(phix, phiy, phiz),
                        phix_z, dtdx);
            }); 

            // yz & yz --------------------
            FArrayBox phiy_xfab (gbx, 1);
            FArrayBox phiy_zfab (gbx, 1);
            Elixir phiy_xeli = phiy_xfab.elixir();
            Elixir phiy_zeli = phiy_zfab.elixir();
            Array4<Real> phiy_x = phiy_xfab.array();
            Array4<Real> phiy_z = phiy_zfab.array();

            amrex::ParallelFor(amrex::growHi(gbxz, 1, 1),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                flux_yx(i, j, k,
                        AMREX_D_DECL(vel[0], vel[1], vel[2]),
                        AMREX_D_DECL(phix, phiy, phiz),
                        phiy_x, dtdx);
            }); 

            amrex::ParallelFor(amrex::growHi(gbxx, 1, 1),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                flux_yz(i, j, k,
                        AMREX_D_DECL(vel[0], vel[1], vel[2]),
                        AMREX_D_DECL(phix, phiy, phiz),
                        phiy_z, dtdx);
            }); 

            // zx & zy --------------------
            FArrayBox phiz_xfab (gbx, 1);
            FArrayBox phiz_yfab (gbx, 1);
            Elixir phiz_xeli = phiz_xfab.elixir();
            Elixir phiz_yeli = phiz_yfab.elixir();
            Array4<Real> phiz_x = phiz_xfab.array();
            Array4<Real> phiz_y = phiz_yfab.array();

            amrex::ParallelFor(amrex::growHi(gbxy, 2, 1),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                flux_zx(i, j, k, 
                        AMREX_D_DECL(vel[0], vel[1], vel[2]),
                        AMREX_D_DECL(phix, phiy, phiz),
                        phiz_x, dtdx);
            }); 

            amrex::ParallelFor(amrex::growHi(gbxx, 2, 1),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                flux_zy(i, j, k,
                        AMREX_D_DECL(vel[0], vel[1], vel[2]),
                        AMREX_D_DECL(phix, phiy, phiz),
                        phiz_y, dtdx);
            }); 

            // final edge states 
            // ===========================
            amrex::ParallelFor(amrex::growHi(bx, 0, 1),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                combine_flux_x(i, j, k,
                               vel[0], vel[1], vel[2],
                               phix, phiy_z, phiz_y,
                               flux[0], dtdx);
            });

            amrex::ParallelFor(amrex::growHi(bx, 1, 1),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                combine_flux_y(i, j, k,
                               vel[0], vel[1], vel[2],
                               phiy, phix_z, phiz_x,
                               flux[1], dtdx);
            });

            amrex::ParallelFor(amrex::growHi(bx, 2, 1),
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                combine_flux_z(i, j, k,
                               vel[0], vel[1], vel[2],
                               phiz, phix_y, phiy_x,
                               flux[2], dtdx);
            });

            // compute new state (stateout) and scale fluxes based on face area.
            // ===========================

            // Do a conservative update 
            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                conservative(i, j, k,
                             statein, stateout,
                             AMREX_D_DECL(flux[0], flux[1], flux[2]),
                             dtdx);
            });

            // Scale by face area in order to correctly reflux
            AMREX_D_TERM(
                         amrex::ParallelFor(amrex::growHi(bx, 0, 1),
                         [=] AMREX_GPU_DEVICE (int i, int j, int k)
                         {
                             flux_scale_x(i, j, k, flux[0], dt_lev, dx);
                         });,
 
                         amrex::ParallelFor(amrex::growHi(bx, 1, 1),
                         [=] AMREX_GPU_DEVICE (int i, int j, int k)
                         {
                             flux_scale_y(i, j, k, flux[1], dt_lev, dx);
                         });,

                         amrex::ParallelFor(amrex::growHi(bx, 2, 1),
                         [=] AMREX_GPU_DEVICE (int i, int j, int k)
                         {
                             flux_scale_z(i, j, k, flux[2], dt_lev, dx);
                         });
                        );

            GpuArray<Array4<Real>, AMREX_SPACEDIM> fluxout{ AMREX_D_DECL(fluxes[0].array(mfi),
                                                                         fluxes[1].array(mfi),
                                                                         fluxes[2].array(mfi)) };
          
            if (do_reflux) {
                for (int idim = 0; idim < BL_SPACEDIM; ++idim) {
                    amrex::ParallelFor(nbx[idim],
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        fluxout[idim](i,j,k) = flux[idim](i,j,k);
                    });
                }
            }
        }
    }

    // ======== CFL CHECK, MOVED OUTSIDE MFITER LOOP =========

    AMREX_D_TERM(Real umax = facevel[0].norm0(0,0,false);,
                 Real vmax = facevel[1].norm0(0,0,false);,
                 Real wmax = facevel[2].norm0(0,0,false););

    if (AMREX_D_TERM(umax*dt_lev > dx[0], ||
                     vmax*dt_lev > dx[1], ||
                     wmax*dt_lev > dx[2]))
    {
        amrex::Print() << "umax = " << umax << ", vmax = " << vmax << ", wmax = " << wmax 
                       << ", dt = " << ctr_time << " dx = " << dx[1] << " " << dx[2] << " " << dx[3] << std::endl;
        amrex::Abort("CFL violation. use smaller adv.cfl.");
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
