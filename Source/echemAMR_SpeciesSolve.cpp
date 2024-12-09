#include <echemAMR.H>
#include <Chemistry.H>
#include <Transport.H>
#include <Reactions.H>
#include <bv_utils.H>
#include <compute_flux.H>

// advance a level by dt
// includes a recursive call for finer levels
void echemAMR::timeStep(int lev, Real time, int iteration)
{
    if (regrid_int > 0) // We may need to regrid
    {

        // help keep track of whether a level was already regridded
        // from a coarser level call to regrid
        static Vector<int> last_regrid_step(max_level + 1, 0);

        // regrid changes level "lev+1" so we don't regrid on max_level
        // also make sure we don't regrid fine levels again if
        // it was taken care of during a coarser regrid
        if (lev < max_level && istep[lev] > last_regrid_step[lev])
        {
            if (istep[lev] % regrid_int == 0)
            {
                // regrid could add newly refine levels 
                // (if finest_level < max_level)
                // so we save the previous finest level index
                int old_finest = finest_level;
                regrid(lev, time);

                // mark that we have regridded this level already
                for (int k = lev; k <= finest_level; ++k)
                {
                    last_regrid_step[k] = istep[k];
                }

                // if there are newly created levels, set the time step
                for (int k = old_finest + 1; k <= finest_level; ++k)
                {
                    dt[k] = dt[k - 1] / MaxRefRatio(k - 1);
                }
            }
        }
    }

    if (Verbose())
    {
        amrex::Print() << "[Level " << lev << " step " 
        << istep[lev] + 1 << "] ";
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
        for (int i = 1; i <= nsubsteps[lev + 1]; ++i)
        {
            timeStep(lev + 1, time + (i - 1) * dt[lev + 1], i);
        }

        if (do_reflux)
        {
            // update lev based on coarse-fine flux mismatch
            flux_reg[lev + 1]->Reflux(phi_new[lev], 1.0, 0, 
                                      0, phi_new[lev].nComp(), geom[lev]);
        }

        AverageDownTo(lev); // average lev+1 down to lev
    }
}
void echemAMR::Advance(int lev, Real time, Real dt_lev, int iteration, int ncycle)
{
    constexpr int num_grow = 2;
    std::swap(phi_old[lev], phi_new[lev]); // old becomes new and new becomes old
    t_old[lev] = t_new[lev];               // old time is now current time (time)
    t_new[lev] += dt_lev;                  // new time is ahead
    MultiFab& S_new = phi_new[lev];        // this is the old value, beware!
    MultiFab& S_old = phi_old[lev];        // current value

    // Build flux multiFabs to work on.
    Array<MultiFab, AMREX_SPACEDIM> flux;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        BoxArray ba = amrex::convert(grids[lev], IntVect::TheDimensionVector(idim));
        flux[idim].define(ba, dmap[lev], S_new.nComp(), 0);
        flux[idim].setVal(0.0);
    }

    // State with ghost cells
    MultiFab Sborder(grids[lev], dmap[lev], S_new.nComp(), num_grow);
    // source term
    MultiFab dsdt(grids[lev], dmap[lev], S_new.nComp(), 0);
    dsdt.setVal(0.0);

    // stage 1
    // time is current time which is t_old, so phi_old is picked up!
    // but wait, we swapped phi_old and new, so we get phi_new here
    FillPatch(lev, time, Sborder, 0, Sborder.nComp());
    // compute dsdt for 1/2 timestep
    compute_fluxes(lev, num_grow, Sborder, flux, time);
    compute_dsdt(lev,num_grow, Sborder, flux, dsdt, time, 0.5 * dt_lev, false);
    // S_new=S_old+0.5*dt*dsdt //sold is the current value
    MultiFab::LinComb(S_new, 1.0, Sborder, 0, 0.5 * dt_lev, dsdt, 0, 0, S_new.nComp(), 0);

    // stage 2
    // time+dt_lev lets me pick S_new for sborder
    FillPatch(lev, time + dt_lev, Sborder, 0, Sborder.nComp());
    // dsdt for full time-step
    dsdt.setVal(0.0);
    compute_fluxes(lev, num_grow, Sborder, flux, time);
    compute_dsdt(lev,num_grow, Sborder, flux, dsdt, time + 0.5*dt_lev, dt_lev, true);
    // S_new=S_old+dt*dsdt
    MultiFab::LinComb(S_new, 1.0, S_old, 0, dt_lev, dsdt, 0, 0, S_new.nComp(), 0);
}

// to account for nano divide dsdt by NP_ID
void echemAMR::compute_dsdt(int lev, const int num_grow, MultiFab& Sborder, 
                            Array<MultiFab,AMREX_SPACEDIM>& flux, MultiFab& dsdt,
                            Real time, Real delt, bool reflux_this_stage)
{
    const auto dx = geom[lev].CellSizeArray();
    auto prob_lo = geom[lev].ProbLoArray();
    auto prob_hi = geom[lev].ProbHiArray();
    ProbParm const* localprobparm = d_prob_parm;

    int ncomp = Sborder.nComp();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(dsdt, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const Box& gbx = amrex::grow(bx, 1);
            FArrayBox reactsource_fab(bx, ncomp);

            Elixir reactsource_fab_eli = reactsource_fab.elixir();

            Array4<Real> sborder_arr = Sborder.array(mfi);
            Array4<Real> dsdt_arr = dsdt.array(mfi);
            Array4<Real> reactsource_arr = reactsource_fab.array();

            GpuArray<Array4<Real>, AMREX_SPACEDIM> flux_arr{
                AMREX_D_DECL(flux[0].array(mfi), flux[1].array(mfi), flux[2].array(mfi))};

            reactsource_fab.setVal<RunOn::Device>(0.0);

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                electrochem_reactions::compute_react_source(i, j, k, sborder_arr, 
                                                            reactsource_arr, prob_lo, 
                                                            prob_hi, dx, time, *localprobparm);
            });

            // update residual
            amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                update_residual(i, j, k, n, dsdt_arr, reactsource_arr, 
                                AMREX_D_DECL(flux_arr[0], flux_arr[1], flux_arr[2]), dx);
            });

            // account for nanoporosity 
            FArrayBox ecoeff_fab(gbx, ncomp);
            ecoeff_fab.setVal<RunOn::Device>(1.0);
            Elixir ecoeff_fab_eli = ecoeff_fab.elixir();
            Array4<Real> ecoeff_arr = ecoeff_fab.array();
            amrex::ParallelFor(bx, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {

                electrochem_transport::compute_eps(i, j, k, sborder_arr, ecoeff_arr);

                dsdt_arr(i,j,k)=dsdt_arr(i,j,k) / ecoeff_arr(i,j,k,n);
            });
        }
    }

    if (do_reflux and reflux_this_stage)
    {
        if (flux_reg[lev + 1])
        {
            for (int i = 0; i < BL_SPACEDIM; ++i)
            {
                // update the lev+1/lev flux register (index lev+1)
                const Real dA = (i == 0) ? dx[1] * dx[2] : ((i == 1) ? dx[0] * dx[2] : dx[0] * dx[1]);
                const Real scale = -delt * dA;
                flux_reg[lev + 1]->CrseInit(flux[i], i, 0, 0, ncomp, scale);
            }
        }
        if (flux_reg[lev])
        {
            for (int i = 0; i < BL_SPACEDIM; ++i)
            {
                // update the lev/lev-1 flux register (index lev)
                const Real dA = (i == 0) ? dx[1] * dx[2] : ((i == 1) ? dx[0] * dx[2] : dx[0] * dx[1]);
                const Real scale = delt * dA;
                flux_reg[lev]->FineAdd(flux[i], i, 0, 0, ncomp, scale);
            }
        }
    }
}

// advance a single level for a single time step, updates flux registers
void echemAMR::compute_fluxes(int lev, const int num_grow, MultiFab& Sborder, 
                              Array<MultiFab,AMREX_SPACEDIM>& flux, 
                              Real time, bool implicit_diffusion)
{

    const auto dx = geom[lev].CellSizeArray();
    auto prob_lo = geom[lev].ProbLoArray();
    auto prob_hi = geom[lev].ProbHiArray();
    ProbParm const* localprobparm = d_prob_parm;

    int bvflux = buttler_vohlmer_flux;

    //FIXME: this is a bit ugly
    //ideally use transported_species_list 
    //and not solve for unwanted variables in the 
    //state like potential and efx/efy/efz
    int bvspec[NVAR]={0};

    for(unsigned int i=0;i<bv_specid_list.size();i++)
    {
        bvspec[bv_specid_list[i]]=1;
    }
    int bvlset    = bv_levset_id;

    Real lsgrad_tol=lsgrad_tolerance;

    int ncomp = Sborder.nComp();


#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(Sborder, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const Box& gbx = amrex::grow(bx, 2);
            Box bx_x = convert(bx, {1, 0, 0});
            Box bx_y = convert(bx, {0, 1, 0});
            Box bx_z = convert(bx, {0, 0, 1});

            FArrayBox dcoeff_fab(gbx, ncomp);
            FArrayBox velx_fab(gbx, ncomp);
            FArrayBox vely_fab(gbx, ncomp);
            FArrayBox velz_fab(gbx, ncomp);

            Elixir dcoeff_fab_eli = dcoeff_fab.elixir();
            Elixir velx_fab_eli = velx_fab.elixir();
            Elixir vely_fab_eli = vely_fab.elixir();
            Elixir velz_fab_eli = velz_fab.elixir();

            Array4<Real> sborder_arr = Sborder.array(mfi);
            Array4<Real> dcoeff_arr = dcoeff_fab.array();
            Array4<Real> velx_arr = velx_fab.array();
            Array4<Real> vely_arr = vely_fab.array();
            Array4<Real> velz_arr = velz_fab.array();

            GpuArray<Array4<Real>, AMREX_SPACEDIM> flux_arr{AMREX_D_DECL(flux[0].array(mfi), 
                                    flux[1].array(mfi), flux[2].array(mfi))};

            dcoeff_fab.setVal<RunOn::Device>(0.0);
            velx_fab.setVal<RunOn::Device>(0.0);
            vely_fab.setVal<RunOn::Device>(0.0);
            velz_fab.setVal<RunOn::Device>(0.0);

            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                electrochem_transport::compute_vel(i, j, k, 0, sborder_arr, velx_arr, 
                                                   prob_lo, prob_hi, dx, time, *localprobparm);
            });

            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                electrochem_transport::compute_vel(i, j, k, 1, sborder_arr, vely_arr, 
                                                   prob_lo, prob_hi, dx, time, *localprobparm);
            });

            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                electrochem_transport::compute_vel(i, j, k, 2, sborder_arr, velz_arr, 
                                                   prob_lo, prob_hi, dx, time, *localprobparm);
            });

            if(!implicit_diffusion)
            {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    electrochem_transport::compute_dcoeff(i, j, k, sborder_arr, dcoeff_arr, 
                                                          prob_lo, prob_hi, dx, time, *localprobparm);
                });
            }

            amrex::ParallelFor(bx_x, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                compute_flux(i, j, k, n, 0, sborder_arr, velx_arr, 
                             dcoeff_arr, flux_arr[0], dx, prob_lo, prob_hi,
                             *localprobparm, implicit_diffusion, bvflux, bvlset, bvspec, lsgrad_tol);
            });

            amrex::ParallelFor(bx_y, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                compute_flux(i, j, k, n, 1, sborder_arr, vely_arr, 
                             dcoeff_arr, flux_arr[1], dx, prob_lo, prob_hi,
                             *localprobparm, implicit_diffusion, bvflux, bvlset, bvspec, lsgrad_tol);
            });

            amrex::ParallelFor(bx_z, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                compute_flux(i, j, k, n, 2, sborder_arr, velz_arr, 
                             dcoeff_arr, flux_arr[2], dx, prob_lo, prob_hi, 
                             *localprobparm, implicit_diffusion, bvflux, bvlset, bvspec, lsgrad_tol);
            });
        }
    }
}

void echemAMR::implicit_solve_species(Real current_time,Real dt,int spec_id, 
                                      Vector<MultiFab>& dsdt_expl)
{
    BL_PROFILE("echemAMR::implicit_solve_species(" + std::to_string( spec_id ) + ")");

    //FIXME:create mlmg and mlabec objects outside this function
    LPInfo info;

    // FIXME: add these as inputs
    int agglomeration = 1;
    int consolidation = 1;
    int max_coarsening_level = linsolve_max_coarsening_level;
    bool semicoarsening = false;
    int max_semicoarsening_level = 0;
    int linop_maxorder = 2;
    int max_fmg_iter = 0;
    int verbose = 1;
    int bottom_verbose = 0;

    //==================================================
    // amrex solves
    // read small a as alpha, b as beta

    //(A a - B del.(b del)) phi = f
    //
    // A and B are scalar constants
    // a and b are scalar fields
    // f is rhs
    // in this case: A=0,a=0,B=1,b=conductivity
    // note also the negative sign
    //====================================================
    int bvspec[NUM_SPECIES]={0};

    for(unsigned int i=0;i<bv_specid_list.size();i++)
    {
        bvspec[bv_specid_list[i]]=1;
    }

    ProbParm const* localprobparm = d_prob_parm;

    const Real tol_rel = linsolve_reltol;
    const Real tol_abs = linsolve_abstol;

#ifdef AMREX_USE_HYPRE
    //Hypre::Interface hypre_interface = Hypre::Interface::ij;
    if(use_hypre)
    {
        amrex::Print()<<"using hypre\n";
    }
#endif
    info.setAgglomeration(agglomeration);
    info.setConsolidation(consolidation);
    info.setSemicoarsening(semicoarsening);
    info.setMaxCoarseningLevel(max_coarsening_level);
    info.setMaxSemicoarseningLevel(max_semicoarsening_level);

    MLABecLaplacian mlabec(Geom(0,finest_level), boxArray(0,finest_level), 
                           DistributionMap(0,finest_level), info);
    mlabec.setMaxOrder(linop_maxorder);

    // set A and B, A=1/dt, B=1
    Real ascalar = 1.0/dt;
    //Real ascalar = 0.0;
    Real bscalar = 1.0;
    mlabec.setScalars(ascalar, bscalar);

    // default to inhomogNeumann since it is defaulted to flux = 0.0 anyways
    std::array<LinOpBCType, AMREX_SPACEDIM> bc_linsolve_lo 
    = {LinOpBCType::inhomogNeumann, LinOpBCType::inhomogNeumann, LinOpBCType::inhomogNeumann};

    std::array<LinOpBCType, AMREX_SPACEDIM> bc_linsolve_hi 
    = {LinOpBCType::inhomogNeumann, LinOpBCType::inhomogNeumann, LinOpBCType::inhomogNeumann};

    for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
    {
        if (bc_lo_spec[idim] == BCType::int_dir)
        {
            bc_linsolve_lo[idim] = LinOpBCType::Periodic;
        }
        if (bc_lo_spec[idim] == BCType::ext_dir)
        {
            bc_linsolve_lo[idim] = LinOpBCType::Dirichlet;
        }
        if (bc_lo_spec[idim] == BCType::foextrap)
        {
            bc_linsolve_lo[idim] = LinOpBCType::Neumann;
        }

        if (bc_hi_spec[idim] == BCType::int_dir)
        {
            bc_linsolve_hi[idim] = LinOpBCType::Periodic;
        }
        if (bc_hi_spec[idim] == BCType::ext_dir)
        {
            bc_linsolve_hi[idim] = LinOpBCType::Dirichlet;
        }
        if (bc_hi_spec[idim] == BCType::foextrap)
        {
            bc_linsolve_hi[idim] = LinOpBCType::Neumann;
        }
    }

    mlabec.setDomainBC(bc_linsolve_lo, bc_linsolve_hi);

    Vector<MultiFab> specdata;
    Vector<MultiFab> acoeff;
    Vector<MultiFab> bcoeff;
    Vector<MultiFab> solution;
    Vector<MultiFab> rhs;

    specdata.resize(finest_level + 1);
    acoeff.resize(finest_level + 1);
    bcoeff.resize(finest_level + 1);
    solution.resize(finest_level + 1);
    rhs.resize(finest_level + 1);

    const int num_grow = 2;

    MLMG mlmg(mlabec);
    mlmg.setMaxIter(linsolve_maxiter);
    mlmg.setMaxFmgIter(max_fmg_iter);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);
    mlmg.setBottomTolerance(linsolve_bot_reltol);
    mlmg.setBottomToleranceAbs(linsolve_bot_abstol);

    mlmg.setPreSmooth(linsolve_num_pre_smooth);
    mlmg.setPostSmooth(linsolve_num_post_smooth);
    mlmg.setFinalSmooth(linsolve_num_final_smooth);
    mlmg.setBottomSmooth(linsolve_num_bottom_smooth);

#ifdef AMREX_USE_HYPRE
    if (use_hypre)
    {
        mlmg.setHypreOptionsNamespace("echemamr.hypre");
        mlmg.setBottomSolver(MLMG::BottomSolver::hypre);
        //mlmg.setHypreInterface(hypre_interface);
    }
#endif
#ifdef AMREX_USE_PETSC
    if (use_petsc)
    {
        mlmg.setBottomSolver(MLMG::BottomSolver::petsc);
    }
#endif

    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        specdata[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);
        acoeff[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);
        bcoeff[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);
        solution[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);
        rhs[ilev].define(grids[ilev], dmap[ilev], 1, 0);
    }

    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        // Copy args (FabArray<FAB>& dst, FabArray<FAB> const& src, 
        // int srccomp, int dstcomp, int numcomp, const IntVect& nghost)

        MultiFab Sborder(grids[ilev], dmap[ilev], phi_new[ilev].nComp(), num_grow);
        FillPatch(ilev, current_time, Sborder, 0, Sborder.nComp());

        specdata[ilev].setVal(0.0);
        amrex::Copy(specdata[ilev], Sborder, spec_id, 0, 1, num_grow);

        acoeff[ilev].setVal(1.0);
        bcoeff[ilev].setVal(1.0);

        rhs[ilev].setVal(0.0);
        MultiFab::LinComb(rhs[ilev], 1.0/dt, specdata[ilev], 0, 1.0, 
                          dsdt_expl[ilev], spec_id, 0, 1, 0);

        solution[ilev].setVal(0.0);
        amrex::MultiFab::Copy(solution[ilev], specdata[ilev], 0, 0, 1, 0);
        int ncomp = Sborder.nComp();

        // fill cell centered diffusion coefficients and rhs
        for (MFIter mfi(phi_new[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const Box& gbx = amrex::grow(bx, 1);
            const auto dx = geom[ilev].CellSizeArray();
            auto prob_lo = geom[ilev].ProbLoArray();
            auto prob_hi = geom[ilev].ProbHiArray();
            const Box& domain = geom[ilev].Domain();

            Real time = current_time; // for GPU capture

            Array4<Real> phi_arr = Sborder.array(mfi);
            Array4<Real> acoeff_arr = acoeff[ilev].array(mfi);
            Array4<Real> bcoeff_arr = bcoeff[ilev].array(mfi);

            FArrayBox dcoeff_fab(gbx, ncomp);
            Elixir dcoeff_fab_eli = dcoeff_fab.elixir();
            Array4<Real> dcoeff_arr = dcoeff_fab.array();

            FArrayBox ecoeff_fab(gbx, ncomp);
            ecoeff_fab.setVal<RunOn::Device>(1.0);
            Elixir ecoeff_fab_eli = ecoeff_fab.elixir();
            Array4<Real> ecoeff_arr = ecoeff_fab.array();

            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                //FIXME: use component wise call
                electrochem_transport::compute_dcoeff(i, j, k, phi_arr, 
                                                      dcoeff_arr, prob_lo, 
                                                      prob_hi, dx, time, *localprobparm);
                electrochem_transport::compute_eps(i, j, k, phi_arr, 
                                                   ecoeff_arr);

                bcoeff_arr(i,j,k)=dcoeff_arr(i,j,k,spec_id);
                acoeff_arr(i,j,k)=ecoeff_arr(i,j,k,spec_id);
            });
        }



        // average cell coefficients to faces, this includes boundary faces
        Array<MultiFab, AMREX_SPACEDIM> face_bcoeff;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            const BoxArray& ba = amrex::convert(bcoeff[ilev].boxArray(), IntVect::TheDimensionVector(idim));
            face_bcoeff[idim].define(ba, bcoeff[ilev].DistributionMap(), 1, 0);
        }
        // true argument for harmonic averaging
        amrex::average_cellcenter_to_face(GetArrOfPtrs(face_bcoeff), bcoeff[ilev], geom[ilev], true);

        if (bvspec[spec_id]==1 && buttler_vohlmer_flux)
        {
            int lset_id = bv_levset_id;
            Real gradctol = lsgrad_tolerance;

            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                for (MFIter mfi(phi_new[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const auto dx = geom[ilev].CellSizeArray();
                    const Box& bx = mfi.tilebox();
                    Real min_dx = amrex::min(dx[0], amrex::min(dx[1], dx[2]));

                    // face box
                    Box fbox = convert(bx, IntVect::TheDimensionVector(idim));
                    Array4<Real> phi_arr = Sborder.array(mfi);
                    Array4<Real> dcoeff_arr = face_bcoeff[idim].array(mfi);
                    auto plo = geom[ilev].ProbLoArray();
                    auto phi = geom[ilev].ProbHiArray();
                    const int* domlo_arr = geom[ilev].Domain().loVect();
                    const int* domhi_arr = geom[ilev].Domain().hiVect();
                    GpuArray<int,AMREX_SPACEDIM> domlo={AMREX_D_DECL(domlo_arr[0], domlo_arr[1], domlo_arr[2])};
                    GpuArray<int,AMREX_SPACEDIM> domhi={AMREX_D_DECL(domhi_arr[0], domhi_arr[1], domhi_arr[2])};

                    amrex::ParallelFor(fbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) { 

                        int normaldir = idim;
                        Real mod_gradc=0.0;
                        Real facecolor=0.0;
                        Real potjump=0.0;
                        Real gradc_cutoff=0.0;
                        Real dphidn = 0.0;
                        Real dphidt1 = 0.0;
                        Real dphidt2 = 0.0;
                        Real n_ls[AMREX_SPACEDIM];
                        Real intloc[AMREX_SPACEDIM];

                        bv_get_grads_and_jumps(i, j, k, normaldir, lset_id, dx, phi_arr, gradctol,
                                               mod_gradc, gradc_cutoff, facecolor, potjump, 
                                               dphidn, dphidt1, dphidt2, n_ls, intloc, 
                                               plo,phi);
                       Real activ_func = electrochem_reactions::bv_activation_function(facecolor, mod_gradc, gradc_cutoff);

                        if (mod_gradc > gradc_cutoff && activ_func > 0.0)
                        {
                            dcoeff_arr(i, j, k) *= (1.0 - activ_func);
                        }
                    });
                }
            }
        }

        // set boundary conditions
        for (MFIter mfi(phi_new[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const auto dx = geom[ilev].CellSizeArray();
            auto prob_lo = geom[ilev].ProbLoArray();
            auto prob_hi = geom[ilev].ProbHiArray();
            const Box& domain = geom[ilev].Domain();

            Array4<Real> phi_arr = Sborder.array(mfi);
            Array4<Real> bc_arr = specdata[ilev].array(mfi);
            Real time = current_time; // for GPU capture

            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                if (!geom[ilev].isPeriodic(idim))
                {
                    if (bx.smallEnd(idim) == domain.smallEnd(idim))
                    {
                        amrex::ParallelFor(amrex::bdryLo(bx, idim), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            electrochem_transport::species_linsolve_bc(i, j, k, idim, -1, 
                                                                       spec_id, phi_arr, bc_arr, 
                                                                       prob_lo, prob_hi, dx, time, *localprobparm);
                        });
                    }
                    if (bx.bigEnd(idim) == domain.bigEnd(idim))
                    {
                        amrex::ParallelFor(amrex::bdryHi(bx, idim), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                            electrochem_transport::species_linsolve_bc(i, j, k, idim, +1, spec_id, phi_arr, bc_arr, 
                                                                       prob_lo, prob_hi, dx, time, *localprobparm);
                        });
                    }
                }
            }
        }

        // set b with diffusivities
        mlabec.setACoeffs(ilev, acoeff[ilev]);
        mlabec.setBCoeffs(ilev, amrex::GetArrOfConstPtrs(face_bcoeff));

        // bc's are stored in the ghost cells
        mlabec.setLevelBC(ilev, &(specdata[ilev]));
    }

    mlmg.solve(GetVecOfPtrs(solution), GetVecOfConstPtrs(rhs), tol_rel, tol_abs);

    // copy solution back to phi_new
    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        Print()<<"max of solution:"<<solution[ilev].max(0)<<"\n";
        Print()<<"min of solution:"<<solution[ilev].min(0)<<"\n";
        Print()<<"max of rhs:"<<rhs[ilev].max(0)<<"\n";
        Print()<<"min of rhs:"<<rhs[ilev].min(0)<<"\n";
        amrex::MultiFab::Copy(phi_new[ilev], solution[ilev], 0, spec_id, 1, 0);
    }
    Print()<<"spec id:"<<spec_id<<"\n";
}
