#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_MLTensorOp.H>
#include <Kernels_3d.H>

#ifdef AMREX_MEM_PROFILING
#include <AMReX_MemProfiler.H>
#endif

#include <echemAMR.H>
#include <Chemistry.H>
#include <Transport.H>
#include <Reactions.H>
#include <Mechanics.H>
#include <PostProcessing.H>
#include <ProbParm.H>
#include <AMReX_MLABecLaplacian.H>

// advance solution to final time
void echemAMR::Evolve()
{
    Real cur_time = t_new[0];
    int last_plot_file_step = 0;
    postprocess(cur_time, 0, 0.0, echemAMR::host_global_storage);

    for (int step = istep[0]; step < max_step && cur_time < stop_time; ++step)
    {
        amrex::Print() << "\nCoarse STEP " << step + 1 << " starts ..." << std::endl;
    
        // BL_PROFILE_TINY_FLUSH()
        
        ComputeDt();

        int lev = 0;
        int iteration = 1;

        if (potential_solve == 1 && step % pot_solve_int == 0)
        {
            solve_potential(cur_time);
        }
        if(!species_implicit_solve)
        {
           timeStep(lev, cur_time, iteration);
        }
        else
        {
            if (max_level > 0 && regrid_int > 0)  // We may need to regrid
            {
                if (istep[0] % regrid_int == 0)
                {
                    regrid(0, cur_time);
                }
            }
            
            for (int lev = 0; lev <= finest_level; lev++)
            {
                amrex::Print() << "[Level " << lev << " step " << istep[lev]+1 << "] ";
                amrex::Print() << "ADVANCE with time = " << t_new[lev]
                    << " dt = " << dt[0] << std::endl;
            }
            
            Vector< Array<MultiFab,AMREX_SPACEDIM> > flux(finest_level+1);
            for (int lev = 0; lev <= finest_level; lev++)
            {
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
                {
                    BoxArray ba = grids[lev];
                    ba.surroundingNodes(idim);
                    flux[lev][idim].define(ba, dmap[lev], phi_new[lev].nComp(), 0);
                    flux[lev][idim].setVal(0.0);
                }
            }
            
            Vector<MultiFab> expl_src(finest_level+1);
            for(int lev=0;lev<=finest_level;lev++)
            {
                amrex::MultiFab::Copy(phi_old[lev], phi_new[lev], 0, 0, phi_new[lev].nComp(), 0);
                t_old[lev] = t_new[lev];
                t_new[lev] += dt[0];

                int num_grow=2;
                MultiFab Sborder(grids[lev], dmap[lev], phi_new[lev].nComp(), num_grow);
                FillPatch(lev, cur_time, Sborder, 0, Sborder.nComp());
                compute_fluxes(lev, num_grow, Sborder, flux[lev], cur_time, true);
            }
            // =======================================================
            // Average down the fluxes before using them to update phi
            // =======================================================
            for (int lev = finest_level; lev > 0; lev--)
            {
                average_down_faces(amrex::GetArrOfConstPtrs(flux[lev  ]),
                        amrex::GetArrOfPtrs(flux[lev-1]),
                        refRatio(lev-1), Geom(lev-1));
            }
            for(int lev=0;lev<=finest_level;lev++)
            {
                int num_grow=2;
                MultiFab Sborder(grids[lev], dmap[lev], phi_new[lev].nComp(), num_grow);
                expl_src[lev].define(grids[lev], dmap[lev], phi_new[lev].nComp(), 0);
                expl_src[lev].setVal(0.0);

                //FIXME: need to avoid this fillpatch
                FillPatch(lev, cur_time, Sborder, 0, Sborder.nComp());
                compute_dsdt(lev, num_grow, Sborder,flux[lev], expl_src[lev], 
                        cur_time, dt[0], false);
            }

            for(unsigned int ind=0;ind<transported_species_list.size();ind++)
            {
                implicit_solve_species(cur_time,dt[0],transported_species_list[ind],expl_src);
            }
            AverageDown ();
            
            for (int lev = 0; lev <= finest_level; lev++)
                ++istep[lev];

            for (int lev = 0; lev <= finest_level; lev++)
            {
                amrex::Print() << "[Level " << lev << " step " << istep[lev] << "] ";
                amrex::Print() << "Advanced " << CountCells(lev) << " cells" << std::endl;
            }
        }
        if (mechanics_solve == 1)
        {
            solve_mechanics(cur_time);
        }


        cur_time += dt[0];

        postprocess(cur_time, step+1, dt[0], echemAMR::host_global_storage);

        amrex::Print() << "Coarse STEP " << step + 1 << " ends."
                       << " TIME = " << cur_time << " DT = " << dt[0] << std::endl;

        // sync up time
        for (lev = 0; lev <= finest_level; ++lev)
        {
            t_new[lev] = cur_time;
        }

        if(reset_species_in_solid)
        {
            for(int ilev=0;ilev<=finest_level;ilev++)
            {

                for (MFIter mfi(phi_new[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const Box& bx = mfi.tilebox();
                    Array4<Real> phi_arr = phi_new[ilev].array(mfi);
                    int *species_list=transported_species_list.data();
                    unsigned int nspec_list=transported_species_list.size();
                    int lset_id=bv_levset_id;

                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) 
                    {
                        for(int sp=0;sp<nspec_list;sp++)
                        {
                           phi_arr(i,j,k,sp)*=phi_arr(i,j,k,lset_id);
                        }
                    });
                }
            }
        }

        if (plot_int > 0 && (step + 1) % plot_int == 0)
        {
            last_plot_file_step = step + 1;
            WritePlotFile();
        }

        if (chk_int > 0 && (step + 1) % chk_int == 0)
        {
            WriteCheckpointFile();
        }

#ifdef AMREX_MEM_PROFILING
        {
            std::ostringstream ss;
            ss << "[STEP " << step + 1 << "]";
            MemProfiler::report(ss.str());
        }
#endif

        if (cur_time >= stop_time - 1.e-6 * dt[0]) break;
    }

    if (plot_int > 0 && istep[0] > last_plot_file_step)
    {
        WritePlotFile();
    }
}

void echemAMR::solve_potential(Real current_time)
{
    BL_PROFILE("echemAMR::solve_potential()");
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
    Real ascalar = 0.0;
    Real bscalar = 1.0;
    int num_nonlinear_iters;

    //FIXME:use get_grads_and_jumps function from Src3d for BVflux

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

    // FIXME: had to adjust for constant coefficent,
    // this could be due to missing terms in the 
    // intercalation reaction or sign mistakes...
    ProbParm const* localprobparm = d_prob_parm;

    const Real tol_rel = linsolve_reltol;
    const Real tol_abs = linsolve_abstol;

#ifdef AMREX_USE_HYPRE
    Hypre::Interface hypre_interface = Hypre::Interface::ij;
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


    // set A and B, A=0, B=1
    //
    if (buttler_vohlmer_flux)
    {
        ascalar = bv_relaxfac;
        num_nonlinear_iters=bv_nonlinear_iters;
    } 
    else
    {
        ascalar = 0.0;
        num_nonlinear_iters=1;
    }

    // default to inhomogNeumann since it is defaulted to flux = 0.0 anyways
    std::array<LinOpBCType, AMREX_SPACEDIM> bc_potsolve_lo 
        = {LinOpBCType::inhomogNeumann, LinOpBCType::inhomogNeumann, LinOpBCType::inhomogNeumann};

    std::array<LinOpBCType, AMREX_SPACEDIM> bc_potsolve_hi 
        = {LinOpBCType::inhomogNeumann, LinOpBCType::inhomogNeumann, LinOpBCType::inhomogNeumann};

    bool mixedbc=false;

    for (int idim = 0; idim < AMREX_SPACEDIM; idim++)
    {
        if (bc_lo_pot[idim] == BCType::int_dir)
        {
            bc_potsolve_lo[idim] = LinOpBCType::Periodic;
        }
        if (bc_lo_pot[idim] == BCType::ext_dir)
        {
            bc_potsolve_lo[idim] = LinOpBCType::Dirichlet;
        }
        if (bc_lo_pot[idim] == BCType::foextrap)
        {
            bc_potsolve_lo[idim] = LinOpBCType::Neumann;
        }
        if (bc_lo_pot[idim] == BCType::hoextrapcc)
        {
            bc_potsolve_lo[idim] = LinOpBCType::Robin;
            mixedbc=true;
        }

        if (bc_hi_pot[idim] == BCType::int_dir)
        {
            bc_potsolve_hi[idim] = LinOpBCType::Periodic;
        }
        if (bc_hi_pot[idim] == BCType::ext_dir)
        {
            bc_potsolve_hi[idim] = LinOpBCType::Dirichlet;
        }
        if (bc_hi_pot[idim] == BCType::foextrap)
        {
            bc_potsolve_hi[idim] = LinOpBCType::Neumann;
        }
        if (bc_hi_pot[idim] == BCType::hoextrapcc)
        {
            bc_potsolve_hi[idim] = LinOpBCType::Robin;
            mixedbc=true;
        }
    }


    Vector<MultiFab> potential(finest_level+1);
    Vector<MultiFab> acoeff(finest_level+1);
    Vector<MultiFab> bcoeff(finest_level+1);
    Vector<Array<MultiFab, AMREX_SPACEDIM>> gradsoln(finest_level+1);
    Vector<MultiFab> solution(finest_level+1);
    Vector<MultiFab> residual(finest_level+1);
    Vector<MultiFab> rhs(finest_level+1);
    Vector<MultiFab> err(finest_level+1);
    Vector<MultiFab> rhs_res(finest_level+1);

    Vector<MultiFab> robin_a(finest_level+1);
    Vector<MultiFab> robin_b(finest_level+1);
    Vector<MultiFab> robin_f(finest_level+1);

    const int num_grow = 1;
    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        potential[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);
        acoeff[ilev].define(grids[ilev], dmap[ilev], 1, 0);
        bcoeff[ilev].define(grids[ilev], dmap[ilev], 1, 1);
        solution[ilev].define(grids[ilev], dmap[ilev], 1, 1);
        residual[ilev].define(grids[ilev], dmap[ilev], 1, 1);
        rhs[ilev].define(grids[ilev], dmap[ilev], 1, 0);
        err[ilev].define(grids[ilev], dmap[ilev], 1, 0);
        rhs_res[ilev].define(grids[ilev], dmap[ilev], 1, 0);
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            const BoxArray& faceba = amrex::convert(grids[ilev], 
                    IntVect::TheDimensionVector(idim));
            gradsoln[ilev][idim].define(faceba, dmap[ilev], 1, 0);
        }

        robin_a[ilev].define(grids[ilev], dmap[ilev], 1, 1);
        robin_b[ilev].define(grids[ilev], dmap[ilev], 1, 1);
        robin_f[ilev].define(grids[ilev], dmap[ilev], 1, 1);
    }

    Real errnorm_1st_iter;
    
    MLABecLaplacian mlabec(Geom(0, finest_level), boxArray(0, finest_level), 
                           DistributionMap(0, finest_level), info);
    MLABecLaplacian mlabec_res(Geom(0,finest_level), boxArray(0,finest_level), 
                               DistributionMap(0, finest_level), info);

    mlabec.setMaxOrder(linop_maxorder);
    mlabec_res.setMaxOrder(linop_maxorder);
    
    for (int nl_it = 0; nl_it < num_nonlinear_iters; ++nl_it)
    {
        mlabec.setScalars(ascalar, bscalar);
        mlabec_res.setScalars(0.0, bscalar);
    
        mlabec.setDomainBC(bc_potsolve_lo, bc_potsolve_hi);
        mlabec_res.setDomainBC(bc_potsolve_lo, bc_potsolve_hi);
    
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
            mlmg.setHypreOptionsNamespace("hypre");
            mlmg.setBottomSolver(MLMG::BottomSolver::hypre);
            mlmg.setHypreInterface(hypre_interface);
        }
#endif
#ifdef AMREX_USE_PETSC
        if (use_petsc)
        {
            mlmg.setBottomSolver(MLMG::BottomSolver::petsc);
        }
#endif

        MLMG mlmg_res(mlabec_res);
        
        for (int ilev = 0; ilev <= finest_level; ilev++)
        {
            MultiFab Sborder(grids[ilev], dmap[ilev], phi_new[ilev].nComp(), num_grow);
            FillPatch(ilev, current_time, Sborder, 0, Sborder.nComp());
            potential[ilev].setVal(0.0);

            // Copy (FabArray<FAB>& dst, FabArray<FAB> const& src, int srccomp, 
            // int dstcomp, int numcomp, const IntVect& nghost)
            amrex::Copy(potential[ilev], Sborder, POT_ID, 0, 1, num_grow);

            acoeff[ilev].setVal(1.0); //will be scaled by ascalar
            mlabec.setACoeffs(ilev, acoeff[ilev]);

            bcoeff[ilev].setVal(1.0);
            solution[ilev].setVal(0.0);
            rhs[ilev].setVal(0.0);

            //default to homogenous Neumann
            robin_a[ilev].setVal(0.0);
            robin_b[ilev].setVal(1.0);
            robin_f[ilev].setVal(0.0);


            // copy current solution for better guess
            if (pot_initial_guess)
            {
                amrex::MultiFab::Copy(solution[ilev], potential[ilev], 0, 0, 1, 0);
                // solution[ilev].copy(potential[ilev], 0, 0, 1);
            }

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
                Array4<Real> bcoeff_arr = bcoeff[ilev].array(mfi);
                Array4<Real> rhs_arr = rhs[ilev].array(mfi);

                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        electrochem_transport::compute_potential_dcoeff(i, j, k, phi_arr, bcoeff_arr, 
                                                   prob_lo, prob_hi, dx, time, *localprobparm);
                        });
                amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                        electrochem_reactions::compute_potential_source(i, j, k, phi_arr, 
                                              rhs_arr, prob_lo, prob_hi, dx, time, *localprobparm);
                        });
            }

            // average cell coefficients to faces, this includes boundary faces
            Array<MultiFab, AMREX_SPACEDIM> face_bcoeff;
            Array<MultiFab, AMREX_SPACEDIM> face_bcoeff_res;
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                const BoxArray& ba = amrex::convert(bcoeff[ilev].boxArray(), IntVect::TheDimensionVector(idim));
                face_bcoeff[idim].define(ba, bcoeff[ilev].DistributionMap(), 1, 0);
                face_bcoeff_res[idim].define(ba, bcoeff[ilev].DistributionMap(), 1, 0);
            }
            // true argument for harmonic averaging
            amrex::average_cellcenter_to_face(GetArrOfPtrs(face_bcoeff), bcoeff[ilev], geom[ilev], true);
            amrex::average_cellcenter_to_face(GetArrOfPtrs(face_bcoeff_res), bcoeff[ilev], geom[ilev], true);
            
            Array<MultiFab, AMREX_SPACEDIM> kdterm;
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                const BoxArray& ba = amrex::convert(bcoeff[ilev].boxArray(), 
                                                    IntVect::TheDimensionVector(idim));
                kdterm[idim].define(ba, bcoeff[ilev].DistributionMap(), 1, 0);
            }

            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                for (MFIter mfi(phi_new[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const auto dx = geom[ilev].CellSizeArray();
                    const Box& bx = mfi.tilebox();
                    auto prob_lo = geom[ilev].ProbLoArray();
                    auto prob_hi = geom[ilev].ProbHiArray();

                    Real time = current_time; // for GPU capture

                    // face box
                    Box fbox = convert(bx, IntVect::TheDimensionVector(idim));
                    Array4<Real> phi_arr = Sborder.array(mfi);
                    Array4<Real> kdterm_arr = kdterm[idim].array(mfi);
                    int captured_kd_conc_id = kd_conc_id; //public variable

                    amrex::ParallelFor(fbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                        IntVect left(i, j, k);
                        IntVect right(i, j, k);
                        left[idim] -= 1;

                        //kdstar is kd/conc
                        //del.(kd grad(lnc))=del. (kd/c grad(c))=del.(kdstar grad(c))
                        //del.(-sigma grad(phi))+del.(kdstar grad(c))=0
                        amrex::Real kdstar = electrochem_transport::compute_kdstar_atface(i, j, k, idim,
                                                                                          phi_arr, prob_lo, 
                                                                                          prob_hi, dx, time, 
                                                                                          *localprobparm);

                        kdterm_arr(i,j,k)=kdstar*(phi_arr(right,captured_kd_conc_id)
                                                  -phi_arr(left,captured_kd_conc_id))/dx[idim];
                    });
                }
            }


            if (buttler_vohlmer_flux)
            {
                int lset_id = bv_levset_id;
                Real gradctol = lsgrad_tolerance;

                Array<MultiFab, AMREX_SPACEDIM> bv_explicit_terms;
                Array<MultiFab, AMREX_SPACEDIM> bv_explicit_terms_res;
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
                {
                    const BoxArray& ba = amrex::convert(bcoeff[ilev].boxArray(), 
                                                        IntVect::TheDimensionVector(idim));
                    bv_explicit_terms[idim].define(ba, bcoeff[ilev].DistributionMap(), 1, 0);
                    bv_explicit_terms_res[idim].define(ba, bcoeff[ilev].DistributionMap(), 1, 0);
                }

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
                        Array4<Real> dcoeff_arr_res = face_bcoeff_res[idim].array(mfi);
                        Array4<Real> explterms_arr = bv_explicit_terms[idim].array(mfi);
                        Array4<Real> explterms_arr_res = bv_explicit_terms_res[idim].array(mfi);
                        Array4<Real> kdterm_arr=kdterm[idim].array(mfi);


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

                            bv_get_grads_and_jumps(i, j, k, normaldir, lset_id, dx, phi_arr, gradctol,
                                                   mod_gradc, gradc_cutoff, facecolor, potjump, dphidn, dphidt1, dphidt2, n_ls);


                            explterms_arr(i, j, k) = 0.0;
                            explterms_arr_res(i, j, k) = 0.0;

                            if (mod_gradc > gradc_cutoff)
                            {
                                Real activ_func = electrochem_reactions::bv_activation_function(facecolor, mod_gradc, gradc_cutoff);

                                //if(fabs(potjump) > 1)
                                //{
                                //   Print()<<"potjump:"<<potjump<<"\t"<<dphidn<<"\t"<<dphidt1<<"\t"<<dphidt2<<"\t"
                                //     <<dcdn<<"\t"<<dcdt1<<"\t"<<dcdt2<<"\t"<<mod_gradc<<"\n";
                                //}

                                // FIXME: pass ion concentration also
                                // FIXME: ideally it should be the ion concentration at the closest electrode cell
                                Real j_bv,jdash_bv;
                                electrochem_reactions::bvcurrent_and_der(i,j,k,normaldir,potjump,phi_arr,
                                                                         *localprobparm,j_bv,jdash_bv);

                                dcoeff_arr(i, j, k) *= (1.0 - activ_func);
                                dcoeff_arr_res(i, j, k) = dcoeff_arr(i, j, k);


                                // dcoeff_arr(i,j,k) += -jdash_bv*activ_func/pow(mod_gradc,3.0) * dcdn*dcdn;
                                dcoeff_arr(i, j, k) += -jdash_bv * activ_func / mod_gradc * n_ls[0] * n_ls[0];
                                

                                // expl term1
                                // explterms_arr(i,j,k) =   j_bv*activ_func*dcdn/mod_gradc;
                                explterms_arr(i, j, k) = j_bv * activ_func * n_ls[0];
                                explterms_arr_res(i, j, k) = explterms_arr(i, j, k);

                                // expl term2
                                // explterms_arr(i,j,k) += -jdash_bv*potjump*activ_func*dcdn/mod_gradc;
                                explterms_arr(i, j, k) += -jdash_bv * potjump * activ_func * n_ls[0];

                                // expl term3 (mix derivative terms from tensor product)
                                // explterms_arr(i,j,k) +=  jdash_bv*activ_func/pow(mod_gradc,3.0)*(dcdn*dcdt1*dphidt1+dcdn*dcdt2*dphidt2);
                                explterms_arr(i, j, k) += jdash_bv * activ_func / mod_gradc * (n_ls[0] * n_ls[1] * dphidt1 + n_ls[0] * n_ls[2] * dphidt2);
                                kdterm_arr(i,j,k) *= (1.0-activ_func);
                                explterms_arr_res(i, j, k) += kdterm_arr(i, j, k);
                            }
                        });
                    }
                }


                for (MFIter mfi(phi_new[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const Box& bx = mfi.tilebox();
                    const auto dx = geom[ilev].CellSizeArray();

                    Array4<Real> rhs_arr = rhs[ilev].array(mfi);
                    Array4<Real> rhs_arr_res = rhs_res[ilev].array(mfi);
                    Array4<Real> phi_arr = Sborder.array(mfi);

                    Array4<Real> term_x = bv_explicit_terms[0].array(mfi);
                    Array4<Real> term_y = bv_explicit_terms[1].array(mfi);
                    Array4<Real> term_z = bv_explicit_terms[2].array(mfi);

                    Array4<Real> term_x_res = bv_explicit_terms_res[0].array(mfi);
                    Array4<Real> term_y_res = bv_explicit_terms_res[1].array(mfi);
                    Array4<Real> term_z_res = bv_explicit_terms_res[2].array(mfi);
                    
                    Array4<Real> kdterm_x = kdterm[0].array(mfi);
                    Array4<Real> kdterm_y = kdterm[1].array(mfi);
                    Array4<Real> kdterm_z = kdterm[2].array(mfi);

                    Real relax_fac = bv_relaxfac;

                    amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {

                        rhs_arr_res(i,j,k) = rhs_arr(i,j,k);

                        rhs_arr(i, j, k) += (term_x(i, j, k) - term_x(i + 1, j, k)) / dx[0] + (term_y(i, j, k) - term_y(i, j + 1, k)) / dx[1] +
                        (term_z(i, j, k) - term_z(i, j, k + 1)) / dx[2];

                        rhs_arr(i, j, k) += phi_arr(i, j, k, POT_ID) * relax_fac;

                        rhs_arr(i, j, k) += (kdterm_x(i, j, k) - kdterm_x(i + 1, j, k)) / dx[0] + (kdterm_y(i, j, k) - kdterm_y(i, j + 1, k)) / dx[1] +
                        (kdterm_z(i, j, k) - kdterm_z(i, j, k + 1)) / dx[2];

                        rhs_arr_res(i, j, k) += (term_x_res(i, j, k) - term_x_res(i + 1, j, k)) / dx[0] + (term_y_res(i, j, k) - term_y_res(i, j + 1, k)) / dx[1] +
                        (term_z_res(i, j, k) - term_z_res(i, j, k + 1)) / dx[2];

                    });
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
                Array4<Real> bc_arr = potential[ilev].array(mfi);

                Array4<Real> robin_a_arr = robin_a[ilev].array(mfi);
                Array4<Real> robin_b_arr = robin_b[ilev].array(mfi);
                Array4<Real> robin_f_arr = robin_f[ilev].array(mfi);

                Real time = current_time; // for GPU capture

                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
                {
                    const amrex::Real bclo = host_global_storage->pot_bc_lo[idim];
                    const amrex::Real bchi = host_global_storage->pot_bc_hi[idim];

                    if (!geom[ilev].isPeriodic(idim))
                    {
                        if (bx.smallEnd(idim) == domain.smallEnd(idim))
                        {
                            amrex::ParallelFor(amrex::bdryLo(bx, idim), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                electrochem_transport::potential_bc(i, j, k, idim, -1, phi_arr, bc_arr, prob_lo, prob_hi, dx, time, bclo, bchi);
                            });
                        }
                        if (bx.bigEnd(idim) == domain.bigEnd(idim))
                        {
                            amrex::ParallelFor(amrex::bdryHi(bx, idim), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                electrochem_transport::potential_bc(i, j, k, idim, +1, phi_arr, bc_arr, prob_lo, prob_hi, dx, time, bclo, bchi);
                            });
                        }

                        if(mixedbc)
                        {
                            if (bx.smallEnd(idim) == domain.smallEnd(idim))
                            {
                                amrex::ParallelFor(amrex::bdryLo(bx, idim), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                    electrochem_transport::potential_mixedbc(i, j, k, idim, -1, phi_arr, robin_a_arr, 
                                                                             robin_b_arr, robin_f_arr, prob_lo, prob_hi, dx, time, bclo, bchi);
                                });
                            }
                            if (bx.bigEnd(idim) == domain.bigEnd(idim))
                            {
                                amrex::ParallelFor(amrex::bdryHi(bx, idim), [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                                    electrochem_transport::potential_mixedbc(i, j, k, idim, +1, phi_arr, robin_a_arr, 
                                                                             robin_b_arr, robin_f_arr, prob_lo, prob_hi, dx, time, bclo, bchi);
                                });
                            }
                        }
                    }
                }
            }

            // set b with diffusivities
            mlabec.setBCoeffs(ilev, amrex::GetArrOfConstPtrs(face_bcoeff));
            mlabec_res.setBCoeffs(ilev, amrex::GetArrOfConstPtrs(face_bcoeff_res));

            // bc's are stored in the ghost cells of potential
            mlabec.setLevelBC(ilev, &potential[ilev], &(robin_a[ilev]), &(robin_b[ilev]), &(robin_f[ilev]));
            mlabec_res.setLevelBC(ilev, &(potential[ilev]), &(robin_a[ilev]), &(robin_b[ilev]), &(robin_f[ilev]));
        }

        mlmg.solve(GetVecOfPtrs(solution), GetVecOfConstPtrs(rhs), tol_rel, tol_abs);
        mlmg_res.apply(GetVecOfPtrs(residual),GetVecOfPtrs(solution));

        //error norm calculation
        // copy solution back to phi_new
        if(buttler_vohlmer_flux)
        {
            Real abs_errnorm_all=0.0;
            Real rel_errnorm_all=0.0;
            for (int ilev = 0; ilev <= finest_level; ilev++)
            {
                amrex::MultiFab::Copy(err[ilev], phi_new[ilev], POT_ID, 0, 1, 0);
                amrex::MultiFab::Subtract(err[ilev],solution[ilev], 0, 0, 1 ,0);
                Real errnorm=err[ilev].norm2();

                if(nl_it==0)
                {
                    errnorm_1st_iter=errnorm;
                }
                Real rel_errnorm=errnorm/errnorm_1st_iter;
                amrex::Print()<<"lev, iter, abs errnorm, rel errnorm:"<<ilev<<"\t"
                <<nl_it<<"\t"<<errnorm<<"\t"<<rel_errnorm<<"\n";

                abs_errnorm_all += errnorm;
                rel_errnorm_all += rel_errnorm;
            }

            //            if(abs_errnorm_all < bv_nonlinear_abstol ||
            //                    rel_errnorm_all < bv_nonlinear_reltol)
            //            {
            //                amrex::Print()<<"Converged with final error:"<<rel_errnorm_all
            //                    <<"\t"<<abs_errnorm_all<<"\n";
            //                break;
            //            }
        }

        // copy solution back to phi_new
        for (int ilev = 0; ilev <= finest_level; ilev++) {
            amrex::MultiFab::Copy(phi_new[ilev], solution[ilev], 0, POT_ID, 1, 0);
        }

        Real total_nl_res = 0.0;
        for (int ilev = 0; ilev <= finest_level; ilev++)
        {
            amrex::MultiFab level_mask;
            if (ilev < finest_level) {
                level_mask = makeFineMask(grids[ilev],dmap[ilev],grids[ilev+1], amrex::IntVect(2), 1.0, 0.0);
            } else {
                level_mask.define(grids[ilev], dmap[ilev], 1, 0,
                                  amrex::MFInfo());
                level_mask.setVal(1);
            }
            amrex::MultiFab::Subtract(residual[ilev],rhs_res[ilev], 0, 0, 1 ,0);
            amrex::MultiFab::Multiply(residual[ilev],level_mask, 0, 0, 1, 0);
            Real nl_res = residual[ilev].norm2();
            total_nl_res += nl_res;
        }

        if(nl_it==0) {
            errnorm_1st_iter=total_nl_res;
        }

        amrex::Print() <<"BV NON-LINEAR RESIDUAL (rel,abs): " <<  total_nl_res/errnorm_1st_iter << ' ' << total_nl_res << std::endl;

        if(total_nl_res < bv_nonlinear_abstol ||
           total_nl_res/errnorm_1st_iter < bv_nonlinear_reltol)
        {
            amrex::Print()<<"Converged with final rel,abs error: "<< total_nl_res/errnorm_1st_iter
            <<"\t"<< total_nl_res <<"\n";
            mlmg.getGradSolution(GetVecOfArrOfPtrs(gradsoln));
            break;
        }

        if(nl_it==num_nonlinear_iters-1)
        {
            mlmg.getGradSolution(GetVecOfArrOfPtrs(gradsoln));
        }
    }


    // copy solution back to phi_new
    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        amrex::MultiFab::Copy(phi_new[ilev], solution[ilev], 0, POT_ID, 1, 0);
        //phi_new[ilev].copy(solution[ilev], 0, NVAR-1, 1);
        const Array<const MultiFab*, AMREX_SPACEDIM> allgrad = {&gradsoln[ilev][0], 
            &gradsoln[ilev][1], &gradsoln[ilev][2]};
        average_face_to_cellcenter(phi_new[ilev], EFX_ID, allgrad);
        phi_new[ilev].mult(-1.0, EFX_ID, 3);
    }

    //clear
    potential.clear();
    acoeff.clear();
    bcoeff.clear();
    gradsoln.clear();
    solution.clear();
    residual.clear();
    rhs.clear();
    err.clear();
    rhs_res.clear();

    robin_a.clear();
    robin_b.clear();
    robin_f.clear();
}

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
                for (int k = old_finest + 1; k <= finest_level; ++k)
                {
                    dt[k] = dt[k - 1] / MaxRefRatio(k - 1);
                }
            }
        }
    }

    if (Verbose())
    {
        amrex::Print() << "[Level " << lev << " step " << istep[lev] + 1 << "] ";
        amrex::Print() << "ADVANCE with time = " << t_new[lev] << " dt = " << dt[lev] << std::endl;
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
            flux_reg[lev + 1]->Reflux(phi_new[lev], 1.0, 0, 0, phi_new[lev].nComp(), geom[lev]);
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
                            Real time, Real dt, bool reflux_this_stage)
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
                                                            reactsource_arr, prob_lo, prob_hi, dx, time, *localprobparm);
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
                const Real scale = -dt * dA;
                flux_reg[lev + 1]->CrseInit(flux[i], i, 0, 0, ncomp, scale);
            }
        }
        if (flux_reg[lev])
        {
            for (int i = 0; i < BL_SPACEDIM; ++i)
            {
                // update the lev/lev-1 flux register (index lev)
                const Real dA = (i == 0) ? dx[1] * dx[2] : ((i == 1) ? dx[0] * dx[2] : dx[0] * dx[1]);
                const Real scale = dt * dA;
                flux_reg[lev]->FineAdd(flux[i], i, 0, 0, ncomp, scale);
            }
        }
    }
}

// advance a single level for a single time step, updates flux registers
void echemAMR::compute_fluxes(int lev, const int num_grow, MultiFab& Sborder, 
                              Array<MultiFab,AMREX_SPACEDIM>& flux, Real time, bool implicit_diffusion)
{

    const auto dx = geom[lev].CellSizeArray();
    auto prob_lo = geom[lev].ProbLoArray();
    auto prob_hi = geom[lev].ProbHiArray();
    ProbParm const* localprobparm = d_prob_parm;

    int bvflux = buttler_vohlmer_flux;
    int bvspec[NUM_SPECIES]={0};

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
            const Box& gbx = amrex::grow(bx, 1);
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

            GpuArray<Array4<Real>, AMREX_SPACEDIM> flux_arr{AMREX_D_DECL(flux[0].array(mfi), flux[1].array(mfi), flux[2].array(mfi))};

            dcoeff_fab.setVal<RunOn::Device>(0.0);
            velx_fab.setVal<RunOn::Device>(0.0);
            vely_fab.setVal<RunOn::Device>(0.0);
            velz_fab.setVal<RunOn::Device>(0.0);

            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                electrochem_transport::compute_vel(i, j, k, 0, sborder_arr, velx_arr, prob_lo, prob_hi, dx, time, *localprobparm);
            });

            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                electrochem_transport::compute_vel(i, j, k, 1, sborder_arr, vely_arr, prob_lo, prob_hi, dx, time, *localprobparm);
            });

            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                electrochem_transport::compute_vel(i, j, k, 2, sborder_arr, velz_arr, prob_lo, prob_hi, dx, time, *localprobparm);
            });

            if(!implicit_diffusion)
            {
                amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) {
                    electrochem_transport::compute_dcoeff(i, j, k, sborder_arr, dcoeff_arr, 
                                                          prob_lo, prob_hi, dx, time, *localprobparm);
                });
            }

            amrex::ParallelFor(bx_x, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                compute_flux(i, j, k, n, 0, sborder_arr, velx_arr, dcoeff_arr, flux_arr[0], dx, *localprobparm, 
                             implicit_diffusion, bvflux, bvlset, bvspec, lsgrad_tol);
            });

            amrex::ParallelFor(bx_y, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                compute_flux(i, j, k, n, 1, sborder_arr, vely_arr, dcoeff_arr, flux_arr[1], dx, *localprobparm, 
                             implicit_diffusion, bvflux, bvlset, bvspec, lsgrad_tol);
            });

            amrex::ParallelFor(bx_z, ncomp, [=] AMREX_GPU_DEVICE(int i, int j, int k, int n) {
                compute_flux(i, j, k, n, 2, sborder_arr, velz_arr, dcoeff_arr, flux_arr[2], dx, *localprobparm, 
                             implicit_diffusion, bvflux, bvlset, bvspec, lsgrad_tol);
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
    Hypre::Interface hypre_interface = Hypre::Interface::ij;
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

    const int num_grow = 1;

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
        mlmg.setHypreOptionsNamespace("hypre");
        mlmg.setBottomSolver(MLMG::BottomSolver::hypre);
        mlmg.setHypreInterface(hypre_interface);
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

                    amrex::ParallelFor(fbox, [=] AMREX_GPU_DEVICE(int i, int j, int k) 
                                       {

                                           int normaldir = idim;
                                           Real mod_gradc=0.0;
                                           Real facecolor=0.0;
                                           Real potjump=0.0;
                                           Real gradc_cutoff=0.0;
                                           Real dphidn = 0.0;
                                           Real dphidt1 = 0.0;
                                           Real dphidt2 = 0.0;
                                           Real n_ls[AMREX_SPACEDIM];

                                           bv_get_grads_and_jumps(i, j, k, normaldir, lset_id, dx, phi_arr, gradctol,
                                                                  mod_gradc, gradc_cutoff, facecolor, potjump, dphidn, dphidt1, dphidt2, n_ls);


                                           if (mod_gradc > gradc_cutoff)
                                           {
                                               Real activ_func = electrochem_reactions::bv_activation_function(facecolor, mod_gradc, gradc_cutoff);
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



void echemAMR::solve_mechanics(Real current_time)
{

    BL_PROFILE("echemAMR::solve_mechanics");
    amrex::Print() << "Performing Mechanics Solve: "<< std::endl;

    // Set parameters
    const int num_grow = 1;
    int max_coarsening_level = 30;//linsolve_max_coarsening_level;
    int verbose = 2;
    int bottom_verbose = 2;
    ProbParm const* localprobparm = d_prob_parm;

    // initialize solver
    LPInfo info;
    info.setMaxCoarseningLevel(max_coarsening_level);
    // std::unique_ptr<MLTensorOp> mltensor;
    // mltensor = std::unique_ptr<MLTensorOp>(new MLTensorOp({geom}, {grids}, {dmap}, info));
    MLTensorOp mltensor(Geom(0,finest_level), boxArray(0,finest_level), 
                        DistributionMap(0,finest_level), info);

    // Setup Dirichlet BC for all sides and components 
    Vector<Array<LinOpBCType,AMREX_SPACEDIM>> mlmg_lobc(AMREX_SPACEDIM);
    Vector<Array<LinOpBCType,AMREX_SPACEDIM>> mlmg_hibc(AMREX_SPACEDIM);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        for (int jdim = 0; jdim < AMREX_SPACEDIM; ++jdim) {
            mlmg_lobc[idim][jdim] = LinOpBCType::Dirichlet;//LinOpBCType::Neumann;
            mlmg_hibc[idim][jdim] = LinOpBCType::Dirichlet;// LinOpBCType::Neumann;
        }
    }

    mltensor.setDomainBC(mlmg_lobc, mlmg_hibc);

    // Init vectors
    Vector<MultiFab> displacement;
    Vector<MultiFab> solution;
    Vector<MultiFab> mech_bc;
    Vector<MultiFab> eta;
    Vector<MultiFab> kappa;
    Vector<MultiFab> lamG_deltaT;
    Vector<MultiFab> lamG_deltaT_gradient;

    // Get Solution ID 
    std::array<int, 8> sln_list = electrochem_mechanics::get_solution_ids();

    // Resize for level
    displacement.resize(finest_level + 1);
    solution.resize(finest_level + 1);
    mech_bc.resize(finest_level + 1);
    eta.resize(finest_level + 1);
    kappa.resize(finest_level + 1);
    lamG_deltaT.resize(finest_level + 1);
    lamG_deltaT_gradient.resize(finest_level + 1);

    for (int ilev = 0; ilev <= finest_level; ilev++)
    {

        // Define Vectors
        displacement[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, num_grow);
        solution[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, num_grow);
        mech_bc[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, num_grow);
        eta[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);
        kappa[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);
        lamG_deltaT[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);
        lamG_deltaT_gradient[ilev].define(grids[ilev], dmap[ilev], AMREX_SPACEDIM, num_grow);

        // Set Default values
        displacement[ilev].setVal(0.0);
        solution[ilev].setVal(0.0);
        mech_bc[ilev].setVal(0.0); 
        eta[ilev].setVal(0.0); 
        kappa[ilev].setVal(0.0); 
        lamG_deltaT[ilev].setVal(0.0); 
        lamG_deltaT_gradient[ilev].setVal(0.0); 

        // Grab current solution (phi) vector
        MultiFab Sborder(grids[ilev], dmap[ilev], phi_new[ilev].nComp(), num_grow);
        FillPatch(ilev, current_time, Sborder, 0, Sborder.nComp());

        // Extract current displacement
        amrex::Copy(displacement[ilev], Sborder, sln_list[0], 0, AMREX_SPACEDIM, num_grow);

        // // Set initial guess from displacement
        // amrex::MultiFab::Copy(solution[ilev], displacement[ilev], 0, 0, AMREX_SPACEDIM, 0);

        // set bcs to be zero 
        mltensor.setLevelBC(ilev, &(mech_bc[ilev]));

        // fill cell centered coefficients
        for (MFIter mfi(phi_new[ilev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Array4<Real> phi_arr = Sborder.array(mfi);
            Array4<Real> eta_arr = eta[ilev].array(mfi);
            Array4<Real> kappa_arr = kappa[ilev].array(mfi);
            Array4<Real> lamG_deltaT_arr = lamG_deltaT[ilev].array(mfi);

            const Box& bx = mfi.tilebox();
            const Box& gbx = amrex::grow(bx, 1);
            amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) 
                               {
                                   //FIXME: use component wise call
                                   electrochem_mechanics::compute_shear_modulus(i, j, k, phi_arr, eta_arr, *localprobparm);
                                   electrochem_mechanics::compute_bulk_modulus(i, j, k, phi_arr, eta_arr, kappa_arr, *localprobparm);
                                   electrochem_mechanics::compute_lamG_deltaT(i, j, k, phi_arr, lamG_deltaT_arr, *localprobparm);
                               });
        }

        // map coeffs to face values
        Array<MultiFab,AMREX_SPACEDIM> face_eta_coef;
        Array<MultiFab,AMREX_SPACEDIM> face_kappa_coef;
        Array<MultiFab,AMREX_SPACEDIM> face_lamG_deltaT;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            const BoxArray& ba = amrex::convert(grids[ilev], IntVect::TheDimensionVector(idim));
            face_eta_coef[idim].define(ba, eta[ilev].DistributionMap(), 1, 0);
            face_kappa_coef[idim].define(ba, kappa[ilev].DistributionMap(), 1, 0);
            face_lamG_deltaT[idim].define(ba, lamG_deltaT[ilev].DistributionMap(), 1, 0);
        }
        amrex::average_cellcenter_to_face(amrex::GetArrOfPtrs(face_eta_coef), eta[ilev], geom[ilev]);
        amrex::average_cellcenter_to_face(amrex::GetArrOfPtrs(face_kappa_coef), kappa[ilev], geom[ilev]);
        amrex::average_cellcenter_to_face(amrex::GetArrOfPtrs(face_lamG_deltaT), lamG_deltaT[ilev], geom[ilev]);

        // Here is where you could set face stresses to zero

        amrex::computeGradient(lamG_deltaT_gradient[ilev],amrex::GetArrOfConstPtrs(face_lamG_deltaT), geom[ilev]);
        mltensor.setShearViscosity(ilev, amrex::GetArrOfConstPtrs(face_eta_coef));
        mltensor.setBulkViscosity(ilev, amrex::GetArrOfConstPtrs(face_kappa_coef));
        mltensor.setACoeffs(ilev, 0.0);
    }



    // Setup solver
    MLMG mlmg(mltensor);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);
    mlmg.setPreSmooth(10);
    mlmg.setPostSmooth(10);
    mlmg.setFinalSmooth(10);
    mlmg.setBottomSmooth(10);
    mlmg.setMaxIter(2000);

    mlmg.setBottomTolerance(1.0e-6);
    mlmg.setBottomToleranceAbs(1.0e-6);

    // Solve
    Real mlmg_err = mlmg.solve(GetVecOfPtrs(solution), GetVecOfConstPtrs(lamG_deltaT_gradient), 0.005, 0.0);
    amrex::Print() << "Mechanics mlmg error: " << mlmg_err << std::endl;



    // compute von Mises stress
    Vector<MultiFab> von_Mises;
    von_Mises.resize(finest_level + 1);

    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        von_Mises[ilev].define(grids[ilev],dmap[ilev],1,0);

        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> fluxes;

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            const BoxArray& ba = amrex::convert(grids[ilev], IntVect::TheDimensionVector(idim));
            fluxes[idim].define(ba, von_Mises[ilev].DistributionMap(), AMREX_SPACEDIM, 0);
        }

        mltensor.compFlux(ilev, amrex::GetArrOfPtrs(fluxes), solution[ilev], amrex::MLMG::Location::FaceCenter);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

        for (MFIter mfi(von_Mises[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box bx = mfi.tilebox();

            AMREX_D_TERM(amrex::Array4<amrex::Real const> const& fx = fluxes[0].const_array(mfi);,
                         amrex::Array4<amrex::Real const> const& fy = fluxes[1].const_array(mfi);,
                         amrex::Array4<amrex::Real const> const& fz = fluxes[2].const_array(mfi););

            amrex::Array4<amrex::Real> const& von = von_Mises[ilev].array(mfi);
            amrex::Array4<amrex::Real const> const& lamG_deltaT_arr = lamG_deltaT[ilev].const_array(mfi);

            AMREX_HOST_DEVICE_PARALLEL_FOR_3D( bx, i, j, k,
            {
                amrex::Real s11 = amrex::Real(0.5) * -( fx(i,j,k,0) + fx(i+1,j,k,0) ) + lamG_deltaT_arr(i,j,k);
                amrex::Real s12 = amrex::Real(0.5) * -( fx(i,j,k,1) + fx(i+1,j,k,1) );
                amrex::Real s13 = amrex::Real(0.5) * -( fx(i,j,k,2) + fx(i+1,j,k,2) );

                amrex::Real s21 = amrex::Real(0.5) * -( fy(i,j,k,0) + fy(i,j+1,k,0) );
                amrex::Real s22 = amrex::Real(0.5) * -( fy(i,j,k,1) + fy(i,j+1,k,1) ) + lamG_deltaT_arr(i,j,k);
                amrex::Real s23 = amrex::Real(0.5) * -( fy(i,j,k,2) + fy(i,j+1,k,2) );

                amrex::Real s31 = amrex::Real(0.5) * -( fz(i,j,k,0) + fz(i,j,k+1,0) );
                amrex::Real s32 = amrex::Real(0.5) * -( fz(i,j,k,1) + fz(i,j,k+1,1) );
                amrex::Real s33 = amrex::Real(0.5) * -( fz(i,j,k,2) + fz(i,j,k+1,2) ) + lamG_deltaT_arr(i,j,k);

                von(i,j,k) = sqrt( 0.5*( (s11-s22)*(s11-s22)+(s22-s33)*(s22-s33)+(s33-s11)*(s33-s11) + 6.0*(s23*s23+s31*s31+s12*s12) ) );
            });
        }
    }

    // compute stress tensor (sigma 11)
    Vector<MultiFab> sigma11;  
    sigma11.resize(finest_level + 1);

    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        sigma11[ilev].define(grids[ilev],dmap[ilev],1,0);
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> fluxes;

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            const BoxArray& ba = amrex::convert(grids[ilev], IntVect::TheDimensionVector(idim));
            fluxes[idim].define(ba, sigma11[ilev].DistributionMap(), AMREX_SPACEDIM, 0);
        }
        mltensor.compFlux(ilev, amrex::GetArrOfPtrs(fluxes), solution[ilev], amrex::MLMG::Location::FaceCenter);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

        for (MFIter mfi(sigma11[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box bx = mfi.tilebox();

            AMREX_D_TERM(amrex::Array4<amrex::Real const> const& fx = fluxes[0].const_array(mfi);,
                         amrex::Array4<amrex::Real const> const& fy = fluxes[1].const_array(mfi);,
                         amrex::Array4<amrex::Real const> const& fz = fluxes[2].const_array(mfi););

            amrex::Array4<amrex::Real> const& s11 = sigma11[ilev].array(mfi);
            amrex::Array4<amrex::Real const> const& lamG_deltaT_arr = lamG_deltaT[ilev].const_array(mfi);

            AMREX_HOST_DEVICE_PARALLEL_FOR_3D( bx, i, j, k,
            {
                amrex::Real s11_value = amrex::Real(0.5) * -( fx(i,j,k,0) + fx(i+1,j,k,0) ) + lamG_deltaT_arr(i,j,k);
                s11(i,j,k) = s11_value;
            });
        }
    }
    // compute stress tensor (sigma 22)
    Vector<MultiFab> sigma22;  
    sigma22.resize(finest_level + 1);

    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        sigma22[ilev].define(grids[ilev],dmap[ilev],1,0);
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> fluxes;

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            const BoxArray& ba = amrex::convert(grids[ilev], IntVect::TheDimensionVector(idim));
            fluxes[idim].define(ba, sigma22[ilev].DistributionMap(), AMREX_SPACEDIM, 0);
        }
        mltensor.compFlux(ilev, amrex::GetArrOfPtrs(fluxes), solution[ilev], amrex::MLMG::Location::FaceCenter);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

        for (MFIter mfi(sigma22[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box bx = mfi.tilebox();

            AMREX_D_TERM(amrex::Array4<amrex::Real const> const& fx = fluxes[0].const_array(mfi);,
                         amrex::Array4<amrex::Real const> const& fy = fluxes[1].const_array(mfi);,
                         amrex::Array4<amrex::Real const> const& fz = fluxes[2].const_array(mfi););

            amrex::Array4<amrex::Real> const& s22 = sigma22[ilev].array(mfi);
            amrex::Array4<amrex::Real const> const& lamG_deltaT_arr = lamG_deltaT[ilev].const_array(mfi);

            AMREX_HOST_DEVICE_PARALLEL_FOR_3D( bx, i, j, k,
            {
                amrex::Real s22_value = amrex::Real(0.5) * -( fy(i,j,k,1) + fy(i,j+1,k,1) ) + lamG_deltaT_arr(i,j,k);
                s22(i,j,k) = s22_value;
            });
        }
    }    
    // compute stress tensor (sigma 33)
    Vector<MultiFab> sigma33;  
    sigma33.resize(finest_level + 1);

    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        sigma33[ilev].define(grids[ilev],dmap[ilev],1,0);
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> fluxes;

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            const BoxArray& ba = amrex::convert(grids[ilev], IntVect::TheDimensionVector(idim));
            fluxes[idim].define(ba, sigma33[ilev].DistributionMap(), AMREX_SPACEDIM, 0);
        }
        mltensor.compFlux(ilev, amrex::GetArrOfPtrs(fluxes), solution[ilev], amrex::MLMG::Location::FaceCenter);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

        for (MFIter mfi(sigma33[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box bx = mfi.tilebox();

            AMREX_D_TERM(amrex::Array4<amrex::Real const> const& fx = fluxes[0].const_array(mfi);,
                         amrex::Array4<amrex::Real const> const& fy = fluxes[1].const_array(mfi);,
                         amrex::Array4<amrex::Real const> const& fz = fluxes[2].const_array(mfi););

            amrex::Array4<amrex::Real> const& s33 = sigma33[ilev].array(mfi);
            amrex::Array4<amrex::Real const> const& lamG_deltaT_arr = lamG_deltaT[ilev].const_array(mfi);

            AMREX_HOST_DEVICE_PARALLEL_FOR_3D( bx, i, j, k,
            {
                amrex::Real s33_value = amrex::Real(0.5) * -( fz(i,j,k,2) + fz(i,j,k+1,2) ) + lamG_deltaT_arr(i,j,k);
                s33(i,j,k) = s33_value;
            });
        }
    }
    // compute stress tensor (sigma 12)
    Vector<MultiFab> sigma12;  
    sigma12.resize(finest_level + 1);

    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        sigma12[ilev].define(grids[ilev],dmap[ilev],1,0);
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> fluxes;

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            const BoxArray& ba = amrex::convert(grids[ilev], IntVect::TheDimensionVector(idim));
            fluxes[idim].define(ba, sigma12[ilev].DistributionMap(), AMREX_SPACEDIM, 0);
        }
        mltensor.compFlux(ilev, amrex::GetArrOfPtrs(fluxes), solution[ilev], amrex::MLMG::Location::FaceCenter);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

        for (MFIter mfi(sigma12[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box bx = mfi.tilebox();

            AMREX_D_TERM(amrex::Array4<amrex::Real const> const& fx = fluxes[0].const_array(mfi);,
                         amrex::Array4<amrex::Real const> const& fy = fluxes[1].const_array(mfi);,
                         amrex::Array4<amrex::Real const> const& fz = fluxes[2].const_array(mfi););

            amrex::Array4<amrex::Real> const& s12 = sigma12[ilev].array(mfi);
            amrex::Array4<amrex::Real const> const& lamG_deltaT_arr = lamG_deltaT[ilev].const_array(mfi);

            AMREX_HOST_DEVICE_PARALLEL_FOR_3D( bx, i, j, k,
            {
                amrex::Real s12_value = amrex::Real(0.5) * -( fx(i,j,k,1) + fx(i+1,j,k,1) );
                s12(i,j,k) = s12_value;
            });
        }
    }         
   // compute stress tensor (sigma 13)
    Vector<MultiFab> sigma13;  
    sigma13.resize(finest_level + 1);

    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        sigma13[ilev].define(grids[ilev],dmap[ilev],1,0);
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> fluxes;

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            const BoxArray& ba = amrex::convert(grids[ilev], IntVect::TheDimensionVector(idim));
            fluxes[idim].define(ba, sigma13[ilev].DistributionMap(), AMREX_SPACEDIM, 0);
        }
        mltensor.compFlux(ilev, amrex::GetArrOfPtrs(fluxes), solution[ilev], amrex::MLMG::Location::FaceCenter);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

        for (MFIter mfi(sigma13[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box bx = mfi.tilebox();

            AMREX_D_TERM(amrex::Array4<amrex::Real const> const& fx = fluxes[0].const_array(mfi);,
                         amrex::Array4<amrex::Real const> const& fy = fluxes[1].const_array(mfi);,
                         amrex::Array4<amrex::Real const> const& fz = fluxes[2].const_array(mfi););

            amrex::Array4<amrex::Real> const& s13 = sigma13[ilev].array(mfi);
            amrex::Array4<amrex::Real const> const& lamG_deltaT_arr = lamG_deltaT[ilev].const_array(mfi);

            AMREX_HOST_DEVICE_PARALLEL_FOR_3D( bx, i, j, k,
            {
                amrex::Real s13_value = amrex::Real(0.5) * -( fx(i,j,k,2) + fx(i+1,j,k,2) );
                s13(i,j,k) = s13_value;
            });
        }
    }         
    // compute stress tensor (sigma 23)
    Vector<MultiFab> sigma23;  
    sigma23.resize(finest_level + 1);

    for (int ilev = 0; ilev <= finest_level; ilev++)
    {
        sigma23[ilev].define(grids[ilev],dmap[ilev],1,0);
        amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> fluxes;

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            const BoxArray& ba = amrex::convert(grids[ilev], IntVect::TheDimensionVector(idim));
            fluxes[idim].define(ba, sigma23[ilev].DistributionMap(), AMREX_SPACEDIM, 0);
        }
        mltensor.compFlux(ilev, amrex::GetArrOfPtrs(fluxes), solution[ilev], amrex::MLMG::Location::FaceCenter);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

        for (MFIter mfi(sigma23[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box bx = mfi.tilebox();

            AMREX_D_TERM(amrex::Array4<amrex::Real const> const& fx = fluxes[0].const_array(mfi);,
                         amrex::Array4<amrex::Real const> const& fy = fluxes[1].const_array(mfi);,
                         amrex::Array4<amrex::Real const> const& fz = fluxes[2].const_array(mfi););

            amrex::Array4<amrex::Real> const& s23 = sigma23[ilev].array(mfi);
            amrex::Array4<amrex::Real const> const& lamG_deltaT_arr = lamG_deltaT[ilev].const_array(mfi);

            AMREX_HOST_DEVICE_PARALLEL_FOR_3D( bx, i, j, k,
            {
                amrex::Real s23_value = amrex::Real(0.5) * -( fx(i,j,k,2) + fx(i+1,j,k,2) );
                s23(i,j,k) = s23_value;
            });
        }
    }  

    // copy solution back to phi_new
    for (int ilev = 0; ilev <= finest_level; ilev++)
    {   
        Print()<<"max of solution:"<<solution[ilev].max(0)<<"\n";
        Print()<<"min of solution:"<<solution[ilev].min(0)<<"\n";
        Print()<<"max of eta:"<<eta[ilev].max(0)<<"\n";
        Print()<<"min of eta:"<<eta[ilev].min(0)<<"\n";
        Print()<<"max of lamG_deltaT:"<<lamG_deltaT[ilev].max(0)<<"\n";
        Print()<<"min of lamG_deltaT:"<<lamG_deltaT[ilev].min(0)<<"\n";
        amrex::MultiFab::Copy(phi_new[ilev], solution[ilev], 0,  sln_list[0], AMREX_SPACEDIM, 0);
        amrex::MultiFab::Copy(phi_new[ilev], von_Mises[ilev], 0, sln_list[1], 1, 0);
        amrex::MultiFab::Copy(phi_new[ilev], sigma11[ilev], 0,   sln_list[2], 1, 0);
        amrex::MultiFab::Copy(phi_new[ilev], sigma22[ilev], 0,   sln_list[3], 1, 0);
        amrex::MultiFab::Copy(phi_new[ilev], sigma33[ilev], 0,   sln_list[4], 1, 0);
        amrex::MultiFab::Copy(phi_new[ilev], sigma12[ilev], 0,   sln_list[5], 1, 0);
        amrex::MultiFab::Copy(phi_new[ilev], sigma13[ilev], 0,   sln_list[6], 1, 0);                
        amrex::MultiFab::Copy(phi_new[ilev], sigma23[ilev], 0,   sln_list[7], 1, 0); 
    }

}
