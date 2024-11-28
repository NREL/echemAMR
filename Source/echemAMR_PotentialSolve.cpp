#include <echemAMR.H>

#include <Chemistry.H>
#include <Transport.H>
#include <Reactions.H>
#include <bv_utils.H>

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

    const int num_grow = 2;
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
        mlmg.setBottomMaxIter(linsolve_bottom_maxiter);

        mlmg.setPreSmooth(linsolve_num_pre_smooth);
        mlmg.setPostSmooth(linsolve_num_post_smooth);
        mlmg.setFinalSmooth(linsolve_num_final_smooth);
        mlmg.setBottomSmooth(linsolve_num_bottom_smooth);

#ifdef AMREX_USE_HYPRE
        if (use_hypre)
        {
            mlmg.setHypreOptionsNamespace("echemamr.hypre");
            mlmg.setBottomSolver(MLMG::BottomSolver::hypre);
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
                        auto plo = geom[ilev].ProbLoArray();
                        auto phi = geom[ilev].ProbHiArray();
                        const int* domlo_arr = geom[ilev].Domain().loVect();
                        const int* domhi_arr = geom[ilev].Domain().hiVect();
                        GpuArray<int,AMREX_SPACEDIM> domlo={AMREX_D_DECL(domlo_arr[0], domlo_arr[1], domlo_arr[2])};
                        GpuArray<int,AMREX_SPACEDIM> domhi={AMREX_D_DECL(domhi_arr[0], domhi_arr[1], domhi_arr[2])};
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
                            Real intloc[AMREX_SPACEDIM];


                            bv_get_grads_and_jumps(i, j, k, normaldir, lset_id, dx, phi_arr, gradctol,
                                                   mod_gradc, gradc_cutoff, facecolor, potjump, dphidn, 
                                                   dphidt1, dphidt2, n_ls, intloc, 
                                                   plo, phi);


                            explterms_arr(i, j, k) = 0.0;
                            explterms_arr_res(i, j, k) = 0.0;
                            Real activ_func = electrochem_reactions::bv_activation_function(facecolor, mod_gradc, gradc_cutoff);

                            if (mod_gradc > gradc_cutoff && activ_func > 0.0)
                            {

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
