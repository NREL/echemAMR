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
#include<PostProcessing.H>
#include<ChemistryProbParm.H>
#include <AMReX_MLABecLaplacian.H>

// advance solution to final time
void echemAMR::Evolve ()
{
    Real cur_time = t_new[0];
    int last_plot_file_step = 0;

    postprocess(cur_time, 0, 0.0, echemAMR::host_global_storage);

    // feel free to delete this commented chunk
    // amrex::Print() << "ProbParm Anode Volume:   " << echemAMR::h_prob_parm->anode_volume << std::endl;
    // amrex::Print() << "ProbParm ELSEP Volume:   " << echemAMR::h_prob_parm->ele_sep_volume << std::endl;
    // amrex::Print() << "ProbParm Cathode Volume: " << echemAMR::h_prob_parm->cathode_volume << std::endl;
    // amrex::Print() << "ProbParm Anode SA:       " << echemAMR::h_prob_parm->anode_SA << std::endl;
    // amrex::Print() << "ProbParm Electrolyte SA: " << echemAMR::h_prob_parm->electrolyte_SA << std::endl;
    // amrex::Print() << "ProbParm Separator SA:   " << echemAMR::h_prob_parm->sepatator_SA << std::endl;
    // amrex::Print() << "ProbParm Cathode SA:     " << echemAMR::h_prob_parm->cathode_SA << std::endl;
    // amrex::Print() << "ProbParm AE SA:          " << echemAMR::h_prob_parm->anode_ele_SA << std::endl;
    // amrex::Print() << "ProbParm CE SA:          " << echemAMR::h_prob_parm->cathode_ele_SA << std::endl;
    // amrex::Print() << "ProbParm Anode CC SA:    " << echemAMR::h_prob_parm->anode_cc_SA << std::endl;
    // amrex::Print() << "ProbParm Cathode CC SA:  " << echemAMR::h_prob_parm->cathode_cc_SA << std::endl;
    // amrex::Print() << "ProbParm Total SA:       " << echemAMR::h_prob_parm->anode_SA + echemAMR::h_prob_parm->electrolyte_SA + echemAMR::h_prob_parm->sepatator_SA + echemAMR::h_prob_parm->cathode_SA - 2.0*echemAMR::h_prob_parm->anode_ele_SA - 2.0*echemAMR::h_prob_parm->cathode_ele_SA - 10e-6*10e-6*4 << std::endl;
    // exit(1);

    for (int step = istep[0]; step < max_step && cur_time < stop_time; ++step)
    {
        amrex::Print() << "\nCoarse STEP " << step+1 << " starts ..." << std::endl;

	ComputeDt();

	int lev = 0;
	int iteration = 1;
        
        if(potential_solve==1 && step % pot_solve_int == 0)
        {
            for(int i=0;i<20;++i) solve_potential(cur_time);
        }
        
	timeStep(lev, cur_time, iteration);

	cur_time += dt[0];
        
        postprocess(cur_time, step, dt[0], echemAMR::host_global_storage);

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

void echemAMR::solve_potential(Real current_time)
{
    LPInfo info;

    // FIXME: add these as inputs
    int agglomeration=1;
    int consolidation=1;
    int max_coarsening_level=30;
    bool semicoarsening = false;
    int max_semicoarsening_level=0;
    int linop_maxorder=2;
    int max_iter=100;
    int max_fmg_iter=0;
    int verbose = 2;
    int bottom_verbose = 0;
    Real ascalar=0.0;
    Real bscalar=1.0;

    //==================================================
    //amrex solves
    //read small a as alpha, b as beta

    //(A a - B del.b) phi = f
    //
    //A and B are scalar constants
    //a and b are scalar fields
    //f is rhs
    //in this case: A=0,a=0,B=1,b=conductivity
    //note also the negative sign
    //====================================================
    
    // FIXME: had to adjust for constant coefficent, 
    // this could be due to missing terms in the intercalation reaction or sign mistakes...
    ProbParm const* localprobparm = d_prob_parm;

    const Real tol_rel = 1.0e-6;
    const Real tol_abs = 1.0e-13;

#ifdef AMREX_USE_HYPRE
        int hypre_interface_i = 1;  // 1. structed, 2. semi-structed, 3. ij
        Hypre::Interface hypre_interface = Hypre::Interface::structed;
#endif

    info.setAgglomeration(agglomeration);
    info.setConsolidation(consolidation);
    info.setSemicoarsening(semicoarsening);
    info.setMaxCoarseningLevel(max_coarsening_level);
    info.setMaxSemicoarseningLevel(max_semicoarsening_level);

    MLABecLaplacian mlabec(geom, grids, dmap, info);

    mlabec.setMaxOrder(linop_maxorder);

    //set A and B, A=0, B=1
    //
    if(buttler_vohlmer_flux)
    {
        ascalar=bv_relaxfac;   
    }
    else
    {
        ascalar=0.0;
    }
    mlabec.setScalars(ascalar, bscalar);

    // default to inhomogNeumann since it is defaulted to flux = 0.0 anyways
    std::array<LinOpBCType,AMREX_SPACEDIM> bc_potential_lo
        ={LinOpBCType::inhomogNeumann,
            LinOpBCType::inhomogNeumann,
            LinOpBCType::inhomogNeumann};

    std::array<LinOpBCType,AMREX_SPACEDIM> bc_potential_hi
        ={LinOpBCType::inhomogNeumann,
            LinOpBCType::inhomogNeumann,
            LinOpBCType::inhomogNeumann};


    for(int idim=0;idim<AMREX_SPACEDIM;idim++)
    {
        if(bc_lo[idim] == BCType::int_dir)
        {
            bc_potential_lo[idim]=LinOpBCType::Periodic;
        }
        else if(bc_lo[idim] == BCType::ext_dir)
        {
            bc_potential_lo[idim]=LinOpBCType::Dirichlet;
        }

        if(bc_hi[idim] == BCType::int_dir)
        {
            bc_potential_hi[idim]=LinOpBCType::Periodic;
        }
        if(bc_hi[idim] == BCType::ext_dir)
        {
            bc_potential_hi[idim]=LinOpBCType::Dirichlet;
        }
    }

    mlabec.setDomainBC(bc_potential_lo,bc_potential_hi);

    Vector<MultiFab> potential;
    Vector<MultiFab> acoeff;
    Vector<MultiFab> bcoeff;
    Vector<Array<MultiFab *,AMREX_SPACEDIM>> gradsoln;
    Vector<MultiFab> solution;
    Vector<MultiFab> rhs;

    acoeff.resize(finest_level+1);
    bcoeff.resize(finest_level+1);
    gradsoln.resize(finest_level+1);
    potential.resize(finest_level+1);
    solution.resize(finest_level+1);
    rhs.resize(finest_level+1);

    const int num_grow=1;

    for(int ilev=0; ilev<=finest_level; ilev++)
    {
        MultiFab Sborder(grids[ilev], dmap[ilev], phi_new[ilev].nComp(), num_grow);
        FillPatch(ilev, current_time, Sborder, 0, Sborder.nComp()); 
        potential[ilev].define(grids[ilev], dmap[ilev], 1, num_grow);
        potential[ilev].setVal(0.0);

        //Copy (FabArray<FAB>& dst, FabArray<FAB> const& src, int srccomp, int dstcomp, int numcomp, const IntVect& nghost)
        amrex::Copy(potential[ilev], Sborder, NVAR-1, 0, 1, num_grow);

        acoeff[ilev].define(grids[ilev], dmap[ilev], 1, 0);
        bcoeff[ilev].define(grids[ilev], dmap[ilev], 1, 1);
        solution[ilev].define(grids[ilev], dmap[ilev], 1, 1);
        rhs[ilev].define(grids[ilev], dmap[ilev], 1, 0);

        acoeff[ilev].setVal(0.0);
        mlabec.setACoeffs(ilev, acoeff[ilev]);

        bcoeff[ilev].setVal(1.0);
        solution[ilev].setVal(0.0);
        rhs[ilev].setVal(0.0);

        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            const BoxArray& faceba = amrex::convert(grids[ilev],
                    IntVect::TheDimensionVector(idim));

            gradsoln[ilev][idim] = new MultiFab(faceba, dmap[ilev], 1, 0);
        }

        //copy current solution for better guess
        if(pot_initial_guess)
        {
            amrex::MultiFab::Copy(solution[ilev],potential[ilev], 0, 0, 1 ,0);
            //solution[ilev].copy(potential[ilev], 0, 0, 1);
        }

        // fill cell centered diffusion coefficients and rhs
        for (MFIter mfi(phi_new[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const Box& gbx = amrex::grow(bx,1);
            const auto dx = geom[ilev].CellSizeArray();
            auto prob_lo = geom[ilev].ProbLoArray();
            auto prob_hi = geom[ilev].ProbHiArray();
            const Box& domain = geom[ilev].Domain();

            Real time=current_time; //for GPU capture

            Array4<Real> phi_arr      = Sborder.array(mfi);
            Array4<Real> bcoeff_arr   = bcoeff[ilev].array(mfi);
            Array4<Real> rhs_arr      = rhs[ilev].array(mfi);
            
            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                electrochem_transport::compute_potential_dcoeff
                (i, j, k, phi_arr, bcoeff_arr, prob_lo, prob_hi, dx, time, *localprobparm);
            });
            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                electrochem_reactions::compute_potential_source
                (i, j, k, phi_arr, rhs_arr, prob_lo, prob_hi, dx, time, *localprobparm);
            });

        }

        // average cell coefficients to faces, this includes boundary faces
        Array<MultiFab,AMREX_SPACEDIM> face_bcoeff;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            const BoxArray& ba = amrex::convert(bcoeff[ilev].boxArray(), IntVect::TheDimensionVector(idim));
            face_bcoeff[idim].define(ba, bcoeff[ilev].DistributionMap(), 1, 0);
        }
        //true argument for harmonic averaging
        amrex::average_cellcenter_to_face(GetArrOfPtrs(face_bcoeff), bcoeff[ilev], geom[ilev],true);

        if(buttler_vohlmer_flux)
        {
            int lset_id=bv_levset_id;
        
            Array<MultiFab,AMREX_SPACEDIM> bv_explicit_terms;
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                const BoxArray& ba = amrex::convert(bcoeff[ilev].boxArray(), IntVect::TheDimensionVector(idim));
                bv_explicit_terms[idim].define(ba, bcoeff[ilev].DistributionMap(), 1, 0);
            }

            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                for (MFIter mfi(phi_new[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const auto dx = geom[ilev].CellSizeArray();
                    const Box& bx = mfi.tilebox();
                    Real min_dx = amrex::min(dx[0],amrex::min(dx[1],dx[2]));

                    //face box
                    Box fbox=convert(bx,IntVect::TheDimensionVector(idim));
                    Array4<Real> phi_arr       = Sborder.array(mfi);
                    Array4<Real> dcoeff_arr    = face_bcoeff[idim].array(mfi);
                    Array4<Real> explterms_arr = bv_explicit_terms[idim].array(mfi);
            
                    amrex::ParallelFor(fbox,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        IntVect left(i,j,k);
                        IntVect right(i,j,k);
                        
                        IntVect top_left(i,j,k);
                        IntVect bottom_left(i,j,k);
                        IntVect top_right(i,j,k);
                        IntVect bottom_right(i,j,k);
                        
                        IntVect front_left(i,j,k);
                        IntVect back_left(i,j,k);
                        IntVect front_right(i,j,k);
                        IntVect back_right(i,j,k);

                        int normaldir=idim;
                        int trans1dir=(idim+1)%AMREX_SPACEDIM;
                        int trans2dir=(idim+2)%AMREX_SPACEDIM;


                        left[idim]              -= 1;
                        top_left[idim]          -= 1;
                        bottom_left[idim]       -= 1;
                        front_left[idim]        -= 1;
                        back_left[idim]         -= 1;

                        top_left[trans1dir]     += 1;
                        top_right[trans1dir]    += 1;
                        bottom_left[trans1dir]  -= 1;
                        bottom_right[trans1dir] -= 1;
                        
                        front_left[trans2dir]   += 1;
                        front_right[trans2dir]  += 1;
                        back_left[trans2dir]    -= 1;
                        back_right[trans2dir]   -= 1;

                        //levelset color
                        Real c_left  = phi_arr(left,lset_id);
                        Real c_right = phi_arr(right,lset_id);

                        Real c_top = 0.5*(phi_arr(top_left,lset_id)    +    phi_arr(top_right,lset_id));
                        Real c_bot = 0.5*(phi_arr(bottom_left,lset_id) + phi_arr(bottom_right,lset_id));
                        
                        Real c_frnt = 0.5*(phi_arr(front_left,lset_id)  + phi_arr(front_right,lset_id));
                        Real c_back = 0.5*(phi_arr(back_left,lset_id)   +  phi_arr(back_right,lset_id));
                        
                        //phi
                        Real pot_left  = phi_arr(left,NVAR-1);
                        Real pot_right = phi_arr(right,NVAR-1);

                        Real pot_top = 0.5*(phi_arr(top_left,NVAR-1)    +    phi_arr(top_right,NVAR-1));
                        Real pot_bot = 0.5*(phi_arr(bottom_left,NVAR-1) + phi_arr(bottom_right,NVAR-1));
                        
                        Real pot_frnt = 0.5*(phi_arr(front_left,NVAR-1)  + phi_arr(front_right,NVAR-1));
                        Real pot_back = 0.5*(phi_arr(back_left,NVAR-1)   +  phi_arr(back_right,NVAR-1));

                        //x,y or z
                        Real dcdn  = (c_right-c_left)/dx[normaldir];
                        Real dcdt1 = (c_top-c_bot)/(2.0*dx[trans1dir]);
                        Real dcdt2 = (c_frnt-c_back)/(2.0*dx[trans2dir]);
                        
                        //grad of potential
                        Real dphidn  = (pot_right-pot_left)/dx[normaldir];
                        Real dphidt1 = (pot_top-pot_bot)/(2.0*dx[trans1dir]);
                        Real dphidt2 = (pot_frnt-pot_back)/(2.0*dx[trans2dir]);

                        Real gradc_tolfac=1e-4;
                        Real gradc_max=1.0/min_dx; //maximum gradient possible on the current grid
                        Real gradc_cutoff=gradc_tolfac*gradc_max;

                        Real mod_gradc = sqrt(dcdn*dcdn + dcdt1*dcdt1 + dcdt2*dcdt2);
                
                        if(mod_gradc > gradc_cutoff)
                        {
                            Real n_ls[3];
                            n_ls[0] = dcdn/mod_gradc;
                            n_ls[1] = dcdt1/mod_gradc;
                            n_ls[2] = dcdt2/mod_gradc;

                            Real activ_func = electrochem_reactions::
                                bv_activation_function(0.5*(c_right+c_left),
                                    mod_gradc, gradc_cutoff);

                            //jump along the level set normal (phi_electrolyte-phi_electrode)
                            Real phi_jump  = (dphidn*n_ls[0] + dphidt1*n_ls[1] + dphidt2*n_ls[2])/mod_gradc;

                            /*if(phi_jump > 100)
                            {
                                Print()<<"phi_jump:"<<phi_jump<<"\t"<<dphidn<<"\t"<<dphidt1<<"\t"<<dphidt2<<"\t"
                                    <<dcdn<<"\t"<<dcdt1<<"\t"<<dcdt2<<"\t"<<mod_gradc<<"\n";
                            }*/

                            //FIXME: pass ion concentration also
                            //FIXME: ideally it should be the ion concentration at the closest electrode cell
                            Real j_bv = electrochem_reactions::bvcurrent(phi_jump);
                            Real jdash_bv = electrochem_reactions::bvcurrent_der(phi_jump);


                            dcoeff_arr(i,j,k) *= (1.0-activ_func);
                            //dcoeff_arr(i,j,k) += -jdash_bv*activ_func/pow(mod_gradc,3.0) * dcdn*dcdn;
                            dcoeff_arr(i,j,k) += -jdash_bv*activ_func/mod_gradc * n_ls[0]*n_ls[0];

                            //expl term1
                            //explterms_arr(i,j,k) =   j_bv*activ_func*dcdn/mod_gradc;
                            explterms_arr(i,j,k) =   j_bv*activ_func*n_ls[0];

                            //expl term2
                            //explterms_arr(i,j,k) += -jdash_bv*phi_jump*activ_func*dcdn/mod_gradc; 
                            explterms_arr(i,j,k) += -jdash_bv*phi_jump*activ_func*n_ls[0]; 

                            //expl term3 (mix derivative terms from tensor product)
                            //explterms_arr(i,j,k) +=  jdash_bv*activ_func/pow(mod_gradc,3.0)*(dcdn*dcdt1*dphidt1+dcdn*dcdt2*dphidt2);
                            explterms_arr(i,j,k) +=  jdash_bv*activ_func/mod_gradc*(n_ls[0]*n_ls[1]*dphidt1+n_ls[0]*n_ls[2]*dphidt2);
                        } 

                    });

                }
            }

            for (MFIter mfi(phi_new[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();
                const auto dx = geom[ilev].CellSizeArray();

                Array4<Real> rhs_arr      = rhs[ilev].array(mfi);
                Array4<Real> phi_arr      = Sborder.array(mfi);

                Array4<Real> term_x = bv_explicit_terms[0].array(mfi);
                Array4<Real> term_y = bv_explicit_terms[1].array(mfi);
                Array4<Real> term_z = bv_explicit_terms[2].array(mfi);

                Real relax_fac=bv_relaxfac;

                amrex::ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                        rhs_arr(i,j,k) =      (term_x(i,j,k) - term_x(i+1,j,k)) / dx[0] 
                        +   (term_y(i,j,k) - term_y(i,j+1,k)) / dx[1] 
                        +   (term_z(i,j,k) - term_z(i,j,k+1)) / dx[2];

                        rhs_arr(i,j,k) += phi_arr(i,j,k,NVAR-1)*relax_fac;
                });
            }
        }

        // set boundary conditions
        for (MFIter mfi(phi_new[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const auto dx = geom[ilev].CellSizeArray();
            auto prob_lo = geom[ilev].ProbLoArray();
            auto prob_hi = geom[ilev].ProbHiArray();
            const Box& domain = geom[ilev].Domain();

            Array4<Real> phi_arr = Sborder.array(mfi);
            Array4<Real> bc_arr = potential[ilev].array(mfi);
            Real time = current_time; //for GPU capture

            for(int idim = 0; idim < AMREX_SPACEDIM; ++idim)
            {
                const amrex::Real bclo = host_global_storage->pot_bc_lo[idim];
                const amrex::Real bchi = host_global_storage->pot_bc_hi[idim];

                if (!geom[ilev].isPeriodic(idim)) 
                {
                    if (bx.smallEnd(idim) == domain.smallEnd(idim)) 
                    {
                        amrex::ParallelFor(
                                amrex::bdryLo(bx, idim),
                                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
                                {
                                electrochem_transport::potential_bc(i, j, k, idim, -1, phi_arr, 
                                        bc_arr, prob_lo, prob_hi, dx, time, bclo, bchi);
                                });
                    }
                    if (bx.bigEnd(idim) == domain.bigEnd(idim)) 
                    {
                        amrex::ParallelFor(
                                amrex::bdryHi(bx, idim),
                                [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept 
                                {
                                electrochem_transport::potential_bc(i, j, k, idim, 1, phi_arr, 
                                        bc_arr, prob_lo, prob_hi, dx, time, bclo, bchi);
                                });
                    }
                }
            }
        }

        //set b with diffusivities
        mlabec.setBCoeffs(ilev, amrex::GetArrOfConstPtrs(face_bcoeff));

        // bc's are stored in the ghost cells of potential
        mlabec.setLevelBC(ilev, &(potential[ilev]));

    }


    MLMG mlmg(mlabec);
    mlmg.setMaxIter(max_iter);
    mlmg.setMaxFmgIter(max_fmg_iter);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);
#ifdef AMREX_USE_HYPRE
    if (use_hypre) 
    {
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

    //need user-defined rhs

    mlmg.solve(GetVecOfPtrs(solution), GetVecOfConstPtrs(rhs), tol_rel, tol_abs);

    mlmg.getGradSolution(gradsoln);
    // reset ghost cells using boundary condition values
    // this is commented out since the fillpatch will overwrite this anyways
    //    for(int ilev=0; ilev<=finest_level; ilev++)
    //    {
    //        for (MFIter mfi(phi_new[ilev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
    //        {
    //            const Box& bx = mfi.tilebox();
    //            const Box& gbx = amrex::grow(bx,1);
//            const auto dx = geom[ilev].CellSizeArray();
//            auto prob_lo = geom[ilev].ProbLoArray();
//            auto prob_hi = geom[ilev].ProbHiArray();
//            const Box& domain = geom[ilev].Domain();
//
//            Array4<Real> bc_arr = potential[ilev].array(mfi);
//            Array4<Real> sol_arr = solution[ilev].array(mfi);
//
//            int idim = 0;
//            if (!geom[ilev].isPeriodic(idim)) {
//                if (bx.smallEnd(idim) == domain.smallEnd(idim)) {
//                      amrex::ParallelFor(
//                          amrex::bdryLo(bx, idim),
//                          [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
//                            //(sol(i) - sol(i-1))/dx = bc -> sol(i-1) = sol(i) - bc*dx
//                            sol_arr(i-1,j,k) = sol_arr(i,j,k) - bc_arr(i-1,j,k)*dx[idim];
//                          });
//                  }
//                  if (bx.bigEnd(idim) == domain.bigEnd(idim)) {
//                      amrex::ParallelFor(
//                          amrex::bdryHi(bx, idim),
//                          [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
//                              sol_arr(i,j,k) = - sol_arr(i-1,j,k) + bc_arr(i,j,k)*dx[idim];
//                          });
//                  }
//            }
//
//            idim = 1;
//            if (!geom[ilev].isPeriodic(idim)) {
//                if (bx.smallEnd(idim) == domain.smallEnd(idim)) {
//                      amrex::ParallelFor(
//                          amrex::bdryLo(bx, idim),
//                          [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
//                            sol_arr(i,j-1,k) = sol_arr(i,j,k) - bc_arr(i,j-1,k)*dx[idim];
//                          });
//                  }
//                  if (bx.bigEnd(idim) == domain.bigEnd(idim)) {
//                      amrex::ParallelFor(
//                          amrex::bdryHi(bx, idim),
//                          [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
//                              sol_arr(i,j,k) = - sol_arr(i,j-1,k) + bc_arr(i,j,k)*dx[idim];
//                          });
//                  }
//            }
//
//            idim = 2;
//            if (!geom[ilev].isPeriodic(idim)) {
//                if (bx.smallEnd(idim) == domain.smallEnd(idim)) {
//                      amrex::ParallelFor(
//                          amrex::bdryLo(bx, idim),
//                          [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
//                            sol_arr(i,j,k-1) = sol_arr(i,j,k) - bc_arr(i,j,k-1)*dx[idim];
//                          });
//                  }
//                  if (bx.bigEnd(idim) == domain.bigEnd(idim)) {
//                      amrex::ParallelFor(
//                          amrex::bdryHi(bx, idim),
//                          [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
//                              sol_arr(i,j,k) = - sol_arr(i,j,k-1) + bc_arr(i,j,k)*dx[idim];
//                          });
//                  }
//            }
//        }
//    }

    //copy solution back to phi_new
    for(int ilev=0; ilev<=finest_level; ilev++)
    {
        amrex::MultiFab::Copy(phi_new[ilev], solution[ilev], 0, NVAR-1, 1, 0);
        //phi_new[ilev].copy(solution[ilev], 0, NVAR-1, 1);
        const Array<const MultiFab *,AMREX_SPACEDIM> 
        allgrad={gradsoln[ilev][0],gradsoln[ilev][1],gradsoln[ilev][2]};
        average_face_to_cellcenter(phi_new[ilev], NVAR-4, allgrad);
        phi_new[ilev].mult(-1.0, NVAR-4, 3);
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
    
    //time is current time which is t_old, so phi_old is picked up!
    //but wait, we swapped phi_old and new, so we get phi_new here
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
    auto prob_hi = geom[lev].ProbHiArray();
    ProbParm const* localprobparm = d_prob_parm;

    int ncomp=Sborder.nComp();

    // Build temporary multiFabs to work on.
    Array<MultiFab, AMREX_SPACEDIM> flux;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) 
    {
        BoxArray ba = amrex::convert(dsdt.boxArray(), IntVect::TheDimensionVector(idim));
        flux[idim].define (ba,dsdt.DistributionMap(), ncomp, 0);
        flux[idim].setVal(0.0);
    }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
	for (MFIter mfi(dsdt,TilingIfNotGPU()); mfi.isValid(); ++mfi)
	{
	    const Box& bx = mfi.tilebox();
            const Box& gbx = amrex::grow(bx,1);
            Box bx_x=convert(bx, {1,0,0});
            Box bx_y=convert(bx, {0,1,0});
            Box bx_z=convert(bx, {0,0,1});

            FArrayBox dcoeff_fab(gbx,ncomp);
            FArrayBox velx_fab(bx_x,ncomp);
            FArrayBox vely_fab(bx_y,ncomp);
            FArrayBox velz_fab(bx_z,ncomp);
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

            Array4<Real> sborder_arr     = Sborder.array(mfi);
            Array4<Real> dcoeff_arr      = dcoeff_fab.array();
            Array4<Real> dsdt_arr        = dsdt.array(mfi);
            Array4<Real> reactsource_arr = reactsource_fab.array();
            Array4<Real> velx_arr        = velx_fab.array();
            Array4<Real> vely_arr        = vely_fab.array();
            Array4<Real> velz_arr        = velz_fab.array();

            GpuArray<Array4<Real>, AMREX_SPACEDIM> flux_arr{ AMREX_D_DECL(flux[0].array(mfi),
                                                                          flux[1].array(mfi),
                                                                          flux[2].array(mfi)) };

            dcoeff_fab.setVal<RunOn::Device>(0.0);
            velx_fab.setVal<RunOn::Device>(0.0);
            vely_fab.setVal<RunOn::Device>(0.0);
            velz_fab.setVal<RunOn::Device>(0.0);
            reactsource_fab.setVal<RunOn::Device>(0.0);
            
            amrex::ParallelFor(bx_x,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                electrochem_transport::compute_velx(i, j, k, sborder_arr, 
                        velx_arr,prob_lo, prob_hi, dx, time, *localprobparm);
            });

            amrex::ParallelFor(bx_y,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                electrochem_transport::compute_vely(i, j, k, sborder_arr,
                        vely_arr, prob_lo, prob_hi, dx, time, *localprobparm);
            });
            
            amrex::ParallelFor(bx_z,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                electrochem_transport::compute_velz(i, j, k, sborder_arr,
                        velz_arr, prob_lo, prob_hi, dx, time, *localprobparm);
            });
            
            amrex::ParallelFor(gbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                electrochem_transport::compute_dcoeff(i, j, k, sborder_arr, 
                        dcoeff_arr, prob_lo, prob_hi, dx, time, *localprobparm);
            });

            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                electrochem_reactions::compute_react_source(i, j, k, sborder_arr, reactsource_arr, 
                        prob_lo, prob_hi, dx, time, *localprobparm);
            });

            amrex::ParallelFor(bx_x,ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                compute_flux_x(i, j, k, n, sborder_arr, velx_arr, dcoeff_arr, flux_arr[0], dx);
            });

            amrex::ParallelFor(bx_y,ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n)
            {
                compute_flux_y(i, j, k, n, sborder_arr, vely_arr, dcoeff_arr, flux_arr[1], dx);
            });

            amrex::ParallelFor(bx_z,ncomp,
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
