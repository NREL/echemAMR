#include <echemAMR.H>
#include <Chemistry.H>
#include <Transport.H>
#include <Reactions.H>
#include <Mechanics.H>

void echemAMR::solve_mechanics(Real current_time)
{

    BL_PROFILE("echemAMR::solve_mechanics");
    amrex::Print() << "Performing Mechanics Solve: "<< std::endl;

    // Set parameters
    const int num_grow = 2;
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
