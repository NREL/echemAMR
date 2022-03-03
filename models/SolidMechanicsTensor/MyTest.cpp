#include "MyTest.H"
#include "MyTest_K.H"

#include <AMReX_MLTensorOp.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>

using namespace amrex;

MyTest::MyTest ()
{
    readParameters();

    initGrids();

    initData();
}

void
MyTest::solve ()
{
    Vector<Array<LinOpBCType,AMREX_SPACEDIM>> mlmg_lobc(AMREX_SPACEDIM);
    Vector<Array<LinOpBCType,AMREX_SPACEDIM>> mlmg_hibc(AMREX_SPACEDIM);

//    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
//        for (int jdim = 0; jdim < AMREX_SPACEDIM; ++jdim) {
//            mlmg_lobc[idim][jdim] = LinOpBCType::Periodic;//LinOpBCType::Neumann;
//            mlmg_hibc[idim][jdim] = LinOpBCType::Periodic;// LinOpBCType::Neumann;
//        }
//    }


    // xsides
    mlmg_lobc[0][0] =  LinOpBCType::Neumann;
    mlmg_lobc[1][0] =  LinOpBCType::Neumann;
#if(AMREX_SPACEDIM==3)
    mlmg_lobc[2][0] =  LinOpBCType::Neumann;
#endif

    mlmg_hibc[0][0] =  LinOpBCType::Neumann;
    mlmg_hibc[1][0] =  LinOpBCType::Neumann;
#if(AMREX_SPACEDIM==3)
    mlmg_hibc[2][0] =  LinOpBCType::Neumann;
#endif


    // ysides
    mlmg_lobc[0][1] =  LinOpBCType::Dirichlet;
    mlmg_lobc[1][1] =  LinOpBCType::Dirichlet;
#if(AMREX_SPACEDIM==3)
    mlmg_lobc[2][1] =  LinOpBCType::Dirichlet;
#endif

    mlmg_hibc[0][1] =  LinOpBCType::Neumann;
    mlmg_hibc[1][1] =  LinOpBCType::Neumann;
#if(AMREX_SPACEDIM==3)
    mlmg_hibc[2][1] =  LinOpBCType::Dirichlet;
#endif



//    std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_lobc;
//    std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_hibc;
//    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
//        mlmg_lobc[idim] = LinOpBCType::Dirichlet;
//        mlmg_hibc[idim] = LinOpBCType::Dirichlet;
//    }
//
//    mlmg_lobc[0] = LinOpBCType::Dirichlet;
//    mlmg_hibc[0] = LinOpBCType::Dirichlet;
//    mlmg_lobc[1] = LinOpBCType::inhomogNeumann;
//    mlmg_hibc[1] = LinOpBCType::inhomogNeumann;

    LPInfo info;
    info.setMaxCoarseningLevel(max_coarsening_level);

    std::unique_ptr<MLTensorOp> mltensor;
    mltensor = std::unique_ptr<MLTensorOp>(new MLTensorOp({geom}, {grids}, {dmap}, info));

    mltensor->setDomainBC(mlmg_lobc, mlmg_hibc);
    mltensor->setLevelBC(0, &exact);

    // random values
//    const Real beta = 1.2;
//    const Real E = 1.4;
//    const Real nu = 0.3;
//    const Real lambda = E * nu / (1.0 + nu) / (1.0 - nu);
//    const Real G = 0.5 * E / (1.0 + nu);

//    // 2/2*G
//    eta.setVal(G);
//    // kappa - 2*eta/3 = lambda
//    // kappa = lambda + 2*eta/3
//    kappa.setVal(lambda);//
//    MultiFab::Saxpy(kappa, 2.0/3.0, eta, 0, 0, 1, 0);


    // delta c(x,y,z) = c(x,y,z) - c0(x,y,z)
//    concentration.mult(-(3*lambda+2*G)*beta);

    // don't forget to set this!
    mltensor->setACoeffs(0, 0.0);

    {
        const int lev = 0;
        Array<MultiFab,AMREX_SPACEDIM> face_eta_coef;
        Array<MultiFab,AMREX_SPACEDIM> face_kappa_coef;
        Array<MultiFab,AMREX_SPACEDIM> face_concentration;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            const BoxArray& ba = amrex::convert(grids, IntVect::TheDimensionVector(idim));
            face_eta_coef[idim].define(ba, dmap, 1, 0);
            face_kappa_coef[idim].define(ba, dmap, 1, 0);
            face_concentration[idim].define(ba, dmap, 1, 0);
        }
        amrex::average_cellcenter_to_face(amrex::GetArrOfPtrs(face_eta_coef), eta, geom);
        amrex::average_cellcenter_to_face(amrex::GetArrOfPtrs(face_kappa_coef), kappa, geom);
        amrex::average_cellcenter_to_face(amrex::GetArrOfPtrs(face_concentration), concentration, geom);
        amrex::computeGradient(concentration_gradient,amrex::GetArrOfConstPtrs(face_concentration), geom);

        mltensor->setShearViscosity(lev, amrex::GetArrOfConstPtrs(face_eta_coef));
        mltensor->setBulkViscosity(lev, amrex::GetArrOfConstPtrs(face_kappa_coef));
    }


    MLMG mlmg(*mltensor);
    mlmg.setVerbose(verbose);
    mlmg.setBottomVerbose(bottom_verbose);
    Real mlmg_err = mlmg.solve({&solution}, {&rhs}, 1.e-11, 0.0);
//    Real mlmg_err = mlmg.solve({&solution}, {&concentration_gradient}, 1.e-11, 0.0);
    amrex::Print() << "mlmg error: " << mlmg_err << std::endl;


    amrex::Array<amrex::MultiFab, AMREX_SPACEDIM> fluxes;

    const BoxArray& bax = amrex::convert(grids, IntVect::TheDimensionVector(0));
    const BoxArray& bay = amrex::convert(grids, IntVect::TheDimensionVector(1));
    fluxes[0].define(bax, dmap, AMREX_SPACEDIM, 0);
    fluxes[1].define(bay, dmap, AMREX_SPACEDIM, 0);

#if(AMREX_SPACEDIM==3)
    const BoxArray& baz = amrex::convert(grids, IntVect::TheDimensionVector(2));
    fluxes[2].define(baz, dmap, AMREX_SPACEDIM, 0);
#endif

    const int lev = 0;
    mltensor->compFlux(lev, amrex::GetArrOfPtrs(fluxes), solution, amrex::MLMG::Location::FaceCenter);

    Vector<std::string> varname_x = {"stress_xx","stress_xy"
#if(AMREX_SPACEDIM==3)
        ,"stress_xz"
#endif
    };

    Vector<std::string> varname_y = {"stress_yx","stress_yy"
#if(AMREX_SPACEDIM==3)
        ,"stress_yz"
#endif
    };

    WriteMultiLevelPlotfile("stress_x", 1, {&fluxes[0]},
                                varname_x, {geom}, 0.0, {0}, {IntVect(2)});
    WriteMultiLevelPlotfile("stress_y", 1, {&fluxes[1]},
                                varname_y, {geom}, 0.0, {0}, {IntVect(2)});
#if(AMREX_SPACEDIM==3)
    Vector<std::string> varname_z = {"stress_zx","stress_zy","stress_zz"};
    WriteMultiLevelPlotfile("stress_z", 1, {&fluxes[2]},
                               varname_z, {geom}, 0.0, {0}, {IntVect(2)});
#endif

}

void
MyTest::writePlotfile ()
{
    Vector<std::string> varname = {"u", "v",
#if(AMREX_SPACEDIM==3)
        "w",
#endif
        "uexact", "vexact",
#if (AMREX_SPACEDIM==3)
        "wexact",
#endif
        "xerror", "yerror",
#if (AMREX_SPACEDIM==3)
        "zerror",
#endif
        "xrhs", "yrhs",
#if (AMREX_SPACEDIM==3)
        "zrhs",
#endif
        "eta", "kappa","concentration","dcdx","dcdy"
#if (AMREX_SPACEDIM==3)
        ,"dcdz"
#endif

    };

    MultiFab plotmf(grids, dmap, varname.size(),  0);
    MultiFab::Copy(plotmf, solution, 0, 0, AMREX_SPACEDIM, 0);
    MultiFab::Copy(plotmf, exact   , 0, AMREX_SPACEDIM, AMREX_SPACEDIM, 0);
    MultiFab::Copy(plotmf, solution, 0, 2*AMREX_SPACEDIM, AMREX_SPACEDIM, 0);
    MultiFab::Copy(plotmf, rhs     , 0, 3*AMREX_SPACEDIM, AMREX_SPACEDIM, 0);
    MultiFab::Copy(plotmf, eta     , 0, 4*AMREX_SPACEDIM, 1, 0);
    MultiFab::Copy(plotmf, kappa   , 0, 4*AMREX_SPACEDIM+1, 1, 0);
    MultiFab::Copy(plotmf, concentration  , 0, 4*AMREX_SPACEDIM+2, 1, 0);
    MultiFab::Copy(plotmf, concentration_gradient  , 0, 4*AMREX_SPACEDIM+3, AMREX_SPACEDIM, 0);
    MultiFab::Subtract(plotmf, exact, 0, 2*AMREX_SPACEDIM, AMREX_SPACEDIM, 0);
    WriteMultiLevelPlotfile("plot", 1, {&plotmf},
                            varname, {geom}, 0.0, {0}, {IntVect(2)});
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        amrex::Print() << "\n";
        amrex::Print() << "  max-norm error = " << plotmf.norm0(2*AMREX_SPACEDIM+idim) << std::endl;
        const auto dx = geom.CellSize();
        amrex::Print() << "    1-norm error = " << plotmf.norm1(2*AMREX_SPACEDIM+idim) << std::endl;
    }
}

void
MyTest::readParameters ()
{
    ParmParse pp;
    pp.query("n_cell", n_cell);
    pp.query("max_grid_size", max_grid_size);

    pp.query("plot_file", plot_file_name);

    pp.query("verbose", verbose);
    pp.query("bottom_verbose", bottom_verbose);
    pp.query("max_coarsening_level", max_coarsening_level);
}

void
MyTest::initGrids ()
{
    const int n = 5;
    RealBox rb({AMREX_D_DECL(-n*1.0,-1.0,-n*1.0)}, {AMREX_D_DECL(n*1.0,1.0,n*1.0)});
    std::array<int,AMREX_SPACEDIM> isperiodic{AMREX_D_DECL(0,0,0)};
    Geometry::Setup(&rb, 0, isperiodic.data());
    Box domain(IntVect{AMREX_D_DECL(0,0,0)}, IntVect{AMREX_D_DECL(n*n_cell-1,n_cell-1,n*n_cell-1)});
    geom.define(domain, rb, CoordSys::cartesian, isperiodic);

    grids.define(domain);
    grids.maxSize(max_grid_size);
}

void
MyTest::initData ()
{
    dmap.define(grids);

    solution.define(grids, dmap, AMREX_SPACEDIM, 1);
    exact.define(grids, dmap, AMREX_SPACEDIM, 1);
    rhs.define(grids, dmap, AMREX_SPACEDIM, 1);
    eta.define(grids, dmap, 1, 1);
    kappa.define(grids, dmap, 1, 1);
    concentration.define(grids, dmap, 1, 1);
    concentration_gradient.define(grids, dmap, AMREX_SPACEDIM, 1);

    const auto problo = geom.ProbLoArray();
    const auto probhi = geom.ProbHiArray();
    const auto dx     = geom.CellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& vbx = mfi.validbox();
        const Box& gbx = mfi.growntilebox(1);
        const Array4<Real> solnfab = solution.array(mfi);
        const Array4<Real> exactfab = exact.array(mfi);
        const Array4<Real> rhsfab = rhs.array(mfi);
        const Array4<Real> confab = concentration.array(mfi);
        const Array4<Real> etafab = eta.array(mfi);
        const Array4<Real> kappafab = kappa.array(mfi);

        amrex::ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real x = (i+0.5)*dx[0] + problo[0];
            Real y = (j+0.5)*dx[1] + problo[1];
#if(AMREX_SPACEDIM==3)
            Real z = (k+0.5)*dx[2] + problo[2];
#else
            Real z = 0.0;
#endif
            x = std::max(-1.0,std::min(1.0,x));
            y = std::max(-1.0,std::min(1.0,y));
            z = std::max(-1.0,std::min(1.0,z));

            Real u,v,w,urhs,vrhs,wrhs,con,eta_,kappa_;
            init(x,y,z,1.0,u,v,w,urhs,vrhs,wrhs,con,eta_,kappa_);

            etafab(i,j,k) = eta_;
            kappafab(i,j,k) = kappa_;

            if(j> 2) {
                exactfab(i,j,k,0) = 0.0;
            }
            else {
                exactfab(i,j,k,0) = 0.0;
            }

            exactfab(i,j,k,1) = 0.0;
#if(AMREX_SPACEDIM==3)
            exactfab(i,j,k,2) = w;
#endif
            rhsfab(i,j,k,0) = urhs;
            rhsfab(i,j,k,1) = vrhs;
#if(AMREX_SPACEDIM==3)
            rhsfab(i,j,k,2) = wrhs;
#endif
            confab(i,j,k) = con;
//            if (!vbx.contains(IntVect(AMREX_D_DECL(i,j,k)))) {
                solnfab(i,j,k,0) = u;
                solnfab(i,j,k,1) = v;
#if(AMREX_SPACEDIM==3)
                solnfab(i,j,k,2) = w;
#endif
//            } else {
//                solnfab(i,j,k,0) = 0;
//                solnfab(i,j,k,1) = 0;
#if(AMREX_SPACEDIM==3)
//                solnfab(i,j,k,2) = 0;
#endif
//            }

        });
    }
}
