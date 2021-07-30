#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MLABecLaplacian.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_MLMG.H>
#include <Prob.H>
#include <global_defines.H>

using namespace amrex;


int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    {
        std::vector<int> n_cell;
        std::vector<Real> prob_lo;
        std::vector<Real> prob_hi;
        Real electrode_dcoeff=1.0;
        Real electrolyte_dcoeff=3.0;
        int transport_dir=0;
        int max_grid_size = 32;
        Real leftdircval=1.0;
        Real rightdircval=0.0;
        int maxnonliniter=10;
    
        Vector<int> is_periodic(AMREX_SPACEDIM,0);
        amrex::Vector<int> bc_lo{0,0,0};
        amrex::Vector<int> bc_hi{0,0,0};

        ParmParse pp;
        pp.getarr("n_cell", n_cell);
        pp.getarr("prob_lo", prob_lo);
        pp.getarr("prob_hi", prob_hi);
        pp.getarr("is_periodic",is_periodic);

        pp.query("max_grid_size", max_grid_size);
        pp.query("transport_dir",transport_dir);
        pp.query("electrode_dcoeff",electrode_dcoeff);
        pp.query("electrolyte_dcoeff",electrolyte_dcoeff);
        pp.query("maxnonliniter",maxnonliniter);
    
        pp.queryarr("lo_bc", bc_lo, 0, AMREX_SPACEDIM);
        pp.queryarr("hi_bc", bc_hi, 0, AMREX_SPACEDIM);

        Geometry geom;
        BoxArray grids;
        DistributionMapping dmap;
        RealBox rb({AMREX_D_DECL(prob_lo[0],prob_lo[1],prob_lo[2])}, 
                {AMREX_D_DECL(prob_hi[0],prob_hi[1],prob_hi[2])});

        Box domain(IntVect{AMREX_D_DECL(0,0,0)},
                IntVect{AMREX_D_DECL(n_cell[0]-1,n_cell[1]-1,n_cell[2]-1)});

        geom.define(domain, &rb, CoordSys::cartesian, is_periodic.data());

        // define the BoxArray to be a single grid
        grids.define(domain); 
        // chop domain up into boxes with length max_Grid_size
        grids.maxSize(max_grid_size); 

        dmap.define(grids); // create a processor distribution mapping given the BoxARray

        int required_coarsening_level = 0; // typically the same as the max AMR level index
        int max_coarsening_level = 100;    // typically a huge number so MG coarsens as much as possible

        MultiFab phi(grids, dmap, 1, 1);
        MultiFab soln(grids, dmap, 1, 0);
        MultiFab err(grids, dmap, 1,  0);

        MultiFab rhs(grids, dmap, 1, 0);
        MultiFab phi_bc(grids, dmap, 1, 1);
        MultiFab levset(grids,dmap, 1, 1);

        rhs.setVal(0.0);
        phi.setVal(0.0);
        soln.setVal(0.0);
        phi_bc.setVal(0.0);
        levset.setVal(0.0);
        
        initlevset(levset,geom);

        LPInfo info;
        MLABecLaplacian mlabec({geom}, {grids}, {dmap}, info);
        MLMG mlmg(mlabec);

        // relative and absolute tolerances for linear solve
        const Real tol_rel = 1.e-10;
        const Real tol_abs = 0.0;
        int verbose = 1;
        mlmg.setVerbose(verbose);

        // define array of LinOpBCType for domain boundary conditions
        std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_bc_lo;
        std::array<LinOpBCType,AMREX_SPACEDIM> mlmg_bc_hi;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) 
        {
            if(bc_lo[idim] == PERIODIC_ID)
            {
                mlmg_bc_lo[idim]=LinOpBCType::Periodic;
            }
            else if(bc_lo[idim] == DIRICHLET_ID)
            {
                mlmg_bc_lo[idim]=LinOpBCType::Dirichlet;
            }
            else if(bc_lo[idim] == H_NEUMANN_ID)
            {
                mlmg_bc_lo[idim]=LinOpBCType::Neumann;
            }
            else
            {
                mlmg_bc_lo[idim]=LinOpBCType::inhomogNeumann;
            }

            if(bc_hi[idim] == PERIODIC_ID)
            {
                mlmg_bc_hi[idim]=LinOpBCType::Periodic;
            }
            else if(bc_hi[idim] == DIRICHLET_ID)
            {
                mlmg_bc_hi[idim]=LinOpBCType::Dirichlet;
            }
            else if(bc_hi[idim] == H_NEUMANN_ID)
            {
                mlmg_bc_hi[idim]=LinOpBCType::Neumann;
            }
            else
            {
                mlmg_bc_hi[idim]=LinOpBCType::inhomogNeumann;
            }
            
        }
        setbc(phi_bc,phi,geom,bc_lo,bc_hi);

        // Boundary of the whole domain. This functions must be called,
        // and must be called before other bc functions.
        mlabec.setDomainBC(mlmg_bc_lo,mlmg_bc_hi);
        mlabec.setLevelBC(0, &phi_bc);

        // operator looks like (ACoef - div BCoef grad) phi = rhs
        // scaling factors; these multiply ACoef and BCoef
        Real ascalar = 0.0;
        Real bscalar = 1.0;
        mlabec.setScalars(ascalar, bscalar);

        // set ACoef to zero
        MultiFab acoef(grids, dmap, 1, 0);
        acoef.setVal(0.);
        mlabec.setACoeffs(0, acoef);


        // set BCoef to 1.0 (and array of face-centered coefficients)
        Array<MultiFab,AMREX_SPACEDIM> bcoef;
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) 
        {
            bcoef[idim].define(amrex::convert(grids,IntVect::TheDimensionVector(idim)), dmap, 1, 0);
        }


        for(int iter=0;iter<maxnonliniter;iter++)
        {

            rhs.setVal(0.0);
            phi.FillBoundary(geom.periodicity());
            amrex::MultiFab::Copy(err,phi, 0, 0, 1 ,0);

            for(int idim=0;idim<AMREX_SPACEDIM;idim++)
            {
            for (MFIter mfi(phi); mfi.isValid(); ++mfi)
            {
                const auto dx = geom.CellSizeArray();
                const Box& bx = mfi.tilebox();
                Box fbx = convert(bx,IntVect::TheDimensionVector(idim));

                Array4<Real> bcoef_arr = bcoef[idim].array(mfi);
                Array4<Real> ls_arr = levset.array(mfi);

                amrex::ParallelFor(fbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k)
                        {
                        Real eps=1e-10;
                        IntVect left(i,j,k);
                        IntVect right(i,j,k);

                        left[idim] -= 1;

                        Real ls_left  = ls_arr(left);
                        Real ls_right = ls_arr(right);

                        Real d_right = ls_right*electrolyte_dcoeff + (1.0-ls_right)*electrode_dcoeff;
                        Real d_left  = ls_left*electrolyte_dcoeff + (1.0-ls_left)*electrode_dcoeff;

                        bcoef_arr(i,j,k)= 2.0*d_right*d_left/(d_right+d_left+eps);
                        });
            }
            }

            #include "bvflux.H"
            mlabec.setBCoeffs(0, amrex::GetArrOfConstPtrs(bcoef));

            mlmg.solve({&soln}, {&rhs}, tol_rel, tol_abs);

            amrex::MultiFab::Subtract(err,soln, 0, 0, 1 ,0);
            Real errnorm=err.norm2();
            amrex::Print()<<"errnorm:"<<errnorm<<"\n";
            amrex::MultiFab::Copy(phi,soln, 0, 0, 1 ,0);

            if(errnorm < 1e-8)
            {
                break;
            }
        }

        // store plotfile variables; q and phi
        MultiFab plotfile_mf(grids, dmap, 2, 0);
        MultiFab::Copy(plotfile_mf, levset,0,0,1,0);
        MultiFab::Copy(plotfile_mf, phi,0,1,1,0);

        WriteSingleLevelPlotfile("plt", plotfile_mf, {"levset","potential"}, geom, 0.0, 0);
    }

    amrex::Finalize();
}
