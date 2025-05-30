#ifndef echemAMR_H_
#define echemAMR_H_

#include <AMReX_MLMG.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_MLTensorOp.H>
#include <AMReX_MLABecLaplacian.H>

#include <string>
#include <limits>
#include <memory>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef AMREX_USE_HYPRE
#include <AMReX_Hypre.H>
#endif

#include <AMReX_AmrCore.H>
#include <AMReX_FluxRegister.H>
#include <AMReX_BCRec.H>
#include <ProbParm.H>


using namespace amrex;

class echemAMR : public amrex::AmrCore
{
public:
    ////////////////
    // public member functions

    // constructor - reads in parameters from inputs file
    //             - sizes multilevel arrays and data structures
    echemAMR();
    virtual ~echemAMR();

    // advance solution to final time
    void Evolve();

    // initializes multilevel data
    void InitData();

    // Make a new level using provided BoxArray and DistributionMapping and
    // fill with interpolated coarse level data.
    // overrides the pure virtual function in AmrCore
    virtual void MakeNewLevelFromCoarse(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;

    // Remake an existing level using provided BoxArray and DistributionMapping and
    // fill with existing fine and coarse data.
    // overrides the pure virtual function in AmrCore
    virtual void RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;

    // Delete level data
    // overrides the pure virtual function in AmrCore
    virtual void ClearLevel(int lev) override;

    // Make a new level from scratch using provided BoxArray and DistributionMapping.
    // Only used during initialization.
    // overrides the pure virtual function in AmrCore
    virtual void MakeNewLevelFromScratch(int lev, amrex::Real time, const amrex::BoxArray& ba, const amrex::DistributionMapping& dm) override;

    // tag all cells for refinement
    // overrides the pure virtual function in AmrCore
    virtual void ErrorEst(int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override;

    // advance a single level for a single time step, updates flux registers
    void Advance(int lev, amrex::Real time, amrex::Real dt_lev, int iteration, int ncycle);

    void compute_fluxes(int lev, const int num_grow, MultiFab& Sborder, 
        Array<MultiFab,AMREX_SPACEDIM>& flux, Real time, bool implicit_diffusion=false);

    void compute_dsdt(int lev, const int num_grow, MultiFab& Sborder, 
        Array<MultiFab,AMREX_SPACEDIM>& flux, MultiFab& dsdt,
        Real time, Real dt, bool reflux_this_stage);

    void solve_potential(Real current_time);
    
    //void implicit_solve_species(Real current_time,Real dt,int spec_id);
    void implicit_solve_species(Real current_time,Real dt,int spec_id,
            Vector<MultiFab>& dsdt_expl);

    void update_interface_cells(Real current_time);

    void solve_mechanics(Real current_time);   

    // compute dt from CFL considerations
    Real EstTimeStep(int lev);

    // a wrapper for computing integrals
    Real VolumeIntegral(int comp1, int comp2, int domain);

    // a wrapper for computing integrals
    Real SurfaceIntegral(int comp, int domain1, int domain2);

    // a wrapper for computing integrals
    Real CurrentCollectorIntegral(int comp, int domain);

    void postprocess(Real time, int timestep, Real dt, GlobalStorage* globalstorage);

    void init_volumes();

private:
    ////////////////
    // private member functions

    // read in some parameters from inputs file
    void ReadParameters();

    // create require directories for saving output data
    void CreateDirectories();

    // set covered coarse cells to be the average of overlying fine cells
    void AverageDown();

    // more flexible version of AverageDown() that lets you average down across multiple levels
    void AverageDownTo(int crse_lev);

    // compute a new multifab by coping in phi from valid region and filling ghost cells
    // works for single level and 2-level cases (fill fine grid ghost by interpolating from coarse)
    void FillPatch(int lev, amrex::Real time, amrex::MultiFab& mf, int icomp, int ncomp);

    // fill an entire multifab by interpolating from the coarser level
    // this comes into play when a new level of refinement appears
    void FillCoarsePatch(int lev, amrex::Real time, amrex::MultiFab& mf, int icomp, int ncomp);

    // utility to copy in data from phi_old and/or phi_new into another multifab
    void GetData(int lev, amrex::Real time, amrex::Vector<amrex::MultiFab*>& data, amrex::Vector<amrex::Real>& datatime);

    // advance a level by dt
    // includes a recursive call for finer levels
    void timeStep(int lev, amrex::Real time, int iteration);

    // a wrapper for EstTimeStep(0
    void ComputeDt();

    // get plotfile name
    std::string PlotFileName(int lev) const;

    // put together an array of multifabs for writing
    amrex::Vector<const amrex::MultiFab*> PlotFileMF() const;

    // set plotfile variables names
    amrex::Vector<std::string> PlotFileVarNames() const;

    // write plotfile to disk
    void WritePlotFile() const;

    // write checkpoint file to disk
    void WriteCheckpointFile() const;

    // read checkpoint file from disk
    void ReadCheckpointFile();

    // utility to skip to next line in Header
    static void GotoNextLine(std::istream& is);

    ////////////////
    // private data members
    amrex::Gpu::ManagedVector<Real> refine_phi;
    amrex::Gpu::ManagedVector<Real> refine_phigrad;
    amrex::Gpu::ManagedVector<int> refine_phi_comps;

    amrex::Vector<int> istep;     // which step?
    amrex::Vector<int> nsubsteps; // how many substeps on each level?

    // keep track of old time, new time, and time step at each level
    amrex::Vector<amrex::Real> t_new;
    amrex::Vector<amrex::Real> t_old;
    amrex::Vector<amrex::Real> dt;

    // array of multifabs to store the solution at each level of refinement
    // after advancing a level we use "swap".
    amrex::Vector<amrex::MultiFab> phi_new;
    amrex::Vector<amrex::MultiFab> phi_old;

    // this is essentially a 2*DIM integer array storing the physical boundary
    // condition types at the lo/hi walls in each direction
    amrex::Vector<BCRec> bcspec; // 1-component

    // stores fluxes at coarse-fine interface for synchronization
    // this will be sized "nlevs_max+1"
    // NOTE: the flux register associated with flux_reg[lev] is associated
    // with the lev/lev-1 interface (and has grid spacing associated with lev-1)
    // therefore flux_reg[0] and flux_reg[nlevs_max] are never actually
    // used in the reflux operation
    amrex::Vector<std::unique_ptr<amrex::FluxRegister>> flux_reg;

    ////////////////
    // runtime parameters

    // maximum number of steps and stop time
    int max_step = std::numeric_limits<int>::max();
    amrex::Real stop_time = std::numeric_limits<amrex::Real>::max();


    // if >= 0 we restart from a checkpoint
    std::string restart_chkfile = "";


    amrex::Vector<int> bc_lo_spec{BCType::foextrap, BCType::foextrap, BCType::foextrap};
    amrex::Vector<int> bc_hi_spec{BCType::foextrap, BCType::foextrap, BCType::foextrap};
    
    amrex::Vector<int> bc_lo_pot{BCType::foextrap, BCType::foextrap, BCType::foextrap};
    amrex::Vector<int> bc_hi_pot{BCType::foextrap, BCType::foextrap, BCType::foextrap};

    // advective cfl number - dt = cfl*dx/umax
    amrex::Real cfl = 0.7;

    amrex::Real dtmin = 1e-20;
    amrex::Real dtmax = 1e10;
    amrex::Real dtgpfactor=1.0;

    int fixed_timestep=0;
    int interface_update_iters=-1;
    int update_species_interface=0;

    // by default no relaxation for
    // BV poisson solves
    amrex::Real bv_relaxfac = 0.0;

    amrex::Real linsolve_reltol = 1e-6;
    amrex::Real linsolve_abstol = 1e-10;
    amrex::Real linsolve_bot_reltol = 1e-4;
    amrex::Real linsolve_bot_abstol = -1.0;

    int linsolve_num_pre_smooth = 2;
    int linsolve_num_post_smooth = 2;
    int linsolve_num_final_smooth = 8;
    int linsolve_num_bottom_smooth = 0;

    int linsolve_maxiter=100;
    int linsolve_bottom_maxiter=100;
    int linsolve_max_coarsening_level=0;

    // how often each level regrids the higher levels of refinement
    // (after a level advances that many time steps)
    int regrid_int = 2;

    // hyperbolic refluxing as part of multilevel synchronization
    int do_reflux = 1;

    // poission solve required
    int potential_solve = 0;
    // frequency to solve potential equation
    int pot_solve_int = 1;
    // use old solution as initial guess for potential
    int pot_initial_guess = 0;
    //potential solve on init
    int potential_solve_init=0;

    int species_implicit_solve=0;
    int reset_species_in_solid=0;
    Real species_implicit_solve_dt=0.01; //sec

    int mechanics_solve = 0;

    Vector<int> transported_species_list{0}; 

    // apply Buttler-Vohlmer conditions
    int buttler_vohlmer_flux = 0;
    // level-set id for buttler-vohlmer
    int bv_levset_id = 0;
    Vector<int> bv_specid_list{0};
    int kd_conc_id=0;

    int bv_nonlinear_iters = 1;
    Real bv_nonlinear_reltol = 1e-8;
    Real bv_nonlinear_abstol = 1e-12;
    Real lsgrad_tolerance = 1e-2;
    int use_hypre=0;

    // frequency to output line plots
    int line_plot_int = -1;
    // direction to create line plot
    int line_plot_dir = 0;
    // number of points along line
    int line_plot_npoints = 100;

    // plotfile prefix and frequency
    std::string plot_file{"plt"};
    int plot_int = -1;

    // checkpoint prefix and frequency
    std::string chk_file{"chk"};
    int chk_int = -1;

#if defined(BL_USE_SENSEI_INSITU)
    amrex::AmrMeshInSituBridge* insitu_bridge;
#endif

public:
    static ProbParm* h_prob_parm;
    static ProbParm* d_prob_parm;
    static GlobalStorage* host_global_storage;
    // set the default output folder
    static std::string output_folder;
};

#endif
