#ifndef echemAMR_H_
#define echemAMR_H_

#include <AMReX_MLMG.H>

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
#include <ChemistryProbParm.H>

using namespace amrex;

class echemAMR
    : public amrex::AmrCore
{
public:

    ////////////////
    // public member functions

    // constructor - reads in parameters from inputs file
    //             - sizes multilevel arrays and data structures
    echemAMR ();
    virtual ~echemAMR();

    // advance solution to final time
    void Evolve ();

    // initializes multilevel data
    void InitData ();

    // Make a new level using provided BoxArray and DistributionMapping and 
    // fill with interpolated coarse level data.
    // overrides the pure virtual function in AmrCore
    virtual void MakeNewLevelFromCoarse (int lev, amrex::Real time, const amrex::BoxArray& ba,
					 const amrex::DistributionMapping& dm) override;

    // Remake an existing level using provided BoxArray and DistributionMapping and 
    // fill with existing fine and coarse data.
    // overrides the pure virtual function in AmrCore
    virtual void RemakeLevel (int lev, amrex::Real time, const amrex::BoxArray& ba,
			      const amrex::DistributionMapping& dm) override;

    // Delete level data
    // overrides the pure virtual function in AmrCore
    virtual void ClearLevel (int lev) override;

    // Make a new level from scratch using provided BoxArray and DistributionMapping.
    // Only used during initialization.
    // overrides the pure virtual function in AmrCore
    virtual void MakeNewLevelFromScratch (int lev, amrex::Real time, const amrex::BoxArray& ba,
					  const amrex::DistributionMapping& dm) override;

    // tag all cells for refinement
    // overrides the pure virtual function in AmrCore
    virtual void ErrorEst (int lev, amrex::TagBoxArray& tags, amrex::Real time, int ngrow) override;

    // advance a single level for a single time step, updates flux registers
    void Advance (int lev, amrex::Real time, amrex::Real dt_lev, int iteration, int ncycle);
    void compute_dsdt (int lev, const int num_grow, MultiFab &Sborder, 
            MultiFab &dsdt, Real time, Real dt, bool reflux_this_stage);

    void solve_potential(Real current_time);

    // compute dt from CFL considerations
    Real EstTimeStep (int lev, bool local=false);

    // a wrapper for computing integrals
    Real VolumeIntegral (int comp, int domain);

    // a wrapper for computing integrals
    Real SurfaceIntegral (int comp, int domain1, int domain2);

    // a wrapper for computing integrals
    Real CurrentCollectorIntegral (int comp, int domain);

    void postprocess(Real time, int timestep);

    void init_volumes();
private:

    ////////////////
    // private member functions

    // read in some parameters from inputs file
    void ReadParameters();

    // set covered coarse cells to be the average of overlying fine cells
    void AverageDown ();

    // more flexible version of AverageDown() that lets you average down across multiple levels
    void AverageDownTo (int crse_lev);

    // compute a new multifab by coping in phi from valid region and filling ghost cells
    // works for single level and 2-level cases (fill fine grid ghost by interpolating from coarse)
    void FillPatch (int lev, amrex::Real time, amrex::MultiFab& mf, int icomp, int ncomp);

    // fill an entire multifab by interpolating from the coarser level
    // this comes into play when a new level of refinement appears
    void FillCoarsePatch (int lev, amrex::Real time, amrex::MultiFab& mf, int icomp, int ncomp);

    // utility to copy in data from phi_old and/or phi_new into another multifab
    void GetData (int lev, amrex::Real time, amrex::Vector<amrex::MultiFab*>& data,
                  amrex::Vector<amrex::Real>& datatime);

    // advance a level by dt
    // includes a recursive call for finer levels
    void timeStep (int lev, amrex::Real time, int iteration);

    // a wrapper for EstTimeStep(0
    void ComputeDt ();

    // get plotfile name
    std::string PlotFileName (int lev) const;

    // put together an array of multifabs for writing
    amrex::Vector<const amrex::MultiFab*> PlotFileMF () const;

    // set plotfile variables names
    amrex::Vector<std::string> PlotFileVarNames () const;

    // write plotfile to disk
    void WritePlotFile () const;

    // write checkpoint file to disk
    void WriteCheckpointFile () const;

    // read checkpoint file from disk
    void ReadCheckpointFile ();

    // utility to skip to next line in Header
    static void GotoNextLine (std::istream& is);


    ////////////////
    // private data members
    amrex::Gpu::ManagedVector<Real> refine_phi;
    amrex::Gpu::ManagedVector<Real> refine_phigrad;
    amrex::Gpu::ManagedVector<int> refine_phi_comps;

    amrex::Vector<int> istep;      // which step?
    amrex::Vector<int> nsubsteps;  // how many substeps on each level?

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
    amrex::Vector<BCRec> bcs;  // 1-component

    // stores fluxes at coarse-fine interface for synchronization
    // this will be sized "nlevs_max+1"
    // NOTE: the flux register associated with flux_reg[lev] is associated
    // with the lev/lev-1 interface (and has grid spacing associated with lev-1)
    // therefore flux_reg[0] and flux_reg[nlevs_max] are never actually 
    // used in the reflux operation
    amrex::Vector<std::unique_ptr<amrex::FluxRegister> > flux_reg; 
    
    ////////////////
    // runtime parameters

    // maximum number of steps and stop time
    int max_step = std::numeric_limits<int>::max();
    amrex::Real stop_time = std::numeric_limits<amrex::Real>::max();

    // if >= 0 we restart from a checkpoint
    std::string restart_chkfile = "";

    amrex::Vector<int> bc_lo{BCType::foextrap, BCType::foextrap, BCType::foextrap};
    amrex::Vector<int> bc_hi{BCType::foextrap, BCType::foextrap, BCType::foextrap};

    // advective cfl number - dt = cfl*dx/umax
    amrex::Real cfl = 0.7;

    // how often each level regrids the higher levels of refinement
    // (after a level advances that many time steps)
    int regrid_int = 2;

    // hyperbolic refluxing as part of multilevel synchronization
    int do_reflux = 1;

    //poission solve required
    int potential_solve = 0;
    // frequency to solve potential equation
    int pot_solve_int = 1;

    // frequency to output line plots
    int line_plot_int = -1;
    // direction to create line plot
    int line_plot_dir = 0;
    // number of points along line
    int line_plot_npoints = 100;

    // plotfile prefix and frequency
    std::string plot_file {"plt"};
    int plot_int = -1;

    // checkpoint prefix and frequency
    std::string chk_file {"chk"};
    int chk_int = -1;

#if defined(BL_USE_SENSEI_INSITU)
    amrex::AmrMeshInSituBridge *insitu_bridge;
#endif

public:
    static ProbParm* h_prob_parm;
    static ProbParm* d_prob_parm;
};

#endif
