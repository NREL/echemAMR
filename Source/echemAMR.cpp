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
#include <Chemistry.H>
#include <ProbParm.H>

using namespace amrex;

ProbParm* echemAMR::h_prob_parm = nullptr;
ProbParm* echemAMR::d_prob_parm = nullptr;
GlobalStorage* echemAMR::host_global_storage = nullptr;
// constructor - reads in parameters from inputs file
//             - sizes multilevel arrays and data structures
//             - initializes BCRec boundary condition object
echemAMR::echemAMR()
{

    h_prob_parm = new ProbParm{};
    d_prob_parm = (ProbParm*)The_Arena()->alloc(sizeof(ProbParm));
    host_global_storage = new GlobalStorage{};
    amrex_probinit(*h_prob_parm, *d_prob_parm);

    ReadParameters();

    // Geometry on all levels has been defined already.

    // No valid BoxArray and DistributionMapping have been defined.
    // But the arrays for them have been resized.

    int nlevs_max = max_level + 1;

    istep.resize(nlevs_max, 0);
    nsubsteps.resize(nlevs_max, 1);
    for (int lev = 1; lev <= max_level; ++lev)
    {
        nsubsteps[lev] = MaxRefRatio(lev - 1);
    }

    t_new.resize(nlevs_max, 0.0);
    t_old.resize(nlevs_max, -1.e100);
    dt.resize(nlevs_max, 1.e100);

    phi_new.resize(nlevs_max);
    phi_old.resize(nlevs_max);

    ParmParse pp("echemamr");
    pp.queryarr("lo_bc_spec", bc_lo_spec, 0, AMREX_SPACEDIM);
    pp.queryarr("hi_bc_spec", bc_hi_spec, 0, AMREX_SPACEDIM);
    
    pp.queryarr("lo_bc_pot", bc_lo_pot, 0, AMREX_SPACEDIM);
    pp.queryarr("hi_bc_pot", bc_hi_pot, 0, AMREX_SPACEDIM);

    /*
        // walls (Neumann)
        int bc_lo[] = {FOEXTRAP, FOEXTRAP, FOEXTRAP};
        int bc_hi[] = {FOEXTRAP, FOEXTRAP, FOEXTRAP};
    */
    bcspec.resize(NVAR);
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // lo-side BCs
        if (bc_lo_spec[idim] == BCType::int_dir ||  // periodic uses "internal Dirichlet"
            bc_lo_spec[idim] == BCType::foextrap || // first-order extrapolation
            bc_lo_spec[idim] == BCType::ext_dir || bc_lo[idim] == BCType::hoextrapcc)
        {
            for (int sp = 0; sp < NVAR; sp++)
            {
                bcspec[sp].setLo(idim, bc_lo[idim]);
            }
        } else
        {
            amrex::Abort("Invalid bc_lo");
        }

        // hi-side BCSs
        if (bc_hi_spec[idim] == BCType::int_dir ||  // periodic uses "internal Dirichlet"
            bc_hi_spec[idim] == BCType::foextrap || // first-order extrapolation
            bc_hi_spec[idim] == BCType::ext_dir || bc_hi[idim] == BCType::hoextrapcc)
        {
            for (int sp = 0; sp < NVAR; sp++)
            {
                bcspec[sp].setHi(idim, bc_hi[idim]);
            }
        } else
        {
            amrex::Abort("Invalid bc_hi");
        }
    }

    // stores fluxes at coarse-fine interface for synchronization
    // this will be sized "nlevs_max+1"
    // NOTE: the flux register associated with flux_reg[lev] is associated
    // with the lev/lev-1 interface (and has grid spacing associated with lev-1)
    // therefore flux_reg[0] is never actually used in the reflux operation
    flux_reg.resize(nlevs_max + 1);
}

echemAMR::~echemAMR()
{
    delete h_prob_parm;
    delete host_global_storage;
    The_Arena()->free(d_prob_parm);
}
// initializes multilevel data
void echemAMR::InitData()
{
    ProbParm* localprobparm = d_prob_parm;

    if (restart_chkfile == "")
    {
        // start simulation from the beginning
        const Real time = 0.0;
        InitFromScratch(time);
        AverageDown();

        // Calculate the initial volumes and append to probparm
        init_volumes();

        // Initialize the concentration and potential fields
        for (int lev = 0; lev <= finest_level; ++lev)
        {

            MultiFab& state = phi_new[lev];
            for (MFIter mfi(state); mfi.isValid(); ++mfi)
            {
                Array4<Real> fab = state[mfi].array();
                GeometryData geomData = geom[lev].data();
                const Box& box = mfi.validbox();

                amrex::launch(box, [=] AMREX_GPU_DEVICE(Box const& tbx) { initproblemdata(box, fab, geomData, localprobparm); });
            }
        }

        // print_init_data(echemAMR::h_prob_parm);

        if (chk_int > 0)
        {
            WriteCheckpointFile();
        }

    } else
    {
        // restart from a checkpoint
        ReadCheckpointFile();
    }

    if (plot_int > 0)
    {
        WritePlotFile();
    }
}

// tag all cells for refinement
// overrides the pure virtual function in AmrCore
void echemAMR::ErrorEst(int lev, TagBoxArray& tags, Real time, int ngrow)
{
    static bool first = true;

    // only do this during the first call to ErrorEst
    if (first)
    {
        first = false;
        // read in an array of "phierr", which is the tagging threshold
        // in this example, we tag values of "phi" which are greater than phierr
        // for that particular level
        // in subroutine state_error, you could use more elaborate tagging, such
        // as more advanced logical expressions, or gradients, etc.
        ParmParse pp("echemamr");
        if (pp.contains("tagged_vars"))
        {
            int nvars = pp.countval("tagged_vars");
            refine_phi.resize(nvars);
            refine_phigrad.resize(nvars);
            refine_phi_comps.resize(nvars);
            std::string varname;
            for (int i = 0; i < nvars; i++)
            {
                pp.get("tagged_vars", varname, i);
                pp.get((varname + "_refine").c_str(), refine_phi[i]);
                pp.get((varname + "_refinegrad").c_str(), refine_phigrad[i]);
                int varname_id = electrochem::find_id(varname);
                if (varname_id == -1)
                {
                    Print() << "Variable name:" << varname << " not found for tagging\n";
                    amrex::Abort("Invalid tagging variable");
                }
                refine_phi_comps[i] = varname_id;
            }
        }
    }

    if (refine_phi.size() == 0) return;

    //    const int clearval = TagBox::CLEAR;
    const int tagval = TagBox::SET;

    const MultiFab& state = phi_new[lev];
    MultiFab Sborder(grids[lev], dmap[lev], state.nComp(), 1);
    FillPatch(lev, time, Sborder, 0, Sborder.nComp());

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {

        for (MFIter mfi(Sborder, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const auto statefab = Sborder.array(mfi);
            const auto tagfab = tags.array(mfi);

            amrex::Real* refine_phi_dat = refine_phi.data();
            amrex::Real* refine_phigrad_dat = refine_phigrad.data();
            int* refine_phi_comps_dat = refine_phi_comps.data();
            int ntagged_comps = refine_phi_comps.size();

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                state_based_refinement(i, j, k, tagfab, statefab, refine_phi_dat, refine_phi_comps_dat, ntagged_comps, tagval);
            });

            amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
                stategrad_based_refinement(i, j, k, tagfab, statefab, refine_phigrad_dat, refine_phi_comps_dat, ntagged_comps, tagval);
            });
        }
    }
}

// read in some parameters from inputs file
void echemAMR::ReadParameters()
{
    {
        ParmParse pp; // Traditionally, max_step and stop_time do not have prefix.
        pp.query("max_step", max_step);
        pp.query("stop_time", stop_time);
    }

    {
        ParmParse pp("amr"); // Traditionally, these have prefix, amr.

        pp.query("regrid_int", regrid_int);
        pp.query("plot_file", plot_file);
        pp.query("plot_int", plot_int);
        pp.query("line_plot_int", line_plot_int);
        pp.query("line_plot_dir", line_plot_dir);
        pp.query("line_plot_npoints", line_plot_npoints);
        pp.query("chk_file", chk_file);
        pp.query("chk_int", chk_int);
        pp.query("restart", restart_chkfile);
    }

    {
        ParmParse pp("echemamr");

        pp.query("cfl", cfl);
        pp.query("dtmin", dtmin);
        pp.query("dtmax", dtmax);
        pp.query("do_reflux", do_reflux);
        pp.query("potential_solve", potential_solve);
        pp.query("potential_solve_int", pot_solve_int);
        pp.query("potential_initial_guess", pot_initial_guess);

        pp.query("buttler_vohlmer_flux", buttler_vohlmer_flux);
        pp.query("bv_levelset_id", bv_levset_id);
        pp.query("bv_species_id", bv_spec_id);

        pp.query("bv_relaxation_factor", bv_relaxfac);
        pp.query("bv_nonlinear_iters", bv_nonlinear_iters);
        pp.query("bv_nonlinear_reltol", bv_nonlinear_reltol);
        pp.query("bv_nonlinear_abstol", bv_nonlinear_abstol);

        pp.query("use_hypre",use_hypre);

        pp.query("linsolve_reltol",linsolve_reltol);
        pp.query("linsolve_abstol",linsolve_abstol);
        pp.query("linsolve_maxiter",linsolve_maxiter);
        pp.query("linsolve_max_coarsening_level",linsolve_max_coarsening_level);
        pp.query("lsgrad_tol",lsgrad_tolerance);
    }
}

// utility to copy in data from phi_old and/or phi_new into another multifab
void echemAMR::GetData(int lev, Real time, Vector<MultiFab*>& data, Vector<Real>& datatime)
{
    data.clear();
    datatime.clear();

    const Real teps = (t_new[lev] - t_old[lev]) * 1.e-3;

    if (time > t_new[lev] - teps && time < t_new[lev] + teps)
    {
        data.push_back(&phi_new[lev]);
        datatime.push_back(t_new[lev]);
    } else if (time > t_old[lev] - teps && time < t_old[lev] + teps)
    {
        data.push_back(&phi_old[lev]);
        datatime.push_back(t_old[lev]);
    } else
    {
        data.push_back(&phi_old[lev]);
        data.push_back(&phi_new[lev]);
        datatime.push_back(t_old[lev]);
        datatime.push_back(t_new[lev]);
    }
}
