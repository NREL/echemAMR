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

using namespace amrex;

// constructor - reads in parameters from inputs file
//             - sizes multilevel arrays and data structures
//             - initializes BCRec boundary condition object
echemAMR::echemAMR ()
{
    ReadParameters();

    // Geometry on all levels has been defined already.

    // No valid BoxArray and DistributionMapping have been defined.
    // But the arrays for them have been resized.

    int nlevs_max = max_level + 1;

    istep.resize(nlevs_max, 0);
    nsubsteps.resize(nlevs_max, 1);
    for (int lev = 1; lev <= max_level; ++lev) 
    {
	nsubsteps[lev] = MaxRefRatio(lev-1);
    }

    t_new.resize(nlevs_max, 0.0);
    t_old.resize(nlevs_max, -1.e100);
    dt.resize(nlevs_max, 1.e100);

    phi_new.resize(nlevs_max);
    phi_old.resize(nlevs_max);

    amrex::Vector<int> bc_lo{BCType::foextrap, BCType::foextrap, BCType::foextrap};
    amrex::Vector<int> bc_hi{BCType::foextrap, BCType::foextrap, BCType::foextrap};

    ParmParse pp("echemamr");
    pp.queryarr("lo_bc", bc_lo, 0, AMREX_SPACEDIM);
    pp.queryarr("hi_bc", bc_hi, 0, AMREX_SPACEDIM);

/*
    // walls (Neumann)
    int bc_lo[] = {FOEXTRAP, FOEXTRAP, FOEXTRAP};
    int bc_hi[] = {FOEXTRAP, FOEXTRAP, FOEXTRAP};
*/

    bcs.resize(electrochem::nspecies);     // Setup 1-component
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        // lo-side BCs
        if (bc_lo[idim] == BCType::int_dir  ||  // periodic uses "internal Dirichlet"
            bc_lo[idim] == BCType::foextrap ||  // first-order extrapolation
            bc_lo[idim] == BCType::ext_dir ) 
        {  
            for(int sp=0;sp<electrochem::nspecies;sp++)
            {
                bcs[sp].setLo(idim, bc_lo[idim]);
            }
        }
        else 
        {
            amrex::Abort("Invalid bc_lo");
        }

        // hi-side BCSs
        if (bc_hi[idim] == BCType::int_dir  ||  // periodic uses "internal Dirichlet"
            bc_hi[idim] == BCType::foextrap ||  // first-order extrapolation
            bc_hi[idim] == BCType::ext_dir ) 
        {  
            for(int sp=0;sp<electrochem::nspecies;sp++)
            {
                bcs[sp].setHi(idim, bc_hi[idim]);
            }
        }
        else 
        {
            amrex::Abort("Invalid bc_hi");
        }
    }

    // stores fluxes at coarse-fine interface for synchronization
    // this will be sized "nlevs_max+1"
    // NOTE: the flux register associated with flux_reg[lev] is associated
    // with the lev/lev-1 interface (and has grid spacing associated with lev-1)
    // therefore flux_reg[0] is never actually used in the reflux operation
    flux_reg.resize(nlevs_max+1);

}

echemAMR::~echemAMR ()
{

}
// initializes multilevel data
void echemAMR::InitData ()
{
    if (restart_chkfile == "") 
    {
        // start simulation from the beginning
        const Real time = 0.0;
        InitFromScratch(time);
        AverageDown();

        if (chk_int > 0) 
        {
            WriteCheckpointFile();
        }

    }
    else 
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
void echemAMR::ErrorEst (int lev, TagBoxArray& tags, Real time, int ngrow)
{
    static bool first = true;
    static Vector<Real> refine_phi;
    static Vector<Real> refine_phi_comps;

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
            refine_phi_comps.resize(nvars);
            std::string varname;
            for(int i=0; i<nvars; i++)
            {
                pp.get("tagged_vars",varname,i);
                pp.get((varname+"_refine").c_str(),refine_phi[i]);
                int varname_id=electrochem::find_id(varname);
                if(varname_id == -1)
                {
                    Print()<<"Variable name:"<<varname<<" not found for tagging\n";
                    amrex::Abort("Invalid tagging variable");
                }
                refine_phi_comps[i]=varname_id;
            }
        }
    }

    if(refine_phi.size() == 0) return;

//    const int clearval = TagBox::CLEAR;
    const int   tagval = TagBox::SET;

    const MultiFab& state = phi_new[lev];

#ifdef _OPENMP
#pragma omp parallel if(Gpu::notInLaunchRegion())
#endif
    {
	
	for (MFIter mfi(state,TilingIfNotGPU()); mfi.isValid(); ++mfi)
	{
	    const Box& bx       = mfi.tilebox();
            const auto statefab = state.array(mfi);
            const auto tagfab   = tags.array(mfi);
	    
            amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                state_error(i, j, k, tagfab, statefab, 
                        refine_phi.dataPtr(), refine_phi_comps.dataPtr(), 
                        refine_phi.size(), tagval);
            });
	}
    }
}

// read in some parameters from inputs file
void echemAMR::ReadParameters ()
{
    {
	ParmParse pp;  // Traditionally, max_step and stop_time do not have prefix.
	pp.query("max_step", max_step);
	pp.query("stop_time", stop_time);
    }

    {
	ParmParse pp("amr"); // Traditionally, these have prefix, amr.

	pp.query("regrid_int", regrid_int);
	pp.query("plot_file", plot_file);
	pp.query("plot_int", plot_int);
	pp.query("chk_file", chk_file);
	pp.query("chk_int", chk_int);
        pp.query("restart",restart_chkfile);
    }

    {
	ParmParse pp("echemamr");
	
	pp.query("cfl", cfl);
        pp.query("do_reflux", do_reflux);
    }
}


// utility to copy in data from phi_old and/or phi_new into another multifab
void echemAMR::GetData (int lev, Real time, Vector<MultiFab*>& data, Vector<Real>& datatime)
{
    data.clear();
    datatime.clear();

    const Real teps = (t_new[lev] - t_old[lev]) * 1.e-3;

    if (time > t_new[lev] - teps && time < t_new[lev] + teps)
    {
	data.push_back(&phi_new[lev]);
	datatime.push_back(t_new[lev]);
    }
    else if (time > t_old[lev] - teps && time < t_old[lev] + teps)
    {
	data.push_back(&phi_old[lev]);
	datatime.push_back(t_old[lev]);
    }
    else
    {
	data.push_back(&phi_old[lev]);
	data.push_back(&phi_new[lev]);
	datatime.push_back(t_old[lev]);
	datatime.push_back(t_new[lev]);
    }
}
