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

// a wrapper for EstTimeStep
void echemAMR::ComputeDt ()
{
    Vector<Real> dt_tmp(finest_level+1);

    for (int lev = 0; lev <= finest_level; ++lev)
    {
	dt_tmp[lev] = EstTimeStep(lev, true);
    }
    ParallelDescriptor::ReduceRealMin(&dt_tmp[0], dt_tmp.size());

    constexpr Real change_max = 1.1;
    Real dt_0 = dt_tmp[0];
    int n_factor = 1;
    for (int lev = 0; lev <= finest_level; ++lev) 
    {
	dt_tmp[lev] = std::min(dt_tmp[lev], change_max*dt[lev]);
	n_factor *= nsubsteps[lev];
	dt_0 = std::min(dt_0, n_factor*dt_tmp[lev]);
    }

    // Limit dt's by the value of stop_time.
    const Real eps = 1.e-3*dt_0;
    if (t_new[0] + dt_0 > stop_time - eps) 
    {
	dt_0 = stop_time - t_new[0];
    }

    dt[0] = dt_0;
    for (int lev = 1; lev <= finest_level; ++lev) 
    {
	dt[lev] = dt[lev-1] / nsubsteps[lev];
    }
}

// compute dt from CFL considerations
Real echemAMR::EstTimeStep (int lev, bool local)
{
    BL_PROFILE("echemAMR::EstTimeStep()");

    Real dt_est = std::numeric_limits<Real>::max();
    auto prob_lo = geom[lev].ProbLoArray();
    auto prob_hi = geom[lev].ProbHiArray();
    const auto dx = geom[lev].CellSizeArray();
    
    const Real cur_time = t_new[lev];
    MultiFab& S_new = phi_new[lev];
   
    //need fillpatched data for velocity calculation 
    constexpr int num_grow = 2; 
    MultiFab Sborder(grids[lev], dmap[lev], S_new.nComp(), num_grow);
    FillPatch(lev, cur_time, Sborder, 0, Sborder.nComp()); 

    MultiFab dcoeff(S_new.boxArray(), S_new.DistributionMap(), S_new.nComp(), 0);
    MultiFab vel(S_new.boxArray(), S_new.DistributionMap(), S_new.nComp(), 0);
    
    //set sane default values
    dcoeff.setVal(1.0);
    vel.setVal(1.0);

    int ncomp=S_new.nComp();

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi) 
        {
            const Box& bx = mfi.tilebox();
            Box bx_x=convert(bx, {1,0,0});
            Box bx_y=convert(bx, {0,1,0});
            Box bx_z=convert(bx, {0,0,1});
            
            Array4<Real> statearray = Sborder.array(mfi);
            Array4<Real> dcoeffarray = dcoeff.array(mfi);
            Array4<Real> velarray  = vel.array(mfi);

            FArrayBox velx_fab(bx_x,ncomp);
            FArrayBox vely_fab(bx_y,ncomp);
            FArrayBox velz_fab(bx_z,ncomp);
            
            Elixir velx_fab_eli=velx_fab.elixir();
            Elixir vely_fab_eli=vely_fab.elixir();
            Elixir velz_fab_eli=velz_fab.elixir();
            
            velx_fab.setVal<RunOn::Device>(0.0);
            vely_fab.setVal<RunOn::Device>(0.0);
            velz_fab.setVal<RunOn::Device>(0.0);
            
            Array4<Real> velxarray = velx_fab.array();
            Array4<Real> velyarray = vely_fab.array();
            Array4<Real> velzarray = velz_fab.array();

            amrex::ParallelFor(bx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                electrochem_transport::compute_dcoeff(i, j, k, statearray, 
                        dcoeffarray, prob_lo, prob_hi,
                        dx, cur_time, *d_prob_parm);
            });

            amrex::ParallelFor(bx_x,
                  [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                electrochem_transport::compute_velx(i, j, k, statearray,
                        velxarray,prob_lo, prob_hi, dx, cur_time);
            });
            
            amrex::ParallelFor(bx_y,
                  [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                electrochem_transport::compute_vely(i, j, k, statearray,
                        velyarray,prob_lo, prob_hi, dx, cur_time);
            });

            amrex::ParallelFor(bx_z,
                  [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                electrochem_transport::compute_velz(i, j, k, statearray,
                        velzarray,prob_lo, prob_hi, dx, cur_time);
            });

            amrex::ParallelFor(bx,
                  [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                for(int comp=0;comp<ncomp;comp++)
                {
                    Real vx=0.5*(velxarray(i,j,k,comp)+velxarray(i+1,j,k,comp));
                    Real vy=0.5*(velyarray(i,j,k,comp)+velyarray(i,j+1,k,comp));
                    Real vz=0.5*(velzarray(i,j,k,comp)+velzarray(i,j,k+1,comp));

                    velarray(i,j,k,comp)=sqrt(vx*vx + vy*vy + vz*vz);
                }
            });
        }
    }


    Real maxdcoeff=dcoeff.norm0(0,0,true);
    //only loop over species
    for(int comp=0;comp<NUM_SPECIES;comp++)
    {
        Real diffcomp=dcoeff.norm0(comp,0,true);
        if(diffcomp > maxdcoeff)
        {
           maxdcoeff=diffcomp;
        }
    }
    for(int i=0;i<AMREX_SPACEDIM;i++)
    {
        dt_est = std::min(dt_est, (0.5/AMREX_SPACEDIM)*(dx[i]*dx[i])/maxdcoeff);
    }
    
    Real maxvel=vel.norm0(0,0,true);
    for(int comp=0;comp<NUM_SPECIES;comp++)
    {
        Real velcomp=vel.norm0(comp,0,true);
        if(velcomp > maxvel)
        {
           maxvel=velcomp;
        }
    }
    if(maxvel > 0.0)
    {
        for(int i=0;i<AMREX_SPACEDIM;i++)
        {
            dt_est = std::min(dt_est, (dx[i]/maxvel));
        }
    }

    Real maxcon = S_new.norm0(0,0,true);
    std::cout << "max concentration: " << maxcon << std::endl;
//    if(maxcon > 1.0e12) amrex::Abort("con too high");
    Real mincon = S_new.min(0,0,true);
    std::cout << "min concentration: " << maxcon << std::endl;
//    if(mincon < 0.0) amrex::Abort("negative concentration");

    // Currently, this never happens (function called with local = true).
    // Reduction occurs outside this function.
    if (!local) 
    {
        ParallelDescriptor::ReduceRealMin(dt_est);
    }

    dt_est *= cfl;
    return dt_est;
}
