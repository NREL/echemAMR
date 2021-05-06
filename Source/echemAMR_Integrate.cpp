#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>

#include <echemAMR.H>
#include<Chemistry.H>
#include<IntegralUtils.H>

//Returns the volume integral of the state variable <comp> on <domain>
//TODO: incorperate a level mask
Real echemAMR::VolumeIntegral(int comp, int domain)
{
    Real int_tmp = 0;
    Real exact_con;
    for (int lev = 0; lev <= finest_level; ++lev)
    {

        const auto dx = geom[lev].CellSizeArray();
        
        const Real cur_time = t_new[lev];
        MultiFab& S_new = phi_new[lev];
       
        //need fillpatched data for velocity calculation 
        // constexpr int num_grow = 2; 
        // MultiFab Sborder(grids[lev], dmap[lev], S_new.nComp(), num_grow);
        // FillPatch(lev, cur_time, Sborder, 0, Sborder.nComp()); 

        // need to implement the mask feature from https://github.com/Exawind/amr-wind/blob/main/amr-wind/utilities/sampling/Enstrophy.cpp

        Real nm1 = amrex::ReduceSum(S_new, lev,
        [=] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& fab) -> Real
        {
            Real r = 0.0;
            AMREX_LOOP_3D(bx, i, j, k,
            {
                r += electrochem_integral_utils::volume_value(i, j, k, comp, domain, fab, dx);
            });
            return r;
        });

        ParallelAllReduce::Sum(nm1, ParallelContext::CommunicatorSub());

        int_tmp = nm1;

        // exact_con = S_new.norm1(0,0,true);
        // ParallelDescriptor::ReduceRealSum(exact_con);

    }

    // amrex::Print() << "Exact Concentration Integral: " << exact_con << std::endl;

    return int_tmp;

}

//Returns the surface integral <domain>
//TODO: incorperate a level mask
//TODO: Extend to flux 
Real echemAMR::SurfaceIntegral(int comp, int domain1, int domain2)
{
    Real int_tmp = 0;
    Real exact_con;
    for (int lev = 0; lev <= finest_level; ++lev)
    {

        const auto dx = geom[lev].CellSizeArray();
        
        const Real cur_time = t_new[lev];
        MultiFab& S_new = phi_new[lev];
       
        //need fillpatched data for velocity calculation 
        constexpr int num_grow = 1; 
        MultiFab Sborder(grids[lev], dmap[lev], S_new.nComp(), num_grow);
        FillPatch(lev, cur_time, Sborder, 0, Sborder.nComp()); 

        // need to implement the mask feature from https://github.com/Exawind/amr-wind/blob/main/amr-wind/utilities/sampling/Enstrophy.cpp

        Real nm1 = amrex::ReduceSum(Sborder, lev,
        [=] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& fab) -> Real
        {
            Real r = 0.0;
            AMREX_LOOP_3D(bx, i, j, k,
            {
                r += electrochem_integral_utils::surface_value(i, j, k, comp, domain1, domain2, fab, geom[lev]);
            });
            return r;
        });

        ParallelAllReduce::Sum(nm1, ParallelContext::CommunicatorSub());

        int_tmp = nm1;

        // exact_con = S_new.norm1(0,0,true);
        // ParallelDescriptor::ReduceRealSum(exact_con);

    }

    // amrex::Print() << "Exact Concentration Integral: " << exact_con << std::endl;

    return int_tmp;

}

