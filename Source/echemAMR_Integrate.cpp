#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>

#include <echemAMR.H>
#include <Chemistry.H>
#include <IntegralUtils.H>

// Returns the volume integral of the state variable <comp> on <domain>
// TODO: incorperate a level mask
Real echemAMR::VolumeIntegral(int comp, int domain)
{
    Real exact_con;
    int captured_comp = comp;
    int captured_dm = domain;
    Real vol = 0.0;
    for (int lev = 0; lev <= finest_level; ++lev)
    {

        const auto dx = geom[lev].CellSizeArray();

        const Real cur_time = t_new[lev];
        MultiFab& S_new = phi_new[lev];

        // mask the level refinement overlap using: https://github.com/Exawind/amr-wind/blob/main/amr-wind/utilities/sampling/Enstrophy.cpp
        amrex::iMultiFab level_mask;
        if (lev < finest_level) {
            level_mask = makeFineMask(
                phi_new[lev].boxArray(), phi_new[lev].DistributionMap(),
                phi_new[lev+1].boxArray(), amrex::IntVect(2), 1, 0);
        } else {
            level_mask.define(
                phi_new[lev].boxArray(), phi_new[lev].DistributionMap(),
                1, 0, amrex::MFInfo());
            level_mask.setVal(1);
        }

        vol += amrex::ReduceSum(
            S_new, level_mask, 0, 
            [=] AMREX_GPU_HOST_DEVICE(
                Box const& bx, 
                Array4<Real const> const& fab, 
                Array4<int const> const& mask_arr) -> Real {

                    Real vol_part = 0.0;
                    amrex::Loop(bx, [=, &vol_part](int i, int j, int k) noexcept {
                        vol_part += electrochem_integral_utils::volume_value(i, j, k, captured_comp, captured_dm, fab, mask_arr, dx);
                    });

            return vol_part;
        });


    }
    ParallelAllReduce::Sum(vol, ParallelContext::CommunicatorSub());

    return vol;
}

// Returns the surface integral <domain>
// TODO: incorperate a level mask
// TODO: Extend to flux
Real echemAMR::SurfaceIntegral(int comp, int domain1, int domain2)
{
    Real surface_area = 0;
    Real exact_con;
    int captured_comp = comp;
    int captured_dm1 = domain1;
    int captured_dm2 = domain2;
    for (int lev = 0; lev <= finest_level; ++lev)
    {

        const auto dx = geom[lev].CellSizeArray();

        // Get the boundary ids
        const int* domlo = geom[lev].Domain().loVect();
        const int* domhi = geom[lev].Domain().hiVect();

        const Real cur_time = t_new[lev];
        MultiFab& S_new = phi_new[lev];

        // need fillpatched data for velocity calculation
        constexpr int num_grow = 1;
        MultiFab Sborder(grids[lev], dmap[lev], S_new.nComp(), num_grow);
        FillPatch(lev, cur_time, Sborder, 0, Sborder.nComp());

        // mask the level refinement overlap using: https://github.com/Exawind/amr-wind/blob/main/amr-wind/utilities/sampling/Enstrophy.cpp
        amrex::iMultiFab level_mask;
        if (lev < finest_level) {
            level_mask = makeFineMask(
                phi_new[lev].boxArray(), phi_new[lev].DistributionMap(),
                phi_new[lev+1].boxArray(), amrex::IntVect(2), 1, 0);
        } else {
            level_mask.define(
                phi_new[lev].boxArray(), phi_new[lev].DistributionMap(),
                1, 0, amrex::MFInfo());
            level_mask.setVal(1);
        }

        surface_area += amrex::ReduceSum(
            Sborder, level_mask, 0, 
            [=] AMREX_GPU_HOST_DEVICE(
                Box const& bx, 
                Array4<Real const> const& fab, 
                Array4<int const> const& mask_arr) -> Real {

                    Real sa_part = 0.0;
                    amrex::Loop(bx, [=, &sa_part](int i, int j, int k) noexcept {
                         sa_part += electrochem_integral_utils::surface_value(i, j, k, captured_comp, captured_dm1, captured_dm2, fab, mask_arr, domlo, domhi, dx); 
                    });
            return sa_part;
        });

    }

    ParallelAllReduce::Sum(surface_area, ParallelContext::CommunicatorSub());

    return surface_area;
}

// Returns the surface integral <domain>
// TODO: incorperate a level mask
// TODO: Extend to flux
Real echemAMR::CurrentCollectorIntegral(int comp, int domain)
{
    Real surface_area = 0;
    Real exact_con;
    int captured_comp = comp;
    int captured_dm = domain;
    for (int lev = 0; lev <= finest_level; ++lev)
    {

        const auto dx = geom[lev].CellSizeArray();

        // Get the boundary ids
        const int* domlo = geom[lev].Domain().loVect();
        const int* domhi = geom[lev].Domain().hiVect();

        const Real cur_time = t_new[lev];
        MultiFab& S_new = phi_new[lev];

        // need fillpatched data for velocity calculation
        constexpr int num_grow = 1;
        MultiFab Sborder(grids[lev], dmap[lev], S_new.nComp(), num_grow);
        FillPatch(lev, cur_time, Sborder, 0, Sborder.nComp());

        // mask the level refinement overlap using: https://github.com/Exawind/amr-wind/blob/main/amr-wind/utilities/sampling/Enstrophy.cpp
        amrex::iMultiFab level_mask;
        if (lev < finest_level) {
            level_mask = makeFineMask(
                phi_new[lev].boxArray(), phi_new[lev].DistributionMap(),
                phi_new[lev+1].boxArray(), amrex::IntVect(2), 1, 0);
        } else {
            level_mask.define(
                phi_new[lev].boxArray(), phi_new[lev].DistributionMap(),
                1, 0, amrex::MFInfo());
            level_mask.setVal(1);
        }

        surface_area += amrex::ReduceSum(
            Sborder, level_mask, 0, 
                [=] AMREX_GPU_HOST_DEVICE(
                    Box const& bx,
                    Array4<Real const> const& fab, 
                    Array4<int const> const& mask_arr) -> Real {

                        Real sa_part = 0.0;
                        amrex::Loop(bx, [=, &sa_part](int i, int j, int k) noexcept {
                            sa_part += electrochem_integral_utils::current_collector_value(i, j, k, captured_comp, captured_dm, fab, mask_arr, domlo, domhi, dx); 
                        });
                return sa_part;
        });

    }

    ParallelAllReduce::Sum(surface_area, ParallelContext::CommunicatorSub());

    return surface_area;
}
