#ifndef _POSTPROCESSING_H_
#define _POSTPROCESSING_H_

#include <echemAMR.H>
#include <Chemistry.H>
#include <DirectionSelector.H>
using namespace amrex;

template <typename IndexSelector>
void line_plot(const IndexSelector& idxOp,
               int num_points,
               amrex::Real time,
               int timestep,
               int finest_level,
               amrex::Vector<amrex::Geometry> &geom,
               amrex::Vector<amrex::DistributionMapping>& dmap,
               amrex::Vector<amrex::BoxArray>& grids,
               amrex::Vector<amrex::MultiFab>& phi_new)
{

    const int axis = idxOp.dir;
    const int num_comps = 4;

    const bool print_stuff = false;

    AMREX_ALWAYS_ASSERT(num_points > 0);
    AMREX_ALWAYS_ASSERT(num_comps < NVAR);

    const amrex::Real xlo = geom[0].ProbLo(axis);
    const amrex::Real xhi = geom[0].ProbHi(axis);
    const amrex::Real line_dx = (xhi-xlo)/static_cast<amrex::Real>(num_points);

    amrex::Vector<amrex::Real> line_volume;
    amrex::Vector<amrex::Real> line_minimum;
    amrex::Vector<amrex::Real> line_average;
    amrex::Vector<amrex::Real> line_maximum;
    amrex::Vector<amrex::Real> line_stddev;

    line_volume.resize(num_points*num_comps, 0.0);
    line_minimum.resize(num_points*num_comps, 1.0e30);
    line_average.resize(num_points*num_comps, 0.0);
    line_maximum.resize(num_points*num_comps, -1.0e30);
    line_stddev.resize(num_points*num_comps, 0.0);

    amrex::Vector<amrex::Real> line_location(num_points);
    for(int i = 0; i < num_points; ++i){
        line_location[i] = xlo + (i+0.5)*line_dx;
    }

    // use AsyncArray to copy to device
    amrex::AsyncArray<amrex::Real> lmin(line_minimum.data(), line_minimum.size());
    amrex::AsyncArray<amrex::Real> lavg(line_average.data(), line_average.size());
    amrex::AsyncArray<amrex::Real> lmax(line_maximum.data(), line_maximum.size());
    amrex::AsyncArray<amrex::Real> lstd(line_stddev.data(), line_stddev.size());
    amrex::AsyncArray<amrex::Real> lvol(line_volume.data(), line_volume.size());

    amrex::Real* line_min = lmin.data();
    amrex::Real* line_avg = lavg.data();
    amrex::Real* line_max = lmax.data();
    amrex::Real* line_std = lstd.data();
    amrex::Real* line_vol = lvol.data();

    for (int lev = 0; lev <= finest_level; ++lev)
    {

        // mask covered cells
        amrex::iMultiFab level_mask;
        if (lev < finest_level) {
            level_mask = makeFineMask(grids[lev], dmap[lev], grids[lev], amrex::IntVect(2), 1, 0);
        } else {
            level_mask.define(grids[lev], dmap[lev], 1, 0);
            level_mask.setVal(1);
        }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

        for (MFIter mfi(phi_new[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            auto phi_arr = phi_new[lev].const_array(mfi);
            auto mask_arr = level_mask.const_array(mfi);

            amrex::Box pbx = PerpendicularBox<IndexSelector>(bx, amrex::IntVect{0, 0, 0});

            amrex::ParallelFor(
            amrex::Gpu::KernelInfo().setReduction(true), pbx,
            [=] AMREX_GPU_DEVICE(
                int p_i, int p_j, int p_k,
                amrex::Gpu::Handler const& handler) noexcept {

                // parallel box
                amrex::Box lbx = ParallelBox<IndexSelector>(bx, amrex::IntVect{p_i, p_j, p_k});

                for (int k = lbx.smallEnd(2); k <= lbx.bigEnd(2); ++k) {
                    for (int j = lbx.smallEnd(1); j <= lbx.bigEnd(1); ++j) {
                        for (int i = lbx.smallEnd(0); i <= lbx.bigEnd(0); ++i) {

                            if(mask_arr(i,j,k) == 1) {

                                // cell coordinates
                                const amrex::Real cell_xlo = xlo + idxOp(i,j,k)*geom[lev].CellSizeArray()[axis];
                                const amrex::Real cell_xhi = cell_xlo + geom[lev].CellSizeArray()[axis];
                                const amrex::Real dy = geom[lev].CellSizeArray()[idxOp.odir1];
                                const amrex::Real dz = geom[lev].CellSizeArray()[idxOp.odir2];

                                // line indices
                                const int line_ind_lo = amrex::min(amrex::max(static_cast<int>((cell_xlo-xlo)/line_dx), 0), num_points-1);
                                const int line_ind_hi = amrex::min(amrex::max(static_cast<int>((cell_xhi-xlo)/line_dx), 0), num_points-1);

                                if(print_stuff)
                                {
                                    amrex::Print() << std::endl << "index i: " << idxOp(i,j,k) << std::endl;
                                    amrex::Print() << "cell_xlo: " << cell_xlo << " cell_xhi: " << cell_xhi << std::endl;
                                    amrex::Print() << "line_ind_lo: " << line_ind_lo << " line_ind_hi: " << line_ind_hi << std::endl;
                                }

                                AMREX_ALWAYS_ASSERT(line_ind_lo >= 0);
                                AMREX_ALWAYS_ASSERT(line_ind_hi >= 0);
                                AMREX_ALWAYS_ASSERT(line_ind_lo < num_points);
                                AMREX_ALWAYS_ASSERT(line_ind_hi < num_points);

                                for(int ind = line_ind_lo; ind <= line_ind_hi; ++ind){

                                    // line coordinates
                                    const amrex::Real line_xlo = xlo + ind*line_dx;
                                    const amrex::Real line_xhi = line_xlo + line_dx;

                                    amrex::Real dx;

                                    if(line_xlo <= cell_xlo)
                                        dx = line_xhi - cell_xlo;
                                    else if(line_xhi >= cell_xhi)
                                        dx = cell_xhi - line_xlo;
                                    else
                                        dx = line_dx;

                                    dx = amrex::min(dx, geom[lev].CellSizeArray()[axis]);

                                    if(print_stuff)
                                    {
                                        amrex::Print() << "line index: " << ind << std::endl;
                                        amrex::Print() << "line_xlo: " << line_xlo << " line_xhi: " << line_xhi << std::endl;
                                        amrex::Print() << "dx: " << dx << std::endl;
                                        if(dx < 0.0) amrex::Print() << "negative volume error: " << idxOp(i,j,k) << ' ' << dx << std::endl;
                                    }

                                    for(int n = 0; n < num_comps; ++n) {

                                        const amrex::Real concen = phi_arr(i,j,k,0);
                                        const amrex::Real lvlset = amrex::max(phi_arr(i,j,k,n+1), 0.0);
                                        const amrex::Real vol = dx*dy*dz;

                                        // min is annoying can't use level set here
                                        if(lvlset > 0.0) amrex::Gpu::deviceReduceMin(&line_min[num_comps * ind + n], concen, handler);
                                        amrex::Gpu::deviceReduceSum(&line_avg[num_comps * ind + n], concen*lvlset*vol, handler);
                                        amrex::Gpu::deviceReduceMax(&line_max[num_comps * ind + n], concen*lvlset, handler);
                                        amrex::Gpu::deviceReduceSum(&line_vol[num_comps * ind + n], vol*lvlset, handler);


                                    } // component loop
                                } // line loop
                            } // mask if statement
                        } // i loop
                    } // j loop
                } // k loop
            }); // parallel for
        } // mfiter loop
    } // level loop

    lmin.copyToHost(line_minimum.data(), line_minimum.size());
    lavg.copyToHost(line_average.data(), line_average.size());
    lmax.copyToHost(line_maximum.data(), line_maximum.size());
    lvol.copyToHost(line_volume.data(), line_volume.size());

    amrex::ParallelDescriptor::ReduceRealMin(line_minimum.data(), line_minimum.size());
    amrex::ParallelDescriptor::ReduceRealSum(line_average.data(), line_average.size());
    amrex::ParallelDescriptor::ReduceRealMax(line_maximum.data(), line_maximum.size());
    amrex::ParallelDescriptor::ReduceRealSum(line_volume.data(), line_volume.size());

    // FIXME: lots of code duplication here...
    for (int lev = 0; lev <= finest_level; ++lev)
    {

        // mask covered cells
        amrex::iMultiFab level_mask;
        if (lev < finest_level) {
            level_mask = makeFineMask(grids[lev], dmap[lev], grids[lev], amrex::IntVect(2), 1, 0);
        } else {
            level_mask.define(grids[lev], dmap[lev], 1, 0);
            level_mask.setVal(1);
        }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

        for (MFIter mfi(phi_new[lev],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            auto phi_arr = phi_new[lev].const_array(mfi);
            auto mask_arr = level_mask.const_array(mfi);

            amrex::Box pbx = PerpendicularBox<IndexSelector>(bx, amrex::IntVect{0, 0, 0});

            amrex::ParallelFor(
            amrex::Gpu::KernelInfo().setReduction(true), pbx,
            [=] AMREX_GPU_DEVICE(
                int p_i, int p_j, int p_k,
                amrex::Gpu::Handler const& handler) noexcept {

                amrex::Box lbx = ParallelBox<IndexSelector>(bx, amrex::IntVect{p_i, p_j, p_k});

                for (int k = lbx.smallEnd(2); k <= lbx.bigEnd(2); ++k) {
                    for (int j = lbx.smallEnd(1); j <= lbx.bigEnd(1); ++j) {
                        for (int i = lbx.smallEnd(0); i <= lbx.bigEnd(0); ++i) {

                            if(mask_arr(i,j,k) == 1) {

                                // cell coordinates
                                const amrex::Real cell_xlo = xlo + idxOp(i,j,k)*geom[lev].CellSizeArray()[axis];
                                const amrex::Real cell_xhi = cell_xlo + geom[lev].CellSizeArray()[axis];
                                const amrex::Real dy = geom[lev].CellSizeArray()[idxOp.odir1];
                                const amrex::Real dz = geom[lev].CellSizeArray()[idxOp.odir2];

                                // line indices
                                const int line_ind_lo = amrex::min(amrex::max(static_cast<int>((cell_xlo-xlo)/line_dx), 0), num_points-1);
                                const int line_ind_hi = amrex::min(amrex::max(static_cast<int>((cell_xhi-xlo)/line_dx), 0), num_points-1);

                                for(int ind = line_ind_lo; ind <= line_ind_hi; ++ind){

                                    // line coordinates
                                    const amrex::Real line_xlo = xlo + ind*line_dx;
                                    const amrex::Real line_xhi = line_xlo + line_dx;

                                    amrex::Real dx;

                                    if(line_xlo <= cell_xlo)
                                        dx = line_xhi - cell_xlo;
                                    else if(line_xhi >= cell_xhi)
                                        dx = cell_xhi - line_xlo;
                                    else
                                        dx = line_dx;

                                    dx = amrex::min(dx, geom[lev].CellSizeArray()[axis]);

                                    for(int n = 0; n < num_comps; ++n) {
                                        const amrex::Real concen = phi_arr(i,j,k,0);
                                        const amrex::Real lvlset = amrex::max(phi_arr(i,j,k,n+1), 0.0);
                                        const amrex::Real vol = dx*dy*dz;
                                        const amrex::Real mean = line_avg[num_comps * ind + n]/line_volume[num_comps * ind + n];
                                        const amrex::Real stddev = std::pow(concen - mean, 2.0)*lvlset*vol;
                                        amrex::Gpu::deviceReduceSum(&line_std[num_comps * ind + n], stddev, handler);
                                    } // component loop
                                } // line loop
                            } // mask if statement
                        } // i loop
                    } // j loop
                } // k loop
            }); // parallel for
        } // mfiter loop
    } // level loop

    lstd.copyToHost(line_stddev.data(), line_stddev.size());
    amrex::ParallelDescriptor::ReduceRealSum(line_stddev.data(), line_stddev.size());

    // only IO Processor outputs file
    if(amrex::ParallelDescriptor::IOProcessor())
    {
        const int num_files = 4;

        AMREX_ALWAYS_ASSERT(num_files == num_comps);

        std::ofstream outfile[num_files];
        std::string filename[num_files];

        filename[0] = "anode.csv";
        filename[1] = "cathode.csv";
        filename[2] = "electrolyte.csv";
        filename[3] = "separator.csv";

        for(int f=0; f < num_files; ++f){
            if(timestep == 0){
                outfile[f].open(filename[f].c_str(), std::ios_base::out);
                outfile[f] << "Time step    Time[s]    Position[m]    Min    Mean    Max    Std" << std::endl;
            } else {
                outfile[f].open(filename[f].c_str(), std::ios_base::out | std::ios_base::app);
            }

            outfile[f].precision(16);

            for(int i=0;i<num_points; ++i){
                outfile[f] << timestep << ' ' << std::scientific
                           << time << ' '
                           << line_location[i] << ' '
                           << line_min[num_comps*i + f] << ' '
                           << line_average[num_comps*i + f]/line_volume[num_comps*i + f] << ' '
                           << line_max[num_comps*i + f] << ' '
                           << std::sqrt(line_stddev[num_comps*i + f]/line_volume[num_comps*i + f]) << std::endl;
            }

        }


    }

}

void echemAMR::postprocess(Real time, int timestep)
{

    amrex::Print() << std::endl;

    Real A_int = VolumeIntegral(-1, A_ID);
    amrex::Print() << "Anode Volume:       " << A_int << std::endl;
    Real C_int = VolumeIntegral(-1, C_ID);
    amrex::Print() << "Cathode Volume:     " << C_int << std::endl;
    Real E_int = VolumeIntegral(-1, E_ID);
    amrex::Print() << "Electrolyte Volume: " << E_int << std::endl;
    Real S_int = VolumeIntegral(-1, S_ID);
    amrex::Print() << "Separator Volume:   " << S_int << std::endl;
    Real tot_int = A_int+C_int+E_int+S_int;
    amrex::Print() << "Total Volume:       " << tot_int << std::endl;



    amrex::Print() << std::endl;

    Real cA_int = VolumeIntegral(CO_ID, A_ID);
    amrex::Print() << "Anode Volume:       " << cA_int << " (" << cA_int/A_int << ")" << std::endl;
    Real cC_int = VolumeIntegral(CO_ID, C_ID);
    amrex::Print() << "Cathode Volume:     " << cC_int << " (" << cC_int/C_int << ")" << std::endl;
    Real cE_int = VolumeIntegral(CO_ID, E_ID);
    amrex::Print() << "Electrolyte Volume: " << cE_int << " (" << cE_int/E_int << ")" << std::endl;
    Real cS_int = VolumeIntegral(CO_ID, S_ID);
    amrex::Print() << "Separator Volume:   " << cS_int << " (" << cS_int/S_int << ")" << std::endl;
    Real ctot_int = cA_int+cC_int+cE_int+cS_int;
    amrex::Print() << "Total Volume:       " << ctot_int << " (" << ctot_int/tot_int << ")" << std::endl;



    amrex::Print() << std::endl;

    Real test1 = SurfaceIntegral(-1, A_ID, E_ID);
    Real test2 = SurfaceIntegral(-1, C_ID, E_ID);
    amrex::Print() << "Anode-Electrolyte SA:   " << test1 << std::endl;
    amrex::Print() << "Cathode-Electrolyte SA: " << test2 << std::endl;
    amrex::Print() << std::endl;

    if (line_plot_int > 0 && ((timestep+1) % line_plot_int == 0 || timestep == 0))
    {
        if(line_plot_dir == 0)
            line_plot(XDir(), line_plot_npoints, time, timestep, finest_level, geom, dmap, grids, phi_new);
        else if(line_plot_dir == 1)
            line_plot(YDir(), line_plot_npoints, time, timestep, finest_level, geom, dmap, grids, phi_new);
        else
            line_plot(ZDir(), line_plot_npoints, time, timestep, finest_level, geom, dmap, grids, phi_new);

    }

}
#endif
