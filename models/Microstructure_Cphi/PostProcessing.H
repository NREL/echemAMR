#ifndef _POSTPROCESSING_H_
#define _POSTPROCESSING_H_

#include <echemAMR.H>
#include <Chemistry.H>
#include <DirectionSelector.H>
using namespace amrex;

template <typename IndexSelector>
void line_plot(
    const IndexSelector& idxOp,
    int num_points,
    amrex::Real time,
    int timestep,
    int finest_level,
    amrex::Vector<amrex::Geometry>& geom,
    amrex::Vector<amrex::DistributionMapping>& dmap,
    amrex::Vector<amrex::BoxArray>& grids,
    amrex::Vector<amrex::MultiFab>& phi_new,
    int index,
    std::string& filename_type)
{

    const int axis = idxOp.dir;
    const int index_in = index;
    // this is hard coded for the 4 level sets
    const int num_comps = 4;

    AMREX_ALWAYS_ASSERT(num_points > 0);

    const amrex::Real xlo = geom[0].ProbLo(axis);
    const amrex::Real xhi = geom[0].ProbHi(axis);
    const amrex::Real line_dx = (xhi - xlo) / static_cast<amrex::Real>(num_points);

    amrex::Vector<amrex::Real> line_volume;
    amrex::Vector<amrex::Real> line_minimum;
    amrex::Vector<amrex::Real> line_average;
    amrex::Vector<amrex::Real> line_maximum;
    amrex::Vector<amrex::Real> line_stddev;

    line_volume.resize(num_points * num_comps, 0.0);
    line_minimum.resize(num_points * num_comps, 1.0e30);
    line_average.resize(num_points * num_comps, 0.0);
    line_maximum.resize(num_points * num_comps, -1.0e30);
    line_stddev.resize(num_points * num_comps, 0.0);

    amrex::Vector<amrex::Real> line_location(num_points);
    for (int i = 0; i < num_points; ++i)
    {
        line_location[i] = xlo + (i + 0.5) * line_dx;
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
        GeometryData geomData = geom[lev].data();

        // mask covered cells
        amrex::iMultiFab level_mask;
        if (lev < finest_level)
        {
            level_mask = makeFineMask(grids[lev], dmap[lev], grids[lev], amrex::IntVect(2), 1, 0);
        } else
        {
            level_mask.define(grids[lev], dmap[lev], 1, 0);
            level_mask.setVal(1);
        }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

        for (MFIter mfi(phi_new[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            auto phi_arr = phi_new[lev].const_array(mfi);
            auto mask_arr = level_mask.const_array(mfi);

            amrex::Box pbx = PerpendicularBox<IndexSelector>(bx, amrex::IntVect{0, 0, 0});

            amrex::ParallelFor(
                amrex::Gpu::KernelInfo().setReduction(true), pbx, [=] AMREX_GPU_DEVICE(int p_i, int p_j, int p_k, amrex::Gpu::Handler const& handler) noexcept {
                    // parallel box
                    amrex::Box lbx = ParallelBox<IndexSelector>(bx, amrex::IntVect{p_i, p_j, p_k});

                    for (int k = lbx.smallEnd(2); k <= lbx.bigEnd(2); ++k)
                    {
                        for (int j = lbx.smallEnd(1); j <= lbx.bigEnd(1); ++j)
                        {
                            for (int i = lbx.smallEnd(0); i <= lbx.bigEnd(0); ++i)
                            {

                                if (mask_arr(i, j, k) == 1)
                                {

                                    // cell coordinates
                                    const amrex::Real cell_xlo = xlo + idxOp(i, j, k) * geomData.CellSize()[axis];
                                    const amrex::Real cell_xhi = cell_xlo + geomData.CellSize()[axis];
                                    const amrex::Real dy = geomData.CellSize()[idxOp.odir1];
                                    const amrex::Real dz = geomData.CellSize()[idxOp.odir2];

                                    // line indices
                                    const int line_ind_lo = amrex::min(amrex::max(static_cast<int>((cell_xlo - xlo) / line_dx), 0), num_points - 1);
                                    const int line_ind_hi = amrex::min(amrex::max(static_cast<int>((cell_xhi - xlo) / line_dx), 0), num_points - 1);

                                    AMREX_ALWAYS_ASSERT(line_ind_lo >= 0);
                                    AMREX_ALWAYS_ASSERT(line_ind_hi >= 0);
                                    AMREX_ALWAYS_ASSERT(line_ind_lo < num_points);
                                    AMREX_ALWAYS_ASSERT(line_ind_hi < num_points);

                                    for (int ind = line_ind_lo; ind <= line_ind_hi; ++ind)
                                    {

                                        // line coordinates
                                        const amrex::Real line_xlo = xlo + ind * line_dx;
                                        const amrex::Real line_xhi = line_xlo + line_dx;

                                        amrex::Real dx;

                                        if (line_xlo <= cell_xlo)
                                            dx = line_xhi - cell_xlo;
                                        else if (line_xhi >= cell_xhi)
                                            dx = cell_xhi - line_xlo;
                                        else
                                            dx = line_dx;

                                        dx = amrex::min(dx, geomData.CellSize()[axis]);

                                        for (int n = 0; n < num_comps; ++n)
                                        {

                                            const amrex::Real field = phi_arr(i, j, k, index_in);
                                            const amrex::Real lvlset = amrex::max(phi_arr(i, j, k, n + 1), 0.0);
                                            const amrex::Real vol = dx * dy * dz;

                                            // min is annoying can't use level set here
                                            if (lvlset > 0.0) amrex::Gpu::deviceReduceMin(&line_min[num_comps * ind + n], field, handler);
                                            amrex::Gpu::deviceReduceSum(&line_avg[num_comps * ind + n], field * lvlset * vol, handler);
                                            amrex::Gpu::deviceReduceMax(&line_max[num_comps * ind + n], field * lvlset, handler);
                                            amrex::Gpu::deviceReduceSum(&line_vol[num_comps * ind + n], vol * lvlset, handler);

                                        } // component loop
                                    }     // line loop
                                }         // mask if statement
                            }             // i loop
                        }                 // j loop
                    }                     // k loop
                });                       // parallel for
        }                                 // mfiter loop
    }                                     // level loop

    lmin.copyToHost(line_minimum.data(), line_minimum.size());
    lavg.copyToHost(line_average.data(), line_average.size());
    lmax.copyToHost(line_maximum.data(), line_maximum.size());
    lvol.copyToHost(line_volume.data(), line_volume.size());

    amrex::ParallelDescriptor::ReduceRealMin(line_minimum.data(), line_minimum.size(), ParallelDescriptor::IOProcessorNumber());
    amrex::ParallelDescriptor::ReduceRealSum(line_average.data(), line_average.size());
    amrex::ParallelDescriptor::ReduceRealMax(line_maximum.data(), line_maximum.size(), ParallelDescriptor::IOProcessorNumber());
    amrex::ParallelDescriptor::ReduceRealSum(line_volume.data(), line_volume.size());

    // just make new ones until we figure out how to copytodevice...
    amrex::AsyncArray<amrex::Real> lavg2(line_average.data(), line_average.size());
    amrex::AsyncArray<amrex::Real> lvol2(line_volume.data(), line_volume.size());
    line_avg = lavg2.data();
    line_vol = lvol2.data();

    // FIXME: lots of code duplication here...
    for (int lev = 0; lev <= finest_level; ++lev)
    {
        GeometryData geomData = geom[lev].data();

        // mask covered cells
        amrex::iMultiFab level_mask;
        if (lev < finest_level)
        {
            level_mask = makeFineMask(grids[lev], dmap[lev], grids[lev], amrex::IntVect(2), 1, 0);
        } else
        {
            level_mask.define(grids[lev], dmap[lev], 1, 0);
            level_mask.setVal(1);
        }

#ifdef _OPENMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif

        for (MFIter mfi(phi_new[lev], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            auto phi_arr = phi_new[lev].const_array(mfi);
            auto mask_arr = level_mask.const_array(mfi);

            amrex::Box pbx = PerpendicularBox<IndexSelector>(bx, amrex::IntVect{0, 0, 0});

            amrex::ParallelFor(
                amrex::Gpu::KernelInfo().setReduction(true), pbx, [=] AMREX_GPU_DEVICE(int p_i, int p_j, int p_k, amrex::Gpu::Handler const& handler) noexcept {
                    amrex::Box lbx = ParallelBox<IndexSelector>(bx, amrex::IntVect{p_i, p_j, p_k});

                    for (int k = lbx.smallEnd(2); k <= lbx.bigEnd(2); ++k)
                    {
                        for (int j = lbx.smallEnd(1); j <= lbx.bigEnd(1); ++j)
                        {
                            for (int i = lbx.smallEnd(0); i <= lbx.bigEnd(0); ++i)
                            {

                                if (mask_arr(i, j, k) == 1)
                                {

                                    // cell coordinates
                                    const amrex::Real cell_xlo = xlo + idxOp(i, j, k) * geomData.CellSize()[axis];
                                    const amrex::Real cell_xhi = cell_xlo + geomData.CellSize()[axis];
                                    const amrex::Real dy = geomData.CellSize()[idxOp.odir1];
                                    const amrex::Real dz = geomData.CellSize()[idxOp.odir2];

                                    // line indices
                                    const int line_ind_lo = amrex::min(amrex::max(static_cast<int>((cell_xlo - xlo) / line_dx), 0), num_points - 1);
                                    const int line_ind_hi = amrex::min(amrex::max(static_cast<int>((cell_xhi - xlo) / line_dx), 0), num_points - 1);

                                    for (int ind = line_ind_lo; ind <= line_ind_hi; ++ind)
                                    {

                                        // line coordinates
                                        const amrex::Real line_xlo = xlo + ind * line_dx;
                                        const amrex::Real line_xhi = line_xlo + line_dx;

                                        amrex::Real dx;

                                        if (line_xlo <= cell_xlo)
                                            dx = line_xhi - cell_xlo;
                                        else if (line_xhi >= cell_xhi)
                                            dx = cell_xhi - line_xlo;
                                        else
                                            dx = line_dx;

                                        dx = amrex::min(dx, geomData.CellSize()[axis]);

                                        for (int n = 0; n < num_comps; ++n)
                                        {
                                            const amrex::Real field = phi_arr(i, j, k, index_in);
                                            // this assumes level sets are stacked after concentration
                                            const amrex::Real lvlset = amrex::max(phi_arr(i, j, k, n + 1), 0.0);
                                            const amrex::Real vol = dx * dy * dz;
                                            const amrex::Real mean = line_avg[num_comps * ind + n] / line_vol[num_comps * ind + n];
                                            const amrex::Real stddev = std::pow(field - mean, 2.0) * lvlset * vol;
                                            amrex::Gpu::deviceReduceSum(&line_std[num_comps * ind + n], stddev, handler);
                                        } // component loop
                                    }     // line loop
                                }         // mask if statement
                            }             // i loop
                        }                 // j loop
                    }                     // k loop*/
                });                       // parallel for
        }                                 // mfiter loop
    }                                     // level loop

    lstd.copyToHost(line_stddev.data(), line_stddev.size());
    amrex::ParallelDescriptor::ReduceRealSum(line_stddev.data(), line_stddev.size(), ParallelDescriptor::IOProcessorNumber());

    amrex::Print() << "Writing line plot file " << filename_type << "\n";


    // only IO Processor outputs file
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        const int num_files = 4;

        AMREX_ALWAYS_ASSERT(num_files == num_comps);

        std::ofstream outfile[num_files];
        std::string filename[num_files];

        filename[0] = "anode_" + filename_type + ".csv";
        filename[1] = "cathode_" + filename_type + ".csv";
        filename[2] = "electrolyte_" + filename_type + ".csv";
        filename[3] = "separator_" + filename_type + ".csv";

        for (int f = 0; f < num_files; ++f)
        {
            if (timestep == 0)
            {
                outfile[f].open(filename[f].c_str(), std::ios_base::out);
                outfile[f] << "Time_step    Time[s]    Position[m]    Min    Mean    Max    Std" << std::endl;
            } else
            {
                outfile[f].open(filename[f].c_str(), std::ios_base::out | std::ios_base::app);
            }

            outfile[f].precision(16);

            for (int i = 0; i < num_points; ++i)
            {
                outfile[f] << timestep << ' ' << std::scientific << time << ' ' << line_location[i] << ' ' << line_minimum[num_comps * i + f] << ' '
                           << line_average[num_comps * i + f] / line_volume[num_comps * i + f] << ' ' << line_maximum[num_comps * i + f] << ' '
                           << std::sqrt(line_stddev[num_comps * i + f] / line_volume[num_comps * i + f]) << std::endl;
            }
        }
    }
}

void echemAMR::postprocess(Real time, int timestep, Real dt, GlobalStorage* globalstorage)
{

    if (line_plot_int > 0 && ((timestep + 1) % line_plot_int == 0 || timestep == 0))
    {
        std::string co = "concentration";
        std::string po = "potential";

        if (line_plot_dir == 0)
        {
            line_plot(XDir(), line_plot_npoints, time, timestep, finest_level, geom, dmap, grids, phi_new, CO_ID, co);
            line_plot(XDir(), line_plot_npoints, time, timestep, finest_level, geom, dmap, grids, phi_new, POT_ID, po);
        } else if (line_plot_dir == 1)
        {
            line_plot(YDir(), line_plot_npoints, time, timestep, finest_level, geom, dmap, grids, phi_new, CO_ID, co);
            line_plot(YDir(), line_plot_npoints, time, timestep, finest_level, geom, dmap, grids, phi_new, POT_ID, po);
        } else
        {
            line_plot(ZDir(), line_plot_npoints, time, timestep, finest_level, geom, dmap, grids, phi_new, CO_ID, co);
            line_plot(ZDir(), line_plot_npoints, time, timestep, finest_level, geom, dmap, grids, phi_new, POT_ID, po);
        }
    }


    // Calculate domain volume
    Real anode_volume = VolumeIntegral(-1, A_ID);
    Real cathode_volume = VolumeIntegral(-1, C_ID);
    Real electrolyte_volume = VolumeIntegral(-1, E_ID);
    Real separator_volume = VolumeIntegral(-1, S_ID);
    Real total_volume = anode_volume+cathode_volume+electrolyte_volume+separator_volume;

    // Calculate mol per domain (current value)
    Real anode_current_mol = VolumeIntegral(CO_ID, A_ID);
    Real cathode_current_mol = VolumeIntegral(CO_ID, C_ID);
    Real electrolyte_current_mol = VolumeIntegral(CO_ID, E_ID);
    Real separator_current_mol = VolumeIntegral(CO_ID, S_ID);
    Real total_current_mol = anode_current_mol+cathode_current_mol+electrolyte_current_mol+separator_current_mol;

    // Expected mol per domain (target value)
    Real anode_target_dmol = globalstorage->charging * globalstorage->Applied_current*dt/FARADCONST; // [mol]
    Real cathode_target_dmol = -1 * globalstorage->charging * globalstorage->Applied_current*dt/FARADCONST; // [mol]
    Real electrolyte_target_dmol = 0; // [mol]
    // Real anode_target_dSOC = dt/3600 * Crate; // [] Alternative method, FYI
    //Real anode_target_dC = anode_target_dSOC * Csmax; // [mol.m-3]
    //Real anode_target_dmol = anode_target_dC * anode_volume; // [mol]
    //Real cathode_target_dmol = dt/3600 * Crate * anode_target_dSOC * Csmax * anode_volume; // [mol]
    //Real electrolyte_target_dmol = 0; // [mol]

    // Update mass error array
    Real anode_target_mol, electrolyte_target_mol, cathode_target_mol; // Initialization
    if (timestep == 0){
        anode_target_mol = anode_current_mol; // [mol]
        electrolyte_target_mol = electrolyte_current_mol; // [mol]
        cathode_target_mol = cathode_current_mol; // [mol]

    } else {
        anode_target_mol = globalstorage->Mass_error[4] + anode_target_dmol; // [mol] Theortical value
        electrolyte_target_mol = globalstorage->Mass_error[5] + electrolyte_target_dmol; // [mol] Theortical value
        cathode_target_mol = globalstorage->Mass_error[6] + cathode_target_dmol; // [mol] Theortical value
    }

    // Mass error
    Real anode_mass_error = 100*(anode_target_mol-anode_current_mol)/anode_target_mol;
    Real electrolyte_mass_error = 100*(electrolyte_target_mol-electrolyte_current_mol)/electrolyte_target_mol;
    Real cathode_mass_error = 100*(cathode_target_mol-cathode_current_mol)/cathode_target_mol;

    // Calculate potential per domain (current value)
    Real anode_current_potential = VolumeIntegral(POT_ID, A_ID)/anode_volume;
    Real cathode_current_potential = VolumeIntegral(POT_ID, C_ID)/cathode_volume;
    Real electrolyte_current_potential = VolumeIntegral(POT_ID, E_ID)/electrolyte_volume;
    Real separator_current_potential = VolumeIntegral(POT_ID, S_ID)/separator_volume;
    
    // Overwritte
    globalstorage->Mass_error[0]=time;
    globalstorage->Mass_error[1]=anode_current_mol; globalstorage->Mass_error[2]=electrolyte_current_mol; globalstorage->Mass_error[3]=cathode_current_mol; 
    globalstorage->Mass_error[4]=anode_target_mol; globalstorage->Mass_error[5]=electrolyte_target_mol; globalstorage->Mass_error[6]=cathode_target_mol; 

    // PRINT STATMENT
    amrex::Print() << std::endl;

    if (timestep == 0){ // Print static information
        amrex::Print() << " VOLUMES:" << std::endl;
        amrex::Print() << "    Anode Volume:       " << anode_volume << " m3"<< std::endl;
        amrex::Print() << "    Cathode Volume:     " << cathode_volume << " m3"<< std::endl;
        amrex::Print() << "    Electrolyte Volume: " << electrolyte_volume << " m3"<< std::endl;
        amrex::Print() << "    Separator Volume:   " << separator_volume << " m3"<< std::endl;
        amrex::Print() << "    Total Volume:       " << total_volume << " m3"<< std::endl;
        amrex::Print() << std::endl;
    }

    amrex::Print() << std::endl;
    amrex::Print() << "----------------------" << std::endl;
    amrex::Print() << "ITERATION NO: " << timestep << ", (dis)charging time " << time << " s " << std::endl;
    amrex::Print() << "----------------------" << std::endl;
    amrex::Print() << std::endl;

    amrex::Print() << "APPLIED LOADING:" << std::endl;
    amrex::Print() << "    Applied_current " << globalstorage->Applied_current << " A" << " for duration " << dt << " s" << std::endl;
    amrex::Print() << std::endl;

    amrex::Print() << "MOL AND CONCENTRATIONS:       " << std::endl;
    amrex::Print() << "    Anode:       " << anode_current_mol << " mol / " << anode_current_mol/anode_volume << " mol.m-3" << std::endl;
    amrex::Print() << "    Cathode:     " << cathode_current_mol << " mol / " << cathode_current_mol/cathode_volume << " mol.m-3" << std::endl;
    amrex::Print() << "    Electrolyte: " << electrolyte_current_mol << " mol / " << electrolyte_current_mol/electrolyte_volume << " mol.m-3" << std::endl;
    amrex::Print() << "    Separator:   " << separator_current_mol << " mol / " << separator_current_mol/separator_volume << " mol.m-3" << std::endl;
    amrex::Print() << std::endl;

    amrex::Print() << "MASS ERROR:" << std::endl;
    amrex::Print() << "    Anode      : current = " << anode_current_mol << " mol , target = " << anode_target_mol << " mol , error = " << anode_mass_error << " % " << std::endl;
    amrex::Print() << "    Cathode    : current = " << cathode_current_mol << " mol , target = " << cathode_target_mol << " mol , error = " << cathode_mass_error << " % " << std::endl;
    amrex::Print() << "    Electrolyte: current = " << electrolyte_current_mol << " mol , target = " << electrolyte_target_mol << " mol , error = " << electrolyte_mass_error << " % " << std::endl;
    amrex::Print() << std::endl;

    amrex::Print() << "POTENTIAL:       " << std::endl;
    amrex::Print() << "    Anode:       " << anode_current_potential << " V" << std::endl;
    amrex::Print() << "    Cathode:     " << cathode_current_potential << " V" << std::endl;
    amrex::Print() << "    Electrolyte: " << electrolyte_current_potential << " V" << std::endl;
    amrex::Print() << "    Separator:   " << separator_current_potential << " V" << std::endl;
    amrex::Print() << std::endl;

    // SAVE MASS ERROR IN % IN CSV FILE
    if (amrex::ParallelDescriptor::IOProcessor()) // only IO Processor outputs file
    {
        std::ofstream outfile;
        std::string filename;
        filename = "Masserror.csv";
            if (timestep == 0)
            {
                outfile.open(filename.c_str(), std::ios_base::out);
                outfile << "Time_step    Time[s]    Anode    Electrolyte    Cathode" << std::endl;
            } else if (timestep == 0 || ((timestep + 1) % 25 == 0))
            {
                outfile.open(filename.c_str(), std::ios_base::out | std::ios_base::app);
            }
            outfile.precision(16);
            outfile << timestep << ' ' << std::scientific << time << ' ' << anode_mass_error << ' ' << electrolyte_mass_error << ' ' << cathode_mass_error << std::endl;
    }    

}
#endif
