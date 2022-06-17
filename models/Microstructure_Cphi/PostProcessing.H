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

        filename[0] = echemAMR::output_folder + "line_plots/anode_" + filename_type + ".csv";
        filename[1] = echemAMR::output_folder + "line_plots/cathode_" + filename_type + ".csv";
        filename[2] = echemAMR::output_folder + "line_plots/electrolyte_" + filename_type + ".csv";
        filename[3] = echemAMR::output_folder + "line_plots/separator_" + filename_type + ".csv";

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

    // // VOLUMES
    // Calculate domain apparent volume
    Real anode_AM_volume = VolumeIntegral(-1, -1, A_AM_ID);
    Real anode_electrolyte_volume = VolumeIntegral(-1, -1, A_E_ID);
    Real anode_CBD_volume = VolumeIntegral(-1, -1, A_CBD_ID);
    Real separator_volume = VolumeIntegral(-1, -1, S_ID);
    Real cathode_AM_volume = VolumeIntegral(-1, -1, C_AM_ID);
    Real cathode_electrolyte_volume = VolumeIntegral(-1, -1, C_E_ID);
    Real cathode_CBD_volume = VolumeIntegral(-1, -1, C_CBD_ID);    
    Real total_electrolyte_volume = anode_electrolyte_volume + separator_volume + cathode_electrolyte_volume;
    Real total_active_material_volume = anode_AM_volume + cathode_AM_volume;
    Real totalCBD_volume = anode_CBD_volume + cathode_CBD_volume;
    Real total_anode_volume = anode_AM_volume + anode_electrolyte_volume + anode_CBD_volume;
    Real total_cathode_volume = cathode_AM_volume + cathode_electrolyte_volume + cathode_CBD_volume; 
    Real total_volume = total_electrolyte_volume + total_active_material_volume + totalCBD_volume;

    Real case_CBD=0;
    Real Case_SEP=0;
    if (totalCBD_volume==0 && globalstorage->CBD_transport==0 && globalstorage->Nanoscale_coefficients_txtimport_CBD==0) {
        case_CBD = 1;
    } else if (totalCBD_volume==0 && globalstorage->CBD_transport==0 && globalstorage->Nanoscale_coefficients_txtimport_CBD==1) {
        case_CBD = 2;
    } else if (totalCBD_volume>0 && anode_electrolyte_volume==0 && cathode_electrolyte_volume==0 && globalstorage->CBD_transport==1 && globalstorage->Nanoscale_coefficients_txtimport_CBD==1) {
        case_CBD = 3;
    } else if (totalCBD_volume>0 && anode_electrolyte_volume>0 && cathode_electrolyte_volume>0 && globalstorage->CBD_transport==0 && globalstorage->Nanoscale_coefficients_txtimport_CBD==0) {
        case_CBD = 4;
    } else if (totalCBD_volume>0 && anode_electrolyte_volume>0 && cathode_electrolyte_volume>0 && globalstorage->CBD_transport==1 && globalstorage->Nanoscale_coefficients_txtimport_CBD==1) {
        case_CBD = 5;        
    } else {
        amrex::Print() << " >>> WARNING !!! INCOHERENT PARAMETERS FOR CBD." << std::endl;
    }
    if (globalstorage->Nanoscale_coefficients_txtimport_separator==0) {
        Case_SEP = 1;
    } else if (globalstorage->Nanoscale_coefficients_txtimport_separator==1) {
        Case_SEP = 2;
    } else {
        amrex::Print() << " >>> WARNING !!! INCORRECT PARAMETER FOR SEPARATOR." << std::endl;
    }

    // True electrolyte volume
    Real anode_electrolyte_true_volume, cathode_electrolyte_true_volume;
    if (case_CBD==2) {
        anode_electrolyte_true_volume = anode_electrolyte_volume;
        cathode_electrolyte_true_volume = cathode_electrolyte_volume;
    } else { 
        anode_electrolyte_true_volume = VolumeIntegral(NP_ID, -1, A_E_ID);
        cathode_electrolyte_true_volume = VolumeIntegral(NP_ID, -1, C_E_ID);
    }
    Real separator_electrolyte_true_volume = VolumeIntegral(NP_ID, -1, S_ID);

    // True electrolyte and solid volumes for CBD
    Real anode_CBDe_true_volume, cathode_CBDe_true_volume;
    if (case_CBD==1 || case_CBD==2 || case_CBD==4) {
        anode_CBDe_true_volume = 0.0;
        cathode_CBDe_true_volume = 0.0;
    } else if (case_CBD==3 || case_CBD==5) {
        anode_CBDe_true_volume = VolumeIntegral(NP_ID, -1, A_CBD_ID);
        cathode_CBDe_true_volume = VolumeIntegral(NP_ID, -1, C_CBD_ID);
    }
    Real anode_CBDs_true_volume = anode_CBD_volume-anode_CBDe_true_volume;
    Real cathode_CBDs_true_volume = cathode_CBD_volume-cathode_CBDe_true_volume;

    // True electrolyte volume (sum of domains)
    Real anode_CBDelectrolyte_true_volume = anode_electrolyte_true_volume + anode_CBDe_true_volume;
    Real cathode_CBDelectrolyte_true_volume = cathode_electrolyte_true_volume + cathode_CBDe_true_volume;
    Real total_electrolyte_true_volume = anode_CBDelectrolyte_true_volume + separator_electrolyte_true_volume + cathode_CBDelectrolyte_true_volume;

    // True SOLID volume for CBD and separator
    Real totalCBD_solid_volume = anode_CBDs_true_volume + cathode_CBDs_true_volume;
    Real Separator_solid_volume = separator_volume-separator_electrolyte_true_volume;   

    // // VOLUME FRACTIONS AND POROSITY
    // From apparent volume
    Real anode_AM_vf = anode_AM_volume/total_anode_volume;
    Real anode_electrolyte_vf = anode_electrolyte_volume/total_anode_volume;
    Real anode_CBD_vf = anode_CBD_volume/total_anode_volume;
    Real separator_electrolyte_vf = 1.0;
    Real cathode_AM_vf = cathode_AM_volume/total_cathode_volume;
    Real cathode_electrolyte_vf = cathode_electrolyte_volume/total_cathode_volume;
    Real cathode_CBD_vf = cathode_CBD_volume/total_cathode_volume;
    // From true volume
    Real anode_electrolyte_true_vf = anode_CBDelectrolyte_true_volume/total_anode_volume;
    Real anode_CBDsolid_true_vf = anode_CBDs_true_volume/total_anode_volume;
    Real separator_electrolyte_true_vf = separator_electrolyte_true_volume/separator_volume;
    Real separator_solid_true_vf = Separator_solid_volume/separator_volume;
    Real cathode_electrolyte_true_vf = cathode_CBDelectrolyte_true_volume/total_cathode_volume;
    Real cathode_CBDsolid_true_vf = cathode_CBDs_true_volume/total_cathode_volume;

    // // MASS
    // From apparent volume
    Real anode_electrolyte_current_mol = VolumeIntegral(CO_ID, -1, A_E_ID);
    Real separator_current_mol = VolumeIntegral(CO_ID, -1, S_ID);
    Real cathode_electrolyte_current_mol = VolumeIntegral(CO_ID, -1, C_E_ID);

    Real anode_CBD_current_mol, cathode_CBD_current_mol;
    if (case_CBD==1 || case_CBD==2 || case_CBD==4) {
        anode_CBD_current_mol = 0.0;
        cathode_CBD_current_mol = 0.0;
    } else if (case_CBD==3 || case_CBD==5) {
        anode_CBD_current_mol = VolumeIntegral(CO_ID, -1, A_CBD_ID);
        cathode_CBD_current_mol = VolumeIntegral(CO_ID, -1, C_CBD_ID);
    }

    // From true volume
    Real anode_AM_current_true_mol = VolumeIntegral(CO_ID, -1, A_AM_ID);
    Real separator_current_true_mol = VolumeIntegral(CO_ID, NP_ID, S_ID);
    Real cathode_AM_current_true_mol = VolumeIntegral(CO_ID, -1, C_AM_ID);
    Real anode_electrolyte_current_true_mol,cathode_electrolyte_current_true_mol;
    if (case_CBD==2) {
        anode_electrolyte_current_true_mol = VolumeIntegral(CO_ID, -1, A_E_ID);
        cathode_electrolyte_current_true_mol = VolumeIntegral(CO_ID, -1, C_E_ID);
    } else {
        anode_electrolyte_current_true_mol = VolumeIntegral(CO_ID, NP_ID, A_E_ID);
        cathode_electrolyte_current_true_mol = VolumeIntegral(CO_ID, NP_ID, C_E_ID);        
    }

    Real anode_CBD_current_true_mol, cathode_CBD_current_true_mol = 0;
    if (case_CBD==1 || case_CBD==2 || case_CBD==4) {
        anode_CBD_current_true_mol = 0.0;
        cathode_CBD_current_true_mol = 0.0;
    } else if (case_CBD==3 || case_CBD==5) {
        anode_CBD_current_true_mol = VolumeIntegral(CO_ID, NP_ID, A_CBD_ID);
        cathode_CBD_current_true_mol = VolumeIntegral(CO_ID, NP_ID, C_CBD_ID);
    }    
    
    // Total mol in electrolyte (apparent)
    Real anode_CBDelectrolyte_current_mol = anode_electrolyte_current_mol + anode_CBD_current_mol;
    Real cathode_CBDelectrolyte_current_mol = cathode_electrolyte_current_mol + cathode_CBD_current_mol;
    Real total_CBDelectrolyte_current_mol = anode_CBDelectrolyte_current_mol+separator_current_mol+cathode_CBDelectrolyte_current_mol;
     // Total mol in electrolyte (true)
    Real anode_CBDelectrolyte_current_true_mol = anode_electrolyte_current_true_mol + anode_CBD_current_true_mol;
    Real cathode_CBDelectrolyte_current_true_mol = cathode_electrolyte_current_true_mol + cathode_CBD_current_true_mol;
    Real total_CBDelectrolyte_current_true_mol = anode_CBDelectrolyte_current_true_mol+separator_current_true_mol+cathode_CBDelectrolyte_current_true_mol;
    // Total mol in solid
    Real total_solid_current_true_mol = anode_AM_current_true_mol + cathode_AM_current_true_mol;
    // Total mol in domain
    Real total_current_true_mol = total_CBDelectrolyte_current_true_mol + total_solid_current_true_mol;

    // // MASS CONSERVATION ERROR
    // Expected mol per domain (target value)
    Real anode_AM_target_dmol = globalstorage->charging * globalstorage->Applied_current*dt/FARADCONST; // [mol]
    Real cathode_AM_target_dmol = -1 * globalstorage->charging * globalstorage->Applied_current*dt/FARADCONST; // [mol]
    Real total_CBDelectrolyte_target_dmol = 0; // [mol]
    // Real anode_target_dSOC = dt/3600 * Crate; // [] Alternative method, FYI
    //Real anode_target_dC = anode_target_dSOC * Csmax; // [mol.m-3]
    //Real anode_AM_target_dmol = anode_target_dC * anode_AM_volume; // [mol]
    //Real cathode_AM_target_dmol = dt/3600 * Crate * anode_target_dSOC * Csmax * anode_AM_volume; // [mol]
    //Real total_CBDelectrolyte_target_dmol = 0; // [mol]

    // Update mass error array
    Real anode_AM_target_mol, total_CBDelectrolyte_target_mol, cathode_AM_target_mol; // Initialization
    if (timestep == 0){
        anode_AM_target_mol = anode_AM_current_true_mol; // [mol]
        total_CBDelectrolyte_target_mol = total_CBDelectrolyte_current_true_mol; // [mol]
        cathode_AM_target_mol = cathode_AM_current_true_mol; // [mol]

    } else {
        anode_AM_target_mol = globalstorage->Mass_error[4] + anode_AM_target_dmol; // [mol] Theortical value
        total_CBDelectrolyte_target_mol = globalstorage->Mass_error[5] + total_CBDelectrolyte_target_dmol; // [mol] Theortical value
        cathode_AM_target_mol = globalstorage->Mass_error[6] + cathode_AM_target_dmol; // [mol] Theortical value
    }

    // Mass error
    Real anode_mass_error = 100*(anode_AM_target_mol-anode_AM_current_true_mol)/anode_AM_target_mol;
    Real electrolyte_mass_error = 100*(total_CBDelectrolyte_target_mol-total_CBDelectrolyte_current_true_mol)/total_CBDelectrolyte_target_mol;
    Real cathode_mass_error = 100*(cathode_AM_target_mol-cathode_AM_current_true_mol)/cathode_AM_target_mol;

    // Overwritte
    globalstorage->Mass_error[0]=time;
    globalstorage->Mass_error[1]=anode_AM_current_true_mol; globalstorage->Mass_error[2]=total_CBDelectrolyte_current_true_mol; globalstorage->Mass_error[3]=cathode_AM_current_true_mol; 
    globalstorage->Mass_error[4]=anode_AM_target_mol; globalstorage->Mass_error[5]=total_CBDelectrolyte_target_mol; globalstorage->Mass_error[6]=cathode_AM_target_mol; 

    // // POTENTIAL
    // Calculate potential per domain (current value)
    Real anode_CBDsolid_current_potential, cathode_CBDsolid_current_potential, anode_CBDelectrolyte_current_potential, cathode_CBDelectrolyte_current_potential;

    // Solid potential
    Real anode_AM_current_potential = VolumeIntegral(POT_ID, -1, A_AM_ID)/anode_AM_volume;
    Real cathode_AM_current_potential = VolumeIntegral(POT_ID, -1, C_AM_ID)/cathode_AM_volume;
    if (case_CBD==1 || case_CBD==2) {
        anode_CBDsolid_current_potential = 0.0;
        cathode_CBDsolid_current_potential = 0.0;
    } else {
        anode_CBDsolid_current_potential = VolumeIntegral(POTs_ID, -1, A_CBD_ID)/anode_CBD_volume;
        cathode_CBDsolid_current_potential = VolumeIntegral(POTs_ID, -1, C_CBD_ID)/cathode_CBD_volume;        
    } 

    // Electrolyte potential
    Real anode_electrolyte_current_potential = VolumeIntegral(POT_ID, -1, A_E_ID)/anode_electrolyte_volume;
    Real separator_current_potential = VolumeIntegral(POT_ID, -1, S_ID)/separator_volume;
    Real cathode_electrolyte_current_potential = VolumeIntegral(POT_ID, -1, C_E_ID)/cathode_electrolyte_volume;
    if (case_CBD==1 || case_CBD==2 || case_CBD==4) {
        anode_CBDelectrolyte_current_potential = 0.0;
        cathode_CBDelectrolyte_current_potential = 0.0;
    } else {
        anode_CBDelectrolyte_current_potential = VolumeIntegral(POT_ID, -1, A_CBD_ID)/anode_CBD_volume;
        cathode_CBDelectrolyte_current_potential = VolumeIntegral(POT_ID, -1, C_CBD_ID)/cathode_CBD_volume;
    } 
 
    // PRINT STATMENT
    amrex::Print() << std::endl;

    if (timestep == 0){ // Print static information
        amrex::Print() << " NANOSCALE:" << std::endl;
        if (case_CBD==1) {
            amrex::Print() << " - Case 1: CBD is not meshed and there is no nanoporosity (oversimplification!)." << std::endl;
        } else if (case_CBD==2) {
            amrex::Print() << " - Case 2: CBD is not meshed and only impact electrolyte effective transport." << std::endl;
        } else if (case_CBD==3) {
            amrex::Print() << " - Case 3: CBD replaces electrolyte... with nanoporosity (mixed domain)." << std::endl;
        } else if (case_CBD==4) {
            amrex::Print() << " - Case 4: CBD is meshed... without nanoporosity (solid domain)." << std::endl;
        } else if (case_CBD==5) {
            amrex::Print() << " - Case 5: CBD is meshed... with nanoporosity (mixed domain)." << std::endl;
        } else {
            amrex::Print() << " >>> WARNING !!! INCOHERENT PARAMETERS FOR CBD." << std::endl;
        }
        if (Case_SEP==1) {
            amrex::Print() << " - Separator has no nanoporosity (oversimplification!)." << std::endl;
        } else if (Case_SEP==2) {
            amrex::Print() << " - Separator has nanoporosity." << std::endl;
        } else {
            amrex::Print() << " >>> WARNING !!! INCORRECT PARAMETER FOR SEPARATOR." << std::endl;
        }
        amrex::Print() << std::endl;
        amrex::Print() << " APPARENT (MESH, w/o nanoporosity) VOLUMES:" << std::endl;
        amrex::Print() << "    Anode side:  " << std::endl;
        amrex::Print() << "    - Active material volume      : " << anode_AM_volume << " m3"<< std::endl;
        amrex::Print() << "    - Electrolyte w/o CBD volume  : " << anode_electrolyte_volume << " m3"<< std::endl;
        amrex::Print() << "    - CBD volume                  : " << anode_CBD_volume << " m3"<< std::endl;
        amrex::Print() << "    - Total volume                : " << total_anode_volume << " m3"<< std::endl;
        amrex::Print() << "    Separator volume  : " << separator_volume << " m3"<< std::endl;
        amrex::Print() << "    Cathode side:  " << std::endl;
        amrex::Print() << "    - Active material volume      : " << cathode_AM_volume << " m3"<< std::endl;
        amrex::Print() << "    - Electrolyte w/o CBD volume  : " << cathode_electrolyte_volume << " m3"<< std::endl;
        amrex::Print() << "    - CBD volume                  : " << cathode_CBD_volume << " m3"<< std::endl;    
        amrex::Print() << "    - Total volume                : " << total_cathode_volume << " m3"<< std::endl;     
        amrex::Print() << "    Total active material volume : " << total_active_material_volume << " m3"<< std::endl;   
        amrex::Print() << "    Total electrolyte volume     : " << total_electrolyte_volume << " m3"<< std::endl;
        amrex::Print() << "    Total CBD volume             : " << totalCBD_volume << " m3"<< std::endl;
        amrex::Print() << "    Total volume                 : " << total_volume << " m3"<< std::endl;
        amrex::Print() << std::endl;
        amrex::Print() << " TRUE (w/ nanoporosity) VOLUMES:" << std::endl;
        amrex::Print() << "    Anode side:  " << std::endl;
        amrex::Print() << "    - Electrolyte w/o CBD volume : " << anode_electrolyte_true_volume << " m3"<< std::endl;
        amrex::Print() << "    - Electrolyte volume in CBD  : " << anode_CBDe_true_volume << " m3"<< std::endl;
        amrex::Print() << "    Separator true electrolyte volume : " << separator_electrolyte_true_volume << " m3"<< std::endl;
        amrex::Print() << "    Cathode side:  " << std::endl;
        amrex::Print() << "    - Electrolyte w/o CBD volume : " << cathode_electrolyte_true_volume << " m3"<< std::endl;
        amrex::Print() << "    - Electrolyte volume in CBD  : " << cathode_CBDe_true_volume << " m3"<< std::endl;
        amrex::Print() << "    Total active material volume : " << total_active_material_volume << " m3"<< std::endl;   
        amrex::Print() << "    Total electrolyte volume     : " << total_electrolyte_true_volume << " m3"<< std::endl;
        amrex::Print() << "    Total CBD solid volume       : " << totalCBD_solid_volume << " m3"<< std::endl;
        amrex::Print() << "    Separator solid volume       : " << Separator_solid_volume << " m3"<< std::endl;
        // Total volume sanity check
        amrex::Print() << "    Total volume                 : " << total_active_material_volume + total_electrolyte_true_volume + totalCBD_solid_volume + Separator_solid_volume << " m3"<< std::endl;        
        amrex::Print() << std::endl;
        amrex::Print() << " APPARENT (MESH, w/o nanoporosity) VOLUME FRACTIONS:" << std::endl;
        amrex::Print() << "    Anode    : electrolyte | active material | CBD       : " << anode_electrolyte_vf << " | "<< anode_AM_vf << " | "<< anode_CBD_vf << std::endl;
        amrex::Print() << "    Separator: electrolyte | solid                       : " << separator_electrolyte_vf << " | "<< 0.0 << std::endl;
        amrex::Print() << "    Cathode  : electrolyte | active material | CBD       : " << cathode_electrolyte_vf << " | "<< cathode_AM_vf << " | "<< cathode_CBD_vf << std::endl;
        amrex::Print() << " TRUE (w/ nanoporosity) VOLUME FRACTIONS:" << std::endl;
        amrex::Print() << "    Anode    : electrolyte | active material | CBD solid : " << anode_electrolyte_true_vf << " | "<< anode_AM_vf << " | "<< anode_CBDsolid_true_vf << std::endl;
        amrex::Print() << "    Separator: electrolyte | solid                       : " << separator_electrolyte_true_vf << " | "<< separator_solid_true_vf << std::endl;
        amrex::Print() << "    Cathode  : electrolyte | active material | CBD solid : " << cathode_electrolyte_true_vf << " | "<< cathode_AM_vf << " | "<< cathode_CBDsolid_true_vf << std::endl;

        amrex::Print() << std::endl;
        amrex::Print() << "----------------------" << std::endl;
        amrex::Print() << "INITIALIZATION " << std::endl;
        amrex::Print() << "----------------------" << std::endl;
        amrex::Print() << std::endl;

    } else {
        amrex::Print() << std::endl;
        amrex::Print() << "----------------------" << std::endl;
        amrex::Print() << "ITERATION NO: " << timestep << ", (dis)charging time " << time << " s " << std::endl;
        amrex::Print() << "----------------------" << std::endl;
        amrex::Print() << std::endl;
    }

    amrex::Print() << "APPLIED LOADING:" << std::endl;
    amrex::Print() << "    Applied_current " << globalstorage->Applied_current << " A" << " for duration " << dt << " s" << std::endl;
    amrex::Print() << std::endl;

    amrex::Print() << "MOL AND CONCENTRATIONS:       " << std::endl;
    amrex::Print() << "     Anode side:  " << std::endl;
    amrex::Print() << "    - active material (cs)    : " << anode_AM_current_true_mol << " mol (true) | " << anode_AM_current_true_mol/anode_AM_volume << " mol.m-3 (true)" << std::endl;
    amrex::Print() << "    - electrolyte w/o CBD (ce): " << anode_electrolyte_current_mol << " mol (apparent) | " << anode_electrolyte_current_true_mol << " mol (true) | " << anode_electrolyte_current_mol/anode_electrolyte_volume << " mol.m-3 (apparent) | " << anode_electrolyte_current_true_mol/anode_electrolyte_true_volume << " mol.m-3 (true)" << std::endl;
    if (case_CBD==3 || case_CBD==5) {
    amrex::Print() << "    - CBD (ce)                : " << anode_CBD_current_mol << " mol (apparent) | " << anode_CBD_current_true_mol << " mol (true) | " << anode_CBD_current_mol/anode_CBD_volume << " mol.m-3 (apparent) | " << anode_CBD_current_true_mol/anode_CBDe_true_volume << " mol.m-3 (true)" << std::endl;
    amrex::Print() << "    - total electrolyte (ce)  : " << anode_CBDelectrolyte_current_mol << " mol (apparent) | " << anode_CBDelectrolyte_current_true_mol << " mol (true) | " << anode_CBDelectrolyte_current_mol/(anode_electrolyte_volume+anode_CBD_volume) << " mol.m-3 (apparent) | " << anode_CBDelectrolyte_current_true_mol/anode_CBDelectrolyte_true_volume << " mol.m-3 (true)" << std::endl;
    }
    amrex::Print() << "    Separator (ce)            : " << separator_current_mol << " mol (apparent) | " << separator_current_true_mol << " mol (true) | " << separator_current_mol/separator_volume << " mol.m-3 (apparent) | " << separator_current_true_mol/separator_electrolyte_true_volume << " mol.m-3 (true)" << std::endl;
    amrex::Print() << "     Cathode side:  " << std::endl;
    amrex::Print() << "    - active material (cs)    : " << cathode_AM_current_true_mol << " mol (true) | " << cathode_AM_current_true_mol/cathode_AM_volume << " mol.m-3 (true)" << std::endl;
    amrex::Print() << "    - electrolyte w/o CBD (ce): " << cathode_electrolyte_current_mol << " mol (apparent) | "  << cathode_electrolyte_current_true_mol << " mol (true) | " << cathode_electrolyte_current_mol/cathode_electrolyte_volume << " mol.m-3 (apparent) | " << cathode_electrolyte_current_true_mol/cathode_electrolyte_true_volume << " mol.m-3 (true)" << std::endl;
    if (case_CBD==3 || case_CBD==5) {
    amrex::Print() << "    - CBD (ce)                : " << cathode_CBD_current_mol << " mol (apparent) | " << cathode_CBD_current_true_mol << " mol (true) | " << cathode_CBD_current_mol/cathode_CBD_volume << " mol.m-3 (apparent) | " << cathode_CBD_current_true_mol/cathode_CBDe_true_volume << " mol.m-3 (true)" << std::endl;
    amrex::Print() << "    - total electrolyte (ce)  : " << cathode_CBDelectrolyte_current_mol << " mol (apparent) | " << cathode_CBDelectrolyte_current_true_mol << " mol (true) | " << cathode_CBDelectrolyte_current_mol/(cathode_electrolyte_volume+cathode_CBD_volume) << " mol.m-3 (apparent) | " << cathode_CBDelectrolyte_current_true_mol/cathode_CBDelectrolyte_true_volume << " mol.m-3 (true)" << std::endl;
    }
    if (case_CBD==3) {    
    amrex::Print() << "    Ce (total): " << total_CBDelectrolyte_current_mol << " mol (apparent) | " << total_CBDelectrolyte_current_true_mol << " mol (true) | " << total_CBDelectrolyte_current_mol/(total_electrolyte_volume+totalCBD_volume) << " mol.m-3 (apparent) | " << total_CBDelectrolyte_current_true_mol/total_electrolyte_true_volume << " mol.m-3 (true)" << std::endl;
    } else {
    amrex::Print() << "    Ce (total): " << total_CBDelectrolyte_current_mol << " mol (apparent) | " << total_CBDelectrolyte_current_true_mol << " mol (true) | " << total_CBDelectrolyte_current_mol/total_electrolyte_volume << " mol.m-3 (apparent) | " << total_CBDelectrolyte_current_true_mol/total_electrolyte_true_volume << " mol.m-3 (true)" << std::endl;
    }
    amrex::Print() << "    Cs (total): " << total_solid_current_true_mol << " mol (true)" << std::endl;
    amrex::Print() << std::endl;

    amrex::Print() << "MASS ERROR:" << std::endl;
    amrex::Print() << "    Anode      : current = " << anode_AM_current_true_mol << " mol | target = " << anode_AM_target_mol << " mol | error = " << anode_mass_error << " % " << std::endl;
    amrex::Print() << "    Cathode    : current = " << cathode_AM_current_true_mol << " mol | target = " << cathode_AM_target_mol << " mol | error = " << cathode_mass_error << " % " << std::endl;
    amrex::Print() << "    Electrolyte: current = " << total_CBDelectrolyte_current_true_mol << " mol | target = " << total_CBDelectrolyte_target_mol << " mol | error = " << electrolyte_mass_error << " % " << std::endl;
    amrex::Print() << std::endl;

    amrex::Print() << "POTENTIAL:       " << std::endl;
    if (case_CBD==1 || case_CBD==2) {
        amrex::Print() << "      Anode | Separator | Cathode" << std::endl;
        amrex::Print() << "phis: " << anode_AM_current_potential << " | " << "n/a" << " | "  << cathode_AM_current_potential << " V" << std::endl;
        amrex::Print() << "phie: " << anode_electrolyte_current_potential << " | " << separator_current_potential << " | "  << cathode_electrolyte_current_potential << " V" << std::endl;
    } else if (case_CBD==3 ){
        amrex::Print() << "      Anode | Anode (CBD) | Separator | Cathode | Cathode (CBD)" << std::endl;
        amrex::Print() << "phis: " << anode_AM_current_potential << " | " << anode_CBDsolid_current_potential << " | " << "n/a" << " | "  << cathode_AM_current_potential << " | "  << cathode_CBDsolid_current_potential << " V" << std::endl;
        amrex::Print() << "phie: " << "n/a" << " | " << anode_CBDelectrolyte_current_potential << " | " << separator_current_potential << " | "  << "n/a" << " | "  << cathode_CBDelectrolyte_current_potential << " V" << std::endl;
    } else if (case_CBD==4 ){
        amrex::Print() << "      Anode | Anode (CBD) | Separator | Cathode | Cathode (CBD)" << std::endl;
        amrex::Print() << "phis: " << anode_AM_current_potential << " | " << anode_CBDsolid_current_potential << " | " << "n/a" << " | "  << cathode_AM_current_potential << " | "  << cathode_CBDsolid_current_potential << " V" << std::endl;
        amrex::Print() << "phie: " << anode_electrolyte_current_potential << " | " << "n/a" << " | " << separator_current_potential << " | "  << cathode_electrolyte_current_potential << " | "  << "n/a" << " V" << std::endl;
    } else {
      amrex::Print() << "      Anode | Anode (CBD) | Separator | Cathode | Cathode (CBD)" << std::endl;
        amrex::Print() << "phis: " << anode_AM_current_potential << " | " << anode_CBDsolid_current_potential << " | " << "n/a" << " | "  << cathode_AM_current_potential << " | "  << cathode_CBDsolid_current_potential << " V" << std::endl;
        amrex::Print() << "phie: " << anode_electrolyte_current_potential << " | " << anode_CBDelectrolyte_current_potential << " | " << separator_current_potential << " | "  << cathode_electrolyte_current_potential << " | "  << cathode_CBDelectrolyte_current_potential << " V" << std::endl;
    }

    //amrex::Print() << "     Anode  : active material (phis) | CBD (phis) | electrolyte (phie): " << anode_AM_current_potential << " / " << anode_CBDsolid_current_potential << " / "  << anode_electrolyte_current_potential << " V" << std::endl;
    //amrex::Print() << "     Separator (phie) : " << separator_current_potential << " V" << std::endl;
    //amrex::Print() << "     Cathode: active material (phis) | CBD (phis) | electrolyte (phie): " << cathode_AM_current_potential << " / " << cathode_CBDsolid_current_potential << " / "  << cathode_electrolyte_current_potential << " V" << std::endl;
    amrex::Print() << std::endl;

    // SAVE MASS ERROR IN % IN CSV FILE
    if (amrex::ParallelDescriptor::IOProcessor()) // only IO Processor outputs file
    {
        std::ofstream outfile;
        std::string filename;
        filename = echemAMR::output_folder + "data/Masserror.csv";
            if (timestep == 0)
            {
                outfile.open(filename.c_str(), std::ios_base::out);
                outfile << "Time_step    Time[s]    Anode    Electrolyte    Cathode" << std::endl;
            } 
            outfile.open(filename.c_str(), std::ios_base::out | std::ios_base::app);
            outfile.precision(16);
            outfile << timestep << ' ' << std::scientific << time << ' ' << anode_mass_error << ' ' << electrolyte_mass_error << ' ' << cathode_mass_error << std::endl;
    }    

}
#endif
