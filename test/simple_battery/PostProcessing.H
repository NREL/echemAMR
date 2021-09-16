#ifndef _POSTPROCESSING_H_
#define _POSTPROCESSING_H_

#include <echemAMR.H>
#include <Chemistry.H>

using namespace amrex;


void echemAMR::postprocess(Real time, int timestep, Real dt, GlobalStorage *globalstorage)
//void echemAMR::postprocess(Real time, int timestep, Real dt, Real Applied_current, Real Fday, Real charging, Vector<amrex::Real> Mass_error)
{

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
    Real anode_target_dmol = globalstorage->charging*globalstorage->Applied_current*dt/globalstorage->Faraday_const; // [mol]
    Real cathode_target_dmol = globalstorage->charging*globalstorage->Applied_current*dt/globalstorage->Faraday_const; // [mol]
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
    amrex::Print() << "    Anode      : current = " << anode_current_mol << " mol , target = " << anode_target_mol << " mol , error = " << 100*(anode_target_mol-anode_current_mol)/anode_target_mol << " % " << std::endl;
    amrex::Print() << "    Cathode    : current = " << cathode_current_mol << " mol , target = " << cathode_target_mol << " mol , error = " << 100*(cathode_target_mol-cathode_current_mol)/cathode_target_mol << " % " << std::endl;
    amrex::Print() << "    Electrolyte: current = " << electrolyte_current_mol << " mol , target = " << electrolyte_target_mol << " mol , error = " << 100*(electrolyte_target_mol-electrolyte_current_mol)/electrolyte_target_mol << " % " << std::endl;
    amrex::Print() << std::endl;

    // Real test1 = SurfaceIntegral(-1, A_ID, E_ID);
    // Real test2 = SurfaceIntegral(-1, C_ID, E_ID);
    // amrex::Print() << "Anode-Electrolyte SA:   " << test1 << std::endl;
    // amrex::Print() << "Cathode-Electrolyte SA: " << test2 << std::endl;
    // amrex::Print() << std::endl;

/*
    if (line_plot_int > 0 && ((timestep+1) % line_plot_int == 0 || timestep == 0))
    {
        std::string co = "concentration";
        std::string po = "potential";
        
        if(line_plot_dir == 0){
            line_plot(XDir(), line_plot_npoints, time, timestep, finest_level, geom, dmap, grids, phi_new, CO_ID, co);
            line_plot(XDir(), line_plot_npoints, time, timestep, finest_level, geom, dmap, grids, phi_new, POT_ID, po);
        }
        else if(line_plot_dir == 1){
            line_plot(YDir(), line_plot_npoints, time, timestep, finest_level, geom, dmap, grids, phi_new, CO_ID, co);
            line_plot(YDir(), line_plot_npoints, time, timestep, finest_level, geom, dmap, grids, phi_new, POT_ID, po);
        } else {
            line_plot(ZDir(), line_plot_npoints, time, timestep, finest_level, geom, dmap, grids, phi_new, CO_ID, co);
            line_plot(ZDir(), line_plot_npoints, time, timestep, finest_level, geom, dmap, grids, phi_new, POT_ID, po);
        }

    }
*/

}
#endif
