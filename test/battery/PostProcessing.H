#ifndef _POSTPROCESSING_H_
#define _POSTPROCESSING_H_

#include <echemAMR.H>
#include <Chemistry.H>

using namespace amrex;

void echemAMR::post_timestep(Real time, int timestep)
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



}
#endif