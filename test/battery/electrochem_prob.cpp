//#include <AMReX_PROB_AMR_F.H> FIXME: cant seem to locate this file
#include <AMReX_ParmParse.H>
#include "echemAMR.H"
#include "ChemistryProbParm.H"

extern "C" {
    void amrex_probinit (const int* /*init*/,
                         const int* /*name*/,
                         const int* /*namelen*/,
                         const amrex_real* /*problo*/,
                         const amrex_real* /*probhi*/)
    {
        // read problem specific parmparse parameters here

        amrex::ParmParse pp("prob");
        pp.query("anode_conductivity", echemAMR::h_prob_parm->K_a );
        pp.query("cathode_conductivity", echemAMR::h_prob_parm->K_c );
        pp.query("anode_diffusivity", echemAMR::h_prob_parm->D_a );
        pp.query("cathode_diffusivity", echemAMR::h_prob_parm->D_c );
        pp.query("faraday_constant", echemAMR::h_prob_parm->Faraday_const );
        pp.query("R_gas_constant", echemAMR::h_prob_parm-> R_gas_const );


#ifdef AMREX_USE_GPU
        amrex::Gpu::htod_memcpy(echemAMR::d_prob_parm, echemAMR::h_prob_parm, sizeof(ProbParm));
#else
        std::memcpy(echemAMR::d_prob_parm, echemAMR::h_prob_parm, sizeof(ProbParm));
#endif

    }
}
