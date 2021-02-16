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
        pp.query("source1", echemAMR::h_prob_parm->r1 );
        pp.query("source2", echemAMR::h_prob_parm->r2 );

#ifdef AMREX_USE_GPU
        amrex::Gpu::htod_memcpy(echemAMR::d_prob_parm, echemAMR::h_prob_parm, sizeof(ProbParm));
#else
        std::memcpy(echemAMR::d_prob_parm, echemAMR::h_prob_parm, sizeof(ProbParm));
#endif

    }
}
