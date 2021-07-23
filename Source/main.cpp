
#include <iostream>

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParallelDescriptor.H>

#include <echemAMR.H>
#include <Chemistry.H>

using namespace amrex;

int main(int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    // timer for profiling
    BL_PROFILE_VAR("main()", pmain);

    // wallclock time
    const Real strt_total = amrex::second();

    electrochem::init();

    {
        // constructor - reads in parameters from inputs file
        //             - sizes multilevel arrays and data structures
        echemAMR echem_obj;

        // initialize AMR data
        echem_obj.InitData();

        // advance solution to final time
        echem_obj.Evolve();

        // wallclock time
        Real end_total = amrex::second() - strt_total;

        // print wallclock time
        ParallelDescriptor::ReduceRealMax(
            end_total, ParallelDescriptor::IOProcessorNumber());
        if (echem_obj.Verbose()) {
            amrex::Print() << "\nTotal Time: " << end_total << '\n';
        }
    }

    // destroy timer for profiling
    BL_PROFILE_VAR_STOP(pmain);

    electrochem::close();
    amrex::Finalize();
}
