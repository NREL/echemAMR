#ifndef MY_TEST_H_
#define MY_TEST_H_

#include <AMReX_MLMG.H>

class MyTest
{
public:

    MyTest ();

    void solve ();
    void writePlotfile ();
    void initData ();

private:

    void readParameters ();
    void initGrids ();

    int n_cell = 32;
    int n_pad = 0;
    int m_pad = 0;
    int max_grid_size = 64;

    std::string plot_file_name{"output/the_reality/plot"};

    // For MLMG solver
    int verbose = 2;
    int bottom_verbose = 2;
    int max_coarsening_level = 30;

    amrex::Geometry geom;
    amrex::BoxArray grids;
    amrex::DistributionMapping dmap;

    amrex::MultiFab solution;
    amrex::MultiFab exact;
    amrex::MultiFab rhs;
    amrex::MultiFab eta;
    amrex::MultiFab kappa;
    amrex::MultiFab lamG_deltaT;
    amrex::MultiFab lamG_deltaT_gradient;

};

#endif
