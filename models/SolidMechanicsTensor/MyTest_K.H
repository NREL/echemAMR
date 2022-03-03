#ifndef MY_TEST_K_H_
#define MY_TEST_K_H_

#include <AMReX_FArrayBox.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void init (amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real R2,
           amrex::Real& u, amrex::Real& v, amrex::Real& w,
           amrex::Real& urhs, amrex::Real& vrhs, amrex::Real& wrhs,
           amrex::Real& con, amrex::Real& eta, amrex::Real& kappa)
{

    u = 0.0;
    v = 0.0;
    w = 0.0;

    // concentration
    con = 0.1*(x-1.0)*(x+1.0)*(y-1.0)*(y+1.0);

    // rhs
    urhs = 0.01;
    vrhs = 0.0;
    wrhs = 0.0;

    amrex::Real E = 138.73;
    if(y > 0.0) {
        E = 1.78;
    }

    const amrex::Real nu = 0.3;
    const amrex::Real lambda = E * nu / (1.0 + nu) / (1.0 - nu);
    const amrex::Real G = 0.5 * E / (1.0 + nu);

    eta = G;
    kappa = lambda + 2.0/3.0*eta;

}

#endif
