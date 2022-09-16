#ifndef MY_TEST_K_H_
#define MY_TEST_K_H_

#include <AMReX_FArrayBox.H>

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void init (amrex::Real x, amrex::Real y, amrex::Real z, amrex::Real R2,
           amrex::Real& u, amrex::Real& v, amrex::Real& w,
           amrex::Real& urhs, amrex::Real& vrhs, amrex::Real& wrhs,
           amrex::Real& lamG_betadT, amrex::Real& eta, amrex::Real& kappa)
{

    // ic and bc's
    u = 0.0;
    v = 0.0;

//     if(y > 0.0) {
// //        v = 5.0e-9;
//     }

    w = 0.0;

    // rhs not used
    urhs = 0.0;
    vrhs = 0.0;
    wrhs = 0.0;

    amrex::Real beta = 10.0e-6;
    amrex::Real E = 10.0e9;

    if(x > 4e-6) {
        beta = 20.0e-6;
        E = 20.0e9;
    }

    const amrex::Real L = 1.0e-5;

    if(std::fabs(x) > 2*L || std::fabs(z) > 2*L || y > 2*L || y < -L) {
        E = 20.0e3;
    }

    const amrex::Real deltaT = 10.0;
    const amrex::Real nu = 0.3;
    const amrex::Real lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
    const amrex::Real G = 0.5 * E / (1.0 + nu);

    eta = G;
    kappa = lambda + 2.0/3.0*eta;

    lamG_betadT = -(3.0*lambda + 2.0*G)*beta*deltaT;

}

#endif
