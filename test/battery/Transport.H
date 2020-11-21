#ifndef _TRANSPORT_H_
#define _TRANSPORT_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace electrochem_transport
{
    AMREX_GPU_DEVICE AMREX_INLINE
        void compute_dcoeff(int i, int j, int k,
                Array4<Real> const& phi,
                Array4<Real> const& dcoeff,
                GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                GpuArray<Real, AMREX_SPACEDIM> dx,
                const Real time)
        {
            if(phi(i,j,k,electrochem::AC_ID) > 0.0) {
                /* electrodes */
                if(phi(i,j,k,electrochem::ES_ID) > 0.0) {
                    // cathode
                    dcoeff(i,j,k,electrochem::PO_ID) = electrochem::K_c;
                    dcoeff(i,j,k,electrochem::CO_ID) = electrochem::D_c;
                } else {
                    // anode
                    dcoeff(i,j,k,electrochem::PO_ID) = electrochem::K_a;
                    dcoeff(i,j,k,electrochem::CO_ID) = electrochem::D_a;
                }
            } else {

                /* electrolyte */

                constexpr Real eps = 1.0e-12;
                // concentration
                const Real con = phi(i,j,k,electrochem::CO_ID);

                // FIXME: put the actual equation for KeC here
                const Real KeC = amrex::max(-0.00000025*con*(con-4000.0), eps);
                // FIXME: put the actual equation for DeC, tplus, KD here
                const Real DeC = amrex::max(3.5e-10 - 3.5e-10/4000.0*con, eps);
                const Real tplus = amrex::max(0.4 + 0.1/4000.0*con, eps);
                const Real AC = amrex::max(1.0 + 25.0/4000.0*con, eps);
                const Real RTF = electrochem::R_gas_const*electrochem::T0/electrochem::Faraday_const;
                const Real KD = amrex::min((1.0-tplus)*(1.0+AC)*2.0*KeC*RTF, -eps);

                dcoeff(i,j,k,electrochem::CO_ID) = DeC - tplus*KD/electrochem::Faraday_const;
                dcoeff(i,j,k,electrochem::PO_ID) = 1.0e-6;//KeC;


//              amrex::Print() << i << ' ' << j << ' ' << k << ' ' << KeC << ' ' << DeC  << ' ' << -tplus*KD/electrochem::Faraday_const << std::endl;

            }

            // no diffusion yet
            dcoeff(i,j,k,electrochem::AC_ID)=0.0;
            dcoeff(i,j,k,electrochem::ES_ID)=0.0;
        }
}
#endif


