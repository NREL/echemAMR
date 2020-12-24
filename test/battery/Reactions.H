#ifndef _REACTIONS_H_
#define _REACTIONS_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace electrochem_reactions
{
    AMREX_GPU_DEVICE AMREX_INLINE
        void compute_react_source(int i, int j, int k,
                Array4<Real> const& phi,
                Array4<Real> const& reactsource,
                GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                GpuArray<Real, AMREX_SPACEDIM> dx,
                const Real time)
        {

            // FIXME: add Neummann bc's to potential and concentration source terms
            if(electrochem::is_electrode(i,j,k,phi)){

                reactsource(i,j,k,electrochem::PO_ID) = 0.0;
                reactsource(i,j,k,electrochem::CO_ID) = 0.0;

            } else {

                constexpr Real eps = 1.0e-12;
                // concentration
                const Real con = phi(i,j,k,electrochem::CO_ID);

                // FIXME: put the actual equation for KeC here
                const Real KeC = amrex::max(-0.00000025*con*(con-4000.0), eps);

                // FIXME: put the actual equation for tplus, KD here
                const Real tplus = amrex::max(0.4 + 0.1/4000.0*con, eps);
                const Real AC = amrex::max(1.0 + 25.0/4000.0*con, eps);
                const Real RTF = electrochem::R_gas_const*electrochem::T0/electrochem::Faraday_const;
                const Real KD = amrex::min((1.0-tplus)*(1.0+AC)*2.0*KeC*RTF, -eps);

                // FIXME: should probably average KeC and tplus over faces but for now use value at center
                // FIXME: check sign on this
                reactsource(i,j,k,electrochem::CO_ID) = tplus / electrochem::Faraday_const * KeC
                                                        * ( (phi(i-1,j,k,electrochem::PO_ID) - 2.0*phi(i,j,k,electrochem::PO_ID) + phi(i+1,j,k,electrochem::PO_ID))/dx[0]/dx[0]
                                                           +(phi(i,j-1,k,electrochem::PO_ID) - 2.0*phi(i,j,k,electrochem::PO_ID) + phi(i,j+1,k,electrochem::PO_ID))/dx[1]/dx[1]
                                                           +(phi(i,j,k-1,electrochem::PO_ID) - 2.0*phi(i,j,k,electrochem::PO_ID) + phi(i,j,k+1,electrochem::PO_ID))/dx[2]/dx[2]);

                // FIXME: should probably average KD over faces but for now use use value at center
                // FIXME: check sign on this
                // FIXME: this should be KDstar
                reactsource(i,j,k,electrochem::PO_ID) = KD * ( (phi(i-1,j,k,electrochem::CO_ID) - 2.0*phi(i,j,k,electrochem::CO_ID) + phi(i+1,j,k,electrochem::CO_ID))/dx[0]/dx[0]
                                                              +(phi(i,j-1,k,electrochem::CO_ID) - 2.0*phi(i,j,k,electrochem::CO_ID) + phi(i,j+1,k,electrochem::CO_ID))/dx[1]/dx[1]
                                                              +(phi(i,j,k-1,electrochem::CO_ID) - 2.0*phi(i,j,k,electrochem::CO_ID) + phi(i,j,k+1,electrochem::CO_ID))/dx[2]/dx[2]);

            }

            // level sets have no source terms
            reactsource(i,j,k,electrochem::AC_ID) = 0.0;
            reactsource(i,j,k,electrochem::ES_ID) = 0.0;
        }
}
#endif
