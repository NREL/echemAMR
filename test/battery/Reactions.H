#ifndef _REACTIONS_H_
#define _REACTIONS_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>
#include <ChemistryProbParm.H>
#include<AMReX_Tuple.H>
#include<tuple>
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
            const Real time,
            ProbParm const& prob_parm)
    {

        for(int n=0;n<reactsource.nComp();++n) {
            reactsource(i,j,k,n) = 0.0;
        }

        // FIXME: add Neummann bc's to potential and concentration source terms
        if(electrochem::is_electrode(i,j,k,phi)){

            reactsource(i,j,k,PO_ID) = 0.0;
            reactsource(i,j,k,CO_ID) = 0.0;

        } else {

            // concentration
            const Real con = phi(i,j,k,CO_ID);

            const Real KeC = electrochem::KeC(con);
            const Real tplus = electrochem::t_plus(con);
            const Real KD = electrochem::KD(con, prob_parm);

            // FIXME: should probably average KeC and tplus over faces but for now use value at center
            // FIXME: check sign on this
            reactsource(i,j,k,CO_ID) = tplus / prob_parm.Faraday_const * KeC
                                                    * ( (phi(i-1,j,k,PO_ID) - 2.0*phi(i,j,k,PO_ID) + phi(i+1,j,k,PO_ID))/dx[0]/dx[0]
                                                       +(phi(i,j-1,k,PO_ID) - 2.0*phi(i,j,k,PO_ID) + phi(i,j+1,k,PO_ID))/dx[1]/dx[1]
                                                       +(phi(i,j,k-1,PO_ID) - 2.0*phi(i,j,k,PO_ID) + phi(i,j,k+1,PO_ID))/dx[2]/dx[2]);

            // FIXME: should probably average KD over faces but for now use use value at center
            // FIXME: check sign on this
            // FIXME: this should be KDstar
            reactsource(i,j,k,PO_ID) = KD * ( (phi(i-1,j,k,CO_ID) - 2.0*phi(i,j,k,CO_ID) + phi(i+1,j,k,CO_ID))/dx[0]/dx[0]
                                             +(phi(i,j-1,k,CO_ID) - 2.0*phi(i,j,k,CO_ID) + phi(i,j+1,k,CO_ID))/dx[1]/dx[1]
                                             +(phi(i,j,k-1,CO_ID) - 2.0*phi(i,j,k,CO_ID) + phi(i,j,k+1,CO_ID))/dx[2]/dx[2]);

        }


        /* add intercalation reaction */
        for(int dir = 0; dir < AMREX_SPACEDIM; ++dir){
            amrex::Real normal = 1.0/dx[dir];
            reactsource(i,j,k,CO_ID) += electrochem::ic_reaction(i,j,k,dir,+normal,phi,prob_parm);
            reactsource(i,j,k,CO_ID) += electrochem::ic_reaction(i,j,k,dir,-normal,phi,prob_parm);
        }


    }
}
#endif
