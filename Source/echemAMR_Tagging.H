#ifndef TAGGING_H
#define TAGGING_H

#include <AMReX_Array4.H>

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
state_error (int i, int j, int k,
             amrex::Array4<char> const& tag,
             amrex::Array4<amrex::Real const> const& state,
             amrex::Real *refine_phi, amrex::Real *refine_phi_comps,
             int ntagvars,
             char tagval)
{

    for(int c=0;c<ntagvars;c++)
    {
        int comp = refine_phi_comps[c];
        if (state(i,j,k,comp) > refine_phi[comp])
        {
            tag(i,j,k) = tagval;
        }
    }
}

#endif
