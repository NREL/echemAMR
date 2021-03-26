#ifndef _UTILITIES_H_
#define _UTILITIES_H_
#include <AMReX_MultiFabUtil.H>

using namespace amrex;
namespace electrochem_utilities
{

//Returns the state value of <comp> if within <domain>
//TODO: incorperate a level mask
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
Real volume_value(int i, int j, int k, int comp, int domain,
                    Array4<Real const> const& phi,
                    GpuArray<Real, AMREX_SPACEDIM> dx
                    )
{

    Real val = 1.0;

    if( comp > -1 )
        val = phi(i,j,k,comp);

    if( phi(i,j,k,domain) > 0 )
        return val*dx[0]*dx[1]*dx[2];
    else
        return 0.0;

}

//Returns the state value of <comp> if one the interface of <domain1> and <domain1>
//TODO: incorperate a level mask
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
Real surface_value(int i, int j, int k, int comp, int domain1, int domain2,
                    Array4<Real const> const& phi,
                    GpuArray<Real, AMREX_SPACEDIM> dx
                    )
{

    Real val = 1.0;
    Real out = 0.0;

    if( comp > -1 )
        val = phi(i,j,k,comp);

    if( (phi(i,j,k,domain1) > 0 && phi(i+1,j,k,domain2) > 0) || (phi(i,j,k,domain2) > 0 && phi(i+1,j,k,domain1)) > 0 )
        out += val*dx[1]*dx[2];

    if( (phi(i,j,k,domain1) > 0 && phi(i,j+1,k,domain2) > 0) || (phi(i,j,k,domain2) > 0 && phi(i,j+1,k,domain1)) > 0 )
        out += val*dx[0]*dx[2];

    if( (phi(i,j,k,domain1) > 0 && phi(i,j,k+1,domain2) > 0) || (phi(i,j,k,domain2) > 0 && phi(i,j,k+1,domain1)) > 0 )
        out += val*dx[0]*dx[1];

    return out;

}




}
#endif


// Questions: 
// 0: how to calculate surface area? checking i and i+1 only works if I have both available 
// 1: if comp=-1 return 1.0 for volume and surface area?
// 2: if domain2=0 return value if on the boundary (one of the current collectors)?
// 3: passing a pointer to the entire echemAMR object is probably overkill. is there a better way?