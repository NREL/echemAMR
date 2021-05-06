#ifndef _INTEGRALUTILS_H_
#define _INTEGRALUTILS_H_
#include <AMReX_MultiFabUtil.H>
#include <echemAMR.H>

using namespace amrex;
namespace electrochem_integral_utils
{

//Returns the state value of <comp> if within <domain>
//TODO: incorporate a level mask
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
//TODO: incorporate a level mask
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
Real surface_value(int i, int j, int k, int comp, int domain1, int domain2,
                    Array4<Real const> const& phi,
                    amrex::Geometry geom
                    )
{
    // Get the mesh size
    const auto dx = geom.CellSizeArray();

    // Get the boundary ids
    const int* domlo = geom.Domain().loVect();
    const int* domhi = geom.Domain().hiVect();

    // Init some values
    Real val = 1.0;
    Real out = 0.0;
    int same_domain = 1;

    // Check if we are trying to find the total surface area of domain1
    if (domain1 == domain2)
        same_domain = -1;

    // Check if we want just the area
    if( comp > -1 )
        val = phi(i,j,k,comp);

    // Get the values of the six adjacent cells
    Real x_prev = phi(i-1,j,k,domain2);
    Real x_next = phi(i+1,j,k,domain2);
    Real y_prev = phi(i,j-1,k,domain2);
    Real y_next = phi(i,j+1,k,domain2);
    Real z_prev = phi(i,j,k-1,domain2);
    Real z_next = phi(i,j,k+1,domain2);

    // Check if any of the adjacent cells is OOB
    if (i-1 < domlo[0]) x_prev = -1;
    if (i+1 > domhi[0]) x_next = -1;
    if (j-1 < domlo[1]) y_prev = -1;
    if (j+1 > domhi[1]) y_next = -1;
    if (k-1 < domlo[2]) z_prev = -1;
    if (k+1 > domhi[2]) z_next = -1;

    // Check if we are in domain1 and the adjacent cell is domain2 or boundary
    if( (phi(i,j,k,domain1) > 0) )
    {
        if ( same_domain*x_prev > 0) out += val*dx[1]*dx[2];
        if ( same_domain*x_next > 0) out += val*dx[1]*dx[2];
        if ( same_domain*y_prev > 0) out += val*dx[0]*dx[2];
        if ( same_domain*y_next > 0) out += val*dx[0]*dx[2];
        if ( same_domain*z_prev > 0) out += val*dx[0]*dx[1];
        if ( same_domain*z_next > 0) out += val*dx[0]*dx[1];
    }









    // if( (phi(i,j,k,domain1) > 0
    // {
    //     if (phi(i+1,j,k,domain2) < 0 || phi(i+1,j,k,domain2) < 0 || phi(i+1,j,k,domain2) < 0)
    //         out += val*dx[1]*dx[2];

    // }
    //     out += val*dx[1]*dx[2];
    // if( (phi(i,j,k,domain1) > 0 && same_domain*phi(i-1,j,k,domain2) < 0) )
    //     out += val*dx[1]*dx[2];

    // if( (phi(i,j,k,domain1) > 0 && same_domain*phi(i,j+1,k,domain2) < 0) )
    //     out += val*dx[0]*dx[2];
    // if( (phi(i,j,k,domain1) > 0 && same_domain*phi(i,j-1,k,domain2) < 0) )
    //     out += val*dx[0]*dx[2];

    // if( (phi(i,j,k,domain1) > 0 && same_domain*phi(i,j,k+1,domain2) < 0) )
    //     out += val*dx[0]*dx[1];
    // if( (phi(i,j,k,domain1) > 0 && same_domain*phi(i,j,k-1,domain2) < 0) )
    //     out += val*dx[0]*dx[1];








    // if( (phi(i,j,k,domain1) > 0 && same_domain*phi(i+1,j,k,domain2) > 0) )
    //     out += val*dx[1]*dx[2];
    // if( (phi(i,j,k,domain1) > 0 && same_domain*phi(i-1,j,k,domain2) > 0) )
    //     out += val*dx[1]*dx[2];

    // if( (phi(i,j,k,domain1) > 0 && same_domain*phi(i,j+1,k,domain2) > 0) )
    //     out += val*dx[0]*dx[2];
    // if( (phi(i,j,k,domain1) > 0 && same_domain*phi(i,j-1,k,domain2) > 0) )
    //     out += val*dx[0]*dx[2];

    // if( (phi(i,j,k,domain1) > 0 && same_domain*phi(i,j,k+1,domain2) > 0) )
    //     out += val*dx[0]*dx[1];
    // if( (phi(i,j,k,domain1) > 0 && same_domain*phi(i,j,k-1,domain2) > 0) )
    //     out += val*dx[0]*dx[1];






    // if( (phi(i,j,k,domain1) > 0 && same_domain*phi(i+1,j,k,domain2) > 0) )
    //     out += val*dx[1]*dx[2];
    // if( (phi(i,j,k,domain2) > 0 && same_domain*phi(i+1,j,k,domain1)) > 0 )
    //     out += val*dx[1]*dx[2];

    // if( (phi(i,j,k,domain1) > 0 && same_domain*phi(i,j+1,k,domain2) > 0) )
    //     out += val*dx[0]*dx[2];
    // if( (phi(i,j,k,domain2) > 0 && same_domain*phi(i,j+1,k,domain1)) > 0 )
    //     out += val*dx[0]*dx[2];

    // if( (phi(i,j,k,domain1) > 0 && same_domain*phi(i,j,k+1,domain2) > 0) )
    //     out += val*dx[0]*dx[1];
    // if( (phi(i,j,k,domain2) > 0 && same_domain*phi(i,j,k+1,domain1)) > 0 )
    //     out += val*dx[0]*dx[1];

    // if( (phi(i,j,k,domain1) > 0 && same_domain*phi(i+1,j,k,domain2) > 0)) || (phi(i,j,k,domain2) > 0 && same_domain*phi(i+1,j,k,domain1)) > 0 )
    //     out += val*dx[1]*dx[2];

    // if( (phi(i,j,k,domain1) > 0 && same_domain*phi(i,j+1,k,domain2) > 0) || (phi(i,j,k,domain2) > 0 && same_domain*phi(i,j+1,k,domain1)) > 0 )
    //     out += val*dx[0]*dx[2];

    // if( (phi(i,j,k,domain1) > 0 && same_domain*phi(i,j,k+1,domain2) > 0) || (phi(i,j,k,domain2) > 0 && same_domain*phi(i,j,k+1,domain1)) > 0 )
    //     out += val*dx[0]*dx[1];

    return out;

}




}
#endif


void echemAMR::init_volumes(ProbParm & prob_parm)
{
    // Real A_int = VolumeIntegral(-1, A_ID);
    // amrex::Print() << "Direct Anode Volume:       " << A_int << std::endl;
    prob_parm.anode_volume =   VolumeIntegral(-1, A_ID);
    prob_parm.ele_sep_volume = VolumeIntegral(-1, E_ID) + VolumeIntegral(-1, S_ID);
    prob_parm.cathode_volume = VolumeIntegral(-1, C_ID);

    // Surface Areas
    prob_parm.anode_SA =       SurfaceIntegral(-1, A_ID, A_ID);
    prob_parm.electrolyte_SA = SurfaceIntegral(-1, E_ID, E_ID);
    prob_parm.sepatator_SA =   SurfaceIntegral(-1, S_ID, S_ID);
    prob_parm.cathode_SA =     SurfaceIntegral(-1, C_ID, C_ID);
    prob_parm.anode_ele_SA =   SurfaceIntegral(-1, A_ID, E_ID);
    prob_parm.cathode_ele_SA = SurfaceIntegral(-1, C_ID, E_ID);
    prob_parm.anode_cc_SA =    prob_parm.anode_SA - prob_parm.anode_ele_SA;
    prob_parm.cathode_cc_SA =  prob_parm.cathode_SA - prob_parm.cathode_ele_SA;
}
