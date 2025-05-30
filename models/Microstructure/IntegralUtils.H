#ifndef _INTEGRALUTILS_H_
#define _INTEGRALUTILS_H_
#include <AMReX_MultiFabUtil.H>
#include <echemAMR.H>

using namespace amrex;
namespace electrochem_integral_utils
{

    //Returns the state value of <comp> if within <domain>
    //TODO: incorporate a level mask
    AMREX_GPU_HOST_DEVICE
        AMREX_FORCE_INLINE
        Real volume_value(int i, int j, int k, int comp1, int comp2, int domain,
                Array4<Real const> const& phi,
                Array4<int const> const& mask_arr,
                GpuArray<Real, AMREX_SPACEDIM> dx
                )
        {

            Real  val= 1.0;

            if( comp1 > -1 )
                val = phi(i,j,k,comp1);
                if ( comp2 > -1 )
                    val = val * phi(i,j,k,comp2);

            if( phi(i,j,k,domain) > 0 )
                return val*dx[0]*dx[1]*dx[2]*phi(i,j,k,domain);
            else
                return 0.0;

        }

    //Returns the state value of <comp> if on the interface of <domain1> and <domain1>
    //TODO: incorporate a level mask
    AMREX_GPU_HOST_DEVICE
        AMREX_FORCE_INLINE
        Real surface_value(int i, int j, int k, int comp, int domain1, int domain2,
                Array4<Real const> const& phi,
                Array4<int const> const& mask_arr,
                const GpuArray<int, 3> domlo,const GpuArray<int, 3> domhi,
                const GpuArray<Real, AMREX_SPACEDIM>& dx)
        {

            // Init some values
            Real val = 1.0;
            Real out = 0.0;
            int same_domain = 1;

            // Check if we are trying to find the total surface 
            // area of domain1 (a -1 means we are looking for 
            // the neighboring cell to be different)
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

            // Check if any of the adjacent cells is 
            // OOB (a -1 means the neighboring cell is different)
            if (i-1 < domlo[0]) x_prev = -1;
            if (i+1 > domhi[0]) x_next = -1;
            if (j-1 < domlo[1]) y_prev = -1;
            if (j+1 > domhi[1]) y_next = -1;
            if (k-1 < domlo[2]) z_prev = -1;
            if (k+1 > domhi[2]) z_next = -1;

            // Check if we are in domain1 and the 
            // adjacent cell is domain2 or boundary
            if( (phi(i,j,k,domain1) > 0) )
            {
                if ( same_domain*x_prev > 0) out += val*dx[1]*dx[2]*phi(i,j,k,domain1);
                if ( same_domain*x_next > 0) out += val*dx[1]*dx[2]*phi(i,j,k,domain1);
                if ( same_domain*y_prev > 0) out += val*dx[0]*dx[2]*phi(i,j,k,domain1);
                if ( same_domain*y_next > 0) out += val*dx[0]*dx[2]*phi(i,j,k,domain1);
                if ( same_domain*z_prev > 0) out += val*dx[0]*dx[1]*phi(i,j,k,domain1);
                if ( same_domain*z_next > 0) out += val*dx[0]*dx[1]*phi(i,j,k,domain1);
            }

            return out;
        }

    //Returns the state value of <comp> if on the current 
    //collectors located at +/- x
    //TODO: incorporate a level mask
    AMREX_GPU_HOST_DEVICE
        AMREX_FORCE_INLINE
        Real current_collector_value(int i, int j, int k, int comp, int domain,
                Array4<Real const> const& phi,
                Array4<int const> const& mask_arr,
                const GpuArray<int, 3> domlo,const GpuArray<int, 3> domhi,
                const GpuArray<Real, AMREX_SPACEDIM>& dx)
        {
            // Init some values
            Real val = 1.0;
            Real out = 0.0;

            // Check if we want just the area
            if( comp > -1 )
                val = phi(i,j,k,comp);

            // Check if we are in the right domain and at one of the x extremes
            if( (phi(i,j,k,domain) > 0) )
            {
                if (i-1 < domlo[0]) out += val*dx[1]*dx[2]*phi(i,j,k,domain);
                if (i+1 > domhi[0]) out += val*dx[1]*dx[2]*phi(i,j,k,domain);
            }

            return out;
        }
}

void echemAMR::init_volumes()
{

    // Volumes
    host_global_storage->anode_AM_volume =   VolumeIntegral(-1, -1, A_AM_ID);
    host_global_storage->cathode_AM_volume = VolumeIntegral(-1, -1, C_AM_ID);

    // Surface Areas
    // Current collector - solid e- conductor (Applied loading)
    host_global_storage->anode_ccAM_SA     = CurrentCollectorIntegral(-1, A_AM_ID);
    host_global_storage->anode_ccCBDs_SA   = CurrentCollectorIntegral(-1, A_CBD_ID);
    host_global_storage->cathode_ccAM_SA   = CurrentCollectorIntegral(-1, C_AM_ID);
    host_global_storage->cathode_ccCBDs_SA = CurrentCollectorIntegral(-1, C_CBD_ID);
    // Electrolyte - active material (Intercalation reaction)
    host_global_storage->anode_eleAM_SA    = SurfaceIntegral(-1, A_E_ID, A_AM_ID);
    host_global_storage->anode_seppAM_SA  = SurfaceIntegral(-1, S_ID, A_AM_ID);
    host_global_storage->anode_CBDpAM_SA = SurfaceIntegral(-1, A_CBD_ID, A_AM_ID); 
    host_global_storage->cathode_eleAM_SA  = SurfaceIntegral(-1, C_E_ID, C_AM_ID);
    host_global_storage->cathode_seppAM_SA   = SurfaceIntegral(-1, S_ID, C_AM_ID);
    host_global_storage->cathode_CBDpAM_SA = SurfaceIntegral(-1, C_CBD_ID, C_AM_ID);

}

#endif
