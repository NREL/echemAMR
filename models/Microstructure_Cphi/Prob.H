#ifndef _PROB_H_
#define _PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <Chemistry.H>
#include <ProbParm.H>
#include <echemAMR_constants.H>
#include <echemAMR.H>
#include <vector>

using namespace amrex;

AMREX_INLINE
void amrex_probinit (ProbParm & h_prob_parm, ProbParm & d_prob_parm)
{
    // read problem specific parmparse parameters here. Input overwrittes ProbParm default values.
    // See ProbParm.H for definitions
    amrex::ParmParse pp("prob");

    // Geometry
    pp.query("geometry_filename", h_prob_parm.geometry_filename );
    // Nanoporosity
    pp.query("nanoporosity_filename", h_prob_parm.nanoporosity_filename );    
    // Macmullin number
    pp.query("macmullin_filename", h_prob_parm.macmullin_filename );        
    // Loading
    pp.query("Crate", h_prob_parm.Crate);
    pp.query("Temperature", h_prob_parm.Temperature);    
    // Initialization
    pp.query("ce_atrest", h_prob_parm.ce_atrest);
    pp.query("soc_anode_t0", h_prob_parm.soc_anode_t0);
    pp.query("soc_cathode_t0", h_prob_parm.soc_cathode_t0);
    pp.query("phi_reference", h_prob_parm.phi_reference);
    // SOC
    pp.query("cathode_soc_min", h_prob_parm.cathode_soc_min);
    pp.query("cathode_soc_max", h_prob_parm.cathode_soc_max);
    pp.query("anode_soc_min", h_prob_parm.anode_soc_min);
    pp.query("anode_soc_max", h_prob_parm.anode_soc_max);  
    // Materials
    pp.query("mater_cathode", h_prob_parm.mater_cathode);
    pp.query("Sp_corr_cathode", h_prob_parm.Sp_corr_cathode);
    pp.query("mater_anode", h_prob_parm.mater_anode);
    pp.query("Sp_corr_anode", h_prob_parm.Sp_corr_anode);
    pp.query("mater_electrolyte", h_prob_parm.mater_electrolyte);
    pp.query("mater_electrolyte_tpAc", h_prob_parm.mater_electrolyte_tpAc);
    pp.query("mater_CBD", h_prob_parm.mater_CBD);    
    // Nanoscale coefficients modificators
    pp.query("Nanoscale_coefficients_txtimport_CBD", h_prob_parm.Nanoscale_coefficients_txtimport_CBD);  
    pp.query("Nanoscale_coefficients_txtimport_separator", h_prob_parm.Nanoscale_coefficients_txtimport_separator);  
    // CBD transport behavior
    pp.query("CBD_transport", h_prob_parm.CBD_transport);  
    // Replace electrolyte with CBD
    pp.query("ELEisCBD", h_prob_parm.ELEisCBD);      

    // Full cell or half-cell
    pp.query("halfcell_case", h_prob_parm.halfcell_case);
   
    // Physics
    pp.query("Cphi_electrolyte_coupling", h_prob_parm.Cphi_electrolyte_coupling);   
    
    // 0D validation
    pp.query("zeroDmodel", h_prob_parm.zeroDmodel);


    // CREATE 3D ARRAY GEOMETRY READ FILE
    std::string inFileName = h_prob_parm.geometry_filename;
    Vector<char> fileCharPtr;
    ParallelDescriptor::ReadAndBcastFile(inFileName, fileCharPtr);
    std::string fileCharPtrString(fileCharPtr.dataPtr());
    std::istringstream inFile(fileCharPtrString, std::istringstream::in);

    // 1: Get dimension: first 3 integers
    for (int i = 0; i < 3; i++)
    {
        inFile >> h_prob_parm.dimension_geometry[i];
    }
    int geo_N = h_prob_parm.dimension_geometry[0]*h_prob_parm.dimension_geometry[1]*h_prob_parm.dimension_geometry[2];
    // std::vector<int> geo1D(geo_N);
    // h_prob_parm.geo1D.clear();
    // h_prob_parm.geo1D.resize(geo_N);


    // 2: Get domain id
    for (int i = 0; i < geo_N ; i++)
    {
        inFile >> h_prob_parm.geo1D[i];
    }

    // CREATE 3D ARRAY NANOPOROSITY READ FILE
    std::string inFileName_np = h_prob_parm.nanoporosity_filename;
    Vector<char> fileCharPtr_np;
    ParallelDescriptor::ReadAndBcastFile(inFileName_np, fileCharPtr_np);
    std::string fileCharPtrString_np(fileCharPtr_np.dataPtr());
    std::istringstream inFile_np(fileCharPtrString_np, std::istringstream::in);

    // 3: Get nanoporosity
    for (int i = 0; i < geo_N ; i++)
    {
        inFile_np >> h_prob_parm.Nanoporosity1D[i];
    }    

    // CREATE 3D ARRAY MACMULLIN READ FILE
    std::string inFileName_mm = h_prob_parm.macmullin_filename;
    Vector<char> fileCharPtr_mm;
    ParallelDescriptor::ReadAndBcastFile(inFileName_mm, fileCharPtr_mm);
    std::string fileCharPtrString_mm(fileCharPtr_mm.dataPtr());
    std::istringstream inFile_mm(fileCharPtrString_mm, std::istringstream::in);

    // 3: Get MacMullin number
    for (int i = 0; i < geo_N ; i++)
    {
        inFile_mm >> h_prob_parm.MacMullin1D[i];
    }    


#ifdef AMREX_USE_GPU
    amrex::Gpu::htod_memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#else
    std::memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#endif

}

AMREX_GPU_DEVICE
AMREX_INLINE
void initdomaindata(Box const& bx, Array4<Real> const& phi, GeometryData const& geomdata,
        ProbParm *prob_parm)
{

    const auto lo = lbound(bx);
    const auto hi = ubound(bx);    

    const Real* AMREX_RESTRICT prob_lo = geomdata.ProbLo();
    const Real* AMREX_RESTRICT prob_hi = geomdata.ProbHi();
    const Real* AMREX_RESTRICT dx      = geomdata.CellSize();

    const Real Lx = prob_hi[0] - prob_lo[0];
    const Real Ly = prob_hi[1] - prob_lo[1];
    const Real Lz = prob_hi[2] - prob_lo[2];

    const int ncomp = phi.nComp();

    /*
    amrex::Print() << "prob_lo[0]=" << prob_lo[0] << " prob_lo[1]=" << prob_lo[1] << " prob_lo[2]=" << prob_lo[2] << std::endl;
    amrex::Print() << "prob_hi[0]=" << prob_hi[0] << " prob_hi[1]=" << prob_hi[1] << " prob_hi[2]=" << prob_hi[2] << std::endl;
    amrex::Print() << "lo.x=" << lo.x << " lo.y=" << lo.y << " lo.z=" << lo.z << std::endl;
    amrex::Print() << "hi.x=" << hi.x << " hi.y=" << hi.y << " hi.z=" << hi.z << std::endl;
    */
        
#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (GPU::notInLaunchRegion)
#endif

    // amrex::Print() << "Dimension [0]=" << dimension_geometry[0] << " [1]=" << dimension_geometry[1] << " [2]=" << dimension_geometry[2] << std::endl;

    // Label from imported txt file
    const int Anode_AM_ID = 1;
    const int Anode_E_ID = 2;
    const int Anode_CBD_ID = 3;
    const int Sep_ID = 4;
    const int Cathode_AM_ID = 5;
    const int Cathode_E_ID = 6;
    const int Cathode_CBD_ID = 7;

    for (int k = lo.z; k <= hi.z; ++k) 
    {
        const Real z = prob_lo[2] + (k+0.5)*dx[2];
        int idz = std::round( (prob_parm->dimension_geometry[2]-1) / Lz * z + 1 ) -1; // Convert z in the index of imported geometry geo3D 
        for (int j = lo.y; j <= hi.y; ++j)
        {
            const Real y = prob_lo[1] + (j+0.5)*dx[1];
            int idy = std::round( (prob_parm->dimension_geometry[1]-1) / Ly * y +1) -1; // Convert y in the index of imported geometry geo3D 
            for (int i = lo.x; i <= hi.x; ++i) 
            {
                const Real x = prob_lo[0] + (i+0.5)*dx[0];
                int idx = std::round( (prob_parm->dimension_geometry[0]-1) / Lx * x +1 ) -1; // Convert x in the index of imported geometry geo3D 
                AMREX_PRAGMA_SIMD
                for (int n = 0; n < ncomp; ++n)
                {
                    phi(i, j, k, n) = 0.0; // Initialize all composant to 0
                }

                int id = (idx+1 + (idy+1-1)*(prob_parm->dimension_geometry[0]) + (idz+1-1)*(prob_parm->dimension_geometry[1])*(prob_parm->dimension_geometry[0]))-1;
                
                // You can not use amrex::Print(), as it will only show the print process
                /*
                std::cout << "i=" << i << " j=" << j << " k=" << k << std::endl;
                std::cout << "x=" << x << " y=" << y << " z=" << z << std::endl;
                std::cout << "idx=" << idx << " idy=" << idy << " idz=" << idz << std::endl;
                std::cout << "id=" << id << " geo[id]=" << geo1D[id] << std::endl;
                */

                if (prob_parm->geo1D[id] == Anode_AM_ID) {
                    phi(i,j,k,A_AM_ID) = 1;
                    //std::cout << "i=" << i << " j=" << j << " k=" << k << " -> ANODE active material" << std::endl;
                } else if (prob_parm->geo1D[id] == Cathode_AM_ID) {
                    phi(i,j,k,C_AM_ID) = 1;
                    //std::cout << "i=" << i << " j=" << j << " k=" << k << " -> CATHODE active material" << std::endl;                    
                } else if (prob_parm->geo1D[id] == Anode_E_ID && prob_parm->ELEisCBD==0) {
                    phi(i,j,k,A_E_ID) = 1;
                    phi(i,j,k,LS_ID) = 1;
                    //std::cout << "i=" << i << " j=" << j << " k=" << k << " -> ANODE electrolyte" << std::endl;
                } else if (prob_parm->geo1D[id] == Cathode_E_ID && prob_parm->ELEisCBD==0) {
                    phi(i,j,k,C_E_ID) = 1;
                    phi(i,j,k,LS_ID) = 1;
                    //std::cout << "i=" << i << " j=" << j << " k=" << k << " -> CATHODE electrolyte" << std::endl;                    
                } else if (prob_parm->geo1D[id] == Sep_ID) {
                    phi(i,j,k,S_ID) = 1;
                    phi(i,j,k,LS_ID) = 1;
                    //std::cout << "i=" << i << " j=" << j << " k=" << k << " -> SEPARATOR" << std::endl;
                } else if (prob_parm->geo1D[id] == Anode_CBD_ID || (prob_parm->geo1D[id] == Anode_E_ID && prob_parm->ELEisCBD==1)) {
                    phi(i,j,k,A_CBD_ID) = 1;
                    if (prob_parm->CBD_transport==1) { // Mixed domain - Intercalation reaction between CBD and active material
                        phi(i,j,k,LS_ID) = 1;
                    } else { // Purely solid domain - no intercalation reaction between CBD and active material
                        phi(i,j,k,LS_ID) = 0; 
                    } 
                    //std::cout << "i=" << i << " j=" << j << " k=" << k << " -> ANODE CBD" << std::endl;
                } else if (prob_parm->geo1D[id] == Cathode_CBD_ID || (prob_parm->geo1D[id] == Cathode_E_ID && prob_parm->ELEisCBD==1)) {
                    phi(i,j,k,C_CBD_ID) = 1;
                    if (prob_parm->CBD_transport==1) { // Mixed domain - Intercalation reaction between CBD and active material
                        phi(i,j,k,LS_ID) = 1;
                    } else { // Purely solid domain - no intercalation reaction between CBD and active material
                        phi(i,j,k,LS_ID) = 0;
                    } 
                    //std::cout << "i=" << i << " j=" << j << " k=" << k << " -> CATHODE CBD" << std::endl;                    
                } 
                // we cannot print on gpu
                // else {
                //     std::cout << "Error for idx= " << idx << " idy= " << idy << " idz= " << idz << " geo1D[id]= " << prob_parm->geo1D[id] << std::endl;
                //     //phi(i,j,k,A_ID) = 1;
                // }

                // Nanoporosity and MacMullin number 
                if (phi(i,j,k,S_ID)==0) { // (electrodes)

                    if (prob_parm->Nanoscale_coefficients_txtimport_CBD==1) {
                        phi(i,j,k,NP_ID) = prob_parm->Nanoporosity1D[id];
                        phi(i,j,k,MAC_ID) = prob_parm->MacMullin1D[id];
                    } else {
                        phi(i,j,k,NP_ID) = 1.0;
                        phi(i,j,k,MAC_ID) = 1.0;
                    }
                    
                } else { // Separator

                    if (prob_parm->Nanoscale_coefficients_txtimport_separator==1) {
                        phi(i,j,k,NP_ID) = prob_parm->Nanoporosity1D[id];
                        phi(i,j,k,MAC_ID) = prob_parm->MacMullin1D[id];
                    } else {
                        phi(i,j,k,NP_ID) = 1.0;
                        phi(i,j,k,MAC_ID) = 1.0;
                    }

                }            


            }
        }
    }
}

AMREX_INLINE
void initialconditions(ProbParm & h_prob_parm, ProbParm & d_prob_parm)
{
        // INITIAL POTENTIAL
    // Calculate active material volumes
    const amrex::Real volume_anode = echemAMR::host_global_storage->anode_AM_volume; // [m3]
    const amrex::Real volume_cathode = echemAMR::host_global_storage->cathode_AM_volume; // [m3]

    const amrex::Real Csmax_anode = electrochem::Cs_max_anode(h_prob_parm); // [mol.m-3]
    const amrex::Real SOCrange_anode = h_prob_parm.anode_soc_max - h_prob_parm.anode_soc_min; // []
    const amrex::Real theoritical_capacity_anode = volume_anode * Csmax_anode * FARADCONST; // [C]
    const amrex::Real pratical_capacity_anode = theoritical_capacity_anode * SOCrange_anode; // [C] 

    const amrex::Real Csmax_cathode = electrochem::Cs_max_cathode(h_prob_parm); // [mol.m-3]
    const amrex::Real SOCrange_cathode = h_prob_parm.cathode_soc_max - h_prob_parm.cathode_soc_min; // []
    const amrex::Real theoritical_capacity_cathode = volume_cathode * Csmax_cathode * FARADCONST; // [C]
    const amrex::Real pratical_capacity_cathode = theoritical_capacity_cathode * SOCrange_cathode; // [C]

    // Deduce cell pratical capacity
    const amrex::Real pratical_capacity_cell = amrex::min(pratical_capacity_anode, pratical_capacity_cathode); // [C]
    const amrex::Real Cell_NP_ratio = pratical_capacity_anode/pratical_capacity_cathode; // Not required, just FYI. Should be always >1 (if not, you can expect eartly degradation)

    // Applied current w/o sign
    const amrex::Real Crate = h_prob_parm.Crate;
    const amrex::Real time_to_charge = (1/Crate)*3600; // [s] Time to charge or discharge the cell
    const amrex::Real Current1C = pratical_capacity_cell/3600; // [A=C.s-1] Current for 1C
    const amrex::Real Applied_current = Current1C * Crate; // [A] (but we need the applied current density in A.m-2)

    // Copy current in echemAMR as it is needed for post-processing
    echemAMR::host_global_storage->Applied_current = Applied_current;

    // SOLID-CURRENT COLLECTOR INTERFACE
    // Surface calculations
    const amrex::Real surfaceanodeside_currentcollector_activematerial = echemAMR::host_global_storage->anode_ccAM_SA; // [m2]
    const amrex::Real surfacecathodeside_currentcollector_activematerial = echemAMR::host_global_storage->cathode_ccAM_SA; // [m2]
    amrex::Real tmp1 = echemAMR::host_global_storage->anode_ccCBDs_SA; // [m2]
    amrex::Real tmp2 = echemAMR::host_global_storage->cathode_ccCBDs_SA; // [m2]
    if (h_prob_parm.CBD_transport==1) { // Modified with (1-nanoporosity) (hard-coded for now)
        tmp1 = tmp1 * (1-0.47); // [m2]
        tmp2 = tmp2 * (1-0.47); // [m2]        
    }
    echemAMR::host_global_storage->CBD_transport=h_prob_parm.CBD_transport; // requried for PostProcessing.H
    echemAMR::host_global_storage->Nanoscale_coefficients_txtimport_CBD=h_prob_parm.Nanoscale_coefficients_txtimport_CBD;
    echemAMR::host_global_storage->Nanoscale_coefficients_txtimport_separator=h_prob_parm.Nanoscale_coefficients_txtimport_separator;

    const amrex::Real surfaceanodeside_currentcollector_CBDsolid = tmp1;
    const amrex::Real surfacecathodeside_currentcollector_CBDsolid = tmp2;
      
    // Applied current density at the interface active material - current collector, w/ sign
    // These are the two Neumann Boundary conditions value applies at the left and right extremities of the cell
    amrex::Real Applied_current_density_anode = -Applied_current / (surfaceanodeside_currentcollector_activematerial + surfaceanodeside_currentcollector_CBDsolid); // [A.m-2]
    amrex::Real Applied_current_density_cathode = Applied_current / (surfacecathodeside_currentcollector_activematerial + surfacecathodeside_currentcollector_CBDsolid); // [A.m-2]
    // Update potential BC
    echemAMR::host_global_storage->pot_bc_lo[0]=-Applied_current_density_cathode/electrochem::Ks_bulk_cathode(1, h_prob_parm);

    // ELECTROCHEMICAL ACTIVE INTERFACE
    // Surface calculations
    const amrex::Real surfaceanodeside_electrolyte_activematerial = echemAMR::host_global_storage->anode_eleAM_SA;
    const amrex::Real surfacecathodeside_electrolyte_activematerial = echemAMR::host_global_storage->cathode_eleAM_SA;
    const amrex::Real surfaceanodeside_separatorp_activematerial = echemAMR::host_global_storage->anode_seppAM_SA;
    const amrex::Real surfacecathodeside_separatorp_activematerial = echemAMR::host_global_storage->cathode_seppAM_SA;    
    tmp1 = 0; // [m2]
    tmp2 = 0; // [m2]
    if (h_prob_parm.CBD_transport==1) { // Modified with (1-nanoporosity) (hard-coded for now)
        tmp1 = echemAMR::host_global_storage->anode_CBDpAM_SA * 0.47; // [m2]
        tmp2 = echemAMR::host_global_storage->cathode_CBDpAM_SA * 0.47; // [m2]        
    }
    const amrex::Real surfaceanodeside_CBDpore_activematerial = tmp1;
    const amrex::Real surfacecathodeside_CBDpore_activematerial = tmp2;

    // Initial current density at the interface active material - electrolyte
    // These are the two Neumann Boundary conditions value applied at the active interface.
    // At time=0s, value is uniform along the interface, then it will vary locally while the surface integral ([A]) will be constant and equal to "Applied_current"
    // Any deviation from the equality: Applied_current = integral(current density, ds active material-current collector) = integral(current density, ds active material-electrolyte)
    // will result in an error in mass conservation
    const amrex::Real Activeinterface_current_density_anode = -Applied_current / (surfaceanodeside_electrolyte_activematerial + surfaceanodeside_separatorp_activematerial + surfaceanodeside_CBDpore_activematerial); // [A.m-2]
    const amrex::Real Activeinterface_current_density_cathode = Applied_current / (surfacecathodeside_electrolyte_activematerial + surfacecathodeside_separatorp_activematerial + surfacecathodeside_CBDpore_activematerial); // [A.m-2]

    // Deduce potential values that verify this condition: Applied_current = integral(current density, ds active material-current collector) = integral(current density, ds active material-electrolyte) 
    const amrex::Real phis_a_to = h_prob_parm.phi_reference;
    const amrex::Real phie_to = electrochem::get_phie_initial_from_anode(Activeinterface_current_density_anode, phis_a_to, h_prob_parm);
    const amrex::Real phis_c_to = electrochem::get_phic_initial_from_electrolyte(Activeinterface_current_density_cathode, phie_to, h_prob_parm);

    // INITIAL CONCENTRATION
    const amrex::Real cs_a_to = h_prob_parm.soc_anode_t0 * electrochem::Cs_max_anode(h_prob_parm);
    const amrex::Real cs_c_to = h_prob_parm.soc_cathode_t0 * electrochem::Cs_max_cathode(h_prob_parm);
    const amrex::Real ce_to = h_prob_parm.ce_atrest;

    h_prob_parm.phis_a_to = phis_a_to;
    h_prob_parm.phie_to   = phie_to;
    h_prob_parm.phis_c_to = phis_c_to;
    h_prob_parm.cs_a_to   = cs_a_to;
    h_prob_parm.cs_c_to   = cs_c_to;
    h_prob_parm.ce_to     = ce_to;


#ifdef AMREX_USE_GPU
    amrex::Gpu::htod_memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#else
    std::memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#endif

}


AMREX_GPU_DEVICE
    AMREX_INLINE
void initproblemdata(Box const& bx, Array4<Real> const& phi, GeometryData const& geomdata,
        ProbParm* prob_parm)
{
    const auto lo = lbound(bx);
    const auto hi = ubound(bx);    

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (GPU::notInLaunchRegion)
#endif



    for (int k = lo.z; k <= hi.z; ++k) 
    {
        for (int j = lo.y; j <= hi.y; ++j)
        {
            for (int i = lo.x; i <= hi.x; ++i) 
            {
                 
                if (phi(i,j,k,C_AM_ID) == 1) { // Within cathode
                    phi(i,j,k,CO_ID) = prob_parm->cs_c_to;
                    phi(i,j,k,POT_ID) = prob_parm->phis_c_to;
                } else if (phi(i,j,k,A_AM_ID) == 1) { // Within anode
                    phi(i,j,k,CO_ID) = prob_parm->cs_a_to;
                    phi(i,j,k,POT_ID) = prob_parm->phis_a_to;
                } else if (phi(i,j,k,A_E_ID) == 1 || phi(i,j,k,C_E_ID) == 1 || phi(i,j,k,S_ID) == 1) { // Within electrolyte and separator
                    phi(i,j,k,CO_ID) = prob_parm->ce_to;
                    phi(i,j,k,POT_ID) = prob_parm->phie_to;
                } else if (phi(i,j,k,A_CBD_ID) == 1) { // Within CBD (anode)
                if (prob_parm->CBD_transport==1) { // Mixed domain
                    phi(i,j,k,CO_ID) = prob_parm->ce_to; // electrolyte concentration
                    phi(i,j,k,POT_ID) = prob_parm->phie_to; // electrolyte potential 
                    phi(i,j,k,POTs_ID) = prob_parm->phis_a_to; // solid potential    
                } else { // Purely solid domain
                    phi(i,j,k,CO_ID) = prob_parm->cs_a_to;
                    phi(i,j,k,POT_ID) = prob_parm->phis_a_to; // solid potential 
                }       
                } else if (phi(i,j,k,C_CBD_ID) == 1) { // Within CBD (cathode)
                if (prob_parm->CBD_transport==1) { // Mixed domain
                    phi(i,j,k,CO_ID) = prob_parm->ce_to;  // electrolyte concentration
                    phi(i,j,k,POT_ID) = prob_parm->phie_to; // electrolyte potential
                    phi(i,j,k,POTs_ID) = prob_parm->phis_c_to; // solid potential    
                } else { // Purely solid domain
                    phi(i,j,k,CO_ID) = prob_parm->cs_c_to;
                    phi(i,j,k,POT_ID) = prob_parm->phis_c_to; // solid potential 
                }                     
                                        
                }
                 
            }
        }
    }

}

    AMREX_INLINE
void print_init_data(ProbParm* prob_parm)
{

    // Calculate active material volumes
    // echemAMR::post_timestep(0, 0); // FIX ME: add volume calculation from Jeff post-processing
    const amrex::Real volume_anode = echemAMR::host_global_storage->anode_AM_volume; // [m3]
    const amrex::Real volume_cathode = echemAMR::host_global_storage->cathode_AM_volume; // [m3]

    const amrex::Real Csmax_anode = electrochem::Cs_max_anode(*prob_parm); // [mol.m-3]
    const amrex::Real SOCrange_anode = prob_parm->anode_soc_max - prob_parm->anode_soc_min; // []
    const amrex::Real theoritical_capacity_anode = volume_anode * Csmax_anode * FARADCONST; // [C]
    const amrex::Real pratical_capacity_anode = theoritical_capacity_anode * SOCrange_anode; // [C] 

    const amrex::Real Csmax_cathode = electrochem::Cs_max_cathode(*prob_parm); // [mol.m-3]
    const amrex::Real SOCrange_cathode = prob_parm->cathode_soc_max - prob_parm->cathode_soc_min; // []
    const amrex::Real theoritical_capacity_cathode = volume_cathode * Csmax_cathode * FARADCONST; // [C]
    const amrex::Real pratical_capacity_cathode = theoritical_capacity_cathode * SOCrange_cathode; // [C]

    // Deduce cell pratical capacity
    const amrex::Real pratical_capacity_cell = amrex::min(pratical_capacity_anode, pratical_capacity_cathode); // [C]
    const amrex::Real Cell_NP_ratio = pratical_capacity_anode/pratical_capacity_cathode; // Not required, just FYI. Should be always >1 (if not, you can expect eartly degradation)

    // Applied current w/o sign
    const amrex::Real Crate = prob_parm->Crate;
    const amrex::Real time_to_charge = (1/Crate)*3600; // [s] Time to charge or discharge the cell
    const amrex::Real Current1C = pratical_capacity_cell/3600; // [A=C.s-1] Current for 1C
    const amrex::Real Applied_current = Current1C * Crate; // [A] (but we need the applied current density in A.m-2)

    // SOLID-CURRENT COLLECTOR INTERFACE
    // Surface calculations
    const amrex::Real surfaceanodeside_currentcollector_activematerial = echemAMR::host_global_storage->anode_ccAM_SA; // [m2]
    const amrex::Real surfacecathodeside_currentcollector_activematerial = echemAMR::host_global_storage->cathode_ccAM_SA; // [m2]
    amrex::Real tmp1 = echemAMR::host_global_storage->anode_ccCBDs_SA; // [m2]
    amrex::Real tmp2 = echemAMR::host_global_storage->cathode_ccCBDs_SA; // [m2]
    if (prob_parm->CBD_transport==1) { // Modified with (1-nanoporosity) (hard-coded for now)
        tmp1 = tmp1 * (1-0.47); // [m2]
        tmp2 = tmp2 * (1-0.47); // [m2]        
    }
    const amrex::Real surfaceanodeside_currentcollector_CBDsolid = tmp1;
    const amrex::Real surfacecathodeside_currentcollector_CBDsolid = tmp2;
      
    // Applied current density at the interface active material - current collector, w/ sign
    // These are the two Neumann Boundary conditions value applies at the left and right extremities of the cell
    amrex::Real Applied_current_density_anode = -Applied_current / (surfaceanodeside_currentcollector_activematerial + surfaceanodeside_currentcollector_CBDsolid); // [A.m-2]
    amrex::Real Applied_current_density_cathode = Applied_current / (surfacecathodeside_currentcollector_activematerial + surfacecathodeside_currentcollector_CBDsolid); // [A.m-2]

    // ELECTROCHEMICAL ACTIVE INTERFACE
    // Surface calculations
    const amrex::Real surfaceanodeside_electrolyte_activematerial = echemAMR::host_global_storage->anode_eleAM_SA;
    const amrex::Real surfacecathodeside_electrolyte_activematerial = echemAMR::host_global_storage->cathode_eleAM_SA;
    const amrex::Real surfaceanodeside_separatorp_activematerial = echemAMR::host_global_storage->anode_seppAM_SA;
    const amrex::Real surfacecathodeside_separatorp_activematerial = echemAMR::host_global_storage->cathode_seppAM_SA;       
    tmp1 = 0; // [m2]
    tmp2 = 0; // [m2]
    if (prob_parm->CBD_transport==1) { // Modified with (1-nanoporosity) (hard-coded for now)
        tmp1 = echemAMR::host_global_storage->anode_CBDpAM_SA * 0.47; // [m2]
        tmp2 = echemAMR::host_global_storage->cathode_CBDpAM_SA * 0.47; // [m2]        
    }
    const amrex::Real surfaceanodeside_CBDpore_activematerial = tmp1;
    const amrex::Real surfacecathodeside_CBDpore_activematerial = tmp2;

    // Initial current density at the interface active material - electrolyte
    // These are the two Neumann Boundary conditions value applied at the active interface.
    // At time=0s, value is uniform along the interface, then it will vary locally while the surface integral ([A]) will be constant and equal to "Applied_current"
    // Any deviation from the equality: Applied_current = integral(current density, ds active material-current collector) = integral(current density, ds active material-electrolyte)
    // will result in an error in mass conservation
    const amrex::Real Activeinterface_current_density_anode = -Applied_current / (surfaceanodeside_electrolyte_activematerial + surfaceanodeside_separatorp_activematerial + surfaceanodeside_CBDpore_activematerial); // [A.m-2]
    const amrex::Real Activeinterface_current_density_cathode = Applied_current / (surfacecathodeside_electrolyte_activematerial + surfacecathodeside_separatorp_activematerial + surfacecathodeside_CBDpore_activematerial); // [A.m-2]

    // Deduce potential values that verify this condition: Applied_current = integral(current density, ds active material-current collector) = integral(current density, ds active material-electrolyte) 
    const amrex::Real phie_to = electrochem::get_phie_initial_from_anode(Activeinterface_current_density_anode, prob_parm->phi_reference, *prob_parm);
    const amrex::Real phi_c_to = electrochem::get_phic_initial_from_electrolyte(Activeinterface_current_density_cathode, phie_to, *prob_parm);


    amrex::Print() << std::endl;
    amrex::Print() << "CAPACITY" << std::endl;
    amrex::Print() << "Anode                     " << std::endl;
    amrex::Print() << "- Active material volume: " << volume_anode << " [m3] " << std::endl;
    amrex::Print() << "- Theoritical capacity  : " << theoritical_capacity_anode << " [C] " << std::endl;
    amrex::Print() << "- SOC delta range       : " << SOCrange_anode << std::endl;
    amrex::Print() << "- Pratical capacity     : " << pratical_capacity_anode << " [C] " << std::endl;
    amrex::Print() << "Cathode                   " << std::endl;
    amrex::Print() << "- Active material volume: " << volume_cathode << " [m3] " <<  std::endl;
    amrex::Print() << "- Theoritical capacity  : " << theoritical_capacity_cathode << " [C] " << std::endl;
    amrex::Print() << "- SOC delta range       : " << SOCrange_cathode << std::endl;
    amrex::Print() << "- Pratical capacity     : " << pratical_capacity_cathode << " [C] " << std::endl;
    amrex::Print() << "Cell                      " << std::endl;
    amrex::Print() << "- N/P ratio             : " << Cell_NP_ratio << std::endl;
    amrex::Print() << "- Pratical capacity     : " << pratical_capacity_cell << " [C] " << std::endl;
    amrex::Print() << std::endl;
    amrex::Print() << "CURRENT" << std::endl;
    amrex::Print() << "Cell                     " << std::endl;
    amrex::Print() << "- C-rate (pratical)                    : " << Crate << std::endl;
    amrex::Print() << "- Time to (dis)charge within soc range : " << time_to_charge << " [s] " << std::endl;
    amrex::Print() << "- Applied current                      : " << Applied_current << " [A] " << std::endl;
    amrex::Print() << "Anode                     " << std::endl;
    amrex::Print() << "- active material - current collector interface: " << surfaceanodeside_currentcollector_activematerial << " [m2] " << std::endl;
    amrex::Print() << "- CBD solid - current collector interface      : " << surfaceanodeside_currentcollector_CBDsolid << " [m2] " << std::endl;
    amrex::Print() << "- Applied current density                      : " << Applied_current_density_anode << " [A.m-2] " << std::endl;
    amrex::Print() << "- active material - electrolyte interface      : " << surfaceanodeside_electrolyte_activematerial << " [m2] " << std::endl;
    amrex::Print() << "- active material - CBD nanopore interface     : " << surfaceanodeside_CBDpore_activematerial << " [m2] " << std::endl;
    amrex::Print() << "- Current density at the active interface      : " << Activeinterface_current_density_anode << " [A.m-2] " << std::endl;
    amrex::Print() << "Cathode                     " << std::endl;
    amrex::Print() << "- active material - current collector interface: " << surfacecathodeside_currentcollector_activematerial << " [m2] " << std::endl;
    amrex::Print() << "- CBD solid - current collector interface      : " << surfacecathodeside_currentcollector_CBDsolid << " [m2] " << std::endl;
    amrex::Print() << "- Applied current density                      : " << Applied_current_density_cathode << " [A.m-2] " << std::endl;
    amrex::Print() << "- active material - electrolyte interface      : " << surfacecathodeside_electrolyte_activematerial << " [m2] " << std::endl;
    amrex::Print() << "- active material - CBD nanopore interface     : " << surfacecathodeside_CBDpore_activematerial << " [m2] " << std::endl;
    amrex::Print() << "- Current density at the active interface      : " << Activeinterface_current_density_cathode << " [A.m-2] " << std::endl;
    amrex::Print() << std::endl;
    amrex::Print() << "POTENTIAL INITALIZATION" << std::endl;
    amrex::Print() << "- Anode reference potential : " << prob_parm->phi_reference << " [V] " << std::endl;
    amrex::Print() << "- Electrolyte potential     : " << phie_to << " [V] " << std::endl;
    amrex::Print() << "- Cathode potential         : " << phi_c_to << " [V] " << std::endl;
}

    AMREX_INLINE
void print_coefficients(ProbParm* prob_parm)
{
    int num_points=100;
    Real soc_min = -0.01; Real soc_max = 1.01; // Go slighly below and above min-max to check expression is robust
    Real Ce_normalized_min = -0.01; Real Ce_normalized_max = 5; 
    Real Ce_atrest = prob_parm->ce_atrest;
    
    if (amrex::ParallelDescriptor::IOProcessor()) // only IO Processor outputs file
    {
        // Anode
        double coeffs_a[num_points][5];
        for (int i = 0; i < num_points; ++i){
            coeffs_a[i][0] = (soc_max-soc_min)/(num_points-1)*i + soc_min;
            Real Cs = electrochem::Cs_max_anode(*prob_parm) * coeffs_a[i][0];
            coeffs_a[i][1] = electrochem::Ds_bulk_anode(Cs, *prob_parm); 
            coeffs_a[i][2] = electrochem::Ks_bulk_anode(Cs, *prob_parm); 
            coeffs_a[i][3] = electrochem::Io_a(Cs, Ce_atrest, *prob_parm); 
            coeffs_a[i][4] = electrochem::OCP_a(Cs, *prob_parm); 
        }
        std::ofstream out_a("Anode_coefficients.csv");
        for (auto& row : coeffs_a) {
        for (auto col : row)
            out_a << col <<',';
        out_a << '\n';
        }
        
        // Electrolyte
        double coeffs_e[num_points][6];
        for (int i = 0; i < num_points; ++i){
            Real Cenorm = (Ce_normalized_max-Ce_normalized_min)/(num_points-1)*i + Ce_normalized_min;
            coeffs_e[i][0] = Cenorm*Ce_atrest;
            coeffs_e[i][1] = electrochem::De_bulk(coeffs_e[i][0], *prob_parm); 
            coeffs_e[i][2] = electrochem::Ke_bulk(coeffs_e[i][0], *prob_parm); 
            coeffs_e[i][3] = electrochem::t_plus(coeffs_e[i][0], *prob_parm); 
            coeffs_e[i][4] = electrochem::A_c(coeffs_e[i][0], *prob_parm);
            coeffs_e[i][5] = electrochem::Kd(coeffs_e[i][2], coeffs_e[i][4], coeffs_e[i][3], *prob_parm) ;
        }
        std::ofstream out_e("Electrolyte_coefficients.csv");
        for (auto& row : coeffs_e) {
        for (auto col : row)
            out_e << col <<',';
        out_e << '\n';
        }      

        // Cathode
        double coeffs_c[num_points][5];
        for (int i = 0; i < num_points; ++i){
            coeffs_c[i][0] = (soc_max-soc_min)/(num_points-1)*i + soc_min;
            Real Cs = electrochem::Cs_max_cathode(*prob_parm) * coeffs_c[i][0];
            coeffs_c[i][1] = electrochem::Ds_bulk_cathode(Cs, *prob_parm); 
            coeffs_c[i][2] = electrochem::Ks_bulk_cathode(Cs, *prob_parm); 
            coeffs_c[i][3] = electrochem::Io_c(Cs, Ce_atrest, *prob_parm); 
            coeffs_c[i][4] = electrochem::OCP_c(Cs, *prob_parm); 
        }
        std::ofstream out_c("Cathode_coefficients.csv");
        for (auto& row : coeffs_c) {
        for (auto col : row)
            out_c << col <<',';
        out_c << '\n';
        }            
    
    }
}

AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
void externalbc(const amrex::Real x[AMREX_SPACEDIM],
        const amrex::Real s_int[],
        amrex::Real s_ext[],
        const int idir,
        const int sgn,
        const amrex::Real time,
        amrex::GeometryData const& geomdata)
{
    //default to extrapolation
    for(int c=0;c<NVAR;c++)
    {
        s_ext[c]=s_int[c];
    }
}
#endif
