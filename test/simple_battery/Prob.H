#ifndef _PROB_H_
#define _PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <Chemistry.H>
#include <ChemistryProbParm.H>

#include<echemAMR.H>

using namespace amrex;

//static initdata

AMREX_INLINE
void amrex_probinit (ProbParm & h_prob_parm, ProbParm & d_prob_parm)
{
    // read problem specific parmparse parameters here
    amrex::ParmParse pp("prob");

    pp.query("faraday_constant", h_prob_parm.Faraday_const );
    pp.query("R_gas_constant", h_prob_parm.R_gas_const );

    //pp.query("material_cathode", h_prob_parm.mater_cathode);
    //pp.query("material_anode", h_prob_parm.mater_anode);	

    pp.query("Crate", h_prob_parm.Crate);
    pp.query("cathode_soc_min", h_prob_parm.cathode_soc_min);
    pp.query("cathode_soc_max", h_prob_parm.cathode_soc_max);
    pp.query("anode_soc_min", h_prob_parm.anode_soc_min);
    pp.query("anode_soc_max", h_prob_parm.anode_soc_max);

    pp.query("ce_atrest", h_prob_parm.ce_atrest);
    pp.query("soc_anode_t0", h_prob_parm.soc_anode_t0);
    pp.query("soc_cathode_t0", h_prob_parm.soc_cathode_t0);
    pp.query("phi_reference", h_prob_parm.phi_reference);

#ifdef AMREX_USE_GPU
    amrex::Gpu::htod_memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#else
    std::memcpy(&d_prob_parm, &h_prob_parm, sizeof(ProbParm));
#endif

}

// ---------------------------------------------------
// PROBLEM INITIALIZATION
//    STEP 1: IDENTIFY DOMAINS (SET LEVEL SET ID)
// ---------------------------------------------------


AMREX_GPU_DEVICE
AMREX_INLINE
void initdomaindata(Box const& bx, Array4<Real> const& phi, GeometryData const& geomdata,
        ProbParm *prob_parm)
{
    // CREATE 3D ARRAY GEOMETRY READ FILE
    std::string inFileName = prob_parm->geometry_filename;
    std::ifstream inFile;

    // 1: Get dimension: first 3 integers
    int dimension_geometry[3];
    inFile.open(inFileName.c_str());
    if (inFile.is_open())
    {
        for (int i = 0; i < 3; i++)
        {
            inFile >> dimension_geometry[i];
        }
        inFile.close(); // Close input file
    }
    else { //Error message
        amrex::Print() << "Can't find input file " << inFileName << std::endl;
    }

    // 2: Get domain id
    int geo1D [dimension_geometry[0]*dimension_geometry[1]*dimension_geometry[2]];
    inFile.open(inFileName.c_str());
    if (inFile.is_open())
    {
        for (int i = 0; i < 3; i++)
        {
            inFile >> dimension_geometry[i];
        }
        for (int i = 0; i < dimension_geometry[0]*dimension_geometry[1]*dimension_geometry[2]-1; i++)
        {
            inFile >> geo1D[i];
        }
        inFile.close(); // Close input file
    }
    else { //Error message
        amrex::Print() << "Can't find input file " << inFileName << std::endl;
    }

    const auto lo = lbound(bx);
    const auto hi = ubound(bx);    

    const Real* AMREX_RESTRICT prob_lo = geomdata.ProbLo();
    const Real* AMREX_RESTRICT prob_hi = geomdata.ProbHi();
    const Real* AMREX_RESTRICT dx      = geomdata.CellSize();
    
    const Real Lx = prob_hi[0] - prob_lo[0];
    const Real Ly = prob_hi[1] - prob_lo[1];
    const Real Lz = prob_hi[2] - prob_lo[2];
    
    const int ncomp = phi.nComp();

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (GPU::notInLaunchRegion)
#endif
    for (int k = lo.z; k <= hi.z; ++k) 
    {
        const Real z = prob_lo[2] + (k+0.5)*dx[2];
        int idz = std::round( (dimension_geometry[2]-1) / Lz * z + 1 ) -1; // Convert z in the index of imported geometry geo3D 
        for (int j = lo.y; j <= hi.y; ++j)
        {
            const Real y = prob_lo[1] + (j+0.5)*dx[1];
            int idy = std::round( (dimension_geometry[1]-1) / Ly * y +1) -1; // Convert y in the index of imported geometry geo3D 
            for (int i = lo.x; i <= hi.x; ++i) 
            {
                const Real x = prob_lo[0] + (i+0.5)*dx[0];
                int idx = std::round( (dimension_geometry[0]-1) / Lx * x +1 ) -1; // Convert x in the index of imported geometry geo3D 
                for (int n = 0; n < ncomp; ++n)
                {
                    phi(i, j, k, n) = 0.0; // Initialize all composant to 0
                }
                int id = (idx+1 + (idy+1-1)*(dimension_geometry[0]) + (idz+1-1)*(dimension_geometry[1])*(dimension_geometry[0]))-1;
                

                if (geo1D[id] == 1) {
                    phi(i,j,k,A_ID) = 1;
                } else if (geo1D[id] == 3) {
                    phi(i,j,k,E_ID) = 1;
                    phi(i,j,k,LS_ID) = 1;
                } else if (geo1D[id] == 4) {
                    phi(i,j,k,S_ID) = 1;
                    phi(i,j,k,LS_ID) = 1;
                } else if (geo1D[id] == 2) {
                    phi(i,j,k,C_ID) = 1;
                } else {
                    amrex::Print() << "Error for idx= " << idx << " idy= " << idy << " idz= " << idz << " geo1D[id]= " << geo1D[id] << std::endl;
                    phi(i,j,k,A_ID) = 1;
                }
            }
        }
    }
}

// ---------------------------------------------------
// PROBLEM INITIALIZATION
//    STEP 2: CONCENTRATIONS AND POTENTIALS
// ---------------------------------------------------

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

    // INITIAL POTENTIAL
    // Calculate active material volumes
    const amrex::Real volume_anode = echemAMR::host_global_storage->anode_volume; // [m3]
    const amrex::Real volume_cathode = echemAMR::host_global_storage->cathode_volume; // [m3]

    //amrex::Print() << "    volume_anode:       " << volume_anode << std::endl;
    //amrex::Print() << "    volume_cathode:       " << volume_cathode << std::endl;


    // Calculate anode and cathode theoritical (use full soc range) and pratical (use a reduced soc range) capacity
    const amrex::Real Fday = prob_parm->Faraday_const; // [C.mol-1]

    const amrex::Real Csmax_anode = electrochem::Cs_max_anode(*prob_parm); // [mol.m-3]
    const amrex::Real SOCrange_anode = prob_parm->anode_soc_max - prob_parm->anode_soc_min; // []
    const amrex::Real theoritical_capacity_anode = volume_anode * Csmax_anode * Fday; // [C]
    const amrex::Real pratical_capacity_anode = theoritical_capacity_anode * SOCrange_anode; // [C] 

    const amrex::Real Csmax_cathode = electrochem::Cs_max_cathode(*prob_parm); // [mol.m-3]
    const amrex::Real SOCrange_cathode = prob_parm->cathode_soc_max - prob_parm->cathode_soc_min; // []
    const amrex::Real theoritical_capacity_cathode = volume_cathode * Csmax_cathode * Fday; // [C]
    const amrex::Real pratical_capacity_cathode = theoritical_capacity_cathode * SOCrange_cathode; // [C]

    // Deduce cell pratical capacity
    const amrex::Real pratical_capacity_cell = amrex::min(pratical_capacity_anode, pratical_capacity_cathode); // [C]
    const amrex::Real Cell_NP_ratio = pratical_capacity_anode/pratical_capacity_cathode; // Not required, just FYI. Should be always >1 (if not, you can expect eartly degradation)

    //amrex::Print() << "    Cell_NP_ratio:       " << Cell_NP_ratio << std::endl;

    // Applied current
    const amrex::Real Crate = prob_parm->Crate;
    const amrex::Real time_to_charge = (1/Crate)*3600; // [s] Time to charge or discharge the cell
    const amrex::Real Current1C = pratical_capacity_cell/3600; // [A=C.s-1] Current for 1C
    const amrex::Real Applied_current = Current1C * Crate; // [A] (but we need the applied current density in A.m-2)

    // Copy current in echemAMR as it is needed for post-processing
    echemAMR::host_global_storage->Applied_current = Applied_current;

    // Surface calculations // FIX ME: add surface calculations from Jeff post-processing
    const amrex::Real surface_anodesolid_currentcollector = echemAMR::host_global_storage->anode_cc_SA; // [m2], temporary value pi*((5e-6)^2)
    const amrex::Real surface_cathodsolid_currentcollector = echemAMR::host_global_storage->cathode_cc_SA; // [m2], temporary value pi*((5e-6)^2)
    // const amrex::Real surface_anodesolid_currentcollector = 7.8540e-11; // [m2], temporary value pi*((5e-6)^2)
    // const amrex::Real surface_cathodsolid_currentcollector = 7.8540e-11; // [m2], temporary value pi*((5e-6)^2)
    
    // Applied current density at the interface active material - current collector
    // These are the two Neumann Boundary conditions value applies at the left and right extremities of the cell
    // FIX ME: MICHAEL/HARI: could you link g with these two values?
    amrex::Real Applied_current_density_anode = Applied_current / surface_anodesolid_currentcollector; // [A.m-2]
    amrex::Real Applied_current_density_cathode = Applied_current / surface_cathodsolid_currentcollector; // [A.m-2]

    echemAMR::host_global_storage->pot_bc_lo[0] = -Applied_current_density_cathode;
    echemAMR::host_global_storage->pot_bc_hi[0] = -Applied_current_density_anode;

    // Surface calculations // FIX ME: add surface calculations from Jeff post-processing
    const amrex::Real surface_anodesolid_electrolyte = echemAMR::host_global_storage->anode_ele_SA; // 4*pi*((5e-6)^2)
    const amrex::Real surface_cathodesolid_electrolyte = echemAMR::host_global_storage->cathode_ele_SA; // 4*pi*((5e-6)^2)
    // const amrex::Real surface_anodesolid_electrolyte = 3.1416e-10; // 4*pi*((5e-6)^2)
    // const amrex::Real surface_cathodesolid_electrolyte = 3.1416e-10; // 4*pi*((5e-6)^2)

    // Initial current density at the interface active material - electrolyte
    // These are the two Neumann Boundary conditions value applied at the active interface.
    // At time=0s, value is uniform along the interface, then it will vary locally while the surface integral ([A]) will be constant and equal to "Applied_current"
    // Any deviation from the equality: Applied_current = integral(current density, ds active material-current collector) = integral(current density, ds active material-electrolyte)
    // will result in an error in mass conservation
    const amrex::Real Activeinterface_current_density_anode = Applied_current / surface_anodesolid_electrolyte; // [A.m-2]
    const amrex::Real Activeinterface_current_density_cathode = Applied_current / surface_cathodesolid_electrolyte; // [A.m-2]

    // Deduce potential values that verify this condition: Applied_current = integral(current density, ds active material-current collector) = integral(current density, ds active material-electrolyte) 
    const amrex::Real phis_a_to = prob_parm->phi_reference;
    const amrex::Real phie_to = electrochem::get_phie_initial_from_anode(Activeinterface_current_density_anode, phis_a_to, *prob_parm);
    const amrex::Real phis_c_to = electrochem::get_phic_initial_from_electrolyte(Activeinterface_current_density_cathode, phie_to, *prob_parm);

    // INITIAL CONCENTRATION
    const amrex::Real cs_a_to = prob_parm->soc_anode_t0 * electrochem::Cs_max_anode(*prob_parm);
    const amrex::Real cs_c_to = prob_parm->soc_cathode_t0 * electrochem::Cs_max_cathode(*prob_parm);
    const amrex::Real ce_to = prob_parm->ce_atrest;

    for (int k = lo.z; k <= hi.z; ++k) 
    {
        for (int j = lo.y; j <= hi.y; ++j)
        {
            for (int i = lo.x; i <= hi.x; ++i) 
            {
                 if (phi(i,j,k,C_ID) == 1) { // Within cathode
                     phi(i,j,k,CO_ID) = cs_c_to;
                     phi(i,j,k,POT_ID) = phis_c_to;
                 } else if (phi(i,j,k,A_ID) == 1) { // Within anode
                     phi(i,j,k,CO_ID) = cs_a_to;
                     phi(i,j,k,POT_ID) = phis_a_to;
                 } else if ((phi(i,j,k,E_ID) == 1) || (phi(i,j,k,S_ID) == 1)) {// Within electrolyte
                     phi(i,j,k,CO_ID) = ce_to;
                     phi(i,j,k,POT_ID) = phie_to;
                 }
            }
        }
    }
}

// ---------------------------------------------------
// PROBLEM INITIALIZATION
//    PRINTING INTIAL DOFs AND RATE
// ---------------------------------------------------

    AMREX_INLINE
void print_init_data(ProbParm* prob_parm)
{
    // Calculate active material volumes
    // echemAMR::post_timestep(0, 0); // FIX ME: add volume calculation from Jeff post-processing
    const amrex::Real volume_anode = echemAMR::host_global_storage->anode_volume; // [m3]
    const amrex::Real volume_cathode = echemAMR::host_global_storage->cathode_volume; // [m3]

    // Calculate anode and cathode theoritical (use full soc range) and pratical (use a reduced soc range) capacity
    const amrex::Real Fday = prob_parm->Faraday_const; // [C.mol-1]

    const amrex::Real Csmax_anode = electrochem::Cs_max_anode(*prob_parm); // [mol.m-3]
    const amrex::Real SOCrange_anode = prob_parm->anode_soc_max - prob_parm->anode_soc_min; // []
    const amrex::Real theoritical_capacity_anode = volume_anode * Csmax_anode * Fday; // [C]
    const amrex::Real pratical_capacity_anode = theoritical_capacity_anode * SOCrange_anode; // [C] 

    const amrex::Real Csmax_cathode = electrochem::Cs_max_cathode(*prob_parm); // [mol.m-3]
    const amrex::Real SOCrange_cathode = prob_parm->cathode_soc_max - prob_parm->cathode_soc_min; // []
    const amrex::Real theoritical_capacity_cathode = volume_cathode * Csmax_cathode * Fday; // [C]
    const amrex::Real pratical_capacity_cathode = theoritical_capacity_cathode * SOCrange_cathode; // [C]

    // Deduce cell pratical capacity
    const amrex::Real pratical_capacity_cell = amrex::min(pratical_capacity_anode, pratical_capacity_cathode); // [C]
    const amrex::Real Cell_NP_ratio = pratical_capacity_anode/pratical_capacity_cathode; // Not required, just FYI. Should be always >1 (if not, you can expect eartly degradation)

    // Applied current
    const amrex::Real Crate = prob_parm->Crate;
    const amrex::Real time_to_charge = (1/Crate)*3600; // [s] Time to charge or discharge the cell
    const amrex::Real Current1C = pratical_capacity_cell/3600; // [A=C.s-1] Current for 1C
    const amrex::Real Applied_current = Current1C * Crate; // [A] (but we need the applied current density in A.m-2)

    // Copy current in echemAMR as it is needed for post-processing
    echemAMR::host_global_storage->Applied_current = Applied_current;

    // Surface calculations // FIX ME: add surface calculations from Jeff post-processing
    const amrex::Real surface_anodesolid_currentcollector = echemAMR::host_global_storage->anode_cc_SA; // [m2], temporary value pi*((5e-6)^2)
    const amrex::Real surface_cathodsolid_currentcollector = echemAMR::host_global_storage->cathode_cc_SA; // [m2], temporary value pi*((5e-6)^2)

    // Applied current density at the interface active material - current collector
    // These are the two Neumann Boundary conditions value applies at the left and right extremities of the cell
    // FIX ME: MICHAEL/HARI: could you link g with these two values?
    const amrex::Real Applied_current_density_anode = Applied_current / surface_anodesolid_currentcollector; // [A.m-2]
    const amrex::Real Applied_current_density_cathode = Applied_current / surface_cathodsolid_currentcollector; // [A.m-2]

    // Surface calculations // FIX ME: add surface calculations from Jeff post-processing
    const amrex::Real surface_anodesolid_electrolyte = echemAMR::host_global_storage->anode_ele_SA; // 4*pi*((5e-6)^2)
    const amrex::Real surface_cathodesolid_electrolyte = echemAMR::host_global_storage->cathode_ele_SA; // 4*pi*((5e-6)^2)

    // Initial current density at the interface active material - electrolyte
    // These are the two Neumann Boundary conditions value applied at the active interface.
    // At time=0s, value is uniform along the interface, then it will vary locally while the surface integral ([A]) will be constant and equal to "Applied_current"
    // Any deviation from the equality: Applied_current = integral(current density, ds active material-current collector) = integral(current density, ds active material-electrolyte)
    // will result in an error in mass conservation
    const amrex::Real Activeinterface_current_density_anode = Applied_current / surface_anodesolid_electrolyte; // [A.m-2]
    const amrex::Real Activeinterface_current_density_cathode = Applied_current / surface_cathodesolid_electrolyte; // [A.m-2]

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
    amrex::Print() << "- active material - current collector interface: " << surface_anodesolid_currentcollector << " [m2] " << std::endl;
    amrex::Print() << "- Applied current density                      : " << Applied_current_density_anode << " [A.m-2] " << std::endl;
    amrex::Print() << "- active material - electrolyte interface      : " << surface_anodesolid_electrolyte << " [m2] " << std::endl;
    amrex::Print() << "- Current density at the active interface      : " << Activeinterface_current_density_anode << " [A.m-2] " << std::endl;
    amrex::Print() << "Cathode                     " << std::endl;
    amrex::Print() << "- active material - current collector interface: " << surface_cathodsolid_currentcollector << " [m2] " << std::endl;
    amrex::Print() << "- Applied current density                      : " << Applied_current_density_cathode << " [A.m-2] " << std::endl;
    amrex::Print() << "- active material - electrolyte interface      : " << surface_cathodesolid_electrolyte << " [m2] " << std::endl;
    amrex::Print() << "- Current density at the active interface      : " << Activeinterface_current_density_cathode << " [A.m-2] " << std::endl;
    amrex::Print() << std::endl;
    amrex::Print() << "POTENTIAL INITALIZATION" << std::endl;
    amrex::Print() << "- Anode reference potential : " << prob_parm->phi_reference << " [V] " << std::endl;
    amrex::Print() << "- Electrolyte potential     : " << phie_to << " [V] " << std::endl;
    amrex::Print() << "- Cathode potential         : " << phi_c_to << " [V] " << std::endl;
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
