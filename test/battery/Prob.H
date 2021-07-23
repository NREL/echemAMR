#ifndef _PROB_H_
#define _PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <Chemistry.H>
#include <ChemistryProbParm.H>

#include <echemAMR.H>

using namespace amrex;

// static initdata

AMREX_INLINE
void amrex_probinit(ProbParm& h_prob_parm, ProbParm& d_prob_parm)
{
    // read problem specific parmparse parameters here
    amrex::ParmParse pp("prob");

    pp.query("faraday_constant", h_prob_parm.Faraday_const);
    pp.query("R_gas_constant", h_prob_parm.R_gas_const);

    pp.query("material_electrolyte", h_prob_parm.mater_electrolyte);
    pp.query("use_KDstar", h_prob_parm.use_KDstar);
    pp.query("material_cathode", h_prob_parm.mater_cathode);
    pp.query("material_anode", h_prob_parm.mater_anode);

    pp.query("T0", h_prob_parm.T0);
    pp.query("Crate", h_prob_parm.Crate);
    pp.query("cathode_soc_min", h_prob_parm.cathode_soc_min);
    pp.query("cathode_soc_max", h_prob_parm.cathode_soc_max);
    pp.query("anode_soc_min", h_prob_parm.anode_soc_min);
    pp.query("cathode_soc_min", h_prob_parm.anode_soc_max);

    pp.query("lowerscale_vf_pore_anode", h_prob_parm.lowerscale_vf_pore_anode);
    pp.query("lowerscale_vf_pore_separator", h_prob_parm.lowerscale_vf_pore_separator);
    pp.query("lowerscale_vf_pore_cathode", h_prob_parm.lowerscale_vf_pore_cathode);
    pp.query("lowerscale_p_pore_anode", h_prob_parm.lowerscale_p_pore_anode);
    pp.query("lowerscale_p_pore_separator", h_prob_parm.lowerscale_p_pore_separator);
    pp.query("lowerscale_p_pore_cathode", h_prob_parm.lowerscale_p_pore_cathode);

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

AMREX_GPU_DEVICE
AMREX_INLINE
void initdomaindata(Box const& bx, Array4<Real> const& phi,
                    GeometryData const& geomdata) // break into multiple steps
{
    const auto lo = lbound(bx);
    const auto hi = ubound(bx);

    const Real* AMREX_RESTRICT prob_lo = geomdata.ProbLo();
    const Real* AMREX_RESTRICT prob_hi = geomdata.ProbHi();
    const Real* AMREX_RESTRICT dx = geomdata.CellSize();

    const int ncomp = phi.nComp();

    const Real Lx = prob_hi[0] - prob_lo[0];
    const Real Ly = prob_hi[1] - prob_lo[1];
    const Real Lz = prob_hi[2] - prob_lo[2];

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (GPU::notInLaunchRegion)
#endif

    for (int k = lo.z; k <= hi.z; ++k)
    {
        const Real z = prob_lo[2] + (0.5 + k) * dx[2];
        for (int j = lo.y; j <= hi.y; ++j)
        {
            const Real y = prob_lo[1] + (0.5 + j) * dx[1];
            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i)
            {
                const Real x = prob_lo[0] + (0.5 + i) * dx[0];

                for (int n = 0; n < ncomp; ++n)
                {
                    phi(i, j, k, n) = -1.0;
                }

                // FIXME: pass prob parm or more inputs so that we can have
                // control over initial condition parameters Separator
                const amrex::Real s_xlo = 0.45 * Lx;
                const amrex::Real s_xhi = 0.55 * Lx;

                bool comp_found = false;
                for (int jj = 0; jj < 3; ++jj)
                {
                    Real y0 = jj * Ly / 2.0;
                    for (int kk = 0; kk < 3; ++kk)
                    {
                        Real z0 = kk * Lz / 2.0;
                        // cathode
                        for (int ii = 0; ii < 5; ++ii)
                        {
                            Real x0 = ii * .35 * Lx / 4.0;
                            Real r = std::sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0));
                            if (r < Lx / 20.0)
                            {
                                phi(i, j, k, C_ID) = 1.0;
                                comp_found = true;
                            }
                        }
                        // anode
                        for (int ii = 0; ii < 5; ++ii)
                        {
                            Real x0 = 0.65 * Lx + ii * .35 * Lx / 4.0;
                            Real r = std::sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0));
                            if (r < Lx / 20.0)
                            {
                                phi(i, j, k, A_ID) = 1.0;
                                comp_found = true;
                            }
                        }
                    }
                }

                if (!comp_found)
                {
                    if (x > s_xlo && x < s_xhi)
                    {
                        // separator
                        phi(i, j, k, S_ID) = 1.0;
                    } else
                    {
                        // electrolyte
                        phi(i, j, k, E_ID) = 1.0;
                    }
                }
                phi(i, j, k, EFX_ID) = 0.0;
                phi(i, j, k, EFY_ID) = 0.0;
                phi(i, j, k, EFZ_ID) = 0.0;
                // phi(i,j,k,PO_ID)  = 0.0;

                // this has a check in it to make sure we don't have
                // multiple fields specified in the same cell
                // FIXME: disable for now
                //                electrochem::level_set_to_component(i,j,k,phi);
            }
        }
    }
}

AMREX_GPU_DEVICE
AMREX_INLINE
void initproblemdata(Box const& bx, Array4<Real> const& phi,
                     GeometryData const& geomdata) // break into multiple steps
{

    const auto lo = lbound(bx);
    const auto hi = ubound(bx);

    const Real* AMREX_RESTRICT prob_lo = geomdata.ProbLo();
    const Real* AMREX_RESTRICT prob_hi = geomdata.ProbHi();
    const Real* AMREX_RESTRICT dx = geomdata.CellSize();

    const int ncomp = phi.nComp();

    const Real Lx = prob_hi[0] - prob_lo[0];
    const Real Ly = prob_hi[1] - prob_lo[1];
    const Real Lz = prob_hi[2] - prob_lo[2];

#ifdef _OPENMP
#pragma omp parallel for collapse(2) if (GPU::notInLaunchRegion)
#endif

    // ------------------------
    // POTENTIAL INITIALIZATION
    // ------------------------

    // Calculate active material volumes
    const amrex::Real volume_anode = echemAMR::host_global_storage->anode_volume;     // [m3]
    const amrex::Real volume_cathode = echemAMR::host_global_storage->cathode_volume; // [m3]

    // Calculate anode and cathode theoritical (use full soc range) and pratical
    // (use a reduced soc range) capacity
    const amrex::Real Fday = echemAMR::d_prob_parm->Faraday_const; // [C.mol-1]

    const amrex::Real Csmax_anode = electrochem::Cs_max_anode(*echemAMR::d_prob_parm);                              // [mol.m-3]
    const amrex::Real SOCrange_anode = echemAMR::d_prob_parm->anode_soc_max - echemAMR::d_prob_parm->anode_soc_min; // []
    const amrex::Real theoritical_capacity_anode = volume_anode * Csmax_anode * Fday;                               // [C]
    const amrex::Real pratical_capacity_anode = theoritical_capacity_anode * SOCrange_anode;                        // [C]

    const amrex::Real Csmax_cathode = electrochem::Cs_max_cathode(*echemAMR::d_prob_parm);                                // [mol.m-3]
    const amrex::Real SOCrange_cathode = echemAMR::d_prob_parm->cathode_soc_max - echemAMR::d_prob_parm->cathode_soc_min; // []
    const amrex::Real theoritical_capacity_cathode = volume_cathode * Csmax_cathode * Fday;                               // [C]
    const amrex::Real pratical_capacity_cathode = theoritical_capacity_cathode * SOCrange_cathode;                        // [C]

    // Deduce cell pratical capacity
    const amrex::Real pratical_capacity_cell = amrex::min(pratical_capacity_anode, pratical_capacity_cathode); // [C]
    const amrex::Real Cell_NP_ratio = pratical_capacity_anode / pratical_capacity_cathode;                     // Not required, just FYI. Should be always
                                                                                                               // >1 (if not, you can expect eartly
                                                                                                               // degradation)

    // Applied current
    const amrex::Real Crate = echemAMR::d_prob_parm->Crate;
    const amrex::Real time_to_charge = (1 / Crate) * 3600;       // [s] Time to charge or discharge the cell
    const amrex::Real Current1C = pratical_capacity_cell / 3600; // [A=C.s-1] Current for 1C
    const amrex::Real Applied_current = Current1C * Crate;       // [A] (but we need the applied current density in A.m-2)

    // Copy current in echemAMR as it is needed for post-processing
    echemAMR::host_global_storage->Applied_current = Applied_current;

    // Surface calculations // FIX ME: add surface calculations from Jeff
    // post-processing
    const amrex::Real surface_anodesolid_currentcollector = echemAMR::host_global_storage->anode_cc_SA;    // [m2], temporary value pi*((5e-6)^2)
    const amrex::Real surface_cathodsolid_currentcollector = echemAMR::host_global_storage->cathode_cc_SA; // [m2], temporary value pi*((5e-6)^2)
    // const amrex::Real surface_anodesolid_currentcollector = 7.8540e-11; //
    // [m2], temporary value pi*((5e-6)^2) const amrex::Real
    // surface_cathodsolid_currentcollector = 7.8540e-11; // [m2], temporary
    // value pi*((5e-6)^2)

    // Applied current density at the interface active material - current
    // collector These are the two Neumann Boundary conditions value applies at
    // the left and right extremities of the cell FIX ME: MICHAEL/HARI: could
    // you link g with these two values?
    const amrex::Real Applied_current_density_anode = Applied_current / surface_anodesolid_currentcollector;    // [A.m-2]
    const amrex::Real Applied_current_density_cathode = Applied_current / surface_cathodsolid_currentcollector; // [A.m-2]

    echemAMR::host_global_storage->pot_bc_lo[0] = -Applied_current_density_cathode;
    echemAMR::host_global_storage->pot_bc_hi[0] = -Applied_current_density_anode;

    // Surface calculations // FIX ME: add surface calculations from Jeff
    // post-processing
    const amrex::Real surface_anodesolid_electrolyte = echemAMR::host_global_storage->anode_ele_SA;     // 4*pi*((5e-6)^2)
    const amrex::Real surface_cathodesolid_electrolyte = echemAMR::host_global_storage->cathode_ele_SA; // 4*pi*((5e-6)^2)
    // const amrex::Real surface_anodesolid_electrolyte = 3.1416e-10; //
    // 4*pi*((5e-6)^2) const amrex::Real surface_cathodesolid_electrolyte
    // = 3.1416e-10; // 4*pi*((5e-6)^2)

    // Initial current density at the interface active material - electrolyte
    // These are the two Neumann Boundary conditions value applied at the active
    // interface. At time=0s, value is uniform along the interface, then it will
    // vary locally while the surface integral ([A]) will be constant and equal
    // to "Applied_current" Any deviation from the equality: Applied_current =
    // integral(current density, ds active material-current collector) =
    // integral(current density, ds active material-electrolyte) will result in
    // an error in mass conservation
    const amrex::Real Activeinterface_current_density_anode = Applied_current / surface_anodesolid_electrolyte;     // [A.m-2]
    const amrex::Real Activeinterface_current_density_cathode = Applied_current / surface_cathodesolid_electrolyte; // [A.m-2]

    // Deduce potential values that verify this condition: Applied_current =
    // integral(current density, ds active material-current collector) =
    // integral(current density, ds active material-electrolyte)
    const amrex::Real phie_to =
        electrochem::get_phie_initial_from_anode(Activeinterface_current_density_anode, echemAMR::d_prob_parm->phi_reference, *echemAMR::d_prob_parm);
    const amrex::Real phi_c_to = electrochem::get_phic_initial_from_electrolyte(Activeinterface_current_density_cathode, phie_to, *echemAMR::d_prob_parm);

    // TO BE REMOVED
    // ProbParm h_prob_parm; // Create a new instance of the structure ProbParm
    // ProbParm* hh_prob_parm = nullptr;
    // ProbParm* dd_prob_parm = nullptr;
    // hh_prob_parm = new ProbParm{};
    // dd_prob_parm = (ProbParm*)The_Arena()->alloc(sizeof(ProbParm));
    // amrex_probinit(*hh_prob_parm, *dd_prob_parm);

    for (int k = lo.z; k <= hi.z; ++k)
    {
        const Real z = prob_lo[2] + (0.5 + k) * dx[2];
        for (int j = lo.y; j <= hi.y; ++j)
        {
            const Real y = prob_lo[1] + (0.5 + j) * dx[1];
            AMREX_PRAGMA_SIMD
            for (int i = lo.x; i <= hi.x; ++i)
            {
                const Real x = prob_lo[0] + (0.5 + i) * dx[0];

                // FIXME: pass prob parm or more inputs so that we can have
                // control over initial condition parameters Separator
                const amrex::Real s_xlo = 0.45 * Lx;
                const amrex::Real s_xhi = 0.55 * Lx;
                // anode
                const amrex::Real a_yhi = 0.8 * Ly;
                const amrex::Real a_ylo = 0.2 * Ly;
                const amrex::Real a_x = 0.25 * Lx;
                // cathode
                const amrex::Real c_yhi = 0.8 * Ly;
                const amrex::Real c_ylo = 0.2 * Ly;
                const amrex::Real c_x = 0.75 * Lx;

                bool comp_found = false;
                for (int jj = 0; jj < 3; ++jj)
                {
                    Real y0 = jj * Ly / 2.0;
                    for (int kk = 0; kk < 3; ++kk)
                    {
                        Real z0 = kk * Lz / 2.0;
                        // cathode
                        for (int ii = 0; ii < 5; ++ii)
                        {
                            Real x0 = ii * .35 * Lx / 4.0;
                            Real r = std::sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0));
                            if (r < Lx / 20.0)
                            {
                                const amrex::Real soc_init = echemAMR::d_prob_parm->soc_cathode_t0;
                                const amrex::Real csmax = electrochem::Cs_max_cathode(*echemAMR::d_prob_parm);
                                phi(i, j, k, CO_ID) = soc_init * csmax;
                                phi(i, j, k, PO_ID) = phi_c_to;
                                comp_found = true;
                            }
                        }
                        // anode
                        for (int ii = 0; ii < 5; ++ii)
                        {
                            Real x0 = 0.65 * Lx + ii * .35 * Lx / 4.0;
                            Real r = std::sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0));
                            if (r < Lx / 20.0)
                            {
                                const amrex::Real soc_init = echemAMR::d_prob_parm->soc_anode_t0;
                                const amrex::Real csmax = electrochem::Cs_max_anode(*echemAMR::d_prob_parm);
                                phi(i, j, k, CO_ID) = soc_init * csmax;
                                phi(i, j, k, PO_ID) = echemAMR::d_prob_parm->phi_reference;
                                comp_found = true;
                            }
                        }
                    }
                }

                if (!comp_found)
                {
                    if (x > s_xlo && x < s_xhi)
                    {
                        // separator
                        phi(i, j, k, CO_ID) = echemAMR::d_prob_parm->ce_atrest;
                        phi(i, j, k, PO_ID) = phie_to;
                    } else
                    {
                        // electrolyte
                        phi(i, j, k, CO_ID) = echemAMR::d_prob_parm->ce_atrest;
                        phi(i, j, k, PO_ID) = phie_to;
                    }
                }
                phi(i, j, k, EFX_ID) = 0.0;
                phi(i, j, k, EFY_ID) = 0.0;
                phi(i, j, k, EFZ_ID) = 0.0;
                // phi(i,j,k,PO_ID)  = 0.0;

                // this has a check in it to make sure we don't have
                // multiple fields specified in the same cell
                // FIXME: disable for now
                //                electrochem::level_set_to_component(i,j,k,phi);
            }
        }
    }
}

AMREX_INLINE
void print_init_data() // break into multiple steps
{
    // Calculate active material volumes
    // echemAMR::post_timestep(0, 0); // FIX ME: add volume calculation from
    // Jeff post-processing
    const amrex::Real volume_anode = echemAMR::host_global_storage->anode_volume;     // [m3]
    const amrex::Real volume_cathode = echemAMR::host_global_storage->cathode_volume; // [m3]

    // Calculate anode and cathode theoritical (use full soc range) and pratical
    // (use a reduced soc range) capacity
    const amrex::Real Fday = echemAMR::d_prob_parm->Faraday_const; // [C.mol-1]

    const amrex::Real Csmax_anode = electrochem::Cs_max_anode(*echemAMR::d_prob_parm);                              // [mol.m-3]
    const amrex::Real SOCrange_anode = echemAMR::d_prob_parm->anode_soc_max - echemAMR::d_prob_parm->anode_soc_min; // []
    const amrex::Real theoritical_capacity_anode = volume_anode * Csmax_anode * Fday;                               // [C]
    const amrex::Real pratical_capacity_anode = theoritical_capacity_anode * SOCrange_anode;                        // [C]

    const amrex::Real Csmax_cathode = electrochem::Cs_max_cathode(*echemAMR::d_prob_parm);                                // [mol.m-3]
    const amrex::Real SOCrange_cathode = echemAMR::d_prob_parm->cathode_soc_max - echemAMR::d_prob_parm->cathode_soc_min; // []
    const amrex::Real theoritical_capacity_cathode = volume_cathode * Csmax_cathode * Fday;                               // [C]
    const amrex::Real pratical_capacity_cathode = theoritical_capacity_cathode * SOCrange_cathode;                        // [C]

    // Deduce cell pratical capacity
    const amrex::Real pratical_capacity_cell = amrex::min(pratical_capacity_anode, pratical_capacity_cathode); // [C]
    const amrex::Real Cell_NP_ratio = pratical_capacity_anode / pratical_capacity_cathode;                     // Not required, just FYI. Should be always
                                                                                                               // >1 (if not, you can expect eartly
                                                                                                               // degradation)

    // Applied current
    const amrex::Real Crate = echemAMR::d_prob_parm->Crate;
    const amrex::Real time_to_charge = (1 / Crate) * 3600;       // [s] Time to charge or discharge the cell
    const amrex::Real Current1C = pratical_capacity_cell / 3600; // [A=C.s-1] Current for 1C
    const amrex::Real Applied_current = Current1C * Crate;       // [A] (but we need the applied current density in A.m-2)

    // Copy current in echemAMR as it is needed for post-processing
    echemAMR::host_global_storage->Applied_current = Applied_current;

    // Surface calculations // FIX ME: add surface calculations from Jeff
    // post-processing
    const amrex::Real surface_anodesolid_currentcollector = echemAMR::host_global_storage->anode_cc_SA;    // [m2], temporary value pi*((5e-6)^2)
    const amrex::Real surface_cathodsolid_currentcollector = echemAMR::host_global_storage->cathode_cc_SA; // [m2], temporary value pi*((5e-6)^2)

    // Applied current density at the interface active material - current
    // collector These are the two Neumann Boundary conditions value applies at
    // the left and right extremities of the cell FIX ME: MICHAEL/HARI: could
    // you link g with these two values?
    const amrex::Real Applied_current_density_anode = Applied_current / surface_anodesolid_currentcollector;    // [A.m-2]
    const amrex::Real Applied_current_density_cathode = Applied_current / surface_cathodsolid_currentcollector; // [A.m-2]

    // Surface calculations // FIX ME: add surface calculations from Jeff
    // post-processing
    const amrex::Real surface_anodesolid_electrolyte = echemAMR::host_global_storage->anode_ele_SA;     // 4*pi*((5e-6)^2)
    const amrex::Real surface_cathodesolid_electrolyte = echemAMR::host_global_storage->cathode_ele_SA; // 4*pi*((5e-6)^2)

    // Initial current density at the interface active material - electrolyte
    // These are the two Neumann Boundary conditions value applied at the active
    // interface. At time=0s, value is uniform along the interface, then it will
    // vary locally while the surface integral ([A]) will be constant and equal
    // to "Applied_current" Any deviation from the equality: Applied_current =
    // integral(current density, ds active material-current collector) =
    // integral(current density, ds active material-electrolyte) will result in
    // an error in mass conservation
    const amrex::Real Activeinterface_current_density_anode = Applied_current / surface_anodesolid_electrolyte;     // [A.m-2]
    const amrex::Real Activeinterface_current_density_cathode = Applied_current / surface_cathodesolid_electrolyte; // [A.m-2]

    // Deduce potential values that verify this condition: Applied_current =
    // integral(current density, ds active material-current collector) =
    // integral(current density, ds active material-electrolyte)
    const amrex::Real phie_to =
        electrochem::get_phie_initial_from_anode(Activeinterface_current_density_anode, echemAMR::d_prob_parm->phi_reference, *echemAMR::d_prob_parm);
    const amrex::Real phi_c_to = electrochem::get_phic_initial_from_electrolyte(Activeinterface_current_density_cathode, phie_to, *echemAMR::d_prob_parm);

    amrex::Print() << std::endl;
    amrex::Print() << "CAPACITY" << std::endl;
    amrex::Print() << "Anode                     " << std::endl;
    amrex::Print() << "- Active material volume: " << volume_anode << " [m3] " << std::endl;
    amrex::Print() << "- Theoritical capacity  : " << theoritical_capacity_anode << " [C] " << std::endl;
    amrex::Print() << "- SOC delta range       : " << SOCrange_anode << std::endl;
    amrex::Print() << "- Pratical capacity     : " << pratical_capacity_anode << " [C] " << std::endl;
    amrex::Print() << "Cathode                   " << std::endl;
    amrex::Print() << "- Active material volume: " << volume_cathode << " [m3] " << std::endl;
    amrex::Print() << "- Theoritical capacity  : " << theoritical_capacity_cathode << " [C] " << std::endl;
    amrex::Print() << "- SOC delta range       : " << SOCrange_cathode << std::endl;
    amrex::Print() << "- Pratical capacity     : " << pratical_capacity_cathode << " [C] " << std::endl;
    amrex::Print() << "Cell                      " << std::endl;
    amrex::Print() << "- N/P ratio             : " << Cell_NP_ratio << std::endl;
    amrex::Print() << "- Pratical capacity     : " << pratical_capacity_cell << " [C] " << std::endl;
    amrex::Print() << std::endl;
    amrex::Print() << "CURRENT" << std::endl;
    amrex::Print() << "Cell                     " << std::endl;
    amrex::Print() << "- C-rate                : " << Crate << std::endl;
    amrex::Print() << "- Time to (dis)charge   : " << time_to_charge << " [s] " << std::endl;
    amrex::Print() << "- Applied current       : " << Applied_current << " [A] " << std::endl;
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
    amrex::Print() << "- Anode reference potential : " << echemAMR::d_prob_parm->phi_reference << " [V] " << std::endl;
    amrex::Print() << "- Electrolyte potential     : " << phie_to << " [V] " << std::endl;
    amrex::Print() << "- Cathode potential         : " << phi_c_to << " [V] " << std::endl;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void externalbc(
    const amrex::Real x[AMREX_SPACEDIM],
    const amrex::Real s_int[],
    amrex::Real s_ext[],
    const int idir,
    const int sgn,
    const amrex::Real time,
    amrex::GeometryData const& geomdata)
{
    // default to extrapolation
    for (int c = 0; c < NVAR; c++)
    {
        s_ext[c] = s_int[c];
    }

    if (sgn == 1) // left lo side
    {
        s_ext[CO_ID] = 0.0;
        s_ext[PO_ID] = 1.0;
    } else
    {
        s_ext[CO_ID] = 0.0;
        s_ext[PO_ID] = 0.0;
    }
}

#endif
