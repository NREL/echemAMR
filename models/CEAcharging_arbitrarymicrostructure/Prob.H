#ifndef _PROB_H_
#define _PROB_H_

#include <AMReX_Box.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <Chemistry.H>
#include <ProbParm.H>
#include <echemAMR_constants.H>
#include <echemAMR.H>

using namespace amrex;

AMREX_INLINE
void amrex_probinit (ProbParm & h_prob_parm, ProbParm & d_prob_parm)
{
    amrex::Vector<amrex::Real> plo;
    amrex::Vector<amrex::Real> phi;

    amrex::Real lenx,leny,lenz,vol;
    amrex::Real phi_c,phi_a,phi_cell,time_charge;

    // read problem specific parmparse parameters here
    amrex::ParmParse pp("prob");
    pp.query("interface1", h_prob_parm.int1 );
    pp.query("interface2", h_prob_parm.int2 );
    pp.query("crate",h_prob_parm.crate);

    amrex::ParmParse ppg("geometry");
    ppg.getarr("prob_lo",plo);
    ppg.getarr("prob_hi",phi);

    lenx=phi[0]-plo[0];
    leny=phi[1]-plo[1];
    lenz=phi[2]-plo[2];

    vol = lenx*leny*lenz;

    phi_c  =  h_prob_parm.int1 * vol * h_prob_parm.Csmax_c
           * (h_prob_parm.SOC_max_c-h_prob_parm.SOC_min_c) * FARADCONST;


    phi_a =  (1.0 - h_prob_parm.int2) * vol * h_prob_parm.Csmax_a
           * (h_prob_parm.SOC_max_a-h_prob_parm.SOC_min_a) * FARADCONST;

    phi_cell = amrex::min(phi_c,phi_a);
    amrex::Print()<<"phi_c,phi_a:"<<phi_c<<"\t"<<phi_a<<"\t"<<phi_cell<<"\n";
    time_charge = 3600.0/h_prob_parm.crate;

    amrex::Real current_dens = (phi_cell/time_charge)/(leny*lenz);
    amrex::Print()<<"current_dens:"<<current_dens<<"\n";

    echemAMR::host_global_storage->pot_bc_lo[0]=-current_dens/h_prob_parm.cathode_conductivity;

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
                    phi(i,j,k,E_ID) = 1; // phi(i,j,k,S_ID) = 1;
                    phi(i,j,k,LS_ID) = 1; // phi(i,j,k,LS_ID) = 1;
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

    for (int k = lo.z; k <= hi.z; ++k) 
    {
        for (int j = lo.y; j <= hi.y; ++j)
        {
            for (int i = lo.x; i <= hi.x; ++i) 
            {
                 if (phi(i,j,k,C_ID) == 1) { // Within cathode
                     phi(i,j,k,CO_ID) = prob_parm->C_c_init;
                     phi(i,j,k,POT_ID) = 4.0;
                 } else if (phi(i,j,k,A_ID) == 1) { // Within anode
                     phi(i,j,k,CO_ID) = prob_parm->C_a_init;
                     phi(i,j,k,POT_ID) = 0.3;
                 } else if (phi(i,j,k,E_ID) == 1) {// Within electrolyte // else if ((phi(i,j,k,E_ID) == 1) || (phi(i,j,k,S_ID) == 1)) {// Within electrolyte
                     phi(i,j,k,CO_ID) = prob_parm->C_e_init;
                     phi(i,j,k,POT_ID) = 0.0;
                 }
            }
        }
    }
}





    AMREX_INLINE
void print_init_data(ProbParm* prob_parm)
{

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
