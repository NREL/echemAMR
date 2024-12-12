#ifndef _CHEMISTRY_H_
#define _CHEMISTRY_H_

#include<AMReX_REAL.H>
#include<AMReX.H>
#include<string>
#include<AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Box.H>
#include <echemAMR_constants.H>

#define NUM_SPECIES 7
#define NUM_SURF_REACTIONS 2
#define NUM_ACTIVE_SPECIES 4

//species CO2,CO,OH-,HCOO-
#define CO2_ID   0
#define CO_ID    1
#define OHM_ID   2
#define HCOOM_ID 3

//level sets (cathode,electrolyte,levelset)
#define C_ID     4
#define E_ID     5
#define LS_ID    6

//electric fields, potential
#define EFX_ID  7
#define EFY_ID  8
#define EFZ_ID  9
#define POT_ID  10

#define NVAR NUM_SPECIES+4 

namespace electrochem
{
    extern amrex::Vector<std::string> specnames;
    void init();
    void close();
    int find_id(std::string specname);
}
#endif
