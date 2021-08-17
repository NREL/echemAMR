#ifndef _CHEMISTRY_H_
#define _CHEMISTRY_H_

#include<AMReX_REAL.H>
#include<AMReX.H>
#include<string>
#include<AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Box.H>

#define NUM_SPECIES 1
#define S1_ID 0
#define EFX_ID 1
#define EFY_ID 2
#define EFZ_ID 3
#define POT_ID 4
#define NVAR NUM_SPECIES+4 

namespace electrochem
{
    extern amrex::Vector<std::string> specnames;
    void init();
    void close();
    int find_id(std::string specname);
}
#endif
