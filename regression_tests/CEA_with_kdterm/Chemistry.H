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

#define NUM_SPECIES 5
#define CO_ID   0
#define A_ID    1
#define C_ID    2
#define E_ID    3
#define LS_ID   4
#define EFX_ID  5
#define EFY_ID  6
#define EFZ_ID  7
#define POT_ID  8
#define NVAR NUM_SPECIES+4 

namespace electrochem
{
    extern amrex::Vector<std::string> specnames;
    void init();
    void close();
    int find_id(std::string specname);
}
#endif
