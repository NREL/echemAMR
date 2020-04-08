#ifndef _CHEMISTRY_H_
#define _CHEMISTRY_H_

#include<AMReX_REAL.H>
#include<AMReX.H>
#include<string>
#include<AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Box.H>

namespace electrochem
{
    extern const int nspecies;
    extern const int S1_ID,S2_ID;
    extern amrex::Vector<std::string> specnames;
    void init();
    void close();
    int find_id(std::string specname);

}
#endif
