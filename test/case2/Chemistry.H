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
    const int nspecies=2;
    const int S1_ID=0;
    const int S2_ID=1;
    extern amrex::Vector<std::string> specnames;
    void init();
    void close();
    int find_id(std::string specname);

}
#endif
