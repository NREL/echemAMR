#ifndef _CHEMISTRY_H_
#define _CHEMISTRY_H_

#include<AMReX_REAL.H>
#include<AMReX.H>
#include<string>
#include<AMReX_Vector.H>

namespace electrochem
{
    extern const int nspecies;
    extern amrex::Vector<std::string> specnames;
    void init();
    void close();

}
#endif
