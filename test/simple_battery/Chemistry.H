#ifndef _CHEMISTRY_H_
#define _CHEMISTRY_H_

#include<AMReX_REAL.H>
#include<AMReX.H>
#include<string>
#include<AMReX_Vector.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Box.H>

#define NUM_SPECIES 6
#define CO_ID   0 //conc
#define A_ID    1 //anode
#define C_ID    2 //cathode
#define E_ID    3 //electrolyte
#define S_ID    4 //separator
#define LS_ID   5
#define EFX_ID  6
#define EFY_ID  7
#define EFZ_ID  8 
#define POT_ID  9
#define NVAR NUM_SPECIES+4 

#define FARADCONST 96387.0
#define GASCONST   8.31446261815324

namespace electrochem
{
    extern amrex::Vector<std::string> specnames;
    void init();
    void close();
    int find_id(std::string specname);
}
#endif
