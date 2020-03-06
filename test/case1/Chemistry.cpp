#include<Chemistry.H>

namespace electrochem
{
    const int nspecies=2;
    amrex::Vector<std::string> specnames(2);

    void init()
    {
        specnames[0]="S1";
        specnames[1]="S2";
    }    
    void close()
    {
        specnames.clear();
    }
}
