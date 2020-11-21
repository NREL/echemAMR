#include<Chemistry.H>

namespace electrochem
{
    amrex::Vector<std::string> specnames(nspecies);

    void init()
    {
        specnames[CO_ID]="Concentration";
        specnames[PO_ID]="Potential";
        specnames[AC_ID]="Electrode";
        specnames[ES_ID]="Electrolyte";

    }    
    void close()
    {
        specnames.clear();
    }
    int find_id(std::string specname)
    {
        int loc=-1;
        auto it=std::find(specnames.begin(),specnames.end(),specname);
        if(it != specnames.end())
        {
            loc=it-specnames.begin();
        }
        return(loc);
    }
}
