#include<Chemistry.H>

namespace electrochem
{
    amrex::Vector<std::string> specnames(NUM_SPECIES);

    void init()
    {
        specnames[S1_ID]="S1";
        specnames[LS_ID]="levelset";
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