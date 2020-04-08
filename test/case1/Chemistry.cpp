#include<Chemistry.H>

namespace electrochem
{
    const int nspecies=2;
    amrex::Vector<std::string> specnames(2);

    const int S1_ID=0;
    const int S2_ID=1; 

    void init()
    {
        specnames[S1_ID]="S1";
        specnames[S2_ID]="S2";
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
