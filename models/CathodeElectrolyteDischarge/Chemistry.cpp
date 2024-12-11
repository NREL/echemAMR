#include<Chemistry.H>

namespace electrochem
{
    amrex::Vector<std::string> specnames(NVAR);

    void init()
    {
        specnames[LI_ID]="Li";
        specnames[LS_ID]="levelset";
        specnames[EFX_ID] = "Efieldx";
        specnames[EFY_ID] = "Efieldy";
        specnames[EFZ_ID] = "Efieldz";
        specnames[POT_ID] = "Potential";
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
