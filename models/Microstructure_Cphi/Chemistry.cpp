#include<Chemistry.H>

namespace electrochem
{
    amrex::Vector<std::string> specnames(NVAR);

    void init()
    {

        // NO SPACE IN NAMES - OR PARAVIEW WILL NOT READ FILE

        // Degree of freedom
        specnames[CO_ID] = "Concentration";
        specnames[POTs_ID] = "Solid_potential"; // Used only if CBD mixed domain (prob.CBD_transport = 1)
        // Domain
        specnames[A_AM_ID]  = "Anode_active_material";
        specnames[A_E_ID]  = "Anode_electrolyte";
        specnames[A_CBD_ID]  = "Anode_CBD";
        specnames[S_ID]  = "Separator";
        specnames[C_AM_ID]  = "Cathode_active_material";
        specnames[C_E_ID]  = "Cathode_electrolyte";
        specnames[C_CBD_ID]  = "Cathode_CBD";
        // Coefficients/parameter fields
        specnames[NP_ID] = "Nanoporosity";         
        specnames[MAC_ID] = "MacMullin_number";        
        // Level set
        specnames[LS_ID] = "levelset";
        specnames[EFX_ID] = "Efieldx";
        specnames[EFY_ID] = "Efieldy";
        specnames[EFZ_ID] = "Efieldz";
        specnames[POT_ID] = "Potential";
        specnames[DIS_U_ID] = "Displacement_u";
        specnames[DIS_V_ID] = "Displacement_v";
        specnames[DIS_W_ID] = "Displacement_w";
        specnames[VON_M_ID] = "von_Mises";
        specnames[Sigma11_ID] = "Sigma11";
        specnames[Sigma22_ID] = "Sigma22";
        specnames[Sigma33_ID] = "Sigma33";
        specnames[Sigma12_ID] = "Sigma12";
        specnames[Sigma13_ID] = "Sigma13";
        specnames[Sigma23_ID] = "Sigma23";                                                
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
