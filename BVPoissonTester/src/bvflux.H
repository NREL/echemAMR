Array<MultiFab,AMREX_SPACEDIM> bv_explicit_terms;
for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
{
    bv_explicit_terms[idim].define(amrex::convert(grids,IntVect::TheDimensionVector(idim)), dmap, 1, 0);
}

for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
{
    for (MFIter mfi(phi); mfi.isValid(); ++mfi)
    {
        const auto dx = geom.CellSizeArray();
        const Box& bx = mfi.tilebox();
        Real min_dx = amrex::min(dx[0],amrex::min(dx[1],dx[2]));

        //face box
        Box fbox=convert(bx,IntVect::TheDimensionVector(idim));
        Array4<Real> phi_arr       = phi.array(mfi);
        Array4<Real> dcoeff_arr    = bcoef[idim].array(mfi);
        Array4<Real> ls_arr        = levset.array(mfi);
        Array4<Real> explterms_arr = bv_explicit_terms[idim].array(mfi);

        amrex::ParallelFor(fbox,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
                IntVect left(i,j,k);
                IntVect right(i,j,k);

                IntVect top_left(i,j,k);
                IntVect bottom_left(i,j,k);
                IntVect top_right(i,j,k);
                IntVect bottom_right(i,j,k);

                IntVect front_left(i,j,k);
                IntVect back_left(i,j,k);
                IntVect front_right(i,j,k);
                IntVect back_right(i,j,k);

                int normaldir=idim;
                int trans1dir=(idim+1)%AMREX_SPACEDIM;
                int trans2dir=(idim+2)%AMREX_SPACEDIM;

                left[idim]              -= 1;
                top_left[idim]          -= 1;
                front_left[idim]        -= 1;

                top_left[trans1dir]     += 1;
                top_right[trans1dir]    += 1;
                bottom_left[trans1dir]  -= 1;
                bottom_right[trans1dir] -= 1;

                front_left[trans2dir]   += 1;
                front_right[trans2dir]  += 1;
                back_left[trans2dir]    -= 1;
                back_right[trans2dir]   -= 1;

                //levelset color
                Real c_left  = ls_arr(left);
                Real c_right = ls_arr(right);

                Real c_top = 0.5*(ls_arr(top_left)    +    ls_arr(top_right));
                Real c_bot = 0.5*(ls_arr(bottom_left) + ls_arr(bottom_right));

                Real c_frnt = 0.5*(ls_arr(front_left)  + ls_arr(front_right));
                Real c_back = 0.5*(ls_arr(back_left)   +  ls_arr(back_right));

                //phi
                Real pot_left  = phi_arr(left);
                Real pot_right = phi_arr(right);

                Real pot_top = 0.5*(phi_arr(top_left)    +    phi_arr(top_right));
                Real pot_bot = 0.5*(phi_arr(bottom_left) + phi_arr(bottom_right));

                Real pot_frnt = 0.5*(phi_arr(front_left)  + phi_arr(front_right));
                Real pot_back = 0.5*(phi_arr(back_left)   +  phi_arr(back_right));

                //x,y or z
                Real dcdn  = (c_right-c_left)/dx[normaldir];
                Real dcdt1 = (c_top-c_bot)/(2.0*dx[trans1dir]);
                Real dcdt2 = (c_frnt-c_back)/(2.0*dx[trans2dir]);

                //grad of potential
                Real dphidn  = (pot_right-pot_left)/dx[normaldir];
                Real dphidt1 = (pot_top-pot_bot)/(2.0*dx[trans1dir]);
                Real dphidt2 = (pot_frnt-pot_back)/(2.0*dx[trans2dir]);

                Real gradc_tolfac=1e-4;
                Real gradc_max=1.0/min_dx; //maximum gradient possible on the current grid
                Real gradc_cutoff=gradc_tolfac*gradc_max;

                Real mod_gradc = amrex::max(sqrt(dcdn*dcdn + dcdt1*dcdt1 + dcdt2*dcdt2),gradc_cutoff);

                //jump along the level set normal (phi_electrolyte-phi_electrode)
                Real phi_jump  = (dphidn*dcdn + dphidt1*dcdt1 + dphidt2*dcdt2)/pow(mod_gradc,2.0);

                //FIXME: pass ion concentration also
                //FIXME: ideally it should be the ion concentration at the closest electrode cell
                Real j_bv = bvcurrent(phi_jump);
                Real jdash_bv = bvcurrent_der(phi_jump);

                Real activ_func = bv_activation_function(0.5*(c_right+c_left),
                        mod_gradc, gradc_cutoff);

                dcoeff_arr(i,j,k) *= (1.0-activ_func);
                dcoeff_arr(i,j,k) += -jdash_bv*activ_func/pow(mod_gradc,3.0) * dcdn*dcdn;

                //expl term1
                explterms_arr(i,j,k) =   j_bv*activ_func*dcdn/mod_gradc;

                //expl term2
                explterms_arr(i,j,k) += -jdash_bv*phi_jump*activ_func*dcdn/mod_gradc; 

                //expl term3 (mix derivative terms from tensor product)
                explterms_arr(i,j,k) +=  jdash_bv*activ_func/pow(mod_gradc,3.0)*(dcdn*dcdt1*dphidt1+dcdn*dcdt2*dphidt2); 
                });

    }
}
for (MFIter mfi(phi); mfi.isValid(); ++mfi)
{
    const Box& bx = mfi.tilebox();
    const auto dx = geom.CellSizeArray();

    Array4<Real> rhs_arr      = rhs.array(mfi);

    Array4<Real> term_x = bv_explicit_terms[0].array(mfi);
    Array4<Real> term_y = bv_explicit_terms[1].array(mfi);
    Array4<Real> term_z = bv_explicit_terms[2].array(mfi);

    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k)
    {
       rhs_arr(i,j,k)  =     (term_x(i,j,k) - term_x(i+1,j,k)) / dx[0] 
                         +   (term_y(i,j,k) - term_y(i,j+1,k)) / dx[1] 
                         +   (term_z(i,j,k) - term_z(i,j,k+1)) / dx[2];

    });
}
