#ifndef _TRANSPORT_H_
#define _TRANSPORT_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <Chemistry.H>

using namespace amrex;
namespace electrochem_transport
{
    AMREX_GPU_DEVICE AMREX_INLINE
        void compute_dcoeff(int i, int j, int k,
                Array4<Real> const& phi,
                Array4<Real> const& dcoeff,
                GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                GpuArray<Real, AMREX_SPACEDIM> dx,
                const Real time,
                ProbParm const& prob_parm)
        {

            for(int n=0; n < phi.nComp(); ++n) {
                dcoeff(i,j,k,n) = 0.0;
            }

            const int comp = electrochem::level_set_to_component(i,j,k,phi);

            if(comp == electrochem::cathode){
                const Real Cs = phi(i,j,k,CO_ID); // cathode solid concentration
                dcoeff(i,j,k,CO_ID) = electrochem::Dc_bulk(Cs, prob_parm); // Cathode diffusion coefficient

            } else if (comp == electrochem::anode){
                const Real Cs = phi(i,j,k,CO_ID); // anode solid concentration
                dcoeff(i,j,k,CO_ID) = electrochem::Da_bulk(Cs, prob_parm); // Cathode diffusion coefficient

            } else {

                /* electrolyte or separator */
                const Real Ce = phi(i,j,k,CO_ID); // concentration

                const Real DeC = electrochem::De_C(Ce, prob_parm);
                const Real tplus = electrochem::t_plus(Ce, prob_parm);
                const Real KD = electrochem::K_D(Ce, prob_parm); // KD is either KD or KDstar as a function of prob_parm.use_KDstar

                if(prob_parm.use_KDstar) { 
                     dcoeff(i,j,k,CO_ID) = DeC - tplus*KD/prob_parm.Faraday_const; // QUESTION: The concentration flux is -dcoeff*grad(Ce) -t/F*conductivity coefficient*grad(phie). In the code, is -t/F*conductivity coefficient*grad(phie) considered ? 
                } else {
                     dcoeff(i,j,k,CO_ID) = 1.0; // Use the baseline Dec - t/F*Kd*grad(ln(Ce))
                }

            }

        }

    AMREX_GPU_DEVICE AMREX_INLINE
    void compute_potential_dcoeff(int i, int j, int k,
            Array4<Real> const& phi,
            Array4<Real> const& dcoeff,
            GpuArray<Real, AMREX_SPACEDIM> prob_lo,
            GpuArray<Real, AMREX_SPACEDIM> prob_hi,
            GpuArray<Real, AMREX_SPACEDIM> dx,
            const Real time,
            ProbParm const& prob_parm)
    {

        const int comp = electrochem::level_set_to_component(i,j,k,phi);

        if(comp == electrochem::cathode){
            const Real Cs = phi(i,j,k,CO_ID); // cathode solid concentration
            dcoeff(i,j,k) = electrochem::Kc_bulk(Cs, prob_parm); // Cathode diffusion coefficient
            // dcoeff(i,j,k) = prob_parm.K_c; // QUESTION: why dcoeff(i,j,k) and not dcoeff(i,j,k,CO_ID) as done for the diffusion coefficient?

        } else if (comp == electrochem::anode){
            const Real Cs = phi(i,j,k,CO_ID); // anode solid concentration
            dcoeff(i,j,k) = electrochem::Ka_bulk(Cs, prob_parm); // Cathode diffusion coefficient

        } else {

            /* electrolyte or separator */
            // QUESTION: this time we use the conductivity coefficient, while the term before the grad(phie) is much more complex
            const Real Ce = phi(i,j,k,CO_ID); // concentration
            dcoeff(i,j,k) = electrochem::KeC(phi(i,j,k,CO_ID), prob_parm); // QUESTION: the potential fluix is -dcoeff*grad(phie) + Kdstar*grad(Ce). In the code, is Kdstar*grad(Ce) considered?

        }


    }

    AMREX_GPU_DEVICE AMREX_INLINE
    void potential_bc(int i, int j, int k,
                      int dir, int sgn,
                      Array4<Real> const& phi,
                      Array4<Real> const& bc,
                      GpuArray<Real, AMREX_SPACEDIM> prob_lo,
                      GpuArray<Real, AMREX_SPACEDIM> prob_hi,
                      GpuArray<Real, AMREX_SPACEDIM> dx,
                      const Real time,
                      const Real bclo,
                      const Real bchi)
    {

        const int im1 = (dir == 0) ? i-1 : i;
        const int jm1 = (dir == 1) ? j-1 : j;
        const int km1 = (dir == 2) ? k-1 : k;

        bool electrode = false;
        if(sgn == -1) { // lo sides
            electrode  = electrochem::is_electrode(i,j,k,phi);
            bc(im1,jm1,km1) = (electrode) ? bclo : 0.0;
        } else { // hi sides
            electrode = electrochem::is_electrode(im1,jm1,km1,phi);
            bc(i,j,k) = (electrode) ? bchi : 0.0;
        }

    }


    AMREX_GPU_DEVICE AMREX_INLINE
    void compute_velx(int i, int j, int k,
            Array4<Real> const& phi,
            Array4<Real> const& velx,
            GpuArray<Real, AMREX_SPACEDIM> prob_lo,
            GpuArray<Real, AMREX_SPACEDIM> prob_hi,
            GpuArray<Real, AMREX_SPACEDIM> dx,
            const Real time,
            ProbParm const& prob_parm)
    {
        for(int n = 0; n < phi.nComp(); ++n) {
            velx(i,j,k,n) = 0.0;
        }

        // FIXME: this is not in the correct form would need to be grad(Ce*tplus/F*EFX)
        // also grad(j_e) = 0 anyways...
//        if(!electrochem::is_electrode(i,j,k,phi)){
//            const Real Ce = phi(i,j,k,CO_ID);
//            const Real tplus = electrochem::t_plus(Ce, prob_parm);
//            velx(i,j,k,CO_ID) = tplus/prob_parm.Faraday_const * phi(i,j,k,EFX_ID);
//        }

    }

    AMREX_GPU_DEVICE AMREX_INLINE
    void compute_vely(int i, int j, int k,
            Array4<Real> const& phi,
            Array4<Real> const& vely,
            GpuArray<Real, AMREX_SPACEDIM> prob_lo,
            GpuArray<Real, AMREX_SPACEDIM> prob_hi,
            GpuArray<Real, AMREX_SPACEDIM> dx,
            const Real time,
            ProbParm const& prob_parm)
    {
        for(int n = 0; n < phi.nComp(); ++n) {
            vely(i,j,k,n) = 0.0;
        }

//        if(!electrochem::is_electrode(i,j,k,phi)){
//            const Real Ce = phi(i,j,k,CO_ID);
//            const Real tplus = electrochem::t_plus(Ce, prob_parm);
//            vely(i,j,k,CO_ID) = tplus/prob_parm.Faraday_const * phi(i,j,k,EFY_ID);
//        }
    }

    AMREX_GPU_DEVICE AMREX_INLINE
    void compute_velz(int i, int j, int k,
            Array4<Real> const& phi,
            Array4<Real> const& velz,
            GpuArray<Real, AMREX_SPACEDIM> prob_lo,
            GpuArray<Real, AMREX_SPACEDIM> prob_hi,
            GpuArray<Real, AMREX_SPACEDIM> dx,
            const Real time,
            ProbParm const& prob_parm)
    {
        for(int n = 0; n < phi.nComp(); ++n) {
            velz(i,j,k,n) = 0.0;
        }

//        if(!electrochem::is_electrode(i,j,k,phi)){
//            const Real Ce = phi(i,j,k,CO_ID);
//            const Real tplus = electrochem::t_plus(Ce, prob_parm);
//            velz(i,j,k,CO_ID) = tplus/prob_parm.Faraday_const * phi(i,j,k,EFZ_ID);
//        }
    }

}
#endif


