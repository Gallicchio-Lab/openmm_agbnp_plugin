/* -------------------------------------------------------------------------- *
 *                               OpenMM-AGBNP                                *
 * -------------------------------------------------------------------------- */

#include <iostream>
#include <cstdlib>
#include <cmath>
//#include "openmm/reference/SimTKOpenMMRealType.h"
#include "AGBNPUtils.h"
#include "ReferenceAGBNPKernels.h"
#include "AGBNPForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/SplineFitter.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"
#include "gaussvol.h"


using namespace AGBNPPlugin;
using namespace OpenMM;
using namespace std;

static vector<RealVec>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->positions);
}

static vector<RealVec>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->forces);
}


/* a switching function for the inverse born radius (beta)
   so that if beta is negative -> beta' = minbeta
*/ 
 static RealOpenMM agbnp_swf_invbr(RealOpenMM beta, RealOpenMM& fp){
  /* the maximum born radius is max reach of Q4 lookup table */
  static const RealOpenMM  a  = 1./AGBNP_I4LOOKUP_MAXA;
  static const RealOpenMM  a2 = 1./(AGBNP_I4LOOKUP_MAXA*AGBNP_I4LOOKUP_MAXA);

  RealOpenMM t;
  if(beta<0.0){
    t = a;
    fp = 0.0;
  }else{
    t = sqrt(a2 + beta*beta);
    fp  = beta/t;
  }
  return t;
}

// Initializes AGBNP library
void ReferenceCalcAGBNPForceKernel::initialize(const System& system, const AGBNPForce& force) {
   

    numParticles = force.getNumParticles();

    //set version
    version = force.getVersion();
    
    //input lists
    positions.resize(numParticles);
    radii_large.resize(numParticles);//van der Waals radii + offset (large radii)
    radii_vdw.resize(numParticles);//van der Waals radii (small radii)
    gammas.resize(numParticles);
    vdw_alpha.resize(numParticles);
    charge.resize(numParticles);
    ishydrogen.resize(numParticles);

    //output lists
    free_volume.resize(numParticles);
    self_volume.resize(numParticles);
    surface_areas.resize(numParticles);
    vol_force.resize(numParticles);
    
    vector<double> vdwrad(numParticles);
    for (int i = 0; i < numParticles; i++){
      double r, g, alpha, q;
      bool h;
      force.getParticleParameters(i, r, g, alpha, q, h);
      radii_large[i] = r + AGBNP_RADIUS_INCREMENT;
      radii_vdw[i] = r;
      vdwrad[i] = r; //double version for lookup table setup
      gammas[i] = g;
      if(h) gammas[i] = 0.0;
      vdw_alpha[i] = alpha;
      charge[i] = q;
      ishydrogen[i] = h;
    }

    //create and saves GaussVol instance
    //loads some initial values for radii and gamma but they will be
    //reset in execute()
    gvol = new GaussVol(numParticles, radii_large, gammas, ishydrogen);

    //initializes I4 lookup table for Born-radii calculation
    double rmin = 0.;
    double rmax = AGBNP_I4LOOKUP_MAXA;
    int i4size = AGBNP_I4LOOKUP_NA;
    i4_lut = new AGBNPI42DLookupTable(vdwrad, ishydrogen, i4size, rmin, rmax, version);

    //volume scaling factors and born radii
    volume_scaling_factor.resize(numParticles);
    inverse_born_radius.resize(numParticles);
    inverse_born_radius_fp.resize(numParticles);
    born_radius.resize(numParticles);
}

double ReferenceCalcAGBNPForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
  double energy = 0.0;
  if(version == 0){
    energy = executeGVolSA(context, includeForces, includeEnergy);
  }else if(version == 1){
    energy = executeAGBNP1(context, includeForces, includeEnergy);
  }else if(version == 2){
    energy = executeAGBNP2(context, includeForces, includeEnergy);
  }
  return energy;
}


double ReferenceCalcAGBNPForceKernel::executeGVolSA(ContextImpl& context, bool includeForces, bool includeEnergy) {

  //sequence: volume1->volume2

  
  //weights
    RealOpenMM w_evol = 1.0;
    
    vector<RealVec>& pos = extractPositions(context);
    vector<RealVec>& force = extractForces(context);
    RealOpenMM energy = 0.0;
    int verbose_level = 0;
    int init = 0; 

    vector<RealOpenMM> nu(numParticles);


    if(verbose_level > 0) cout << "Executing GVolSA" << endl;
    
    if(verbose_level > 0){
      cout << "-----------------------------------------------" << endl;
    } 

    
    // volume energy function 1
    RealOpenMM volume1, vol_energy1;
    for(int i = 0; i < numParticles; i++){
      nu[i] = gammas[i]/SA_DR;
    }
    gvol->set_radii(radii_large);
    gvol->set_gammas(nu);
    gvol->compute_tree(pos);
    gvol->compute_volume(pos, volume1, vol_energy1, vol_force, free_volume, self_volume);
    
    //returns energy and gradients from volume energy function
    for(int i = 0; i < numParticles; i++){
      force[i] += vol_force[i] * w_evol;
    }
    energy += vol_energy1 * w_evol;
    if(verbose_level > 0){
      cout << "Volume energy 1: " << vol_energy1 << endl;
    }

    // volume energy function 2 (small radii)
    RealOpenMM vol_energy2, volume2;
    for(int i = 0; i < numParticles; i++){
      nu[i] = -gammas[i]/SA_DR;
    }
    gvol->rescan_tree_volumes(pos, radii_vdw, nu);
    gvol->compute_volume(pos, volume2, vol_energy2, vol_force, free_volume, self_volume);
    for(int i = 0; i < numParticles; i++){
      force[i] += vol_force[i] * w_evol;
    }
    energy += vol_energy2 * w_evol;
    if(verbose_level > 0){
      cout << "Volume energy 2: " << vol_energy2 << endl;
      cout << "Surface area energy: " << vol_energy1 + vol_energy2 << endl;
    }
    
    //returns energy
    return (double)energy;
}


double ReferenceCalcAGBNPForceKernel::executeAGBNP1(ContextImpl& context, bool includeForces, bool includeEnergy) {

  //sequence: volume1->volume2->GB->VdW->volume2derivatives

  
    //weights
    RealOpenMM w_evol = 1.0, w_egb = 1.0, w_vdw = 1.0;
    
    vector<RealVec>& pos = extractPositions(context);
    vector<RealVec>& force = extractForces(context);
    RealOpenMM energy = 0.0;
    int verbose_level = 0;
    int init = 0; 

    vector<RealOpenMM> nu(numParticles);


    if(verbose_level > 0) cout << "Executing AGBNP1" << endl;
    
    if(verbose_level > 0){
      cout << "-----------------------------------------------" << endl;
    } 

    
    // volume energy function 1
    RealOpenMM volume1, vol_energy1;
    for(int i = 0; i < numParticles; i++){
      nu[i] = gammas[i]/SA_DR;
    }
    gvol->set_radii(radii_large);
    gvol->set_gammas(nu);
    gvol->compute_tree(pos);
    gvol->compute_volume(pos, volume1, vol_energy1, vol_force, free_volume, self_volume);
    
    //returns energy and gradients from volume energy function
    for(int i = 0; i < numParticles; i++){
      force[i] += vol_force[i] * w_evol;
    }
    energy += vol_energy1 * w_evol;
    if(verbose_level > 0){
      cout << "Volume energy 1: " << vol_energy1 << endl;
    }

    // volume energy function 2 (small radii)
    RealOpenMM vol_energy2, volume2;
    for(int i = 0; i < numParticles; i++){
      nu[i] = -gammas[i]/SA_DR;
    }
    gvol->rescan_tree_volumes(pos, radii_vdw, nu);
    gvol->compute_volume(pos, volume2, vol_energy2, vol_force, free_volume, self_volume);
    for(int i = 0; i < numParticles; i++){
      force[i] += vol_force[i] * w_evol;
    }
    energy += vol_energy2 * w_evol;
    if(verbose_level > 0){
      cout << "Volume energy 2: " << vol_energy2 << endl;
      cout << "Surface area energy: " << vol_energy1 + vol_energy2 << endl;
    }
    
    //now overlap tree is set up with small radii

    
#ifdef NOTNOW
    vector<int> nov, nov_2body;
    gvol->getstat(nov, nov_2body);    
    int nn = 0, nn2 = 0;
    for(int i = 0; i < nov.size(); i++){
      nn += nov[i];
     nn2 += nov_2body[i];
    }
    cout << "Noverlaps: " << nn << " " << nn2 << endl;
#endif

#ifdef NOTNOW    
    {
      //tests i4 lookup table
      double dd = 0.12;
      double dmin = 0.;
      double dmax = AGBNP_I4LOOKUP_MAXA/0.165;
      for(int i=0;i<100;i++){
	double x = dmin + i*dd;
	double y = (x < dmax) ? i4_lut->eval(x, 1.0) : 0.;
	double yp = (x < dmax) ? i4_lut->evalderiv(x, 1.0) : 0.;
	cout << "i4: " << x << " " << y << " " << yp << endl;
      }
    }
#endif

    //volume scaling factors from self volumes (with small radii)
    double tot_vol = 0;
    for(int i = 0; i < numParticles; i++){
      RealOpenMM rad = radii_vdw[i];
      RealOpenMM vol = (4./3.)*M_PI*rad*rad*rad;
      volume_scaling_factor[i] = self_volume[i]/vol;
      if(verbose_level > 4){
	cout << "SV " << i << " " << self_volume[i] << endl;
      }
      tot_vol += self_volume[i];
    }
    if(verbose_level > 0){
      cout << "Volume from self volumes: " << tot_vol << endl;
    }

    RealOpenMM pifac = 1./(4.*M_PI);

    //compute inverse Born radii, prototype, no cutoff
    for(int i = 0; i < numParticles; i++){
      inverse_born_radius[i] = 1./radii_vdw[i];
      for(int j = 0; j < numParticles; j++){
	if(i == j) continue;
	if(ishydrogen[j]) continue;
	RealVec dist = pos[j] - pos[i];
	RealOpenMM d = sqrt(dist.dot(dist));
	if(d < AGBNP_I4LOOKUP_MAXA){
	  int rad_typei = i4_lut->radius_type_screened[i];
	  int rad_typej = i4_lut->radius_type_screener[j];
	  inverse_born_radius[i] -= pifac*volume_scaling_factor[j]*i4_lut->eval(d, rad_typei, rad_typej);
	}	
      }
      RealOpenMM fp;
      born_radius[i] = 1./agbnp_swf_invbr(inverse_born_radius[i], fp);
      inverse_born_radius_fp[i] = fp;
    }

    if(verbose_level > 4){
      cout << "Born radii:" << endl;
      RealOpenMM fp;
      for(int i = 0; i < numParticles; i++){
	cout << "BR " << i << " " << 10.*born_radius[i] << " Si " << volume_scaling_factor[i] << endl;
      }
    }

    //GB energy
    RealOpenMM dielectric_in = 1.0;
    RealOpenMM dielectric_out= 80.0;
    RealOpenMM tokjmol = 4.184*332.0/10.0; //the factor of 10 is the conversion of 1/r from nm to Ang
    RealOpenMM dielectric_factor = tokjmol*(-0.5)*(1./dielectric_in - 1./dielectric_out);
    RealOpenMM pt25 = 0.25;
    vector<RealOpenMM> egb_der_Y(numParticles);
    for(int i = 0; i < numParticles; i++){
      egb_der_Y[i] = 0.0;
    }
    RealOpenMM gb_self_energy = 0.0;
    RealOpenMM gb_pair_energy = 0.0;
    for(int i = 0; i < numParticles; i++){
      double uself = dielectric_factor*charge[i]*charge[i]/born_radius[i];
      gb_self_energy += uself;
      for(int j = i+1; j < numParticles; j++){
	RealVec dist = pos[j] - pos[i];
	RealOpenMM d2 = dist.dot(dist);
	RealOpenMM qqf = charge[j]*charge[i];
	RealOpenMM qq = dielectric_factor*qqf;
	RealOpenMM bb = born_radius[i]*born_radius[j];
	RealOpenMM etij = exp(-pt25*d2/bb);
	RealOpenMM fgb = 1./sqrt(d2 + bb*etij);
	RealOpenMM egb = 2.*qq*fgb;
	gb_pair_energy += egb;
	RealOpenMM fgb3 = fgb*fgb*fgb;
	RealOpenMM mw = -2.0*qq*(1.0-pt25*etij)*fgb3;
	RealVec g = dist * mw;
	force[i] += g * w_egb;
	force[j] -= g * w_egb;
	RealOpenMM ytij = qqf*(bb+pt25*d2)*etij*fgb3;
	egb_der_Y[i] += ytij;
	egb_der_Y[j] += ytij;
      }
    }
    if(verbose_level > 0){
      cout << "GB self energy: " << gb_self_energy << endl;
      cout << "GB pair energy: " << gb_pair_energy << endl;
      cout << "GB energy: " << gb_pair_energy+gb_self_energy << endl;
    }
    energy += w_egb*gb_pair_energy + w_egb*gb_self_energy;

    if(verbose_level > 0){
      cout << "Y parameters: " << endl;
      for(int i = 0; i < numParticles; i++){
	cout << "Y: " << i << " " << egb_der_Y[i] << endl;
      }
    }

    //compute van der Waals energy
    RealOpenMM evdw = 0.;
    for(int i=0;i<numParticles; i++){
      evdw += vdw_alpha[i]/pow(born_radius[i]+AGBNP_HB_RADIUS,3);
    }
    if(verbose_level > 0){
      cout << "Van der Waals energy: " << evdw << endl;
    }
    energy += w_vdw*evdw;

    //compute atom-level property for the calculation of the gradients of Evdw and Egb
    vector<RealOpenMM> evdw_der_brw(numParticles);
    for(int i = 0; i < numParticles; i++){
      RealOpenMM br = born_radius[i];
      evdw_der_brw[i] = -pifac*3.*vdw_alpha[i]*br*br*inverse_born_radius_fp[i]/pow(br+AGBNP_HB_RADIUS,4);
    }
    if(verbose_level > 3){
      cout << "BrW parameters: " << endl;
      for(int i = 0; i < numParticles; i++){
	cout << "BrW: " << i << " " << evdw_der_brw[i] << endl;      
      }
    }

    
    vector<RealOpenMM> egb_der_bru(numParticles);
    for(int i = 0; i < numParticles; i++){
      RealOpenMM br = born_radius[i];
      RealOpenMM qi = charge[i];
      egb_der_bru[i] = -pifac*dielectric_factor*(qi*qi + egb_der_Y[i]*br)*inverse_born_radius_fp[i];
    }
    
    if(verbose_level > 3){
      cout << "BrU parameters: " << endl;
      for(int i = 0; i < numParticles; i++){
	cout << "BrU: " << i << " " << egb_der_bru[i] << endl;      
      }
    }
    
    
    //compute the component of the gradients of the van der Waals and GB energies due to
    //variations of Born radii
    //also accumulates W's and U's for self-volume components of the gradients later
    vector<RealOpenMM> evdw_der_W(numParticles);
    vector<RealOpenMM> egb_der_U(numParticles);
    for(int i = 0; i < numParticles; i++){
      evdw_der_W[i] = egb_der_U[i] = 0.0;
    }
    for(int i = 0; i < numParticles; i++){
      for(int j = 0; j < numParticles; j++){
	if(i == j) continue;
	if(ishydrogen[j]) continue;
	RealVec dist = pos[j] - pos[i];
	RealOpenMM d = sqrt(dist.dot(dist));
	double Qji = 0.0, dQji = 0.0;
	// Qji: j descreens i
	if(d < AGBNP_I4LOOKUP_MAXA){
	  int rad_typei = i4_lut->radius_type_screened[i];
	  int rad_typej = i4_lut->radius_type_screener[j];
	  Qji = i4_lut->eval(d, rad_typei, rad_typej); 
	  dQji = i4_lut->evalderiv(d, rad_typei, rad_typej); 
	}
	RealVec w;
	//van der Waals stuff
	evdw_der_W[j] += evdw_der_brw[i]*Qji;
	w = dist * evdw_der_brw[i]*volume_scaling_factor[j]*dQji/d;
	force[i] += w * w_vdw;
	force[j] -= w * w_vdw;
	//GB stuff
	egb_der_U[j] += egb_der_bru[i]*Qji;
	w = dist * egb_der_bru[i]*volume_scaling_factor[j]*dQji/d;
	force[i] += w * w_egb;
	force[j] -= w * w_egb;
      }
    }

    

    if(verbose_level > 3){
      cout << "U parameters: " << endl;
      for(int i = 0; i < numParticles; i++){
	RealOpenMM vol = 4.*M_PI*pow(radii_vdw[i],3)/3.0; 
	cout << "U: " << i << " " << egb_der_U[i]/vol << endl;      
      }
    }

    if(verbose_level > 3){
      cout << "W parameters: " << endl;
      for(int i = 0; i < numParticles; i++){
	RealOpenMM vol = 4.*M_PI*pow(radii_vdw[i],3)/3.0; 
	cout << "W: " << i << " " << evdw_der_W[i]/vol << endl;      
      }
    }


    
    

#ifdef NOTNOW
    //
    // test derivative of van der Waals energy at constant self volumes
    //
    int probe_atom = 120;
    RealVec dx = RealVec(0.001, 0.0, 0.0);
    pos[probe_atom] += dx;
    //recompute Born radii w/o changing volume scaling factors
    for(int i = 0; i < numParticles; i++){
      inverse_born_radius[i] = 1./(radii[i] - AGBNP_RADIUS_INCREMENT);
      for(int j = 0; j < numParticles; j++){
	if(i == j) continue;
	if(ishydrogen[j]) continue;
	RealOpenMM b = (radii[i] - AGBNP_RADIUS_INCREMENT)/radii[j];
	RealVec dist = pos[j] - pos[i];
	RealOpenMM d = sqrt(dist.dot(dist));
	if(d < AGBNP_I4LOOKUP_MAXA){
	  int rad_typei = i4_lut->radius_type_screened[i];
	  int rad_typej = i4_lut->radius_type_screened[j];
	  inverse_born_radius[i] -= pifac*volume_scaling_factor[j]*i4_lut->eval(d, b); 
	}	
      }
      RealOpenMM fp;
      born_radius[i] = 1./agbnp_swf_invbr(inverse_born_radius[i], fp);
      inverse_born_radius_fp[i] = fp;
    }
    for(int i = 0; i < numParticles; i++){
      RealOpenMM br = born_radius[i];
      evdw_der_brw[i] = -pifac*3.*vdw_alpha[i]*br*br*inverse_born_radius_fp[i]/pow(br+AGBNP_HB_RADIUS,4);
    }
    RealOpenMM evdw_new = 0.;
    for(int i=0;i<numParticles; i++){
      evdw_new += vdw_alpha[i]/pow(born_radius[i]+AGBNP_HB_RADIUS,3);
    }
    if(verbose_level > 0){
      cout << "New Van der Waals energy: " << evdw_new << endl;
    }
    RealOpenMM devdw_from_der = -DOT3(force[probe_atom],dx);
    if(verbose_level > 0){
      cout << "Evdw Change from Direct: " << evdw_new - evdw << endl;
      cout << "Evdw Change from Deriv : " << devdw_from_der  << endl;
    }
#endif

#ifdef NOTNOW
    //
    // test derivative of the GB energy at constant self volumes
    //
    int probe_atom = 120;
    RealVec dx = RealVec(0.00, 0.00, 0.01);
    pos[probe_atom] += dx;
    //recompute Born radii w/o changing volume scaling factors
    for(int i = 0; i < numParticles; i++){
      inverse_born_radius[i] = 1./(radii[i] - AGBNP_RADIUS_INCREMENT);
      for(int j = 0; j < numParticles; j++){
	if(i == j) continue;
	if(ishydrogen[j]) continue;
	RealOpenMM b = (radii[i] - AGBNP_RADIUS_INCREMENT)/radii[j];
	RealVec dist = pos[j] - pos[i];
	RealOpenMM d = sqrt(dist.dot(dist));
	if(d < AGBNP_I4LOOKUP_MAXA){
	  inverse_born_radius[i] -= pifac*volume_scaling_factor[j]*i4_lut->eval(d, b); 
	}	
      }
      RealOpenMM fp;
      born_radius[i] = 1./agbnp_swf_invbr(inverse_born_radius[i], fp);
      inverse_born_radius_fp[i] = fp;
    }
    //new GB energy
    RealOpenMM gb_self_energy_new = 0.0;
    RealOpenMM gb_pair_energy_new = 0.0;
    for(int i = 0; i < numParticles; i++){
      gb_self_energy_new += dielectric_factor*charge[i]*charge[i]/born_radius[i];
      for(int j = i+1; j < numParticles; j++){
	RealVec dist = pos[j] - pos[i];
	RealOpenMM d2 = dist.dot(dist);
	RealOpenMM qq = dielectric_factor*charge[j]*charge[i];
	RealOpenMM bb = born_radius[i]*born_radius[j];
	RealOpenMM etij = exp(-pt25*d2/bb);
	RealOpenMM fgb = 1./sqrt(d2 + bb*etij);
	RealOpenMM egb = 2.*qq*fgb;
	gb_pair_energy_new += egb;
      }
    }
    if(verbose_level > 0){
      cout << "GB self energy new: " << gb_self_energy_new << endl;
      cout << "GB pair energy new: " << gb_pair_energy_new << endl;
      cout << "GB energy new: " << gb_pair_energy_new+gb_self_energy_new << endl;
    }
    RealOpenMM degb_from_der = -DOT3(force[probe_atom],dx);
    if(verbose_level > 0){
      cout << "Egb Change from Direct: " << gb_pair_energy_new+gb_self_energy_new - (gb_pair_energy+gb_self_energy) << endl;
      cout << "Egb Change from Deriv : " << degb_from_der  << endl;
    }
#endif


    RealOpenMM volume_tmp, vol_energy_tmp;
    //set up the parameters of the pseudo-volume energy function and
    //compute the component of the gradient of Evdw due to the variations
    //of self volumes
    for(int i = 0; i < numParticles; i++){
      RealOpenMM vol = 4.*M_PI*pow(radii_vdw[i],3)/3.0; 
      nu[i] = evdw_der_W[i]/vol;
    }
    gvol->rescan_tree_gammas(nu);
    gvol->compute_volume(pos, volume_tmp, vol_energy_tmp, vol_force, free_volume, self_volume);
    for(int i = 0; i < numParticles; i++){
      force[i] += vol_force[i] * w_vdw;
    }

    //set up the parameters of the pseudo-volume energy function and
    //compute the component of the gradient of Egb due to the variations
    //of self volumes
    for(int i = 0; i < numParticles; i++){
      RealOpenMM vol = 4.*M_PI*pow(radii_vdw[i],3)/3.0; 
      nu[i] = egb_der_U[i]/vol;
    }
    gvol->rescan_tree_gammas(nu);
    gvol->compute_volume(pos, volume_tmp, vol_energy_tmp, vol_force, free_volume, self_volume);
    for(int i = 0; i < numParticles; i++){
      force[i] += vol_force[i] * w_egb;
    }

    
    //returns energy
    return (double)energy;
}

double ReferenceCalcAGBNPForceKernel::executeAGBNP2(ContextImpl& context, bool includeForces, bool includeEnergy) {
  
    //weights
    RealOpenMM w_evol = 1.0, w_egb = 1.0, w_vdw = 1.0;
    
    vector<RealVec>& pos = extractPositions(context);
    vector<RealVec>& force = extractForces(context);
    RealOpenMM energy = 0.0;
    int verbose_level = 0;
    int init = 0; 

    if(verbose_level > 0) cout << "Executing AGBNP2" << endl;
    
    if(verbose_level > 0){
      cout << "-----------------------------------------------" << endl;
    } 
    
    // volume energy function 1
    RealOpenMM volume1, vol_energy1;
    vector<RealOpenMM> nu(numParticles);
    for(int i = 0; i < numParticles; i++){
      nu[i] = gammas[i]/SA_DR;
    }

    gvol->set_radii(radii_large);
    gvol->set_gammas(nu);
    gvol->compute_tree(pos);
    gvol->compute_volume(pos, volume1, vol_energy1, vol_force, free_volume, self_volume);
    
    //returns energy and gradients from volume energy function
    for(int i = 0; i < numParticles; i++){
      force[i] += vol_force[i] * w_evol;
    }
    energy += vol_energy1 * w_evol;
    if(verbose_level > 0){
      cout << "Volume energy 1: " << vol_energy1 << endl;
    }

    
#ifdef NOTNOW
    vector<int> nov, nov_2body;
    gvol->getstat(nov, nov_2body);    
    int nn = 0, nn2 = 0;
    for(int i = 0; i < nov.size(); i++){
      nn += nov[i];
     nn2 += nov_2body[i];
    }
    cout << "Noverlaps: " << nn << " " << nn2 << endl;
#endif

#ifdef NOTNOW    
    {
      //tests i4 lookup table
      double ang = 0.1;
      double dr = 0.1*ang;
      double rmin = 0.;
      double rmax = AGBNP_I4LOOKUP_MAXA;
      for(int i=0;i<300;i++){
	double x = rmin + i*dr;
	double y = (x < rmax) ? i4_lut->eval(x, 0.925926) : 0.;
	double yp = (x < rmax) ? i4_lut->evalderiv(x, 0.925926) : 0.;
	cout << "i4: " << x << " " << y << " " << yp << endl;
      }
    }
#endif

    //volume scaling factors from self volumes (with large radii)
    double tot_vol = 0;
    for(int i = 0; i < numParticles; i++){
      RealOpenMM rad = radii_large[i];
      RealOpenMM vol = (4./3.)*M_PI*rad*rad*rad;
      volume_scaling_factor[i] = self_volume[i]/vol;
      if(verbose_level > 4){
	cout << "SV " << i << " " << self_volume[i] << endl;
      }
      tot_vol += self_volume[i];
    }
    if(verbose_level > 0){
      cout << "Volume from self volumes: " << tot_vol << endl;
    }

    RealOpenMM pifac = 1./(4.*M_PI);

    //compute inverse Born radii, prototype, no cutoff
    for(int i = 0; i < numParticles; i++){
      inverse_born_radius[i] = 1./radii_vdw[i];
      for(int j = 0; j < numParticles; j++){
	if(i == j) continue;
	if(ishydrogen[j]) continue;
	RealVec dist = pos[j] - pos[i];
	RealOpenMM d = sqrt(dist.dot(dist));
	if(d < AGBNP_I4LOOKUP_MAXA){
	  int rad_typei = i4_lut->radius_type_screened[i];
	  int rad_typej = i4_lut->radius_type_screener[j];
	  inverse_born_radius[i] -= pifac*volume_scaling_factor[j]*i4_lut->eval(d, rad_typei, rad_typej);
	}	
      }
      RealOpenMM fp;
      born_radius[i] = 1./agbnp_swf_invbr(inverse_born_radius[i], fp);
      inverse_born_radius_fp[i] = fp;
    }

    if(verbose_level > 4){
      cout << "Born radii:" << endl;
      RealOpenMM fp;
      for(int i = 0; i < numParticles; i++){
	cout << "BR " << i << " " << 10.*born_radius[i] << " Si " << volume_scaling_factor[i] << endl;
      }
    }

    //GB energy
    RealOpenMM dielectric_in = 1.0;
    RealOpenMM dielectric_out= 80.0;
    RealOpenMM tokjmol = 4.184*332.0/10.0; //the factor of 10 is the conversion of 1/r from nm to Ang
    RealOpenMM dielectric_factor = tokjmol*(-0.5)*(1./dielectric_in - 1./dielectric_out);
    RealOpenMM pt25 = 0.25;
    vector<RealOpenMM> egb_der_Y(numParticles);
    for(int i = 0; i < numParticles; i++){
      egb_der_Y[i] = 0.0;
    }
    RealOpenMM gb_self_energy = 0.0;
    RealOpenMM gb_pair_energy = 0.0;
    for(int i = 0; i < numParticles; i++){
      double uself = dielectric_factor*charge[i]*charge[i]/born_radius[i];
      gb_self_energy += uself;
      for(int j = i+1; j < numParticles; j++){
	RealVec dist = pos[j] - pos[i];
	RealOpenMM d2 = dist.dot(dist);
	RealOpenMM qqf = charge[j]*charge[i];
	RealOpenMM qq = dielectric_factor*qqf;
	RealOpenMM bb = born_radius[i]*born_radius[j];
	RealOpenMM etij = exp(-pt25*d2/bb);
	RealOpenMM fgb = 1./sqrt(d2 + bb*etij);
	RealOpenMM egb = 2.*qq*fgb;
	gb_pair_energy += egb;
	RealOpenMM fgb3 = fgb*fgb*fgb;
	RealOpenMM mw = -2.0*qq*(1.0-pt25*etij)*fgb3;
	RealVec g = dist * mw;
	force[i] += g * w_egb;
	force[j] -= g * w_egb;
	RealOpenMM ytij = qqf*(bb+pt25*d2)*etij*fgb3;
	egb_der_Y[i] += ytij;
	egb_der_Y[j] += ytij;
      }
    }
    if(verbose_level > 0){
      cout << "GB self energy: " << gb_self_energy << endl;
      cout << "GB pair energy: " << gb_pair_energy << endl;
      cout << "GB energy: " << gb_pair_energy+gb_self_energy << endl;
    }
    energy += w_egb*gb_pair_energy + w_egb*gb_self_energy;

    if(verbose_level > 0){
      cout << "Y parameters: " << endl;
      for(int i = 0; i < numParticles; i++){
	cout << "Y: " << i << " " << egb_der_Y[i] << endl;
      }
    }

    //compute van der Waals energy
    RealOpenMM evdw = 0.;
    for(int i=0;i<numParticles; i++){
      evdw += vdw_alpha[i]/pow(born_radius[i]+AGBNP_HB_RADIUS,3);
    }
    if(verbose_level > 0){
      cout << "Van der Waals energy: " << evdw << endl;
    }
    energy += w_vdw*evdw;

    //compute atom-level property for the calculation of the gradients of Evdw and Egb
    vector<RealOpenMM> evdw_der_brw(numParticles);
    for(int i = 0; i < numParticles; i++){
      RealOpenMM br = born_radius[i];
      evdw_der_brw[i] = -pifac*3.*vdw_alpha[i]*br*br*inverse_born_radius_fp[i]/pow(br+AGBNP_HB_RADIUS,4);
    }
    if(verbose_level > 3){
      cout << "BrW parameters: " << endl;
      for(int i = 0; i < numParticles; i++){
	cout << "BrW: " << i << " " << evdw_der_brw[i] << endl;      
      }
    }

    
    vector<RealOpenMM> egb_der_bru(numParticles);
    for(int i = 0; i < numParticles; i++){
      RealOpenMM br = born_radius[i];
      RealOpenMM qi = charge[i];
      egb_der_bru[i] = -pifac*dielectric_factor*(qi*qi + egb_der_Y[i]*br)*inverse_born_radius_fp[i];
    }
    
    if(verbose_level > 3){
      cout << "BrU parameters: " << endl;
      for(int i = 0; i < numParticles; i++){
	cout << "BrU: " << i << " " << egb_der_bru[i] << endl;      
      }
    }

    
    
    //compute the component of the gradients of the van der Waals and GB energies due to
    //variations of Born radii
    //also accumulates W's and U's for self-volume components of the gradients later
    vector<RealOpenMM> evdw_der_W(numParticles);
    vector<RealOpenMM> egb_der_U(numParticles);
    for(int i = 0; i < numParticles; i++){
      evdw_der_W[i] = egb_der_U[i] = 0.0;
    }
    for(int i = 0; i < numParticles; i++){
      for(int j = 0; j < numParticles; j++){
	if(i == j) continue;
	if(ishydrogen[j]) continue;
	RealVec dist = pos[j] - pos[i];
	RealOpenMM d = sqrt(dist.dot(dist));
	double Qji = 0.0, dQji = 0.0;
	// Qji: j descreens i
	if(d < AGBNP_I4LOOKUP_MAXA){
	  int rad_typei = i4_lut->radius_type_screened[i];
	  int rad_typej = i4_lut->radius_type_screener[j];
	  Qji = i4_lut->eval(d, rad_typei, rad_typej); 
	  dQji = i4_lut->evalderiv(d, rad_typei, rad_typej);
	}
	RealVec w;
	//van der Waals stuff
	evdw_der_W[j] += evdw_der_brw[i]*Qji;
	w = dist * evdw_der_brw[i]*volume_scaling_factor[j]*dQji/d;
	force[i] += w * w_vdw;
	force[j] -= w * w_vdw;
	//GB stuff
	egb_der_U[j] += egb_der_bru[i]*Qji;
	w = dist * egb_der_bru[i]*volume_scaling_factor[j]*dQji/d;
	force[i] += w * w_egb;
	force[j] -= w * w_egb;
      }
    }

    

    if(verbose_level > 3){
      cout << "U parameters: " << endl;
      for(int i = 0; i < numParticles; i++){
	RealOpenMM vol = 4.*M_PI*pow(radii_large[i],3)/3.0; 
	cout << "U: " << i << " " << egb_der_U[i]/vol << endl;      
      }
    }

    if(verbose_level > 3){
      cout << "W parameters: " << endl;
      for(int i = 0; i < numParticles; i++){
	RealOpenMM vol = 4.*M_PI*pow(radii_large[i],3)/3.0; 
	cout << "W: " << i << " " << evdw_der_W[i]/vol << endl;      
      }
    }


    
    

#ifdef NOTNOW
    //
    // test derivative of van der Waals energy at constant self volumes
    //
    int probe_atom = 120;
    RealVec dx = RealVec(0.001, 0.0, 0.0);
    pos[probe_atom] += dx;
    //recompute Born radii w/o changing volume scaling factors
    for(int i = 0; i < numParticles; i++){
      inverse_born_radius[i] = 1./(radii[i] - AGBNP_RADIUS_INCREMENT);
      for(int j = 0; j < numParticles; j++){
	if(i == j) continue;
	if(ishydrogen[j]) continue;
	RealOpenMM b = radii_vdw[i]/radii_large[j];
	RealVec dist = pos[j] - pos[i];
	RealOpenMM d = sqrt(dist.dot(dist));
	if(d < AGBNP_I4LOOKUP_MAXA){
	  inverse_born_radius[i] -= pifac*volume_scaling_factor[j]*i4_lut->eval(d, b); 
	}	
      }
      RealOpenMM fp;
      born_radius[i] = 1./agbnp_swf_invbr(inverse_born_radius[i], fp);
      inverse_born_radius_fp[i] = fp;
    }
    for(int i = 0; i < numParticles; i++){
      RealOpenMM br = born_radius[i];
      evdw_der_brw[i] = -pifac*3.*vdw_alpha[i]*br*br*inverse_born_radius_fp[i]/pow(br+AGBNP_HB_RADIUS,4);
    }
    RealOpenMM evdw_new = 0.;
    for(int i=0;i<numParticles; i++){
      evdw_new += vdw_alpha[i]/pow(born_radius[i]+AGBNP_HB_RADIUS,3);
    }
    if(verbose_level > 0){
      cout << "New Van der Waals energy: " << evdw_new << endl;
    }
    RealOpenMM devdw_from_der = -DOT3(force[probe_atom],dx);
    if(verbose_level > 0){
      cout << "Evdw Change from Direct: " << evdw_new - evdw << endl;
      cout << "Evdw Change from Deriv : " << devdw_from_der  << endl;
    }
#endif

#ifdef NOTNOW
    //
    // test derivative of the GB energy at constant self volumes
    //
    int probe_atom = 120;
    RealVec dx = RealVec(0.00, 0.00, 0.01);
    pos[probe_atom] += dx;
    //recompute Born radii w/o changing volume scaling factors
    for(int i = 0; i < numParticles; i++){
      inverse_born_radius[i] = 1./(radii[i] - AGBNP_RADIUS_INCREMENT);
      for(int j = 0; j < numParticles; j++){
	if(i == j) continue;
	if(ishydrogen[j]) continue;
	RealOpenMM b = (radii[i] - AGBNP_RADIUS_INCREMENT)/radii[j];
	RealVec dist = pos[j] - pos[i];
	RealOpenMM d = sqrt(dist.dot(dist));
	if(d < AGBNP_I4LOOKUP_MAXA){
	  inverse_born_radius[i] -= pifac*volume_scaling_factor[j]*i4_lut->eval(d, b); 
	}	
      }
      RealOpenMM fp;
      born_radius[i] = 1./agbnp_swf_invbr(inverse_born_radius[i], fp);
      inverse_born_radius_fp[i] = fp;
    }
    //new GB energy
    RealOpenMM gb_self_energy_new = 0.0;
    RealOpenMM gb_pair_energy_new = 0.0;
    for(int i = 0; i < numParticles; i++){
      gb_self_energy_new += dielectric_factor*charge[i]*charge[i]/born_radius[i];
      for(int j = i+1; j < numParticles; j++){
	RealVec dist = pos[j] - pos[i];
	RealOpenMM d2 = dist.dot(dist);
	RealOpenMM qq = dielectric_factor*charge[j]*charge[i];
	RealOpenMM bb = born_radius[i]*born_radius[j];
	RealOpenMM etij = exp(-pt25*d2/bb);
	RealOpenMM fgb = 1./sqrt(d2 + bb*etij);
	RealOpenMM egb = 2.*qq*fgb;
	gb_pair_energy_new += egb;
      }
    }
    if(verbose_level > 0){
      cout << "GB self energy new: " << gb_self_energy_new << endl;
      cout << "GB pair energy new: " << gb_pair_energy_new << endl;
      cout << "GB energy new: " << gb_pair_energy_new+gb_self_energy_new << endl;
    }
    RealOpenMM degb_from_der = -DOT3(force[probe_atom],dx);
    if(verbose_level > 0){
      cout << "Egb Change from Direct: " << gb_pair_energy_new+gb_self_energy_new - (gb_pair_energy+gb_self_energy) << endl;
      cout << "Egb Change from Deriv : " << degb_from_der  << endl;
    }
#endif


    RealOpenMM volume_tmp, vol_energy_tmp;
    //set up the parameters of the pseudo-volume energy function and
    //compute the component of the gradient of Evdw due to the variations
    //of self volumes
    for(int i = 0; i < numParticles; i++){
      RealOpenMM vol = 4.*M_PI*pow(radii_large[i],3)/3.0; 
      nu[i] = evdw_der_W[i]/vol;
    }
    gvol->rescan_tree_gammas(nu);
    gvol->compute_volume(pos, volume_tmp, vol_energy_tmp, vol_force, free_volume, self_volume);
    for(int i = 0; i < numParticles; i++){
      force[i] += vol_force[i] * w_vdw;
    }

    //set up the parameters of the pseudo-volume energy function and
    //compute the component of the gradient of Egb due to the variations
    //of self volumes
    for(int i = 0; i < numParticles; i++){
      RealOpenMM vol = 4.*M_PI*pow(radii_large[i],3)/3.0; 
      nu[i] = egb_der_U[i]/vol;
    }
    gvol->rescan_tree_gammas(nu);
    gvol->compute_volume(pos, volume_tmp, vol_energy_tmp, vol_force, free_volume, self_volume);
    for(int i = 0; i < numParticles; i++){
      force[i] += vol_force[i] * w_egb;
    }

    // volume energy function 2
    vector<RealOpenMM> rads(numParticles);
    RealOpenMM vol_energy2, volume2;
    for(int i = 0; i < numParticles; i++){
      nu[i] = -gammas[i]/SA_DR;
    }
    for(int i = 0; i < numParticles; i++){
      rads[i] = radii_vdw[i];
    }
    gvol->rescan_tree_volumes(pos, rads, nu);
    gvol->compute_volume(pos, volume2, vol_energy2, vol_force, free_volume, self_volume);
    for(int i = 0; i < numParticles; i++){
      force[i] += vol_force[i] * w_evol;
    }
    energy += vol_energy2 * w_evol;
    if(verbose_level > 0){
      cout << "Volume energy 2: " << vol_energy2 << endl;
      cout << "Surface area energy: " << vol_energy1 + vol_energy2 << endl;
    }
    
    //returns energy
    return (double)energy;
}


void ReferenceCalcAGBNPForceKernel::copyParametersToContext(ContextImpl& context, const AGBNPForce& force) {
  if (force.getNumParticles() != numParticles)
    throw OpenMMException("updateParametersInContext: The number of AGBNP particles has changed");

  for (int i = 0; i < numParticles; i++){
    double r, g, alpha, q;
    bool h;
    force.getParticleParameters(i, r, g, alpha, q, h);
    if(pow(radii_vdw[i]-r,2) > 1.e-6){
      throw OpenMMException("updateParametersInContext: AGBNP plugin does not support changing atomic radii.");
    }
    if(ishydrogen[i] != h){
      throw OpenMMException("updateParametersInContext: AGBNP plugin does not support changing heavy/hydrogen atoms.");
    }
    gammas[i] = g;
    if(h) gammas[i] = 0.0;
    vdw_alpha[i] = alpha;
    charge[i] = q;
  }
}
