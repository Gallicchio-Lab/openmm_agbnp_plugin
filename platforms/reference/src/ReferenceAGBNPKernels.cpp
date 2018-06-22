/* -------------------------------------------------------------------------- *
 *                               OpenMM-AGBNP                                *
 * -------------------------------------------------------------------------- */

#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <cfloat>
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

    //set radius offset
    if(version == 0){
      roffset = AGBNP_RADIUS_INCREMENT;
    }else if(version == 1){
      roffset = AGBNP_RADIUS_INCREMENT;
    }else{
      roffset = AGBNP2_RADIUS_INCREMENT;
    }

    
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
    vol_dv.resize(numParticles);

    vector<double> vdwrad(numParticles);
    common_gamma = -1;
    for (int i = 0; i < numParticles; i++){
      double r, g, alpha, q;
      bool h;
      force.getParticleParameters(i, r, g, alpha, q, h);
      radii_large[i] = r + roffset;
      radii_vdw[i] = r;
      vdwrad[i] = r; //double version for lookup table setup
      gammas[i] = g;
      if(h) gammas[i] = 0.0;
      vdw_alpha[i] = alpha;
      charge[i] = q;
      ishydrogen[i] = h ? 1 : 0;

      //make sure that all gamma's are the same
      if(common_gamma < 0 && !h){
	common_gamma = g; //first occurrence of a non-zero gamma
      }else{
	if(!h && pow(common_gamma - g,2) > FLT_MIN){
	  throw OpenMMException("initialize(): AGBNP does not support multiple gamma values.");
	}
      }

    }

    //create and saves GaussVol instance
    //radii, volumes, etc. will be set in execute()
    gvol = new GaussVol(numParticles, ishydrogen);

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

    solvent_radius = force.getSolventRadius();
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
    int verbose_level = 1;
    int init = 0; 

    vector<RealOpenMM> nu(numParticles);


    if(verbose_level > 0) cout << "Executing GVolSA" << endl;
    
    if(verbose_level > 0){
      cout << "-----------------------------------------------" << endl;
    } 

    
    // volume energy function 1 (large radii)
    RealOpenMM volume1, vol_energy1;

    gvol->setRadii(radii_large);

    vector<RealOpenMM> volumes_large(numParticles);
    for(int i = 0; i < numParticles; i++){
      volumes_large[i] = ishydrogen[i]>0 ? 0.0 : 4.*M_PI*pow(radii_large[i],3)/3.;
    }
    gvol->setVolumes(volumes_large);
    
    for(int i = 0; i < numParticles; i++){
      nu[i] = gammas[i]/roffset;
    }
    gvol->setGammas(nu);
    
    gvol->compute_tree(pos);
    gvol->compute_volume(pos, volume1, vol_energy1, vol_force, vol_dv, free_volume, self_volume);
      
    //returns energy and gradients from volume energy function
    for(int i = 0; i < numParticles; i++){
      force[i] += vol_force[i] * w_evol;
    }
    energy += vol_energy1 * w_evol;
    if(verbose_level > 0){
      cout << "Volume energy 1: " << vol_energy1 << endl;
    }

#ifdef NOTNOW
    //test of vol_dv
    vector<RealOpenMM> vol_dv2(numParticles);
    double energy_save = vol_energy1;
    for(int test_atom = 0; test_atom < numParticles; test_atom++){
      if(ishydrogen[test_atom]>0) continue;
      double deltav = -0.001*volumes_large[test_atom];
      double save_vol = volumes_large[test_atom];
      volumes_large[test_atom] += deltav;
      gvol->setVolumes(volumes_large);
      gvol->compute_tree(pos);
      gvol->compute_volume(pos, volume1, vol_energy1, vol_force, vol_dv2, free_volume, self_volume);
      cout << "DVV " << test_atom << " " << vol_energy1 - energy_save << " " << deltav*vol_dv[test_atom] << endl;
      volumes_large[test_atom] = save_vol;
    }
#endif
    
    // volume energy function 2 (small radii)
    RealOpenMM vol_energy2, volume2;

    gvol->setRadii(radii_vdw);

    vector<RealOpenMM> volumes_vdw(numParticles);
    for(int i = 0; i < numParticles; i++){
      volumes_vdw[i] = ishydrogen[i]>0 ? 0.0 : 4.*M_PI*pow(radii_vdw[i],3)/3.;
    }
    gvol->setVolumes(volumes_vdw);
    
    for(int i = 0; i < numParticles; i++){
      nu[i] = -gammas[i]/roffset;
    }
    gvol->setGammas(nu);

    gvol->rescan_tree_volumes(pos);
    gvol->compute_volume(pos, volume2, vol_energy2, vol_force, vol_dv, free_volume, self_volume);


 #ifdef NOTNOW
    //test of vol_dv
    vector<RealOpenMM> vol_dv2(numParticles);
    double energy_save = vol_energy2;
    for(int test_atom = 0; test_atom < numParticles; test_atom++){
      if(ishydrogen[test_atom]>0) continue;
      double deltav = -0.001*volumes_vdw[test_atom];
      double save_vol = volumes_vdw[test_atom];
      volumes_vdw[test_atom] += deltav;
      gvol->setVolumes(volumes_vdw);
      gvol->compute_tree(pos);
      gvol->compute_volume(pos, volume2, vol_energy2, vol_force, vol_dv2, free_volume, self_volume);
      cout << "DVV " << test_atom << " " << vol_energy2 - energy_save << " " << deltav*vol_dv[test_atom] << endl;
      volumes_vdw[test_atom] = save_vol;
    }
#endif


    
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
    //weights
    RealOpenMM w_evol = 1.0, w_egb = 1.0, w_vdw = 1.0;
  
    vector<RealVec>& pos = extractPositions(context);
    vector<RealVec>& force = extractForces(context);
    RealOpenMM energy = 0.0;
    int verbose_level = 0;
    bool verbose = verbose_level > 1;
    int init = 0;
    
    if(verbose_level > 0) {
      cout << "Executing AGBNP1" << endl;
      cout << "-----------------------------------------------" << endl;
    } 
    
    // volume energy function 1 (large radii)
    RealOpenMM volume1, vol_energy1;

    gvol->setRadii(radii_large);
    
    vector<RealOpenMM> nu(numParticles);
    for(int i = 0; i < numParticles; i++){
      nu[i] = gammas[i]/roffset;
    }
    gvol->setGammas(nu);
    
    vector<RealOpenMM> volumes_large(numParticles);
    for(int i = 0; i < numParticles; i++){
      volumes_large[i] = ishydrogen[i]>0 ? 0.0 : 4.*M_PI*pow(radii_large[i],3)/3.;
    }
    gvol->setVolumes(volumes_large);
    
    gvol->compute_tree(pos);
    if(verbose_level > 4){
      gvol->print_tree();
    }
    gvol->compute_volume(pos, volume1, vol_energy1, vol_force, vol_dv, free_volume, self_volume);


    if(verbose_level > 0){
      vector<int> noverlaps(numParticles);
      for(int i = 0; i<numParticles; i++) noverlaps[i] = 0;
      gvol->getstat(noverlaps);
      
      //compute maximum number of overlaps
      int nn = 0;
      for(int i = 0; i < noverlaps.size(); i++){
	nn += noverlaps[i];
      }

      cout << "Number of overlaps: " << nn << endl;
    }



    
    //returns energy and gradients from volume energy function
    for(int i = 0; i < numParticles; i++){
      force[i] += vol_force[i] * w_evol;
    }
    energy += vol_energy1 * w_evol;
    if(verbose_level > 0){
      cout << "Volume energy 1: " << vol_energy1 << endl;
    }

    double tot_vol = 0;
    double vol_energy = 0;
    for(int i = 0; i < numParticles; i++){
      tot_vol += self_volume[i];
      vol_energy += nu[i]*self_volume[i];
    }
    if(verbose_level > 0){
      cout << "Volume from self volumes(1): " << tot_vol << endl;
      cout << "Volume energy from self volumes(1): " << vol_energy << endl;
    }



    
    // volume energy function 2 (small radii)
    RealOpenMM vol_energy2, volume2;

    gvol->setRadii(radii_vdw);
    
    for(int i = 0; i < numParticles; i++){
      nu[i] = -gammas[i]/roffset;
    }
    gvol->setGammas(nu);
    
    vector<RealOpenMM> volumes_vdw(numParticles);
    for(int i = 0; i < numParticles; i++){
      volumes_vdw[i] = ishydrogen[i]>0 ? 0.0 : 4.*M_PI*pow(radii_vdw[i],3)/3.;
    }
    gvol->setVolumes(volumes_vdw);

    gvol->rescan_tree_volumes(pos);
    gvol->compute_volume(pos, volume2, vol_energy2, vol_force, vol_dv, free_volume, self_volume);
    
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
    tot_vol = 0;
    for(int i = 0; i < numParticles; i++){
      RealOpenMM rad = radii_vdw[i];
      RealOpenMM vol = (4./3.)*M_PI*rad*rad*rad;
      volume_scaling_factor[i] = self_volume[i]/vol;
      if(verbose_level > 3){
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
	if(ishydrogen[j] > 0) continue;
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

    if(verbose_level > 3){
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
	if(ishydrogen[j]>0) continue;
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
      inverse_born_radius[i] = 1./(radii[i] - roffset);
      for(int j = 0; j < numParticles; j++){
	if(i == j) continue;
	if(ishydrogen[j]>0) continue;
	RealOpenMM b = (radii[i] - roffset)/radii[j];
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
      inverse_born_radius[i] = 1./(radii[i] - roffset);
      for(int j = 0; j < numParticles; j++){
	if(i == j) continue;
	if(ishydrogen[j]>0) continue;
	RealOpenMM b = (radii[i] - roffset)/radii[j];
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
    gvol->setGammas(nu);
    gvol->rescan_tree_gammas();
    gvol->compute_volume(pos, volume_tmp, vol_energy_tmp, vol_force, vol_dv, free_volume, self_volume);
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
    gvol->setGammas(nu);
    gvol->rescan_tree_gammas();
    gvol->compute_volume(pos, volume_tmp, vol_energy_tmp, vol_force, vol_dv, free_volume, self_volume);
    for(int i = 0; i < numParticles; i++){
      force[i] += vol_force[i] * w_egb;
    }


    if(verbose_level > 3){
      //creates input for test program
      double nm2ang = 10.0;
      double kjmol2kcalmol = 1/4.184;
      double gf = kjmol2kcalmol/(nm2ang*nm2ang);
      cout << "---- input for test program begins ----" << endl;
      cout << numParticles << endl;
      for(int i = 0; i < numParticles; i++){
	cout << std::setprecision(6) << std::setw(5) << i << " " << std::setw(12) << nm2ang*pos[i][0] << " " << std::setw(12) << nm2ang*pos[i][1] << " " << std::setw(12) << nm2ang*pos[i][2] << " " << std::setw(12) << nm2ang*radii_vdw[i] << " " << std::setw(12) << charge[i] << " " << std::setw(12) << gf*gammas[i] << " " << std::setw(2) << ishydrogen[i] << endl;
      }
      cout << "--- input for test program ends ----" << endl;
    }


    if(verbose_level > 3){
      //creates input for mkws program
      double nm2ang = 10.0;
      cout << "---- input for mkws program begins ----" << endl;
      cout << numParticles << endl;
      for(int i = 0; i < numParticles; i++){
	string at_symbol = "A";
	if(ishydrogen[i] > 0){
	  at_symbol = "H";
	}
	cout << std::setprecision(6) << std::setw(5) << i << " " << at_symbol << "" << std::setw(12) << nm2ang*pos[i][0] << " " << std::setw(12) << nm2ang*pos[i][1] << " " << std::setw(12) << nm2ang*pos[i][2] << " " << std::setw(12) << nm2ang*radii_vdw[i] << endl;
      }
      cout << "--- input for mkws program ends ----" << endl;
    }
    
    //returns energy
    return (double)energy;
}

double ReferenceCalcAGBNPForceKernel::executeAGBNP2(ContextImpl& context, bool includeForces, bool includeEnergy) {
  //weights
  RealOpenMM w_evol = 1.0, w_evol_ms = 1.0 , w_egb = 0.0, w_vdw = 0.0;
  
  vector<RealVec>& pos = extractPositions(context);
  vector<RealVec>& force = extractForces(context);
  RealOpenMM energy = 0.0;
  int verbose_level = 1;
  bool verbose = verbose_level > 0;
  
  if(verbose_level > 0) {
    cout << "Executing AGBNP2" << endl;
    cout << "-----------------------------------------------" << endl;
  } 


  // volume energy function 1 (large radii)
  RealOpenMM volume1, vol_energy1;

  gvol->setRadii(radii_large);
  
  vector<RealOpenMM> nu(numParticles);
  for(int i = 0; i < numParticles; i++) {
    nu[i] = gammas[i]/roffset;
  }
  gvol->setGammas(nu);

  vector<RealOpenMM> volumes_large(numParticles);
  for(int i = 0; i < numParticles; i++){
    volumes_large[i] = ishydrogen[i]>0 ? 0.0 : 4.*M_PI*pow(radii_large[i],3)/3.;
  }
  gvol->setVolumes(volumes_large);
  
  gvol->compute_tree(pos);
  if(verbose_level > 4){
    gvol->print_tree();
  }
  gvol->compute_volume(pos, volume1, vol_energy1, vol_force, vol_dv, free_volume, self_volume);
  energy += w_evol * vol_energy1;
  for(int i = 0; i < numParticles; i++){
    force[i] += vol_force[i] * w_evol;
  }

  if(verbose_level > 0){
      cout << "vol 1: " << volume1 << endl;
      cout << "energy 1: " << vol_energy1 << endl;
  }

  
  
  for(int i = 0; i < numParticles; i++){
    RealOpenMM rad = radii_vdw[i];
    RealOpenMM vol = (4./3.)*M_PI*rad*rad*rad;
    volume_scaling_factor[i] = self_volume[i]/vol;
    if(verbose_level > 3){
      cout << "SV " << i << " " << self_volume[i] << endl;
    }
  }

  

  //constructs molecular surface particles for volume energy function 1 (large radii)
  vector<MSParticle> msparticles1;
  double radw = solvent_radius;
  double volw = 4.*M_PI*pow(radw,3)/3.;
  double vol_coeff = 0.17;
  int nms = 0;
  for(int i = 0; i < numParticles; i++){
    if(ishydrogen[i]>0) continue;
    double rad1 = radii_vdw[i];
    for(int j = i + 1; j < numParticles; j++){
      if(ishydrogen[j]>0) continue;
      double rad2 = radii_vdw[j];
      double q = sqrt(rad1*rad2)/radw;
      RealVec dist = pos[j] - pos[i];
      RealOpenMM d = sqrt(dist.dot(dist));
      //      cout << d << " " << rad1 + rad2 << " " << rad1 + rad2 + 2*radw << endl;
      if(d < rad1 + rad2 + 2*radw ){
	//pair ms volume when the two atoms overlap
	double dms = rad1 + rad2 + 0.5*radw;
	double volms0 = vol_coeff*q*q*volw;
	double sigma = 0.5*sqrt(q)*radw;
	double volms = volms0*exp(-0.5*(d-dms)*(d-dms)/(sigma*sigma));
	double sp;
	double s = pol_switchfunc(volms, VOLMINMSA, VOLMINMSB, sp);
	double volmsw = volms*s;
	double sder = s + volms*sp;
	if(volmsw > FLT_MIN){
	  //position of MS particle
	  //double distms_from_1 = 0.5*(d - rad1 - rad2) + rad1;//midpoint of the two surfaces
	  //RealVec distu = dist * (1/d);
	  //RealVec posms = distu * distms_from_1 + pos[i];
	  RealOpenMM fms = 0.5 * (1. + (rad1 - rad2)/d);
	  RealOpenMM gms = 1. - fms;
	  RealVec posms = pos[j] * fms + pos[i] * gms;
	  if(verbose_level > 3){
	    cout << "S  " << 10.0*posms[0] << " " << 10.0*posms[1] << " " << 10.0*posms[2] << " " << nms << " " << volms << endl;
	  }
	  MSParticle msp;
	  msp.vol = volmsw;
	  msp.vol0 = volmsw;
	  msp.pos = posms;
	  msp.parent1 = i;
	  msp.parent2 = j;
	  msp.gder = dist *  sder*(d-dms)*volms/(d*sigma*sigma); //used for volume derivatives
	  msp.hder = dist * 0.5*(rad1 - rad2)/(d*d*d); //used to compute positional derivatives
	  msp.fms = fms;
	  msparticles1.push_back(msp);
	  nms += 1;
	}

#ifdef NOTNOW
	if(nms == 7){
	  //test derivative of MS position


	  RealVec h = dist * 0.5*(rad1 - rad2)/(d*d*d);
	  RealOpenMM nfms = fms;
	  RealOpenMM ngms = gms;

	  RealVec displ = RealVec(0.001, -0.001, 0.001);
	  
	  RealVec derdispl = displ * ngms; //init with diagonal
	  RealOpenMM dd = displ.dot(h);
	  derdispl += dist * dd;
	  
	  //move second atom by a small amount
	  pos[i] += displ;

	  RealVec dist = pos[j] - pos[i];
	  RealOpenMM d = sqrt(dist.dot(dist));
	  double dms = rad1 + rad2 + 0.5*radw;
	  double volms0 = vol_coeff*q*q*volw;
	  double sigma = 0.5*sqrt(q)*radw;
	  double volms = volms0*exp(-0.5*(d-dms)*(d-dms)/(sigma*sigma));
	  RealOpenMM fms = 0.5 * (1. + (rad1 - rad2)/d);
	  RealOpenMM gms = 1. - fms;
	  RealVec posms = pos[j] * fms + pos[i] * gms;

	  cout << std::setprecision(10) << " Displacement of MS particle " <<  nms << " " << std::setw(12) <<  posms[0] - msparticles1[nms-1].pos[0] << " " << posms[1] - msparticles1[nms-1].pos[1] << " " <<   posms[2] - msparticles1[nms-1].pos[2] << endl;
	  
	  cout << std::setprecision(10) << " Same from derivative: " << std::setw(12) << derdispl[0] << " " << derdispl[1] << " " << derdispl[2] << endl;
	  

	}
#endif
	
      }
    }
  }

#ifdef NOTNOW
  {
    //test derivative of volms0 of MS sphere
    for(int itms = 0; itms < msparticles1.size(); itms++){
      int i = msparticles1[itms].parent1;
      int j = msparticles1[itms].parent2;
      
      RealVec displ = RealVec(0.003, -0.005, 0.003);
      RealVec save_pos = pos[j];
      pos[j] += displ;
      
      double rad1 = radii_vdw[i];
      double rad2 = radii_vdw[j];
      double q = sqrt(rad1*rad2)/radw;
      RealVec dist = pos[j] - pos[i];
      RealOpenMM d = sqrt(dist.dot(dist));
      double dms = rad1 + rad2 + 0.5*radw;
      double volms0 = vol_coeff*q*q*volw;
      double sigma = 0.5*sqrt(q)*radw;
      double volms = volms0*exp(-0.5*(d-dms)*(d-dms)/(sigma*sigma));
      double sp;
      double s = pol_switchfunc(volms, VOLMINMSA, VOLMINMSB, sp);
      double volmsw = volms*s;
      double de = -msparticles1[itms].gder.dot(displ);
      
      cout << "V0MS: " << volmsw - msparticles1[itms].vol << " " << de << endl;
      pos[j] = save_pos;
    }
  }
#endif
  
  //obtain free volumes of ms spheres by summing over atoms scaled by their self volumes
  //saves into new list those with non-zero volume
  vector<MSParticle> msparticles2;
  double ams = KFC/(radw*radw);
  GaussianVca gms, gatom, g12;
  double energy_ms1, vol_ms1; 
  for(int ims = 0; ims < msparticles1.size() ; ims++){
    gms.a = ams;
    gms.v = msparticles1[ims].vol;
    gms.c = msparticles1[ims].pos;
    RealOpenMM freevolms = msparticles1[ims].vol;
    double G0m = 0;
    for(int i=0;i<numParticles;i++){
      if(ishydrogen[i]>0) continue;
      RealOpenMM rad = radii_vdw[i];
      RealOpenMM ai = KFC/(rad*rad);
      RealOpenMM voli = self_volume[i];
      gatom.a = ai;
      gatom.v = voli;
      gatom.c = pos[i];
      RealOpenMM dVdr, dVdV, sfp;
      freevolms -= ogauss_alpha(gms, gatom, g12, dVdr, dVdV, sfp);
      G0m += sfp*g12.v;
    }
    if(freevolms > VOLMINMSA){
      MSParticle msp;
      double sp;
      msp = msparticles1[ims];
      double s = pol_switchfunc(freevolms, VOLMINMSA, VOLMINMSB, sp);
      msp.vol = freevolms*s;
      msp.ssp = s + sp*freevolms;
      msp.pos = gms.c;
      msp.G0 = G0m;
      msparticles2.push_back(msp);
      if(verbose_level > 3){
	cout << "O " << 10.0*msp.pos[0] << " " << 10.0*msp.pos[1] << " " <<  10.0*msp.pos[2] << " " << msparticles2.size()-1 << " " << msp.vol << " " << msp.parent1 << " " << msp.parent2 << endl;
}
    }
  }

  if(verbose_level > 0){
    cout << "Number of ms particles: " << msparticles1.size() << endl;
    cout << "Number of ms particles with Vf > 0: " << msparticles2.size() << endl;
  }



  // now get the self-volumes of MS particles among themselves
  // and add them to the self-volumes of the parent atoms
  
  if(msparticles2.size() > 0){

    // self-volumes of ms particles
    int num_ms = msparticles2.size();
    vector<RealOpenMM> radii_ms(num_ms);
    for(int i=0;i<num_ms;i++) radii_ms[i] = radw;
    vector<RealOpenMM> volumes_ms(num_ms);
    for(int i=0;i<num_ms;i++) volumes_ms[i] = msparticles2[i].vol;
    vector<RealOpenMM> gammas_ms(num_ms);
    for(int i=0;i<num_ms;i++) gammas_ms[i] = common_gamma/roffset;
    vector<int> ishydrogen_ms(num_ms);
    for(int i=0;i<num_ms;i++) ishydrogen_ms[i] = 0;
    vector<RealVec> pos_ms(num_ms);
    for(int i=0;i<num_ms;i++) pos_ms[i] = msparticles2[i].pos;
    GaussVol *gvolms = new GaussVol(msparticles2.size(), ishydrogen_ms);
    gvolms->setRadii(radii_ms);
    gvolms->setVolumes(volumes_ms);
    gvolms->setGammas(gammas_ms);
    gvolms->compute_tree(pos_ms);
    vector<RealVec> forces_ms(num_ms);
    vector<RealOpenMM> vol_dv_ms(num_ms), freevols_ms(num_ms), selfvols_ms(num_ms);
    gvolms->compute_volume(pos_ms, vol_ms1, energy_ms1, forces_ms, vol_dv_ms, freevols_ms, selfvols_ms);
    
    energy += w_evol_ms * energy_ms1;
    
    for(int i=0;i<num_ms;i++){
      msparticles2[i].selfvol = selfvols_ms[i];
    }
    
    if(verbose_level > 0){
      cout << "vol_ms 1: " << vol_ms1 << endl;
      cout << "energy_ms 1: " << energy_ms1 << endl;
    }

    if(verbose_level > 3){
      cout << "MS Self Volumes:" << endl;
      for(int i=0;i<num_ms;i++){
	cout << i << " " << selfvols_ms[i] << endl;
      }
    }

#ifdef NOTNOW
    //test of forces_ms
    vector<RealVec> forces_ms2(num_ms);
    double energy_save = energy_ms1;
    for(int test_atom = 0; test_atom < num_ms ; test_atom++){
      RealVec displ = RealVec(0.001, -0.001, 0.001);
      RealVec save_pos = pos_ms[test_atom];
      pos_ms[test_atom] += displ;
      gvolms->compute_tree(pos_ms);
      double vol_ms, energy_ms;
      gvolms->compute_volume(pos_ms, vol_ms, energy_ms, forces_ms2, vol_dv_ms, freevols_ms, selfvols_ms);
      cout << "DVX " << test_atom << " " << energy_ms - energy_save << " " << -forces_ms[test_atom].dot(displ) << endl;
      pos_ms[test_atom] = save_pos;
    }
#endif
    
#ifdef NOTNOW
    //test of vol_dv
    vector<RealOpenMM> vol_dv2(num_ms);
    double energy_save = energy_ms1;
    for(int test_atom = 0; test_atom < num_ms ; test_atom++){
      double deltav = 0.001*volumes_ms[test_atom];
      double save_vol = volumes_ms[test_atom];
      volumes_ms[test_atom] += deltav;
      gvolms->setVolumes(volumes_ms);
      gvolms->compute_tree(pos_ms);
      double vol_ms, energy_ms;
      gvolms->compute_volume(pos_ms, vol_ms, energy_ms, forces_ms, vol_dv2, freevols_ms, selfvols_ms);
      cout << "DVV " << test_atom << " " << energy_ms - energy_save << " " << deltav*vol_dv_ms[test_atom] << endl;
      volumes_ms[test_atom] = save_vol;
    }
#endif

#ifdef NOTNOW
    //test of gradient of energy_ms with respect to the change of Vol0 of atoms 
    vector<RealVec> forces_ms2(num_ms);
    vector<RealOpenMM> vol_dv2(num_ms);
    double energy_save = energy_ms1;
    for(int test_atom = 0; test_atom < num_ms ; test_atom++){
      double deltav = 0.001*msparticles2[test_atom].vol0;
      double save_vol = msparticles2[test_atom].vol0;
      double dems = deltav * vol_dv_ms[test_atom] * msparticles2[test_atom].ssp*(1. - msparticles2[test_atom].G0/msparticles2[test_atom].vol0);
      msparticles2[test_atom].vol0 += deltav;

      //compute new MS volume
      gms.a = ams;
      gms.v = msparticles2[test_atom].vol0;
      gms.c = msparticles2[test_atom].pos;
      RealOpenMM freevolms = msparticles2[test_atom].vol0;
      for(int i=0;i<numParticles;i++){
	RealOpenMM rad = radii_vdw[i];
	RealOpenMM ai = KFC/(rad*rad);
	RealOpenMM voli = self_volume[i];
	gatom.a = ai;
	gatom.v = voli;
	gatom.c = pos[i];
	RealOpenMM dVdr, dVdV, sfp;
	freevolms -= ogauss_alpha(gms, gatom, g12, dVdr, dVdV, sfp);
      }
      double sp;
      double s = pol_switchfunc(freevolms, VOLMINMSA, VOLMINMSB, sp);
      double save_vol_ms = volumes_ms[test_atom];
      volumes_ms[test_atom] = freevolms*s;
      
      gvolms->setVolumes(volumes_ms);
      gvolms->compute_tree(pos_ms);
      double vol_ms, energy_ms;
      gvolms->compute_volume(pos_ms, vol_ms, energy_ms, forces_ms2, vol_dv2, freevols_ms, selfvols_ms);

      cout << "DVV0 " << test_atom << " " << energy_ms - energy_save << " " << dems << endl;

      volumes_ms[test_atom] = save_vol_ms;
      msparticles2[test_atom].vol0 = save_vol;
    }
#endif

#ifdef NOTNOW
    vector<RealVec> der1p(numParticles);//debug
    for(int i = 0; i < numParticles; i++){
      der1p[i] = RealVec(0.,0.,0.);
    }//debug
#endif
    
    //forces for energy_ms due to MS particle displacement OK
    for(int ims = 0; ims < msparticles2.size() ; ims++){
      int i = msparticles2[ims].parent1;
      int j = msparticles2[ims].parent2;
      RealVec dist = pos[j] - pos[i];
      RealVec hder = msparticles2[ims].hder;
      double fms = msparticles2[ims].fms;
      double gms = 1. - fms;
      double evprod =  w_evol_ms * forces_ms[ims].dot(dist);
      force[i] +=  hder * (+w_evol_ms*evprod) + forces_ms[ims] *  w_evol_ms*gms;
      force[j] +=  hder * (-w_evol_ms*evprod) + forces_ms[ims] *  w_evol_ms*fms;


      //der1p[i] -= hder * (+w_evol_ms*evprod) + forces_ms[ims] *  w_evol_ms*gms;//debug
      //der1p[j] -= hder * (-w_evol_ms*evprod) + forces_ms[ims] *  w_evol_ms*fms;//debug

    }

    //forces of energy_ms wrt of changing MS volumes due to changes of overlap volumes OK
    for(int ims = 0; ims < msparticles2.size() ; ims++){
      int parent1 = msparticles2[ims].parent1;
      int parent2 = msparticles2[ims].parent2;
      RealVec gder = msparticles2[ims].gder;
      double ssp = msparticles2[ims].ssp;
      double vol = msparticles2[ims].vol;
      double vol0 = msparticles2[ims].vol0;
      double G0m = msparticles2[ims].G0;
      double fv = w_evol_ms*ssp*vol_dv_ms[ims]*(1. - G0m/vol0);
      force[parent1] -= gder * fv;
      force[parent2] += gder * fv;
      
      //der1p[parent1] += gder * fv;//debug
      //der1p[parent2] -= gder * fv;//debug
    }

    //forces of energy_ms wrt of changing MS volumes due to changes of overlap volumes OK
    vector<double> numsder(numParticles);
    for(int i=0;i<numParticles;i++) numsder[i] = 0.0;
    for(int i=0;i<numParticles;i++){
      if (ishydrogen[i]>0) continue;
      RealOpenMM rad = radii_vdw[i];
      RealOpenMM ai = KFC/(rad*rad);
      RealOpenMM voli = self_volume[i];
      gatom.a = ai;
      gatom.v = voli;
      gatom.c = pos[i];
      for(int ims = 0; ims < msparticles2.size() ; ims++){
	gms.a = ams;
	gms.v = msparticles2[ims].vol0;
	gms.c = msparticles2[ims].pos;
	double ssp = msparticles2[ims].ssp;
	RealOpenMM dVdr, dVdV, sfp;
	ogauss_alpha(gms, gatom, g12, dVdr, dVdV, sfp);
	force[i] += (gatom.c - gms.c) * w_evol_ms*ssp*sfp*dVdr*vol_dv_ms[ims];


	
	//der1p[i] -= (gatom.c - gms.c) * w_evol_ms*ssp*sfp*dVdr*vol_dv_ms[ims];//debug

	numsder[i] += w_evol_ms*ssp*sfp*g12.v*vol_dv_ms[ims];
	
      }

      numsder[i] /= -voli;
    }

    //derivatives of energy_ms from change in self volumes OK
    gvol->setGammas(numsder);
    gvol->rescan_tree_gammas();
    double volume, vol_energy;
    gvol->compute_volume(pos, volume, vol_energy, vol_force, vol_dv, free_volume, self_volume);
    for(int i = 0; i < numParticles; i++){
      force[i] += vol_force[i] * w_evol_ms;

      //der1p[i] -= vol_force[i] * w_evol_ms;//debug
    }

#ifdef NOTNOW
    {
      //test gradients
      RealVec displ = RealVec(-0.001, -0.001, 0.002);
      double old_energy_ms = energy_ms1;
      vector<MSParticle> msparticles3(num_ms);
      vector<RealOpenMM> volumes_ms3(num_ms);
      
      for(int test_atom = 0; test_atom < numParticles ; test_atom++){
      
	msparticles3 = msparticles2;
	volumes_ms3 = volumes_ms;
	
	int test_particle = test_atom;
	RealVec save_pos = pos[test_particle];
	pos[test_particle] += displ;

	//get new self-volumes //debug
	gvol->compute_tree(pos);
	double volume, vol_energy;
	gvol->compute_volume(pos, volume, vol_energy, vol_force, vol_dv, free_volume, self_volume);

	//get new vol0's
	//pos[test_particle] = save_pos;//debug
	for(int tims = 0; tims < msparticles3.size() ; tims++){
	  int i = msparticles3[tims].parent1;
	  int j = msparticles3[tims].parent2;
	  RealVec dist = pos[j] - pos[i];
	  RealOpenMM d = sqrt(dist.dot(dist));
	  double rad1 = radii_vdw[i];
	  double rad2 = radii_vdw[j];
	  double q = sqrt(rad1*rad2)/radw;
	  RealOpenMM fms = 0.5 * (1. + (rad1 - rad2)/d);
	  RealOpenMM gms = 1. - fms;
	  RealVec posms = pos[j] * fms + pos[i] * gms;
	  double dms = rad1 + rad2 + 0.5*radw;
	  double volms0 = vol_coeff*q*q*volw;
	  double sigma = 0.5*sqrt(q)*radw;
	  double volms = volms0*exp(-0.5*(d-dms)*(d-dms)/(sigma*sigma));
	  double sp;
	  double s = pol_switchfunc(volms, VOLMINMSA, VOLMINMSB, sp);
	  double volmsw = volms*s;
	  msparticles3[tims].vol0 = volmsw;
	  msparticles3[tims].pos = posms;
	}

	//get new free volumes 
	//pos[test_particle] = save_pos; //without changing the position of the test atom
	//pos[test_particle] += displ;
	for(int ims = 0; ims < msparticles3.size() ; ims++){
	  gms.a = ams;
	  gms.v = msparticles3[ims].vol0;
	  gms.c = msparticles3[ims].pos;
	  RealOpenMM freevolms = msparticles3[ims].vol0;
	  for(int i=0;i<numParticles;i++){
	    if(ishydrogen[i]>0) continue;
	    RealOpenMM rad = radii_vdw[i];
	    RealOpenMM ai = KFC/(rad*rad);
	    RealOpenMM voli = self_volume[i];
	    gatom.a = ai;
	    gatom.v = voli;
	    gatom.c = pos[i];
	    RealOpenMM dVdr, dVdV, sfp;
	    freevolms -= ogauss_alpha(gms, gatom, g12, dVdr, dVdV, sfp);
	  }
	  double sp;
	  double s = pol_switchfunc(freevolms, VOLMINMSA, VOLMINMSB, sp);
	  volumes_ms3[ims] = freevolms*s;
	}

	for(int i=0;i<num_ms;i++) pos_ms[i] = msparticles3[i].pos;
	
	gvolms->setVolumes(volumes_ms3);
	gvolms->compute_tree(pos_ms);
	double vol_ms, energy_ms;
	gvolms->compute_volume(pos_ms, vol_ms, energy_ms, forces_ms, vol_dv_ms, freevols_ms, selfvols_ms);


	cout << "DVX0 " << test_atom << " " << energy_ms - old_energy_ms << " " << der1p[test_particle].dot(displ) << endl;

	pos[test_particle] = save_pos; //restore particle position
      }
    }
#endif


    //no longer needs gvol instance
    delete gvolms;
  }

  // volume energy function 2 (small radii)
  RealOpenMM vol_energy2, volume2;
  
  gvol->setRadii(radii_vdw);
  
  for(int i = 0; i < numParticles; i++){
    nu[i] = -gammas[i]/roffset;
  }
  gvol->setGammas(nu);
  
  vector<RealOpenMM> volumes_vdw(numParticles);
  for(int i = 0; i < numParticles; i++){
    volumes_vdw[i] = ishydrogen[i]>0 ? 0.0 : 4.*M_PI*pow(radii_vdw[i],3)/3.;
  }
  gvol->setVolumes(volumes_vdw);
  
  gvol->rescan_tree_volumes(pos);
  gvol->compute_volume(pos, volume2, vol_energy2, vol_force, vol_dv, free_volume, self_volume);
  
  for(int i = 0; i < numParticles; i++){
    force[i] += vol_force[i] * w_evol;
  }
  energy += vol_energy2 * w_evol;
  if(verbose_level > 0){
    cout << "Volume energy 2: " << vol_energy2 << endl;
    cout << "Atom Surface area energy: " << vol_energy1 + vol_energy2 << endl;
  }

  //now main overlap tree is set up with small radii

  //constructs molecular surface particles for volume energy function 2 (small radii)

  msparticles1.clear();
  radw = solvent_radius;
  nms = 0;
  for(int i = 0; i < numParticles; i++){
    if(ishydrogen[i]>0) continue;
    double rad1 = radii_vdw[i];
    for(int j = i + 1; j < numParticles; j++){
      if(ishydrogen[j]>0) continue;
      double rad2 = radii_vdw[j];
      double q = sqrt(rad1*rad2)/radw;
      RealVec dist = pos[j] - pos[i];
      RealOpenMM d = sqrt(dist.dot(dist));
      //      cout << d << " " << rad1 + rad2 << " " << rad1 + rad2 + 2*radw << endl;
      if(d < rad1 + rad2 + 2*radw ){
	//pair ms volume when the two atoms overlap
	double dms = rad1 + rad2 + 0.5*radw;
	double volms0 = vol_coeff*q*q*volw;
	double sigma = 0.5*sqrt(q)*radw;
	double volms = volms0*exp(-0.5*(d-dms)*(d-dms)/(sigma*sigma));
	double sp;
	double s = pol_switchfunc(volms, VOLMINMSA, VOLMINMSB, sp);
	double volmsw = volms*s;
	double sder = s + volms*sp;
	if(volmsw > FLT_MIN){
	  //position of MS particle
	  //double distms_from_1 = 0.5*(d - rad1 - rad2) + rad1;//midpoint of the two surfaces
	  //RealVec distu = dist * (1/d);
	  //RealVec posms = distu * distms_from_1 + pos[i];
	  RealOpenMM fms = 0.5 * (1. + (rad1 - rad2)/d);
	  RealOpenMM gms = 1. - fms;
	  RealVec posms = pos[j] * fms + pos[i] * gms;
	  if(verbose_level > 3){
	    cout << "S  " << 10.0*posms[0] << " " << 10.0*posms[1] << " " << 10.0*posms[2] << " " << nms << " " << volms << endl;
	  }
	  MSParticle msp;
	  msp.vol = volmsw;
	  msp.vol0 = volmsw;
	  msp.pos = posms;
	  msp.parent1 = i;
	  msp.parent2 = j;
	  msp.gder = dist *  sder*(d-dms)*volms/(d*sigma*sigma); //used for volume derivatives
	  msp.hder = dist * 0.5*(rad1 - rad2)/(d*d*d); //used to compute positional derivatives
	  msp.fms = fms;
	  msparticles1.push_back(msp);
	  nms += 1;
	}

      }
    }
  }
    
  //obtain free volumes of ms spheres by summing over atoms scaled by their self volumes
  //saves into new list those with non-zero volume
  msparticles2.clear();
  ams = KFC/(radw*radw);
  double energy_ms2, vol_ms2;
  for(int ims = 0; ims < msparticles1.size() ; ims++){
    gms.a = ams;
    gms.v = msparticles1[ims].vol;
    gms.c = msparticles1[ims].pos;
    RealOpenMM freevolms = msparticles1[ims].vol;
    double G0m = 0;
    for(int i=0;i<numParticles;i++){
      if(ishydrogen[i]>0) continue;
      RealOpenMM rad = radii_vdw[i];
      RealOpenMM ai = KFC/(rad*rad);
      RealOpenMM voli = self_volume[i];
      gatom.a = ai;
      gatom.v = voli;
      gatom.c = pos[i];
      RealOpenMM dVdr, dVdV, sfp;
      freevolms -= ogauss_alpha(gms, gatom, g12, dVdr, dVdV, sfp);
      G0m += sfp*g12.v;
    }
    if(freevolms > VOLMINMSA){
      MSParticle msp;
      double sp;
      msp = msparticles1[ims];
      double s = pol_switchfunc(freevolms, VOLMINMSA, VOLMINMSB, sp);
      msp.vol = freevolms*s;
      msp.ssp = s + sp*freevolms;
      msp.pos = gms.c;
      msp.G0 = G0m;
      msparticles2.push_back(msp);
      if(verbose_level > 3){
	cout << "O " << 10.0*msp.pos[0] << " " << 10.0*msp.pos[1] << " " <<  10.0*msp.pos[2] << " " << msparticles2.size()-1 << " " << msp.vol << " " << msp.parent1 << " " << msp.parent2 << endl;
}
    }
  }

  if(verbose_level > 0){
    cout << "Number of ms particles: " << msparticles1.size() << endl;
    cout << "Number of ms particles with Vf > 0: " << msparticles2.size() << endl;
  }

  // now get the self-volumes of MS particles among themselves
  // and add them to the self-volumes of the parent atoms
  
  if(msparticles2.size() > 0){

    // self-volumes of ms particles
    int num_ms = msparticles2.size();
    vector<RealOpenMM> radii_ms(num_ms);
    for(int i=0;i<num_ms;i++) radii_ms[i] = radw;
    vector<RealOpenMM> volumes_ms(num_ms);
    for(int i=0;i<num_ms;i++) volumes_ms[i] = msparticles2[i].vol;
    vector<RealOpenMM> gammas_ms(num_ms);
    for(int i=0;i<num_ms;i++) gammas_ms[i] = -common_gamma/roffset;
    vector<int> ishydrogen_ms(num_ms);
    for(int i=0;i<num_ms;i++) ishydrogen_ms[i] = 0;
    vector<RealVec> pos_ms(num_ms);
    for(int i=0;i<num_ms;i++) pos_ms[i] = msparticles2[i].pos;
    GaussVol *gvolms = new GaussVol(msparticles2.size(), ishydrogen_ms);
    gvolms->setRadii(radii_ms);
    gvolms->setVolumes(volumes_ms);
    gvolms->setGammas(gammas_ms);
    gvolms->compute_tree(pos_ms);
    vector<RealVec> forces_ms(num_ms);
    vector<RealOpenMM> vol_dv_ms(num_ms), freevols_ms(num_ms), selfvols_ms(num_ms);
    gvolms->compute_volume(pos_ms, vol_ms2, energy_ms2, forces_ms, vol_dv_ms, freevols_ms, selfvols_ms);
    
    energy += w_evol_ms * energy_ms2;
    
    for(int i=0;i<num_ms;i++){
      msparticles2[i].selfvol = selfvols_ms[i];
    }
    
    if(verbose_level > 0){
      cout << "vol_ms 2: " << vol_ms2 << endl;
      cout << "energy_ms 2: " << energy_ms2 << endl;

      cout << "MS Surface area energy: " << energy_ms1 + energy_ms2 << endl;
      cout << "Total Surface area energy: " << vol_energy1 + vol_energy2 + energy_ms1 + energy_ms2 << endl;
      
      cout << endl;
    }

    

    
    if(verbose_level > 3){
      cout << "MS Self Volumes:" << endl;
      for(int i=0;i<num_ms;i++){
	cout << i << " " << selfvols_ms[i] << endl;
      }
    }

    //forces for energy_ms due to MS particle displacement OK
    for(int ims = 0; ims < msparticles2.size() ; ims++){
      int i = msparticles2[ims].parent1;
      int j = msparticles2[ims].parent2;
      RealVec dist = pos[j] - pos[i];
      RealVec hder = msparticles2[ims].hder;
      double fms = msparticles2[ims].fms;
      double gms = 1. - fms;
      double evprod =  w_evol_ms * forces_ms[ims].dot(dist);
      force[i] +=  hder * (+w_evol_ms*evprod) + forces_ms[ims] *  w_evol_ms*gms;
      force[j] +=  hder * (-w_evol_ms*evprod) + forces_ms[ims] *  w_evol_ms*fms;
    }

    //forces of energy_ms wrt of changing MS volumes due to changes of overlap volumes OK
    for(int ims = 0; ims < msparticles2.size() ; ims++){
      int parent1 = msparticles2[ims].parent1;
      int parent2 = msparticles2[ims].parent2;
      RealVec gder = msparticles2[ims].gder;
      double ssp = msparticles2[ims].ssp;
      double vol = msparticles2[ims].vol;
      double vol0 = msparticles2[ims].vol0;
      double G0m = msparticles2[ims].G0;
      double fv = w_evol_ms*ssp*vol_dv_ms[ims]*(1. - G0m/vol0);
      force[parent1] -= gder * fv;
      force[parent2] += gder * fv;
    }

    //forces of energy_ms wrt of changing MS volumes due to changes of overlap volumes OK
    vector<double> numsder(numParticles);
    for(int i=0;i<numParticles;i++) numsder[i] = 0.0;
    for(int i=0;i<numParticles;i++){
      if (ishydrogen[i]>0) continue;
      RealOpenMM rad = radii_vdw[i];
      RealOpenMM ai = KFC/(rad*rad);
      RealOpenMM voli = self_volume[i];
      gatom.a = ai;
      gatom.v = voli;
      gatom.c = pos[i];
      for(int ims = 0; ims < msparticles2.size() ; ims++){
	gms.a = ams;
	gms.v = msparticles2[ims].vol0;
	gms.c = msparticles2[ims].pos;
	double ssp = msparticles2[ims].ssp;
	RealOpenMM dVdr, dVdV, sfp;
	ogauss_alpha(gms, gatom, g12, dVdr, dVdV, sfp);
	force[i] += (gatom.c - gms.c) * w_evol_ms*ssp*sfp*dVdr*vol_dv_ms[ims];
	numsder[i] += w_evol_ms*ssp*sfp*g12.v*vol_dv_ms[ims];
      }

      numsder[i] /= -voli;
    }

    //derivatives of energy_ms from change in self volumes OK
    gvol->setGammas(numsder);
    gvol->rescan_tree_gammas();
    gvol->compute_volume(pos, volume1, vol_energy1, vol_force, vol_dv, free_volume, self_volume);
    for(int i = 0; i < numParticles; i++){
      force[i] += vol_force[i] * w_evol_ms;
    }

    //no longer needs gvol instance
    delete gvolms;
 
  
    //add ms self-volumes (with small radii) to parents
    vector<RealOpenMM> svadd(numParticles);
    for(int iat=0;iat<numParticles;iat++){
      svadd[iat] = 0.0;
    }
    
    for(int i=0;i<num_ms;i++) {
      int iat = msparticles2[i].parent1;
      int jat = msparticles2[i].parent2;
      svadd[iat] += 0.5*msparticles2[i].selfvol;
      svadd[jat] += 0.5*msparticles2[i].selfvol;
    }
    
    if(verbose_level > 2){
      cout << "Updated Self Volumes:" << endl;
      for(int iat=0;iat<numParticles;iat++){
	double r = 0.;
	if(self_volume[iat] > 0){
	  r = 100.0*svadd[iat]/self_volume[iat];
	}
	RealOpenMM rad = radii_vdw[iat];
	RealOpenMM vol = (4./3.)*M_PI*rad*rad*rad;
	cout << "SV " <<  iat << " " << self_volume[iat] << " " << self_volume[iat]+svadd[iat] << " " << r << " " << self_volume[iat]/vol << endl;
      }
    }
    
    for(int iat=0;iat<numParticles;iat++){
      self_volume[iat] += svadd[iat];
    }
    
  }



#ifdef NOTNOW
  //from this point on the calculation of the energy function,
  //with the exception of derivatives, is identical to AGBNP1
  
  //volume scaling factors from self volumes
  RealOpenMM tot_vol = 0;
  for(int i = 0; i < numParticles; i++){
    RealOpenMM rad = radii_vdw[i];
    RealOpenMM vol = (4./3.)*M_PI*rad*rad*rad;
    volume_scaling_factor[i] = self_volume[i]/vol;
    tot_vol += self_volume[i];
  }
  if(verbose_level > 0){
    cout << "Volume from self volumes + MS self volumes: " << tot_vol << endl;
  }

  //compute inverse Born radii, prototype, no cutoff
  RealOpenMM pifac = 1./(4.*M_PI);
  for(int i = 0; i < numParticles; i++){
    RealOpenMM rad = radii_vdw[i];
    RealOpenMM rad_add = pifac*svadd[i]/(rad*rad);//from Taylor expansion
    if(verbose_level > 3){
      cout << "Radd " << i << " " << 10*rad_add << endl;
    }
    inverse_born_radius[i] = 1./(rad + rad_add);
    for(int j = 0; j < numParticles; j++){
      if(i == j) continue;
      if(ishydrogen[j] > 0) continue;
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

  if(verbose_level > 2){
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
  
  if(verbose_level > 3){
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
      if(ishydrogen[j]>0) continue;
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
#endif
  
  //returns energy
  if(verbose) cout << "energy: " << energy << endl;
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
    if(h && ishydrogen[i] == 0){
      throw OpenMMException("updateParametersInContext: AGBNP plugin does not support changing heavy/hydrogen atoms.");
    }
    gammas[i] = g;
    if(h) gammas[i] = 0.0;
    vdw_alpha[i] = alpha;
    charge[i] = q;
  }
}
