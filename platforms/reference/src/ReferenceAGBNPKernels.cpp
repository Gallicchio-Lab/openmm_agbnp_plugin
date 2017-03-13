/* -------------------------------------------------------------------------- *
 *                               OpenMM-AGBNP                                *
 * -------------------------------------------------------------------------- */

#include <iostream>
#include <cstdlib>
#include <cmath>
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
   and otherwise beta' = beta^3/(beta^2+a^2) + minbeta
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

    cout << "In initialize ..." << numParticles <<  endl;

    //input lists
    positions.resize(numParticles);
    radii.resize(numParticles);
    gammas.resize(numParticles);
    vdw_alpha.resize(numParticles);
    ishydrogen.resize(numParticles);

    //output lists
    free_volume.resize(numParticles);
    self_volume.resize(numParticles);
    surface_areas.resize(numParticles);
    surf_force.resize(numParticles);
    
    double ang2nm = 0.1;
    for (int i = 0; i < numParticles; i++){
      double r, g, alpha;
      bool h;
      force.getParticleParameters(i, r, g, alpha, h);
      radii[i] = r;
      gammas[i] = g;
      vdw_alpha[i] = alpha;
      ishydrogen[i] = h;
    }

    //create and saves GaussVol instance
    gvol = new GaussVol(numParticles, radii, gammas, ishydrogen);

    //initializes I4 lookup table for Born-radii calculation
    double rmin = 0.;
    double rmax = AGBNP_I4LOOKUP_MAXA;
    int i4size = AGBNP_I4LOOKUP_NA;
    i4_lut = new AGBNPI42DLookupTable(radii, ishydrogen, i4size, rmin, rmax);

    //volume scaling factors and born radii
    volume_scaling_factor.resize(numParticles);
    inverse_born_radius.resize(numParticles);
    inverse_born_radius_fp.resize(numParticles);
    born_radius.resize(numParticles);

    cout << "Done" << endl;
}

double ReferenceCalcAGBNPForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    vector<RealVec>& pos = extractPositions(context);
    vector<RealVec>& force = extractForces(context);
    RealOpenMM energy = 0.0;
    bool verbose = true;
    int init = 0;

    // surface area energy function
    gvol->enerforc(init, pos, surf_energy, surf_force, free_volume, self_volume, surface_areas);

    //returns energy and gradients from surface area energy function
    for(int i = 0; i < numParticles; i++){
      force[i] += surf_force[i];
    }
    energy += surf_energy;

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

    if(verbose){
      //to input to freesasa
      double nm2ang = 10.0;
      cout << "id x y z radius: (Ang)" << endl;
      for(int i = 0; i < numParticles; i++){
	double rad = radii[i];
	if(ishydrogen[i] == 1) rad = 0.0;
	cout << "xyzr: " << i << " " << nm2ang*pos[i][0] << " " << nm2ang*pos[i][1] << " " << nm2ang*pos[i][2] << " " << nm2ang*rad << endl;
      }
    }


    if(verbose){
      cout << "Surface areas:" << endl;
      double tot_surf_area = 0.0;
      for(int i = 0; i < numParticles; i++){
	cout << "SA " << i << " " << surface_areas[i] << endl;
	tot_surf_area += surface_areas[i];
      }
      cout << "Total surface area: (Ang^2) " << 100.0*tot_surf_area << endl;
      cout << "Energy: " << surf_energy << endl;
    }

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
    for(int i = 0; i < numParticles; i++){
      RealOpenMM rad = radii[i];
      RealOpenMM vol = (4./3.)*M_PI*rad*rad*rad;
      volume_scaling_factor[i] = self_volume[i]/vol;
    }    

    //compute inverse Born radii, prototype, no cutoff
    RealOpenMM pifac = 1./(4.*M_PI);
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
    
    if(verbose){
      cout << "Born radii:" << endl;
      RealOpenMM fp;
      for(int i = 0; i < numParticles; i++){
	cout << "BR " << i << " " << 10.*born_radius[i] << " Si " << volume_scaling_factor[i] << endl;
      }
    }

    //compute van der Waals energy
    RealOpenMM evdw = 0.;
    for(int i=0;i<numParticles; i++){
      evdw += vdw_alpha[i]/pow(born_radius[i]+AGBNP_HB_RADIUS,3);
    }
    if(verbose){
      cout << "Van der Waals energy: " << evdw << endl;
    }


    //returns energy
    return (double)energy;
}

void ReferenceCalcAGBNPForceKernel::copyParametersToContext(ContextImpl& context, const AGBNPForce& force) {
  std::vector<int> neighbors;
    if (force.getNumParticles() != numParticles)
        throw OpenMMException("updateParametersInContext: The number of AGBNP particles has changed");
    for (int i = 0; i < force.getNumParticles(); i++) {
      double r, g, alpha;
      bool h;
      force.getParticleParameters(i, r, g, alpha, h);
    }
}
