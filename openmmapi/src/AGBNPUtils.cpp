#include <cmath>
#include <vector>
#include <set>
#include "gaussvol.h"
#include "AGBNPForce.h"
#include "AGBNPUtils.h"
#include "openmm/OpenMMException.h"
#include <iostream>

using namespace AGBNPPlugin;
using namespace std;

double AGBNPI4LookupTable::switching_function(double x, double xa, double xb){
  if(x > xb) {
    return 0.0;
  }
  if(x < xa) {
    return 1.0;
  }
  double d = 1./(xb - xa);
  double u = (x - xa)*d;
  double u2 = u*u;
  double u3 = u*u2;
  return 1. - u3*(10.-15.*u+6.*u2);
}

double AGBNPI4LookupTable::ogauss(double d2, double pi, double pj, double ai, double aj){
  double deltai = 1./(ai+aj);
  double p = pi*pj;
  double kappa = exp(-ai*aj*d2*deltai);
  return p*kappa*pow(pi*deltai,1.5);
}

double AGBNPI4LookupTable::i4(double rij, double Ri, double Rj){ 
  double u1, u2, u3,u4,u5,u6,a, ad;
  double u4sq,u5sq;
  double rij2 = rij*rij;
  double q;
  static const double twopi = 2.0*M_PI;
  static const double twothirds = 2.0/3.0;

  if(rij>(Ri+Rj)){
    u1 = rij+Rj;
    u2 = rij-Rj;
    u3 = u1*u2;
    u4 = 0.5*log(u1/u2);
    q = twopi*(Rj/u3 - u4/rij);
  }else{
    u1 = Rj-Ri;
    if (rij2 > u1*u1){
      /* overlap */
      u1 = rij+Rj;
      u2 = rij-Rj;
      u3 = u1*u2;
      u4 = 1./u1;
      u4sq = u4*u4;
      u5 = 1./Ri;
      u5sq = u5*u5;
      u6 = 0.5*log(u1/Ri);
      q = twopi*(-(u4-u5) + (0.25*u3*(u4sq-u5sq) - u6)/rij);
    }else{
      /* inclusion */
      if(Ri>Rj){
	q = 0.0;
      }else{
	u1 = rij+Rj;
	u2 = Rj - rij;
	u3 = -u1*u2; /* rij^2 - Rj^2 */
	if(rij < .001*Rj){
	  // removable singularity of (1/2a)*log((1+a)/(1-a)) at a=0.
	  a = rij/Rj;
	  ad = a*a - 1;
	  u6 = (1. + twothirds*a*a)/Rj;
	  q = twopi*(2./Ri + Rj/u3 - u6); 
	}else{
	  u6 =  0.5*log(u1/u2);
	  q = twopi*(2./Ri + Rj/u3 - u6/(rij)); 
	}
      }
    }

  }

  return q;
}

double AGBNPI4LookupTable::i4ov(double rij, double Ri, double Rj, double gvol12_factor){
  double ai = KFC/(Ri*Ri);
  double pii = PFC;
  double aj = KFC/(Rj*Rj); 
  double pjj = PFC;
  double d2 = rij*rij;
  double gvol =  ogauss(d2, pii, pjj, ai, aj);
  double volj = 4.*M_PI*Rj*Rj*Rj/3.;
  double newRj = pow((volj+gvol12_factor*gvol)/volj,1./3.)*Rj;
  return i4(rij, Ri, newRj);
}

//lookup table for I4ij integral kernel
//using units of Rj, the result will be in units of Rj^-1
//to get the actual numerical value divide by Rj
AGBNPI4LookupTable::AGBNPI4LookupTable(const unsigned int size, 
				       const double rmin, const double rmax, 
				       const double Ri, const double Rj,
				       int version){
  //dimensionless parameters
  //  double dmin = rmin/Rj;
  //double dmax = rmax/Rj;
  //double ri = Ri/Rj;
  //double rj = Rj/Rj; //I know: it's 1

  //spacing
  double dr = (rmax - rmin)/(size-1);

  //limits of switching function
  double xa = 0.5*(rmax + rmin); //midpoint
  double xb = rmax;              //upper limit

  vector<double> x(size);
  vector<double> y(size);
  double gvol12_factor = 0.0; //default for AGBNP1
  //maybe not doing this after all
  //if(version == 2) gvol12_factor = 2.0; //default for AGBNP2
  for(unsigned int i = 0; i < size ; i++){
    x[i] = i*dr + rmin;
    double s = switching_function(x[i], xa, xb);
    y[i] = s*i4ov(x[i], Ri, Rj, gvol12_factor);
  }
  table = new AGBNPLookupTable(x,y);
}

//sets of overlap look-up tables indexed in term of bij = Ri/Rj
//input radii are van der Waals radii (small radii)
AGBNPI42DLookupTable::AGBNPI42DLookupTable(const vector<double>& Radii, const vector<int>& ishydrogen,
					   const unsigned int rnodes_count, 
					   const double rmin, const double rmax,
					   const unsigned int version){
  //constructs set of unique radii in system
  set<double, compare_pp10t> unique_radii_i, unique_radii_j;
  set<int> unique_keys;

  for(int i = 0; i < Radii.size() ; i++){
    unique_radii_i.insert(Radii[i]); //screened atom: small radii
  }
  //maybe not using this for AGBNP2 after all
  //double roffset = (version == 2) ? AGBNP_RADIUS_INCREMENT : 0.0;
  double roffset = 0.0; 
  for(int i = 0; i < Radii.size() ; i++){
    //screener atoms get "large" radii with AGBNP2
    if(!ishydrogen[i]) unique_radii_j.insert(Radii[i]+roffset); //hydrogens never descreen
  }
  ntypes_screened = unique_radii_i.size();
  ntypes_screener = unique_radii_j.size();

  //tables are accessed by typei * ntypes_screener + typej
  //where typei is the type id assigned to Ri and typej the id assigned to Rj
  tables.resize(ntypes_screened*ntypes_screener); //holds list of tables
  for(int i = 0; i < tables.size() ; i++){
    tables[i] = (AGBNPI4LookupTable *)0;
  }

  //constructs lookup tables for all ratios of radii
  set<double, compare_pp10t>::iterator it, jt;
  for (it = unique_radii_i.begin(); it != unique_radii_i.end(); it++) {
    double Ri = *it;
    int typei = distance(unique_radii_i.begin(),it);
    for (jt = unique_radii_j.begin(); jt != unique_radii_j.end(); jt++) {
      double Rj = *jt;
      int typej = distance(unique_radii_j.begin(),jt);
      unsigned int index = typei * ntypes_screener + typej;
      tables[index] = new AGBNPI4LookupTable(rnodes_count, rmin, rmax, Ri, Rj, version);
    }
  }

  //assign types to each atom
  int numParticles = Radii.size();
  //screened types
  radius_type_screened.resize(numParticles);
  for(int i = 0; i < numParticles; i++){
    it = unique_radii_i.find(Radii[i]);
    if( it == unique_radii_i.end())
      throw OpenMMException("AGBNPI42DLookupTable(): internal error: van der Waals radius not in list");
    int typei = distance(unique_radii_i.begin(),it);
    radius_type_screened[i] = typei;
  }
  //screener types
  radius_type_screener.resize(numParticles);
  for(int i = 0; i < numParticles; i++){
    radius_type_screener[i] = -1;//invalid index for hydrogens
  }
  for(int i = 0; i < numParticles; i++){
    if(ishydrogen[i]) continue;
    jt = unique_radii_j.find(Radii[i]+roffset);
    if( jt == unique_radii_j.end())
      throw OpenMMException("AGBNPI42DLookupTable(): internal error: enlarged van der Waals radius not in list");
    int typej = distance(unique_radii_j.begin(),jt);
    radius_type_screener[i] = typej;
  }

}

double AGBNPI42DLookupTable::eval(const double x, const int rad_typei, const int rad_typej){
  if(rad_typei < 0 || rad_typej < 0 || rad_typei >= ntypes_screened || rad_typej >= ntypes_screener)
    throw OpenMMException("AGBNPI42DLookupTable::eval(): invalid radius type index");
  int index = rad_typei * ntypes_screener + rad_typej;
  return tables[index]->eval(x);
}

double AGBNPI42DLookupTable::evalderiv(const double x, const int rad_typei, const int rad_typej){
  if(rad_typei < 0 || rad_typej < 0 || rad_typei >= ntypes_screened || rad_typej >= ntypes_screener)
    throw OpenMMException("AGBNPI42DLookupTable::evalderiv(): invalid radius type index");  
  int index = rad_typei * ntypes_screener + rad_typej;
  return tables[index]->evalderiv(x);
}

