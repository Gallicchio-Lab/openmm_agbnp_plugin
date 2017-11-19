#ifndef AGBNP_UTILS_H_
#define AGBNP_UTILS_H_

#include <iostream>
#include "gaussvol.h"
#include "openmm/internal/SplineFitter.h"

using namespace OpenMM;
using namespace std;


/* A simple open addressing hash table. It is implemented explicitly here mainly to prototype the GPU version.

   The table maps values (positive int's) to positions into an array. Positions in array are initially determined by
   applying an hashing function (value % hmask). If the location is occupied, another is tried by jumping to
   another until an empty location is found.
*/
class AGBNPHtable {
 public:

  // creates a table
 AGBNPHtable(const unsigned int size, const unsigned int jump){
   h_create(size, jump);
 }
 AGBNPHtable(const unsigned int size){
   h_create(size, 1);
 }

  //destroys table
  ~AGBNPHtable(){
    values.clear();
  }

  //enter a value, returns position in array, or -1 if table is full
  int h_enter(const unsigned int value){
    if(nvalues >= hsize) return -1;
    unsigned int k = (value & hmask);
    while(values[k] >= 0 && values[k] != value){
      k = ( (k + hjump) & hmask);
    }
    values[k] = value;
    nvalues += 1;
    return (int)k;
  }

  //finds position of a value, or -1 if not found
  int h_find(const unsigned int value){
    unsigned int k = (value & hmask);
    unsigned int ntries = 0;
    while(values[k] >= 0 &&  values[k] !=  value && ntries < hsize){
      k = ( (k+hjump) & hmask);
      ntries += 1;
    }
    if(values[k] < 0 || ntries >= hsize) return -1;
    return (int)k;
  }

  unsigned int size(void){
    return hsize;
  }

  void h_print(){
    for(unsigned int k = 0; k < hsize; k++){
      cout << k << ":" << values[k] << endl;
    }
  }

  // private:
  void h_create(const unsigned int size, const unsigned int jump){
    hsize = two2n_size(size);
    hmask = hsize - 1;
    hjump = jump;
    for(unsigned int k = 0; k < hsize; k++){
      values.push_back(-1); //-1 means empty
    }
    nvalues = 0;
  }

  //returns smallest power of 2 larger than the input
  unsigned int two2n_size(unsigned int m){
    unsigned int s = 1;
    unsigned int l = m;
    if(m<=0) return 0;
    while(s<l){
      s = (s<<1);
    }
    return s;
  }

  unsigned int hsize;
  unsigned int hmask;
  unsigned int hjump;
  unsigned int nvalues;
  vector<int> values;
};

//a lookup table based on OpenMM's spline functions
class AGBNPLookupTable {
 public:
  AGBNPLookupTable(vector<double> &x, vector<double> &y){
    xt = x;
    yt = y;
    SplineFitter::createNaturalSpline(xt, yt, y2t);
  }
  ~AGBNPLookupTable(){
    xt.clear();
    yt.clear();
    y2t.clear();
  }
  double eval(const double x){
    return SplineFitter::evaluateSpline(xt, yt, y2t, x);
  }
  double evalderiv(const double x){
    return SplineFitter::evaluateSplineDerivative(xt, yt, y2t, x);
  }
  vector<double> xt;
  vector<double> yt;
  vector<double> y2t;
};

//max distance in nanometers in Q4ij lookup table
//Q4ij=0 beyond this distance
#define AGBNP_I4LOOKUP_MAXA (2.0) 
//max points in table along distance
#define AGBNP_I4LOOKUP_NA (16)

//a lookup table for AGBNP's Q4ij function for a specific Ri/Rj combination
class AGBNPI4LookupTable {
 public:
  AGBNPI4LookupTable(const unsigned int size, 
		     const double rmin, const double rmax, 
		     const double Ri, const double Rj,
		     int version);
  double eval(const double x){
    return table->eval(x);
  }
  double evalderiv(const double x){
    return table->evalderiv(x);
  }
  AGBNPLookupTable *table;
  double switching_function(double x, double xa, double xb);
  double ogauss(double d2, double pi, double pj, double ai, double aj);
  double i4(double rij, double Ri, double Rj);
  double i4ov(double rij, double Ri, double Rj, double gvol12_factor);
};


//1. constructs Q4ij lookup tables for each (Ri,Rj) combination in system
//2. evaluates Q4ij given distance and Ri/Rj ratio
// Ri's are van der Waals radii (= Radii - Roffset)
// Rj = van der Waals radius + Roffset (enlarged radii)
// Radii in input are assumed to be Rj's: van der Waals radii + Roffset

//radius offset for Born radii calculation
//same as radius offset in GaussVol
#define AGBNP_RADIUS_INCREMENT (SA_DR)
#define AGBNP_RADIUS_PRECISION      (10000)
#define AGBNP_HB_RADIUS (1.4*ANG) //radius of a water molecule

class AGBNPI42DLookupTable {
 public:
  AGBNPI42DLookupTable(const vector<double>& Radii, const vector<bool>& ishydrogen,
		       const unsigned int size, 
		       const double rmin, const double rmax,
		       const unsigned int version);
  double eval(const double x, const int rad_typei, const int rad_typej);
  double evalderiv(const double x, const int rad_typei, const int rad_typej);
  vector<AGBNPI4LookupTable*> tables;
  //tables are accessed using type_screened * ntypes_screener + type_screener
  int ntypes_screened; 
  int ntypes_screener;
  vector<int> radius_type_screened; //"screened" radius types for each atom
  vector<int> radius_type_screener; //"screener" radius types for each atom

  //compares two radii with some precision (rad_precision is set by default in class constructor)
  struct compare_pp10t {
    bool operator() (const double& dlhs, const double& drhs) const{
      long int ilhs = dlhs * AGBNP_RADIUS_PRECISION;
      long int irhs = drhs * AGBNP_RADIUS_PRECISION;
      return ilhs < irhs;
    }
  };

};

#endif /* AGBNP_UTILS_H_ */
