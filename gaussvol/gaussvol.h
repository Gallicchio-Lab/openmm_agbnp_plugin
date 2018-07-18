/* -------------------------------------------------------------------------- *
 *                                 GaussVol                                   *
 * -------------------------------------------------------------------------- *
 * This file is part of the AGBNP/OpenMM implicit solvent model software      *
 * implementation funded by the National Science Foundation under grant:      *
 * NSF SI2 1440665  "SI2-SSE: High-Performance Software for Large-Scale       *
 * Modeling of Binding Equilibria"                                            *
 *                                                                            *
 * copyright (c) 2016 Emilio Gallicchio                                       *
 * Authors: Emilio Gallicchio <egallicchio@brooklyn.cuny.edu>                 *
 * Contributors:                                                              *
 *                                                                            *
 *  AGBNP/OpenMM is free software: you can redistribute it and/or modify      *
 *  it under the terms of the GNU Lesser General Public License version 3     *
 *  as published by the Free Software Foundation.                             *
 *                                                                            *
 *  AGBNP/OpenMM is distributed in the hope that it will be useful,           *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of            *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *
 *  GNU General Public License for more details.                              *
 *                                                                            *
 *  You should have received a copy of the GNU General Public License         *
 *  along with AGBNP/OpenMM.  If not, see <http://www.gnu.org/licenses/>      *
 *                                                                            *
 * -------------------------------------------------------------------------- */

#ifndef GAUSSVOL_H
#define GAUSSVOL_H

#include <cmath>
#include <cfloat>
#include <vector>
using std::vector;
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/SimTKOpenMMRealType.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"

using namespace OpenMM;

#define GAUSSVOL_OK (2)
#define GAUSSVOL_ERR (-1)

/* conversion factors from spheres to Gaussians */
#define KFC (2.2269859253f)
#define PFC (2.5f);

#define PI (M_PI)

// have switching function
#define MIN_GVOL (FLT_MIN)

// maximum overlap level
#define MAX_ORDER (8)

//use nm and kj
#define ANG (0.1f)
#define ANG3 (0.001f)

//volume cutoffs in switching function
#define VOLMINA (0.01f*ANG3)
#define VOLMINB (0.1f*ANG3)

/* 3D Gaussian, V,c,a representation */
class GaussianVca {
 public:
  RealOpenMM v; /* Gaussian volume */
  RealOpenMM a; /* Gaussian exponent */
  RealVec  c; /* center */
};

// switching function used in Gaussian overlap function
RealOpenMM pol_switchfunc(RealOpenMM gvol, RealOpenMM volmina, RealOpenMM volminb, RealOpenMM &sp);

/* overlap between two Gaussians represented by a (V,c,a) triplet
   V: volume of Gaussian
   c: position of Gaussian
   a: exponential coefficient

   g(x) = V (a/pi)^(3/2) exp(-a(x-c)^2)

   this version is based on V=V(V1,V2,r1,r2,alpha)
   alpha = (a1 + a2)/(a1 a2)

   dVdr is (1/r)*(dV12/dr)
   dVdV is dV12/dV1 
   dVdalpha is dV12/dalpha
   d2Vdalphadr is (1/r)*d^2V12/dalpha dr
   d2VdVdr is (1/r) d^2V12/dV1 dr

*/
RealOpenMM ogauss_alpha(GaussianVca &g1, GaussianVca &g2, GaussianVca &g12, RealOpenMM &dVdr, RealOpenMM &dVdV, RealOpenMM &sfp);

/* an overlap */
class GOverlap {
  public:
    int level;                      //level (0=root, 1=atoms, 2=2-body, 3=3-body, etc.)
    GaussianVca g;                  // Gaussian representing overlap
    RealOpenMM volume;                   //volume of overlap (also stores Psi1..i in GPU version)
    RealOpenMM dvv1;                     // derivative wrt volume of first atom (also stores F1..i in GPU version)
    RealVec  dv1;                   // derivative wrt position of first atom (also stores P1..i in GPU version) 
    RealOpenMM gamma1i;                  // sum gammai for this overlap
    RealOpenMM self_volume;              //self volume accumulator (also stores Psi'1..i in GPU version)
    RealOpenMM sfp;                     //switching function derivatives    
    int atom;                      // the atomic index of the last atom of the overlap list (i, j, k, ..., atom) 
                                   //    = (Parent, atom)
    int parent_index;              // index in tree list of parent overlap
    int children_startindex;       // start index in tree array of children
    int children_count;            // number of children
    void print_overlap(void);
};


/* overlap comparison function */
bool goverlap_compare( const GOverlap &overlap1, const GOverlap &overlap2);


/*
  A collection of, mainly, recursive routines to constructs and analyze the overlap tree.
  Not meant to be called directly. It is used by GaussVol.
 */
class GOverlap_Tree {
 public:
  GOverlap_Tree(int natoms){
    this->natoms = natoms;
  }

  ~GOverlap_Tree(void){
    overlaps.clear();
  }

  //intializes the tree at 1-body level
  int init_overlap_tree(vector<RealVec> &pos,
			vector<RealOpenMM> &radii, //atomic radii
			vector<RealOpenMM> &volumes, //atomic volumes
			vector<RealOpenMM> &gammas,
			vector<int> &ishydrogen);
  
  // adds to the tree the children of overlap identified by "parent_index" in the tree
  int add_children(int parent_index, vector<GOverlap> &children_overlaps);

  /* scans the siblings of overlap identified by "root_index" to create children overlaps,
   returns them into the "children_overlaps" buffer: (root) + (atom) -> (root, atom) */
  int compute_children(int root_index, vector<GOverlap> &children_overlaps);

  //grow the tree with more children starting at the given root slot (recursive)
  int compute_andadd_children_r(int root);
  
  //compute the tree starting from the 1-body level
  int compute_overlap_tree_r(vector<RealVec> &pos, vector<RealOpenMM> &radius,
			     vector<RealOpenMM> &volume,
			     vector<RealOpenMM> &gamma, vector<int> &ishydrogen);

  /* compute volumes, energy of the overlap at slot and calls itself recursively to get 
     the volumes of the children */
  int compute_volume_underslot2_r(
     int slot,
     RealOpenMM &psi1i, RealOpenMM &f1i, RealVec &p1i, //subtree accumulators for free volume
     RealOpenMM &psip1i, RealOpenMM &fp1i, RealVec &pp1i, //subtree accumulators for self volume
     RealOpenMM &energy1i, RealOpenMM &fenergy1i, RealVec &penergy1i, //subtree accumulators for volume-based energy
     vector<RealVec>  &dr,          //gradients of volume-based energy wrt to atomic positions
     vector<RealOpenMM>  &dv,          //gradients of volume-based energy wrt to atomic volumes				
     vector<RealOpenMM> &free_volume, //atomic free volumes
     vector<RealOpenMM> &self_volume  //atomic self volumes
				  );
  
  /* recursively traverses tree and computes volumes, etc. */
  int compute_volume2_r(vector<RealVec> &pos,
			RealOpenMM &volume, RealOpenMM &energy, 
			vector<RealVec> &dr,
			vector<RealOpenMM> &dv,
			vector<RealOpenMM> &free_volume,
			vector<RealOpenMM> &self_volume);

  /*rescan the sub-tree to recompute the volumes, does not modify the tree */
  int rescan_r(int slot);
  
  /*rescan the tree to recompute the volumes, does not modify the tree */
  int rescan_tree_v(vector<RealVec> &pos,
		    vector<RealOpenMM> &radii,
		    vector<RealOpenMM> &volumes,
		    vector<RealOpenMM> &gammas,
		    vector<int> &ishydrogen);

  /*rescan the sub-tree to recompute the gammas, does not modify the volumes nor the tree */
  int rescan_gamma_r(int slot);

  /*rescan the tree to recompute the gammas only, does not modify volumes and the tree */
  int rescan_tree_g(vector<RealOpenMM> &gammas);
  
  //print the contents of the tree
  void print_tree(void);

  //print the contents of the tree (recursive)
  void print_tree_r(int slot);

  //counts number of overlaps under the one given
  int nchildren_under_slot_r(int slot);

  int natoms;
  vector<GOverlap> overlaps; //the root is at index 0, atoms are at 1..natoms+1
};

/*
A class that implements the Gaussian description of an object (molecule) made of a overlapping spheres
 */
class GaussVol {
 public: 
  /* Creates/Initializes a GaussVol instance*/
  GaussVol(const int natoms,
	   vector<int> &ishydrogen){
    tree = new GOverlap_Tree(natoms);
    this->natoms = natoms;
    this->radii.resize(natoms);
    for(int i=0;i<natoms;i++) radii[i] = 1.;
    this->volumes.resize(natoms);
    for(int i=0;i<natoms;i++) volumes[i] = 0.;
    this->gammas.resize(natoms);
    for(int i=0;i<natoms;i++) gammas[i] = 0.;
    this->ishydrogen = ishydrogen;
  }
  GaussVol(const int natoms,
	   vector<RealOpenMM> &radii,
	   vector<RealOpenMM> &volumes,
	   vector<RealOpenMM> &gammas,
	   vector<int> &ishydrogen){
    tree = new GOverlap_Tree(natoms);
    this->natoms = natoms;
    this->radii = radii;
    this->volumes = volumes;
    this->gammas = gammas;
    this->ishydrogen = ishydrogen;
  }
  ~GaussVol(void){
    delete tree;
    radii.clear();
    volumes.clear();
    gammas.clear();
    ishydrogen.clear();
  }
  
  
  int setRadii(vector<RealOpenMM> &radii){
    if(natoms == radii.size()){
      this->radii = radii;
      return natoms;
    }else{
      throw OpenMMException("setRadii: number of atoms does not match");
      return -1;
    }
  }

  int setVolumes(vector<RealOpenMM> &volumes){
    if(natoms == volumes.size()){
      this->volumes = volumes;
      return natoms;
    }else{
      throw OpenMMException("setVolumes: number of atoms does not match");
      return -1;
    }
  }

  int setGammas(vector<RealOpenMM> &gammas){
    if(natoms == gammas.size()){
      this->gammas = gammas;
      return natoms;
    }else{
      throw OpenMMException("setGammas: number of atoms does not match");
      return -1;
    }
  }
  
  //constructs the tree
  void compute_tree(vector<RealVec> &positions);

  /* returns GaussVol volume energy function and forces */
  /* also returns gradients with respect to atomic volumes and 
     atomic free-volumes and self-volumes */
  void compute_volume(vector<RealVec> &positions,
		      RealOpenMM &volume,
		      RealOpenMM &energy,
		      vector<RealVec> &force,
		      vector<RealOpenMM> &gradV,
		      vector<RealOpenMM> &free_volume,  vector<RealOpenMM> &self_volume);

  //rescan the tree after resetting gammas, radii and volumes
  void rescan_tree_volumes(vector<RealVec> &positions);

  //rescan the tree resetting gammas only with current values
  void rescan_tree_gammas(void);

  // returns number of overlaps for each atom 
  void getstat(vector<int>& nov);

  void print_tree(void){
    tree->print_tree();
  }


  
 private:
  GOverlap_Tree *tree;

  int natoms;
  vector<RealOpenMM> radii;
  vector<RealOpenMM> volumes;
  vector<RealOpenMM> gammas;
  vector<int> ishydrogen;
};

#endif //GAUSSVOL_H
