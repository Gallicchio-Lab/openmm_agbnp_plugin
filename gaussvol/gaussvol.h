/* -------------------------------------------------------------------------- *
 *                                 GaussVol                                   *
 * -------------------------------------------------------------------------- *
 * This file is part of the AGBNP3 implicit solvent model software            *
 * implementation funded by the National Science Foundation under grant:      *
 * NSF SI2 1440665  "SI2-SSE: High-Performance Software for Large-Scale       *
 * Modeling of Binding Equilibria"                                            *
 *                                                                            *
 * copyright (c) 2016 Emilio Gallicchio                                       *
 * Authors: Emilio Gallicchio <egallicchio@brooklyn.cuny.edu>                 *
 * Contributors:                                                              *
 *                                                                            *
 *  GaussVol is free software: you can redistribute it and/or modify          *
 *  it under the terms of the GNU Lesser General Public License version 3     *
 *  as published by the Free Software Foundation.                             *
 *                                                                            *
 *  GaussVol is distributed in the hope that it will be useful,               *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of            *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             *
 *  GNU General Public License for more details.                              *
 *                                                                            *
 *  You should have received a copy of the GNU General Public License         *
 *  along with GaussVol.  If not, see <http://www.gnu.org/licenses/>.         *
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
#define KFC (2.2269859253)
#define PFC (2.5);

#define PI (M_PI)

// have switching function
#define MIN_GVOL (FLT_MIN)

// maximum overlap level
#define MAX_ORDER (12)

//use nm and kj
#define ANG (0.1f)
#define ANG3 (0.001f)

//volume cutoffs in switching function
#define VOLMINA (0.01f*ANG3)
#define VOLMINB (0.1f*ANG3)

//radius offset for surf energy calc.
#define SA_DR (0.5f*ANG)

/* 3D Gaussian, V,c,a representation */
class GaussianVca {
 public:
  RealOpenMM v; /* Gaussian volume */
  RealOpenMM a; /* Gaussian exponent */
  RealVec  c; /* center */
};

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
};


class GOverlap_Tree {
 public:
  int natoms;
  vector<GOverlap> overlaps; //the root is at index 0, atoms are at 1..natoms+1
};

class GaussVol {

 public: 
  /* Creates/Initializes a GaussVol instance*/
  GaussVol(const int natoms, vector<RealOpenMM> &radii, 
  	   vector<RealOpenMM> &gammas, vector<bool> &ishydrogen_in);

  /* Terminate/Delete GaussVol  */
  ~GaussVol( void ){
    tree.overlaps.clear();
  };
  
  //reset atomic radii
  void set_radii(vector<RealOpenMM> &radii){
    radius1 = radii;
  }

  //reset atomic gammas
  void set_gammas(vector<RealOpenMM> &gammas){
    gamma = gammas;  
  }
  
  //constructs the tree
  void compute_tree(vector<RealVec> &positions);

  /* returns GaussVol volume area energy function and forces */
  /* also returns atomic free-volumes and self-volumes */
  void compute_volume(vector<RealVec> &positions,
		      RealOpenMM &volume,
		      RealOpenMM &energy,
		      vector<RealVec> &force,
		      vector<RealOpenMM> &free_volume,  vector<RealOpenMM> &self_volume);

  //rescan the tree resetting gammas and volumes
  void rescan_tree_volumes(vector<RealVec> &positions,
			   vector<RealOpenMM> &radius,
			   vector<RealOpenMM> &gamma);

  //rescan the tree resetting gammas only
  void rescan_tree_gammas(vector<RealOpenMM> &gamma);

  /* returns GaussVol surface area energy function and forces */
  /* also returns atomic free-volumes, self-volumes, and surface areas */
  void compute_surface(const int init,
		       vector<RealVec> &positions,
		       RealOpenMM &energy,
		       vector<RealVec> &force,
		       vector<RealOpenMM> &surface_areas);




  // returns number of overlaps for each atom 
  void getstat(vector<int>& nov, vector<int>& nov_2body);

 private:
  GOverlap_Tree tree;
  vector<RealOpenMM> radius1;
  vector<RealOpenMM> radius2;
  vector<RealOpenMM> gamma;
  vector<RealVec> grad1, grad2;
  vector<RealOpenMM> self_volume1, self_volume2;
  vector<RealOpenMM> free_volume1, free_volume2;
  vector<bool> ishydrogen;
};

#endif //GAUSSVOL_H
