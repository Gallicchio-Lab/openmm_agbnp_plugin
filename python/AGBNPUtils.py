%pythoncode %{
#from __future__ import print_function
#import simtk.unit as unit

class AGBNPUtils(object):
    
    def __init__(self, system, dms):
        self.system = system
        self.hodist = 1.0 * angstrom
        self.hydration_sites = []
        self.nbforces = []
        self.custom_nbforces = []
        self.agbnp_force = dms._agbnp_force
        print("AGBNPUtils: Info: system has AGBNP force")
        for force in system.getForces():
            description = str(force)
            #test for presence of non-bonded force
            if isinstance(force, NonbondedForce):
                self.nbforces.append(force)
            #test for presence of custom non-bonded force
            if isinstance(force, CustomNonbondedForce):
                self.custom_nbforces.append(force)
                
    def addHydrogenBondingSite(self, heavyparticle, hydrogenparticle, distance):
        #
        # places a hydration site in a HB position along the heavy atom--hydrogen axis
        # at the given distance from the heavy atom.
        #
        wheavy    = 1. - distance/self.hodist
        whydrogen = distance/self.hodist
        vsite = TwoParticleAverageSite(heavyparticle, hydrogenparticle, wheavy, whydrogen);
        p = self.system.addParticle(0.0)
        self.system.setVirtualSite(p, vsite)
        self._addParticleToNonBondedForces(p)
        if self.agbnp_force:
            self._addParticleToAGBNPForce(p)
        self.hydration_sites.append(p)

    def _addParticleToNonBondedForces(self, particle):
        #
        # add the virtual particle to the non-bonded forces of the system
        # Assumes a custom non-bonded force of the same form as the OPLS
        # custom non-bonded force in desmonddmsfile.py
        #
        charge = 0.
        epsilon = 0. * kilocalorie_per_mole
        sigma = 1. * angstrom
        for force in self.nbforces:
            force.addParticle(charge, sigma, epsilon)
            print("adding particle to non-bonded force")
        for force in self.custom_nbforces:
            force.addParticle([sigma, epsilon])
            print("adding particle to custom non-bonded force")


    def _addParticleToAGBNPForce(self, particle):
        #
        # add the virtual particle to the AGBNP force of the system
        #
        radius = 0.15 # 1.5 A
        charge = 0.0
        gamma = 0.0
        alpha = 0.0
        hbw = -4.186 # 1 kcal/mol
        h_flag = False
        print("adding particle to AGBNP force")
        self.agbnp_force.addParticle(radius, gamma, alpha, charge, hbw, h_flag)


    
%}
