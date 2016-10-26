from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import os, time, shutil
from desmonddmsfile import *
from datetime import datetime

shutil.copyfile('trpcage_2.dms','trpcage_3.dms')
testDes = DesmondDMSFile('trpcage_3.dms')
system = testDes.createSystem(nonbondedMethod=CutoffNonPeriodic,nonbondedCutoff=1*nanometer, OPLS = True, implicitSolvent='GVolSA')

#Choose Reference or OpenCL platform

#platform = Platform.getPlatformByName('Reference')
#prop = {}
platform = Platform.getPlatformByName('OpenCL')
prop = {"OpenCLPrecision" : "single"}
#prop= {"OpenCLPrecision" : "single", "OpenCLPlatformIndex" : "1", "OpenCLDeviceIndex": "0"};

integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.001*picoseconds)
simulation = Simulation(testDes.topology, system, integrator, platform, prop)
print "Using platform %s" % simulation.context.getPlatform().getName()

simulation.context.setPositions(testDes.positions)
simulation.context.setVelocities(testDes.velocities)
state = simulation.context.getState(getEnergy = True)
simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True,totalEnergy=True,temperature=True))

start=datetime.now()
simulation.step(10000)
end=datetime.now()
elapsed=end - start
print("elapsed time="+str(elapsed.seconds+elapsed.microseconds*1e-6)+"s")

positions = simulation.context.getState(getPositions=True).getPositions()
velocities = simulation.context.getState(getVelocities=True).getVelocities()

#print "Updating positions and velocities"
testDes.setPositions(positions)
testDes.setVelocities(velocities)
testDes.close()#
