from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout
import os, time, shutil
from desmonddmsfile import *
from datetime import datetime


print("Started at: " + str(time.asctime()))
start=datetime.now()
shutil.copyfile('trpcage.dms','trpcage2.dms')
testDes = DesmondDMSFile('trpcage2.dms')
system = testDes.createSystem(nonbondedMethod=NoCutoff, OPLS = True, implicitSolvent='GVolSA')

#Choose Reference or OpenCL platform

#platform = Platform.getPlatformByName('Reference')
#prop = {}

platform = Platform.getPlatformByName('OpenCL')
prop = {"OpenCLPrecision" : "single"}
#prop= {"OpenCLPrecision" : "single", "OpenCLPlatformIndex" : "1", "OpenCLDeviceIndex": "0"};



print "Minimization/equilibration ..."

integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.0005*picoseconds)
simulation = Simulation(testDes.topology, system, integrator,platform)
print "Using platform %s" % simulation.context.getPlatform().getName()
simulation.context.setPositions(testDes.positions)
simulation.context.setVelocities(testDes.velocities)
state = simulation.context.getState(getEnergy = True)
print(state.getPotentialEnergy())

simulation.minimizeEnergy()
simulation.reporters.append(StateDataReporter(stdout, 10, step=True, potentialEnergy=True,totalEnergy=True,temperature=True))
simulation.step(1000)
positions = simulation.context.getState(getPositions=True).getPositions()
velocities = simulation.context.getState(getVelocities=True).getVelocities()

simulation.step(1000)
positions = simulation.context.getState(getPositions=True).getPositions()
velocities = simulation.context.getState(getVelocities=True).getVelocities()

print "Test energy conservation ..."

integrator = VerletIntegrator(0.001*picoseconds)
simulation = Simulation(testDes.topology, system, integrator,platform)
simulation.context.setPositions(positions)
simulation.context.setVelocities(velocities)
state = simulation.context.getState(getEnergy = True)
simulation.reporters.append(StateDataReporter(stdout, 10, step=True, potentialEnergy=True,totalEnergy=True,temperature=True))
simulation.step(1000)
positions = simulation.context.getState(getPositions=True).getPositions()
velocities = simulation.context.getState(getVelocities=True).getVelocities()

#print "Updating positions and velocities"
testDes.setPositions(positions)
testDes.setVelocities(velocities)
testDes.close()#
#
end=datetime.now()
elapsed=end - start
print("elapsed time="+str(elapsed.seconds+elapsed.microseconds*1e-6)+"s")
