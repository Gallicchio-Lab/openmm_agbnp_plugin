#ifndef OPENMM_GVolFORCEIMPL_H_
#define OPENMM_GVolFORCEIMPL_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM-GVol                            *
 * -------------------------------------------------------------------------- */

#include "GVolForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <utility>
#include <set>
#include <string>

namespace GVolPlugin {

class System;

/**
 * This is the internal implementation of GVolForce.
 */

class OPENMM_EXPORT_GVol GVolForceImpl : public OpenMM::ForceImpl {
public:
    GVolForceImpl(const GVolForce& owner);
    ~GVolForceImpl();
    void initialize(OpenMM::ContextImpl& context);
    const GVolForce& getOwner() const {
        return owner;
    }
    void updateContextState(OpenMM::ContextImpl& context) {
        // This force field doesn't update the state directly.
    }
    double calcForcesAndEnergy(OpenMM::ContextImpl& context,  bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters() {
        return std::map<std::string, double>(); // This force field doesn't define any parameters.
    }
    std::vector<std::string> getKernelNames();
    void updateParametersInContext(OpenMM::ContextImpl& context);
private:
    const GVolForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace GVolPlugin

#endif /*OPENMM_GVolFORCEIMPL_H_*/
