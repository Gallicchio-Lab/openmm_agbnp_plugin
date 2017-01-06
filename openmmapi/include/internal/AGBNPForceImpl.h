#ifndef OPENMM_AGBNPFORCEIMPL_H_
#define OPENMM_AGBNPFORCEIMPL_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM-AGBNP                            *
 * -------------------------------------------------------------------------- */

#include "AGBNPForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <utility>
#include <set>
#include <string>

namespace AGBNPPlugin {

class System;

/**
 * This is the internal implementation of AGBNPForce.
 */

class OPENMM_EXPORT_AGBNP AGBNPForceImpl : public OpenMM::ForceImpl {
public:
    AGBNPForceImpl(const AGBNPForce& owner);
    ~AGBNPForceImpl();
    void initialize(OpenMM::ContextImpl& context);
    const AGBNPForce& getOwner() const {
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
    const AGBNPForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace AGBNPPlugin

#endif /*OPENMM_AGBNPFORCEIMPL_H_*/
