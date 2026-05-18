#pragma once

#include <random>

#include "electron.hpp"

namespace ic {

struct ThermalElectronParams {
    double theta_e = 0.1;
};

Electron sample_isotropic_maxwell_juttner_electron(const ThermalElectronParams& params,
                                                   std::mt19937_64& rng);
Electron sample_conditioned_thermal_electron_for_scatter(const Photon& photon,
                                                         const ThermalElectronParams& params,
                                                         std::mt19937_64& rng);

}  // namespace ic
