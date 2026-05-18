#pragma once

#include <random>

#include "electron.hpp"
#include "photon.hpp"

namespace ic {

struct ScatterResult {
    Photon scattered_photon{};
    double outgoing_electron_gamma = 0.0;
    double initial_total_energy = 0.0;
    double final_total_energy = 0.0;
    double energy_error = 0.0;
    double outgoing_electron_mass2 = 0.0;
    double mass_shell_error = 0.0;
    double scattering_mu_lab = 0.0;
    double incoming_photon_energy_erf = 0.0;
    double outgoing_photon_energy_erf = 0.0;
    double photon_energy_ratio_erf = 0.0;
    bool success = false;
};

ScatterResult scatter_single_kn(const Photon& photon,
                                const Electron& electron,
                                std::mt19937_64& rng);
double sigma_kn_total_over_sigma_t(double photon_energy_mec2);

}  // namespace ic
