#pragma once

#include "vec3.hpp"

#include <random>
#include <string>

namespace ic {

struct Photon {
    double energy = 0.0;
    Vec3 direction{0.0, 0.0, 1.0};
};

Photon make_incident_monoenergetic_photon(
    double energy,
    const Vec3& direction = Vec3{0.0, 0.0, 1.0});

double ev_to_mec2(double energy_ev);
double blackbody_temperature_ev_to_mec2(double temperature_ev);
double mean_blackbody_photon_energy_mec2(double temperature_ev);
double sample_blackbody_photon_energy_mec2(double temperature_ev, std::mt19937_64& rng);
double sample_seed_photon_energy_mec2(const std::string& seed_photon_model,
                                      double monoenergetic_energy_mec2,
                                      double seed_temperature_ev,
                                      std::mt19937_64& rng);

}  // namespace ic
