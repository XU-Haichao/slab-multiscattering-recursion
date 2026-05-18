#pragma once

#include <functional>
#include <random>

#include "photon.hpp"
#include "vec3.hpp"

namespace ic {

struct Electron {
    double gamma = 1.0;
    Vec3 direction{0.0, 0.0, 1.0};

    [[nodiscard]] double beta() const;
    [[nodiscard]] Vec3 beta_vector() const;
};

using ElectronSampler = std::function<Electron(const Photon&, std::mt19937_64&)>;

Electron sample_isotropic_monoenergetic_electron(double gamma, std::mt19937_64& rng);

}  // namespace ic
