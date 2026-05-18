#include "electron.hpp"

#include <cmath>
#include <stdexcept>

namespace ic {

double Electron::beta() const {
    if (gamma < 1.0) {
        throw std::runtime_error("electron gamma must be >= 1");
    }
    return std::sqrt(1.0 - 1.0 / (gamma * gamma));
}

Vec3 Electron::beta_vector() const {
    return direction.normalized() * beta();
}

Electron sample_isotropic_monoenergetic_electron(double gamma, std::mt19937_64& rng) {
    Electron electron;
    electron.gamma = gamma;
    electron.direction = sample_isotropic_direction(rng);
    return electron;
}

}  // namespace ic
