#include "vec3.hpp"

#include <algorithm>

#include "constants.hpp"

namespace ic {

Vec3 sample_isotropic_direction(std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const double mu = 2.0 * unit(rng) - 1.0;
    const double phi = constants::two_pi * unit(rng);
    const double sin_theta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
    return {sin_theta * std::cos(phi), sin_theta * std::sin(phi), mu};
}

Vec3 direction_from_axis(const Vec3& axis, double mu, double phi) {
    const Vec3 w = axis.normalized();
    const double sin_theta = std::sqrt(std::max(0.0, 1.0 - mu * mu));

    const Vec3 helper = (std::abs(w.z) < 0.9) ? Vec3{0.0, 0.0, 1.0} : Vec3{1.0, 0.0, 0.0};
    const Vec3 u = cross(helper, w).normalized();
    const Vec3 v = cross(w, u);

    return (u * (sin_theta * std::cos(phi)) +
            v * (sin_theta * std::sin(phi)) +
            w * mu)
        .normalized();
}

}  // namespace ic
