#include "kinematics.hpp"

#include <cmath>
#include <stdexcept>

namespace ic {

FourVector operator+(const FourVector& lhs, const FourVector& rhs) {
    return {lhs.t + rhs.t, lhs.x + rhs.x};
}

FourVector operator-(const FourVector& lhs, const FourVector& rhs) {
    return {lhs.t - rhs.t, lhs.x - rhs.x};
}

FourVector photon_four_vector(const Photon& photon) {
    return {photon.energy, photon.direction.normalized() * photon.energy};
}

FourVector electron_four_vector(const Electron& electron) {
    const Vec3 direction = electron.direction.normalized();
    const double beta = electron.beta();
    return {electron.gamma, direction * (electron.gamma * beta)};
}

FourVector boost_to_rest_frame(const FourVector& v, const Vec3& beta_vec) {
    const double beta2 = beta_vec.norm_squared();
    if (beta2 == 0.0) {
        return v;
    }

    const double gamma = 1.0 / std::sqrt(1.0 - beta2);
    const double beta_dot_p = dot(beta_vec, v.x);
    const Vec3 spatial = v.x + beta_vec * (((gamma - 1.0) * beta_dot_p / beta2) - gamma * v.t);
    return {gamma * (v.t - beta_dot_p), spatial};
}

FourVector boost_from_rest_frame(const FourVector& v, const Vec3& beta_vec) {
    const double beta2 = beta_vec.norm_squared();
    if (beta2 == 0.0) {
        return v;
    }

    const double gamma = 1.0 / std::sqrt(1.0 - beta2);
    const double beta_dot_p = dot(beta_vec, v.x);
    const Vec3 spatial = v.x + beta_vec * (((gamma - 1.0) * beta_dot_p / beta2) + gamma * v.t);
    return {gamma * (v.t + beta_dot_p), spatial};
}

double minkowski_mass_squared(const FourVector& v) {
    return v.t * v.t - v.x.norm_squared();
}

Photon photon_from_four_vector(const FourVector& v) {
    if (v.t <= 0.0) {
        throw std::runtime_error("scattered photon energy became non-positive");
    }
    return Photon{v.t, v.x.normalized()};
}

}  // namespace ic
