#pragma once

#include "electron.hpp"
#include "photon.hpp"

namespace ic {

struct FourVector {
    double t = 0.0;
    Vec3 x{};
};

FourVector operator+(const FourVector& lhs, const FourVector& rhs);
FourVector operator-(const FourVector& lhs, const FourVector& rhs);
FourVector photon_four_vector(const Photon& photon);
FourVector electron_four_vector(const Electron& electron);
FourVector boost_to_rest_frame(const FourVector& v, const Vec3& beta_vec);
FourVector boost_from_rest_frame(const FourVector& v, const Vec3& beta_vec);
double minkowski_mass_squared(const FourVector& v);
Photon photon_from_four_vector(const FourVector& v);

}  // namespace ic
