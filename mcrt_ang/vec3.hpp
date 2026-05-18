#pragma once

#include <cmath>
#include <random>
#include <stdexcept>

namespace ic {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    [[nodiscard]] double norm_squared() const {
        return x * x + y * y + z * z;
    }

    [[nodiscard]] double norm() const {
        return std::sqrt(norm_squared());
    }

    [[nodiscard]] Vec3 normalized() const {
        const double length = norm();
        if (!(length > 0.0)) {
            throw std::runtime_error("cannot normalize a zero-length vector");
        }
        return {x / length, y / length, z / length};
    }
};

inline Vec3 operator+(const Vec3& lhs, const Vec3& rhs) {
    return {lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z};
}

inline Vec3 operator-(const Vec3& lhs, const Vec3& rhs) {
    return {lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z};
}

inline Vec3 operator*(const Vec3& vec, double scale) {
    return {vec.x * scale, vec.y * scale, vec.z * scale};
}

inline Vec3 operator*(double scale, const Vec3& vec) {
    return vec * scale;
}

inline Vec3 operator/(const Vec3& vec, double scale) {
    return {vec.x / scale, vec.y / scale, vec.z / scale};
}

inline double dot(const Vec3& lhs, const Vec3& rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

inline Vec3 cross(const Vec3& lhs, const Vec3& rhs) {
    return {
        lhs.y * rhs.z - lhs.z * rhs.y,
        lhs.z * rhs.x - lhs.x * rhs.z,
        lhs.x * rhs.y - lhs.y * rhs.x,
    };
}

Vec3 sample_isotropic_direction(std::mt19937_64& rng);
Vec3 direction_from_axis(const Vec3& axis, double mu, double phi);

}  // namespace ic
