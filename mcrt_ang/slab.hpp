#pragma once

#include <limits>

#include "vec3.hpp"

namespace ic {

struct Slab {
    double z_min = 0.0;
    double z_max = 1.0;
};

enum class SlabBoundaryFace {
    none,
    lower,
    upper,
};

struct BoundaryHit {
    SlabBoundaryFace face = SlabBoundaryFace::none;
    double distance = std::numeric_limits<double>::infinity();
    Vec3 point{};
};

bool is_inside_slab(const Slab& slab, const Vec3& point, double tol = 1.0e-12);
double distance_to_upper_boundary(const Slab& slab, const Vec3& point, const Vec3& direction);
double distance_to_lower_boundary(const Slab& slab, const Vec3& point, const Vec3& direction);
BoundaryHit first_boundary_hit(const Slab& slab, const Vec3& point, const Vec3& direction);
const char* slab_boundary_face_name(SlabBoundaryFace face);

}  // namespace ic
