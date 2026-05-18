#include "slab.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ic {

bool is_inside_slab(const Slab& slab, const Vec3& point, double tol) {
    return point.z >= slab.z_min - tol && point.z <= slab.z_max + tol;
}

double distance_to_upper_boundary(const Slab& slab, const Vec3& point, const Vec3& direction) {
    const Vec3 dir = direction.normalized();
    if (dir.z <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::max(0.0, (slab.z_max - point.z) / dir.z);
}

double distance_to_lower_boundary(const Slab& slab, const Vec3& point, const Vec3& direction) {
    const Vec3 dir = direction.normalized();
    if (dir.z >= 0.0) {
        return std::numeric_limits<double>::infinity();
    }
    return std::max(0.0, (slab.z_min - point.z) / dir.z);
}

BoundaryHit first_boundary_hit(const Slab& slab, const Vec3& point, const Vec3& direction) {
    if (!is_inside_slab(slab, point)) {
        throw std::runtime_error("boundary hit requested for a point outside the slab");
    }

    const Vec3 dir = direction.normalized();
    const double upper_distance = distance_to_upper_boundary(slab, point, dir);
    const double lower_distance = distance_to_lower_boundary(slab, point, dir);

    if (upper_distance < lower_distance) {
        return {SlabBoundaryFace::upper, upper_distance, point + dir * upper_distance};
    }
    if (lower_distance < upper_distance) {
        return {SlabBoundaryFace::lower, lower_distance, point + dir * lower_distance};
    }
    if (std::isfinite(upper_distance)) {
        return {SlabBoundaryFace::upper, upper_distance, point + dir * upper_distance};
    }
    return {};
}

const char* slab_boundary_face_name(SlabBoundaryFace face) {
    switch (face) {
        case SlabBoundaryFace::lower:
            return "lower";
        case SlabBoundaryFace::upper:
            return "upper";
        case SlabBoundaryFace::none:
        default:
            return "none";
    }
}

}  // namespace ic
