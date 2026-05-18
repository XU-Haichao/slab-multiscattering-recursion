#include "slab_injection.hpp"

#include <cmath>
#include <stdexcept>

#include "constants.hpp"

namespace ic {

bool is_supported_slab_injection_model(const std::string& model) {
    return model == "beam" || model == "lambert" || model == "internal_iso";
}

Vec3 sample_slab_injection_direction(const std::string& model, std::mt19937_64& rng) {
    if (model == "beam") {
        return {0.0, 0.0, 1.0};
    }

    if (model == "lambert") {
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        // Cosine-law injection from the lower boundary: mu = cos(theta) has PDF 2 mu
        // on [0, 1], so the inverse CDF is mu = sqrt(u).
        const double mu = std::sqrt(unit(rng));
        const double phi = constants::two_pi * unit(rng);
        const double sin_theta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
        return {sin_theta * std::cos(phi), sin_theta * std::sin(phi), mu};
    }

    if (model == "internal_iso") {
        return sample_isotropic_direction(rng);
    }

    throw std::runtime_error("unsupported slab injection model: " + model);
}

Photon inject_photon_from_lower_boundary(double energy,
                                         const std::string& model,
                                         std::mt19937_64& rng) {
    return make_incident_monoenergetic_photon(energy, sample_slab_injection_direction(model, rng));
}

SlabInjectionSample sample_slab_injection(const Slab& slab,
                                          double energy,
                                          const std::string& model,
                                          std::mt19937_64& rng) {
    if (model == "beam" || model == "lambert") {
        return {{0.0, 0.0, slab.z_min}, inject_photon_from_lower_boundary(energy, model, rng)};
    }

    if (model == "internal_iso") {
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        // Uniform physical depth is equivalent to t0 ~ Uniform(0, tau) for the
        // constant-density slab optical-depth coordinate used by this code.
        const double z = slab.z_min + unit(rng) * (slab.z_max - slab.z_min);
        return {{0.0, 0.0, z},
                make_incident_monoenergetic_photon(energy, sample_isotropic_direction(rng))};
    }

    throw std::runtime_error("unsupported slab injection model: " + model);
}

}  // namespace ic
