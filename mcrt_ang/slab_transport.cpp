#include "slab_transport.hpp"

#include <cstddef>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "thermal_kn_transport.hpp"

namespace ic {
namespace {

double transport_cross_section_ratio(double photon_energy_mec2,
                                     const TransportCrossSectionSettings& transport_settings) {
    if (transport_settings.mode == "thomson") {
        return 1.0;
    }
    if (transport_settings.mode == "kn") {
        return sigma_kn_total_over_sigma_t(photon_energy_mec2);
    }
    if (transport_settings.mode == "thermal_kn") {
        if (!(transport_settings.electron_kTe_mec2 > 0.0)) {
            throw std::runtime_error("thermal_kn transport requires electron_kTe_mec2 > 0");
        }
        if (transport_settings.thermal_kn_table_path.empty()) {
            throw std::runtime_error("thermal_kn transport requires a non-empty HDF5 table path");
        }
        const ThermalKnTransportTable& table =
            cached_thermal_kn_transport_table(transport_settings.thermal_kn_table_path);
        return interpolate_thermal_kn_transport_ratio(
            table, photon_energy_mec2, transport_settings.electron_kTe_mec2);
    }
    throw std::runtime_error("unsupported transport cross section: " + transport_settings.mode);
}

}  // namespace

double sample_free_path_in_slab(const Slab& slab,
                                double slab_optical_depth,
                                std::mt19937_64& rng,
                                double photon_energy_mec2,
                                const TransportCrossSectionSettings& transport_settings) {
    const double height = slab.z_max - slab.z_min;
    if (!(height > 0.0)) {
        throw std::runtime_error("slab height must be positive");
    }
    if (slab_optical_depth <= 0.0) {
        return std::numeric_limits<double>::infinity();
    }

    const double cross_section_ratio =
        transport_cross_section_ratio(photon_energy_mec2, transport_settings);
    const double attenuation = (slab_optical_depth / height) * cross_section_ratio;
    if (!(attenuation > 0.0)) {
        return std::numeric_limits<double>::infinity();
    }

    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const double u = unit(rng);
    return -std::log(u) / attenuation;
}

PrescatterTransportResult propagate_until_scatter_or_escape(const Slab& slab,
                                                            double slab_optical_depth,
                                                            const Vec3& position,
                                                            const Vec3& direction,
                                                            std::mt19937_64& rng,
                                                            double photon_energy_mec2,
                                                            const TransportCrossSectionSettings& transport_settings) {
    const Vec3 dir = direction.normalized();
    const BoundaryHit hit = first_boundary_hit(slab, position, dir);
    if (hit.face == SlabBoundaryFace::none) {
        throw std::runtime_error("failed to determine slab boundary hit");
    }

    const double free_path =
        sample_free_path_in_slab(slab,
                                 slab_optical_depth,
                                 rng,
                                 photon_energy_mec2,
                                 transport_settings);
    if (free_path < hit.distance) {
        return {free_path, hit.distance, hit.face, true, position + dir * free_path};
    }
    return {free_path, hit.distance, hit.face, false, hit.point};
}

SlabTransportResult transport_photon_through_slab(const Slab& slab,
                                                  double slab_optical_depth,
                                                  const Vec3& injection_position,
                                                  const Photon& initial_photon,
                                                  std::size_t max_scatters,
                                                  std::mt19937_64& rng,
                                                  const ElectronSampler& electron_sampler,
                                                  const TransportCrossSectionSettings& transport_settings) {
    if (max_scatters == 0) {
        throw std::runtime_error("max_scatters must be > 0");
    }
    if (!is_inside_slab(slab, injection_position)) {
        throw std::runtime_error("slab transport started outside the slab");
    }

    SlabTransportResult transport;
    transport.final_position = injection_position;
    transport.final_photon = initial_photon;

    // Preserve the validated single-scatter path exactly when max_scatters=1 so
    // existing slab validation modes remain backward-compatible.
    if (max_scatters == 1) {
        const PrescatterTransportResult prescatter =
            propagate_until_scatter_or_escape(slab,
                                              slab_optical_depth,
                                              injection_position,
                                              initial_photon.direction,
                                              rng,
                                              initial_photon.energy,
                                              transport_settings);
        if (!prescatter.scattered) {
            transport.final_position = prescatter.scatter_position;
            transport.escape_face = prescatter.boundary_face_before_scatter;
            transport.termination =
                prescatter.boundary_face_before_scatter == SlabBoundaryFace::upper
                    ? SlabTransportTermination::escaped_upper
                    : SlabTransportTermination::escaped_lower;
            transport.success = true;
            return transport;
        }

        const Electron electron = electron_sampler(initial_photon, rng);
        const ScatterResult result = scatter_single_kn(initial_photon, electron, rng);
        if (!result.success) {
            return transport;
        }

        transport.scatter_events.push_back({prescatter.scatter_position, electron, result});
        transport.final_position = prescatter.scatter_position;
        transport.final_photon = result.scattered_photon;

        const BoundaryHit exit_hit =
            first_boundary_hit(slab, prescatter.scatter_position, result.scattered_photon.direction);
        if (exit_hit.face == SlabBoundaryFace::none) {
            throw std::runtime_error("scattered photon did not hit a slab boundary");
        }

        transport.escape_face = exit_hit.face;
        transport.termination =
            exit_hit.face == SlabBoundaryFace::upper
                ? SlabTransportTermination::escaped_upper
                : SlabTransportTermination::escaped_lower;
        transport.success = true;
        return transport;
    }

    Vec3 position = injection_position;
    Photon photon = initial_photon;

    while (transport.scatter_events.size() < max_scatters) {
        const PrescatterTransportResult prescatter =
            propagate_until_scatter_or_escape(slab,
                                              slab_optical_depth,
                                              position,
                                              photon.direction,
                                              rng,
                                              photon.energy,
                                              transport_settings);
        if (!prescatter.scattered) {
            transport.final_position = prescatter.scatter_position;
            transport.final_photon = photon;
            transport.escape_face = prescatter.boundary_face_before_scatter;
            transport.termination =
                prescatter.boundary_face_before_scatter == SlabBoundaryFace::upper
                    ? SlabTransportTermination::escaped_upper
                    : SlabTransportTermination::escaped_lower;
            transport.success = true;
            return transport;
        }

        const Electron electron = electron_sampler(photon, rng);
        const ScatterResult result = scatter_single_kn(photon, electron, rng);
        if (!result.success) {
            return transport;
        }

        transport.scatter_events.push_back({prescatter.scatter_position, electron, result});
        position = prescatter.scatter_position;
        photon = result.scattered_photon;
    }

    transport.final_position = position;
    transport.final_photon = photon;
    transport.termination = SlabTransportTermination::hit_max_scatters;
    transport.success = true;
    return transport;
}

const char* slab_transport_termination_name(SlabTransportTermination termination) {
    switch (termination) {
        case SlabTransportTermination::escaped_upper:
            return "escaped_upper";
        case SlabTransportTermination::escaped_lower:
            return "escaped_lower";
        case SlabTransportTermination::hit_max_scatters:
            return "hit_max_scatters";
        case SlabTransportTermination::none:
        default:
            return "none";
    }
}

}  // namespace ic
