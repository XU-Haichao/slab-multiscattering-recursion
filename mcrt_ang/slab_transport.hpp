#pragma once

#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "electron.hpp"
#include "kn_scatter.hpp"
#include "photon.hpp"
#include "slab.hpp"

namespace ic {

struct TransportCrossSectionSettings {
    std::string mode = "thomson";
    double electron_kTe_mec2 = 0.0;
    std::string thermal_kn_table_path;
};

struct PrescatterTransportResult {
    double free_path = 0.0;
    double distance_to_boundary_before_scatter = 0.0;
    SlabBoundaryFace boundary_face_before_scatter = SlabBoundaryFace::none;
    bool scattered = false;
    Vec3 scatter_position{};
};

struct ScatterEvent {
    Vec3 position{};
    Electron electron{};
    ScatterResult result{};
};

enum class SlabTransportTermination {
    none,
    escaped_upper,
    escaped_lower,
    hit_max_scatters,
};

struct SlabTransportResult {
    Vec3 final_position{};
    Photon final_photon{};
    std::vector<ScatterEvent> scatter_events;
    SlabBoundaryFace escape_face = SlabBoundaryFace::none;
    SlabTransportTermination termination = SlabTransportTermination::none;
    bool success = false;

    [[nodiscard]] bool escaped() const {
        return termination == SlabTransportTermination::escaped_upper ||
               termination == SlabTransportTermination::escaped_lower;
    }
};

double sample_free_path_in_slab(const Slab& slab,
                                double slab_optical_depth,
                                std::mt19937_64& rng,
                                double photon_energy_mec2,
                                const TransportCrossSectionSettings& transport_settings);
PrescatterTransportResult propagate_until_scatter_or_escape(
    const Slab& slab,
    double slab_optical_depth,
    const Vec3& position,
    const Vec3& direction,
    std::mt19937_64& rng,
    double photon_energy_mec2,
    const TransportCrossSectionSettings& transport_settings);
SlabTransportResult transport_photon_through_slab(
    const Slab& slab,
    double slab_optical_depth,
    const Vec3& injection_position,
    const Photon& initial_photon,
    std::size_t max_scatters,
    std::mt19937_64& rng,
    const ElectronSampler& electron_sampler,
    const TransportCrossSectionSettings& transport_settings);
const char* slab_transport_termination_name(SlabTransportTermination termination);

}  // namespace ic
