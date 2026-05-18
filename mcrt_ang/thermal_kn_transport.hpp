#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ic {

struct ThermalKnIntegrationSettings {
    std::size_t z_points = 64;
    std::size_t mu_points = 64;
    double z_max = 60.0;
};

struct ThermalKnTableGenerationSettings {
    std::size_t energy_points = 120;
    std::size_t theta_points = 80;
    double energy_min = 1.0e-6;
    double energy_max = 10.0;
    double theta_min = 1.0e-2;
    double theta_max = 1.0;
    ThermalKnIntegrationSettings integration{};
};

struct ThermalKnTransportTable {
    std::vector<double> energy_grid;
    std::vector<double> theta_grid;
    std::vector<double> log_ratio_table;
    ThermalKnIntegrationSettings integration{};
    std::string generator_version;
    std::string physics_definition_note;
    std::string creation_timestamp;

    [[nodiscard]] std::size_t energy_size() const { return energy_grid.size(); }
    [[nodiscard]] std::size_t theta_size() const { return theta_grid.size(); }
};

double compute_thermal_kn_transport_ratio(double photon_energy_mec2,
                                          double theta_e,
                                          const ThermalKnIntegrationSettings& settings);
ThermalKnTransportTable generate_thermal_kn_transport_table(
    const ThermalKnTableGenerationSettings& settings);
void write_thermal_kn_transport_table_hdf5(const std::string& path,
                                           const ThermalKnTransportTable& table);
ThermalKnTransportTable load_thermal_kn_transport_table_hdf5(const std::string& path);
double interpolate_thermal_kn_transport_ratio(const ThermalKnTransportTable& table,
                                              double photon_energy_mec2,
                                              double theta_e);
const ThermalKnTransportTable& cached_thermal_kn_transport_table(const std::string& path);

}  // namespace ic
