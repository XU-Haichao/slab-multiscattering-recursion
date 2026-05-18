#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace ic {

struct Config {
    std::uint64_t num_events = 5000;
    std::uint64_t seed = 123456789ULL;
    double incident_photon_energy = 1.0e-6;
    std::string seed_photon_model = "blackbody";
    double seed_temperature_eV = 5.0;
    std::string geometry_model = "none";
    std::string transport_cross_section = "thomson";
    std::string thermal_kn_table_path = "output/inv_compton_build/thermal_kn_transport_table.h5";
    std::string electron_model = "monoenergetic";
    double electron_gamma = 10.0;
    double electron_kTe = 0.1;
    double slab_height = 1.0;
    double slab_optical_depth = 0.1;
    std::string slab_injection_model = "beam";
    std::size_t max_scatters = 16;
    std::size_t energy_bins = 256;
    std::size_t mu_bins = 180;
    double energy_min = 0.0;
    double energy_max = 0.0;
    std::string energy_bin_spacing = "linear";
    std::size_t thermal_kn_energy_points = 120;
    std::size_t thermal_kn_theta_points = 80;
    std::size_t thermal_kn_z_points = 64;
    std::size_t thermal_kn_mu_points = 64;
    double thermal_kn_z_max = 60.0;
    double thermal_kn_energy_min = 1.0e-8;
    double thermal_kn_energy_max = 10.0;
    double thermal_kn_theta_min = 1.0e-3;
    double thermal_kn_theta_max = 10.0;
    std::string mode = "run";
    std::string run_label;
    std::string output_dir = "output/inv_compton";
    std::string tau_list;
    std::string injection_list;
};

double recommended_energy_max(const Config& cfg);
Config parse_config(int argc, char** argv);

}  // namespace ic
