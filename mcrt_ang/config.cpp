#include "config.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "photon.hpp"

namespace ic {
namespace {

void require_value(int& i, int argc, const char* arg) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value after ") + arg);
    }
    ++i;
}

}  // namespace

double recommended_energy_max(const Config& cfg) {
    double characteristic_gamma = cfg.electron_gamma;
    if (cfg.electron_model == "thermal") {
        characteristic_gamma =
            std::max(1.0 + 12.0 * cfg.electron_kTe + 48.0 * cfg.electron_kTe * cfg.electron_kTe, 1.0);
    }

    const double epsilon =
        (cfg.seed_photon_model == "blackbody" || cfg.seed_photon_model == "bb")
            ? mean_blackbody_photon_energy_mec2(cfg.seed_temperature_eV)
            : cfg.incident_photon_energy;
    const double single_scatter_estimate =
        (4.0 * characteristic_gamma * characteristic_gamma * epsilon) /
        (1.0 + 4.0 * characteristic_gamma * epsilon);
    double energy_max = std::max(10.0 * epsilon, 4.0 * single_scatter_estimate);

    if (cfg.geometry_model == "slab" && cfg.max_scatters > 1) {
        // Multi-scatter slab runs can build a much broader high-energy tail than
        // single-scatter benchmarks. Use a simple widening rule that grows with
        // max_scatters but is also anchored to the electron energy scale.
        const double scatter_depth_factor =
            1.0 + std::sqrt(static_cast<double>(cfg.max_scatters));
        energy_max = std::max(energy_max * scatter_depth_factor, 2.0 * characteristic_gamma);
    }

    return energy_max;
}

std::string config_usage(const char* argv0) {
    return std::string("Usage: ") + argv0 + " [options]\n"
           "  --events N                Number of Monte Carlo photon histories\n"
           "  --seed N                  RNG seed\n"
           "  --photon-energy X         Incident photon energy in mec^2 units\n"
           "  --seed-photon-model NAME  monoenergetic or blackbody\n"
           "  --seed-temperature-eV X   Blackbody seed radiation temperature in eV\n"
           "  --geometry NAME           none or slab\n"
           "  --transport-cross-section NAME  thomson, kn, or thermal_kn for slab free-path sampling\n"
           "  --thermal-kn-table PATH   HDF5 lookup table for thermal_kn transport\n"
           "  --electron-model NAME     monoenergetic or thermal\n"
           "  --electron-gamma X        Monoenergetic electron Lorentz factor\n"
           "  --electron-kTe X          Thermal electron kT_e in units of m_e c^2\n"
           "  --slab-height X           Slab thickness H for geometry=slab\n"
           "  --slab-tau X              Vertical optical depth across the slab\n"
           "  --slab-injection NAME     beam, lambert, or internal_iso (or both for production-slab-high-tau-sweep)\n"
           "  --tau-list LIST           Comma-separated optical depths for sweep-slab-paper\n"
           "  --injection-list LIST     Comma-separated injection modes for sweep-slab-paper\n"
           "  --max-scatters N          Maximum number of scatterings tracked in slab mode\n"
           "  --energy-bins N           Number of scattered-energy histogram bins\n"
           "  --energy-bin-spacing NAME Histogram spacing for scattered energy: linear or log\n"
           "  --mu-bins N               Number of angular histogram bins\n"
           "  --energy-min X            Lower histogram edge for scattered energy\n"
           "  --energy-max X            Upper histogram edge for scattered energy\n"
           "  --thermal-kn-energy-points N  Energy-grid size for thermal_kn table generation\n"
           "  --thermal-kn-theta-points N   Temperature-grid size for thermal_kn table generation\n"
           "  --thermal-kn-z-points N       Gamma quadrature size in scaled z variable\n"
           "  --thermal-kn-mu-points N      Angle quadrature size for thermal_kn table generation\n"
           "  --thermal-kn-z-max X          Upper scaled-z cutoff for thermal_kn table generation\n"
           "  --thermal-kn-energy-min X     Minimum photon energy for thermal_kn table, in mec^2\n"
           "  --thermal-kn-energy-max X     Maximum photon energy for thermal_kn table, in mec^2\n"
           "  --thermal-kn-theta-min X      Minimum electron kT_e for thermal_kn table, in mec^2\n"
           "  --thermal-kn-theta-max X      Maximum electron kT_e for thermal_kn table, in mec^2\n"
           "  --mode NAME               run, sweep-slab-paper, generate-thermal-kn-transport-table, validate-thermal-kn-transport-table, production-slab-thermal-case, production-slab-thermal-sweep, production-slab-high-tau-sweep, production-slab-high-tau-dense-split, production-slab-high-tau-dense-sweep, production-slab-seed-energy-dense-split, production-slab-seed-energy-dense-sweep, production-slab-seed-energy-broad-split, production-slab-seed-energy-broad-sweep, production-slab-thermal-refined-split, production-slab-thermal-refined-sweep, production-slab-thermal-expanded-split, production-slab-thermal-expanded-sweep, validate-thomson, validate-kn, validate-conservation, validate-thermal, validate-thermal-sweep, validate-slab-thin, validate-slab-moderate, validate-slab-tau-sweep, validate-slab-electron-sweep, validate-slab-multi-thin, validate-slab-multi-moderate, validate-slab-multi-convergence, validate-slab-high-tau-convergence, validate-slab-high-tau-seed-energy, validate-slab-transport-cross-section, validate-slab-paper, validate-all\n"
           "  --label TEXT              Optional run label added to output files\n"
           "  --output-dir PATH         Directory for CSV outputs\n"
           "  --help                    Show this message\n";
}

Config parse_config(int argc, char** argv) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--events") {
            require_value(i, argc, argv[i - 1]);
            cfg.num_events = std::stoull(argv[i]);
        } else if (arg == "--seed") {
            require_value(i, argc, argv[i - 1]);
            cfg.seed = std::stoull(argv[i]);
        } else if (arg == "--photon-energy") {
            require_value(i, argc, argv[i - 1]);
            cfg.incident_photon_energy = std::stod(argv[i]);
        } else if (arg == "--seed-photon-model") {
            require_value(i, argc, argv[i - 1]);
            cfg.seed_photon_model = argv[i];
        } else if (arg == "--seed-temperature-eV") {
            require_value(i, argc, argv[i - 1]);
            cfg.seed_temperature_eV = std::stod(argv[i]);
        } else if (arg == "--geometry") {
            require_value(i, argc, argv[i - 1]);
            cfg.geometry_model = argv[i];
        } else if (arg == "--transport-cross-section") {
            require_value(i, argc, argv[i - 1]);
            cfg.transport_cross_section = argv[i];
        } else if (arg == "--thermal-kn-table") {
            require_value(i, argc, argv[i - 1]);
            cfg.thermal_kn_table_path = argv[i];
        } else if (arg == "--electron-model") {
            require_value(i, argc, argv[i - 1]);
            cfg.electron_model = argv[i];
        } else if (arg == "--electron-gamma") {
            require_value(i, argc, argv[i - 1]);
            cfg.electron_gamma = std::stod(argv[i]);
        } else if (arg == "--electron-kTe") {
            require_value(i, argc, argv[i - 1]);
            cfg.electron_kTe = std::stod(argv[i]);
        } else if (arg == "--slab-height") {
            require_value(i, argc, argv[i - 1]);
            cfg.slab_height = std::stod(argv[i]);
        } else if (arg == "--slab-tau") {
            require_value(i, argc, argv[i - 1]);
            cfg.slab_optical_depth = std::stod(argv[i]);
        } else if (arg == "--slab-injection") {
            require_value(i, argc, argv[i - 1]);
            cfg.slab_injection_model = argv[i];
        } else if (arg == "--tau-list") {
            require_value(i, argc, argv[i - 1]);
            cfg.tau_list = argv[i];
        } else if (arg == "--injection-list") {
            require_value(i, argc, argv[i - 1]);
            cfg.injection_list = argv[i];
        } else if (arg == "--max-scatters") {
            require_value(i, argc, argv[i - 1]);
            cfg.max_scatters = static_cast<std::size_t>(std::stoull(argv[i]));
        } else if (arg == "--energy-bins") {
            require_value(i, argc, argv[i - 1]);
            cfg.energy_bins = static_cast<std::size_t>(std::stoull(argv[i]));
        } else if (arg == "--energy-bin-spacing") {
            require_value(i, argc, argv[i - 1]);
            cfg.energy_bin_spacing = argv[i];
        } else if (arg == "--mu-bins") {
            require_value(i, argc, argv[i - 1]);
            cfg.mu_bins = static_cast<std::size_t>(std::stoull(argv[i]));
        } else if (arg == "--energy-min") {
            require_value(i, argc, argv[i - 1]);
            cfg.energy_min = std::stod(argv[i]);
        } else if (arg == "--energy-max") {
            require_value(i, argc, argv[i - 1]);
            cfg.energy_max = std::stod(argv[i]);
        } else if (arg == "--thermal-kn-energy-points") {
            require_value(i, argc, argv[i - 1]);
            cfg.thermal_kn_energy_points = static_cast<std::size_t>(std::stoull(argv[i]));
        } else if (arg == "--thermal-kn-theta-points") {
            require_value(i, argc, argv[i - 1]);
            cfg.thermal_kn_theta_points = static_cast<std::size_t>(std::stoull(argv[i]));
        } else if (arg == "--thermal-kn-z-points") {
            require_value(i, argc, argv[i - 1]);
            cfg.thermal_kn_z_points = static_cast<std::size_t>(std::stoull(argv[i]));
        } else if (arg == "--thermal-kn-mu-points") {
            require_value(i, argc, argv[i - 1]);
            cfg.thermal_kn_mu_points = static_cast<std::size_t>(std::stoull(argv[i]));
        } else if (arg == "--thermal-kn-z-max") {
            require_value(i, argc, argv[i - 1]);
            cfg.thermal_kn_z_max = std::stod(argv[i]);
        } else if (arg == "--thermal-kn-energy-min") {
            require_value(i, argc, argv[i - 1]);
            cfg.thermal_kn_energy_min = std::stod(argv[i]);
        } else if (arg == "--thermal-kn-energy-max") {
            require_value(i, argc, argv[i - 1]);
            cfg.thermal_kn_energy_max = std::stod(argv[i]);
        } else if (arg == "--thermal-kn-theta-min") {
            require_value(i, argc, argv[i - 1]);
            cfg.thermal_kn_theta_min = std::stod(argv[i]);
        } else if (arg == "--thermal-kn-theta-max") {
            require_value(i, argc, argv[i - 1]);
            cfg.thermal_kn_theta_max = std::stod(argv[i]);
        } else if (arg == "--mode") {
            require_value(i, argc, argv[i - 1]);
            cfg.mode = argv[i];
        } else if (arg == "--label") {
            require_value(i, argc, argv[i - 1]);
            cfg.run_label = argv[i];
        } else if (arg == "--output-dir") {
            require_value(i, argc, argv[i - 1]);
            cfg.output_dir = argv[i];
        } else if (arg == "--help") {
            std::cout << config_usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (cfg.num_events == 0) {
        throw std::runtime_error("--events must be > 0");
    }
    if (cfg.incident_photon_energy <= 0.0) {
        throw std::runtime_error("--photon-energy must be > 0");
    }
    if (cfg.seed_photon_model != "monoenergetic" &&
        cfg.seed_photon_model != "mono" &&
        cfg.seed_photon_model != "blackbody" &&
        cfg.seed_photon_model != "bb") {
        throw std::runtime_error("--seed-photon-model must be monoenergetic or blackbody");
    }
    if (cfg.seed_temperature_eV <= 0.0) {
        throw std::runtime_error("--seed-temperature-eV must be > 0");
    }
    if (cfg.electron_gamma < 1.0) {
        throw std::runtime_error("--electron-gamma must be >= 1");
    }
    if (cfg.electron_kTe <= 0.0) {
        throw std::runtime_error("--electron-kTe must be > 0");
    }
    if (cfg.slab_height <= 0.0) {
        throw std::runtime_error("--slab-height must be > 0");
    }
    if (cfg.slab_optical_depth < 0.0) {
        throw std::runtime_error("--slab-tau must be >= 0");
    }
    if (cfg.energy_bins == 0 || cfg.mu_bins == 0) {
        throw std::runtime_error("histogram bin counts must be > 0");
    }
    if (cfg.energy_bin_spacing != "linear" &&
        cfg.energy_bin_spacing != "log" &&
        cfg.energy_bin_spacing != "logarithmic") {
        throw std::runtime_error("--energy-bin-spacing must be linear or log");
    }
    if (cfg.max_scatters == 0) {
        throw std::runtime_error("--max-scatters must be > 0");
    }
    if (cfg.geometry_model != "none" && cfg.geometry_model != "slab") {
        throw std::runtime_error("--geometry must be none or slab");
    }
    if (cfg.transport_cross_section != "thomson" &&
        cfg.transport_cross_section != "kn" &&
        cfg.transport_cross_section != "thermal_kn") {
        throw std::runtime_error("--transport-cross-section must be thomson, kn, or thermal_kn");
    }
    if (cfg.thermal_kn_table_path.empty()) {
        throw std::runtime_error("--thermal-kn-table must not be empty");
    }
    if (cfg.thermal_kn_energy_points < 2 || cfg.thermal_kn_theta_points < 2 ||
        cfg.thermal_kn_z_points == 0 || cfg.thermal_kn_mu_points == 0 ||
        !(cfg.thermal_kn_z_max > 0.0)) {
        throw std::runtime_error("thermal_kn table-generation settings are invalid");
    }
    if (!(cfg.thermal_kn_energy_min > 0.0) ||
        !(cfg.thermal_kn_energy_max > cfg.thermal_kn_energy_min) ||
        !(cfg.thermal_kn_theta_min > 0.0) ||
        !(cfg.thermal_kn_theta_max > cfg.thermal_kn_theta_min)) {
        throw std::runtime_error("thermal_kn table grid limits are invalid");
    }
    const bool high_tau_production_allows_both =
        cfg.mode == "production-slab-high-tau-sweep" ||
        cfg.mode == "production-slab-high-tau-dense-sweep";
    if (cfg.slab_injection_model != "beam" &&
        cfg.slab_injection_model != "lambert" &&
        cfg.slab_injection_model != "internal_iso" &&
        !(high_tau_production_allows_both && cfg.slab_injection_model == "both")) {
        throw std::runtime_error(
            high_tau_production_allows_both
                ? "--slab-injection must be beam, lambert, internal_iso, or both"
                : "--slab-injection must be beam, lambert, or internal_iso");
    }
    if (cfg.electron_model != "monoenergetic" && cfg.electron_model != "thermal") {
        throw std::runtime_error("--electron-model must be monoenergetic or thermal");
    }

    return cfg;
}

}  // namespace ic
