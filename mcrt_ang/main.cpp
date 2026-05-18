#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "constants.hpp"
#include "config.hpp"
#include "electron.hpp"
#include "histogram.hpp"
#include "kn_scatter.hpp"
#include "photon.hpp"
#include "slab.hpp"
#include "slab_injection.hpp"
#include "slab_transport.hpp"
#include "thermal_kn_transport.hpp"
#include "thermal_electron.hpp"

namespace {

constexpr double kOverflowWarningThreshold = 1.0e-3;
constexpr double kValidationOverflowAcceptableThreshold = 5.0e-2;
constexpr double kThermalSweepIncidentPhotonEnergy = 1.0e-6;
constexpr double kBeamThinTransmissionTolerance = 2.0e-2;
constexpr double kConvergenceMeanScatterCountTolerance = 5.0e-2;
constexpr double kConvergenceEscapeFractionTolerance = 2.0e-2;
constexpr double kConvergenceMeanScatteredEnergyTolerance = 5.0e-2;
constexpr double kHighTauConvergenceMeanScatterCountAbsTolerance = 2.0e-1;
constexpr double kHighTauConvergenceMeanScatterCountRelTolerance = 5.0e-2;
constexpr double kHighTauConvergenceEscapeFractionTolerance = 2.5e-2;
constexpr double kHighTauConvergenceMeanScatteredEnergyAbsTolerance = 7.5e-2;
constexpr double kHighTauConvergenceMeanScatteredEnergyRelTolerance = 5.0e-2;
constexpr double kHighTauControlledTerminationFractionThreshold = 1.0e-2;
// Development-only threshold for "weak seed dependence" in the high-tau seed
// diagnostic. If the mean escaped scattered energy varies by less than 20%
// across a 1e3 seed-energy sweep, the output energy scale is no longer
// dominated by the seed scale itself.
constexpr double kHighTauSeedEnergyWeakDependenceThreshold = 2.0e-1;
constexpr double kKnTransportWeakDifferenceThreshold = 5.0e-2;
constexpr double kThermalKnLowEnergyTolerance = 5.0e-2;
constexpr double kThermalKnLowThetaMeanAbsTolerance = 5.0e-2;
constexpr double kThermalKnLowThetaMaxAbsTolerance = 1.0e-1;
constexpr double kThermalKnSmoothnessSecondDiffThreshold = 2.5e-1;
constexpr double kPaperValidationThinTau = 1.0e-3;
constexpr double kPaperValidationThinBeamP0Tolerance = 1.5e-2;
constexpr double kPaperValidationClosureTolerance = 1.0e-12;
constexpr double kPaperValidationSmallProbabilityTolerance = 1.5e-2;
constexpr double kPaperValidationInternalSymmetryTolerance = 2.0e-2;

struct AngularSpectrumWindow {
    std::string label;
    double theta_center_deg = 0.0;
    double theta_min_deg = 0.0;
    double theta_max_deg = 0.0;
};

const std::vector<AngularSpectrumWindow>& angular_spectrum_windows() {
    static const std::vector<AngularSpectrumWindow> windows{
        {"th000", 0.0, 0.0, 5.0},
        {"th030", 30.0, 25.0, 35.0},
        {"th060", 60.0, 55.0, 65.0},
    };
    return windows;
}

std::vector<ic::Histogram> make_angular_spectrum_histograms(const ic::Config& cfg) {
    std::vector<ic::Histogram> histograms;
    histograms.reserve(angular_spectrum_windows().size());
    const ic::HistogramSpacing spacing =
        ic::histogram_spacing_from_string(cfg.energy_bin_spacing);
    for (std::size_t i = 0; i < angular_spectrum_windows().size(); ++i) {
        histograms.emplace_back(cfg.energy_bins, cfg.energy_min, cfg.energy_max, spacing);
    }
    return histograms;
}

std::vector<ic::Histogram> make_mu_escape_histograms_by_order(const ic::Config& cfg) {
    std::vector<ic::Histogram> histograms;
    histograms.reserve(cfg.max_scatters + 1);
    for (std::size_t i = 0; i <= cfg.max_scatters; ++i) {
        histograms.emplace_back(cfg.mu_bins, 0.0, 1.0);
    }
    return histograms;
}

std::vector<ic::Histogram> make_energy_escape_histograms_by_order(const ic::Config& cfg) {
    std::vector<ic::Histogram> histograms;
    histograms.reserve(cfg.max_scatters + 1);
    const ic::HistogramSpacing spacing =
        ic::histogram_spacing_from_string(cfg.energy_bin_spacing);
    for (std::size_t i = 0; i <= cfg.max_scatters; ++i) {
        histograms.emplace_back(cfg.energy_bins, cfg.energy_min, cfg.energy_max, spacing);
    }
    return histograms;
}

struct RunStats {
    explicit RunStats(const ic::Config& cfg)
        : energy_hist(cfg.energy_bins,
                      cfg.energy_min,
                      cfg.energy_max,
                      ic::histogram_spacing_from_string(cfg.energy_bin_spacing)),
          mu_hist(cfg.mu_bins, -1.0, 1.0),
          mu_hist_up_scattered(cfg.mu_bins, -1.0, 1.0),
          mu_hist_down_scattered(cfg.mu_bins, -1.0, 1.0),
          energy_hist_up_scattered(cfg.energy_bins,
                                   cfg.energy_min,
                                   cfg.energy_max,
                                   ic::histogram_spacing_from_string(cfg.energy_bin_spacing)),
          energy_hist_down_scattered(cfg.energy_bins,
                                     cfg.energy_min,
                                     cfg.energy_max,
                                     ic::histogram_spacing_from_string(cfg.energy_bin_spacing)),
          scatter_count_hist(cfg.max_scatters + 1, 0),
          escaped_up_by_order(cfg.max_scatters + 1, 0),
          escaped_down_by_order(cfg.max_scatters + 1, 0),
          mu_escape_hist_up_by_order(make_mu_escape_histograms_by_order(cfg)),
          mu_escape_hist_down_by_order(make_mu_escape_histograms_by_order(cfg)),
          energy_escape_hist_up_by_order(make_energy_escape_histograms_by_order(cfg)),
          energy_escape_hist_down_by_order(make_energy_escape_histograms_by_order(cfg)),
          angular_energy_hist_up(make_angular_spectrum_histograms(cfg)),
          angular_energy_hist_down(make_angular_spectrum_histograms(cfg)) {}

    ic::Histogram energy_hist;
    // Legacy angular histogram: this stores the lab-frame scattering angle
    // cosine for the recorded scatter, ScatterResult::scattering_mu_lab.
    // It is not the final slab escape-direction cosine used in the paper
    // validation probabilities and dP_n/dmu outputs.
    ic::Histogram mu_hist;
    ic::Histogram mu_hist_up_scattered;
    ic::Histogram mu_hist_down_scattered;
    ic::Histogram energy_hist_up_scattered;
    ic::Histogram energy_hist_down_scattered;
    std::vector<std::uint64_t> scatter_count_hist;
    std::vector<std::uint64_t> escaped_up_by_order;
    std::vector<std::uint64_t> escaped_down_by_order;
    // Paper-validation histograms: mu_escape is the final escaped photon
    // direction cosine relative to the outward slab normal, in [0, 1].
    // These are separated by escape boundary and exact scatter order.
    std::vector<ic::Histogram> mu_escape_hist_up_by_order;
    std::vector<ic::Histogram> mu_escape_hist_down_by_order;
    std::vector<ic::Histogram> energy_escape_hist_up_by_order;
    std::vector<ic::Histogram> energy_escape_hist_down_by_order;
    // Angle-resolved escaping photon spectra.  theta is measured from the
    // local outward normal of the escaping surface: theta=acos(mu_escape).
    // These spectra include all escaping histories, including zero-scatter
    // photons, and are meant for direct angle-resolved MCRT spectrum checks.
    std::vector<ic::Histogram> angular_energy_hist_up;
    std::vector<ic::Histogram> angular_energy_hist_down;
    std::uint64_t events_completed = 0;
    std::uint64_t scattered_events = 0;
    std::uint64_t total_scatter_interactions = 0;
    double sum_abs_energy_error = 0.0;
    double max_abs_energy_error = 0.0;
    double sum_mass_shell_error = 0.0;
    double max_mass_shell_error = 0.0;
    double sum_scattered_energy = 0.0;
    double min_scattered_energy = std::numeric_limits<double>::infinity();
    double max_scattered_energy = 0.0;
    double sum_up_scattered_energy = 0.0;
    double sum_down_scattered_energy = 0.0;
    double sum_mu_up_scattered = 0.0;
    double sum_mu_down_scattered = 0.0;
    double sum_incoming_photon_energy_erf = 0.0;
    double sum_outgoing_photon_energy_erf = 0.0;
    double sum_photon_energy_ratio_erf = 0.0;
    double max_abs_recoil_indicator = 0.0;
    double sum_sampled_electron_gamma = 0.0;
    double sum_sampled_electron_gamma2 = 0.0;
    double min_sampled_electron_gamma = std::numeric_limits<double>::infinity();
    double max_sampled_electron_gamma = 0.0;
    std::uint64_t escaped_up_unscattered = 0;
    std::uint64_t escaped_down_unscattered = 0;
    std::uint64_t escaped_up_scattered = 0;
    std::uint64_t escaped_down_scattered = 0;
    std::uint64_t up_scattered_hard_count = 0;
    std::uint64_t down_scattered_hard_count = 0;
    std::uint64_t total_escaped_up = 0;
    std::uint64_t total_escaped_down = 0;
    std::uint64_t terminated_at_max_scatters = 0;
    std::uint64_t sum_scatter_count = 0;
};

struct CompletedRun {
    ic::Config cfg;
    std::string tag;
    RunStats stats;

    CompletedRun(ic::Config cfg_in, std::string tag_in, RunStats stats_in)
        : cfg(std::move(cfg_in)), tag(std::move(tag_in)), stats(std::move(stats_in)) {}
};

CompletedRun run_and_record(ic::Config cfg);

struct ConvergenceSnapshot {
    double tau = 0.0;
    double theta_e = 0.0;
    std::string injection_model;
    std::size_t max_scatters = 0;
    std::string source_run_tag;
    std::uint64_t events_completed = 0;
    std::uint64_t scattered_events = 0;
    std::uint64_t total_scatter_interactions = 0;
    double scatter_fraction = 0.0;
    double escaped_up_total_fraction = 0.0;
    double escaped_down_total_fraction = 0.0;
    double mean_scatter_count = 0.0;
    std::uint64_t terminated_at_max_scatters = 0;
    double max_scatter_termination_fraction = 0.0;
    double mean_scattered_energy_mec2 = 0.0;
    double mean_energy_amplification = 0.0;
    double energy_hist_overflow_fraction = 0.0;
    bool bookkeeping_consistent = false;
};

double convergence_difference_sum(const ConvergenceSnapshot& lhs,
                                  const ConvergenceSnapshot& rhs) {
    return std::abs(rhs.mean_scatter_count - lhs.mean_scatter_count) +
           std::abs(rhs.escaped_up_total_fraction - lhs.escaped_up_total_fraction) +
           std::abs(rhs.escaped_down_total_fraction - lhs.escaped_down_total_fraction) +
           std::abs(rhs.mean_scattered_energy_mec2 - lhs.mean_scattered_energy_mec2);
}

double convergence_difference_sum_vs_reference(const ConvergenceSnapshot& snapshot,
                                               const ConvergenceSnapshot& reference) {
    return std::abs(reference.mean_scatter_count - snapshot.mean_scatter_count) +
           std::abs(reference.escaped_up_total_fraction - snapshot.escaped_up_total_fraction) +
           std::abs(reference.escaped_down_total_fraction - snapshot.escaped_down_total_fraction) +
           std::abs(reference.mean_scattered_energy_mec2 - snapshot.mean_scattered_energy_mec2);
}

bool approximately_stable(double lhs,
                          double rhs,
                          double abs_tolerance,
                          double rel_tolerance) {
    const double scale = std::max(std::abs(lhs), std::abs(rhs));
    return std::abs(lhs - rhs) <= std::max(abs_tolerance, rel_tolerance * scale);
}

std::vector<std::string> production_slab_thermal_table_header() {
    return {
        "slab_tau",
        "slab_injection_model",
        "electron_kTe_mec2",
        "max_scatters",
        "source_run_tag",
        "events_completed",
        "scattered_events",
        "scatter_fraction",
        "escaped_up_total_fraction",
        "escaped_down_total_fraction",
        "escaped_up_scattered",
        "escaped_down_scattered",
        "mean_scatter_count",
        "mean_scattered_energy_mec2",
        "mean_photon_energy_ratio_erf",
        "mean_energy_amplification",
        "downward_scattered_escape_fraction",
        "mean_up_scattered_energy_mec2",
        "mean_down_scattered_energy_mec2",
        "up_scattered_hard_fraction_gt_2x_incident",
        "down_scattered_hard_fraction_gt_2x_incident",
        "mean_mu_up_scattered",
        "mean_mu_down_scattered",
        "mu_up_down_asymmetry",
        "bookkeeping_consistent",
        "energy_hist_overflow_fraction",
        "energy_hist_overflow_acceptable",
    };
}

std::vector<std::string> production_slab_seed_energy_table_header() {
    return {
        "slab_tau",
        "slab_injection_model",
        "electron_kTe_mec2",
        "incident_photon_energy_mec2",
        "max_scatters",
        "source_run_tag",
        "events_completed",
        "scattered_events",
        "scatter_fraction",
        "escaped_up_total_fraction",
        "escaped_down_total_fraction",
        "escaped_up_scattered",
        "escaped_down_scattered",
        "mean_scatter_count",
        "mean_scattered_energy_mec2",
        "mean_photon_energy_ratio_erf",
        "mean_energy_amplification",
        "downward_scattered_escape_fraction",
        "mean_up_scattered_energy_mec2",
        "mean_down_scattered_energy_mec2",
        "up_scattered_hard_fraction_gt_2x_incident",
        "down_scattered_hard_fraction_gt_2x_incident",
        "mean_mu_up_scattered",
        "mean_mu_down_scattered",
        "mu_up_down_asymmetry",
        "bookkeeping_consistent",
        "energy_hist_overflow_fraction",
        "energy_hist_overflow_acceptable",
    };
}

const std::vector<double>& production_slab_thermal_tau_grid_stage1() {
    static const std::vector<double> grid{1.0e-3, 1.0e-2, 1.0e-1, 2.0e-1, 5.0e-1, 1.0};
    return grid;
}

const std::vector<double>& production_slab_thermal_tau_grid_stage2() {
    static const std::vector<double> grid{
        1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1, 1.5e-1, 2.0e-1, 3.0e-1, 5.0e-1, 7.0e-1, 1.0};
    return grid;
}

const std::vector<std::string>& production_slab_thermal_injection_models() {
    static const std::vector<std::string> models{"beam", "lambert"};
    return models;
}

const std::vector<double>& production_slab_thermal_theta_grid_stage1() {
    static const std::vector<double> grid{0.02, 0.05, 0.1, 0.2, 0.5};
    return grid;
}

const std::vector<double>& production_slab_thermal_theta_grid_stage2() {
    static const std::vector<double> grid{0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5};
    return grid;
}

std::vector<double> make_log_grid(double min_value, double max_value, std::size_t count) {
    if (count < 2) {
        return {min_value};
    }

    std::vector<double> grid;
    grid.reserve(count);
    const double log_min = std::log(min_value);
    const double log_max = std::log(max_value);
    const double denom = static_cast<double>(count - 1);

    for (std::size_t i = 0; i < count; ++i) {
        const double t = static_cast<double>(i) / denom;
        grid.push_back(std::exp(log_min + t * (log_max - log_min)));
    }
    return grid;
}

const std::vector<double>& production_slab_thermal_tau_grid_stage3() {
    static const std::vector<double> grid = make_log_grid(1.0e-2, 1.0, 80);
    return grid;
}

const std::vector<double>& production_slab_thermal_theta_grid_stage3() {
    static const std::vector<double> grid = make_log_grid(0.02, 0.5, 80);
    return grid;
}

const std::vector<double>& production_slab_high_tau_grid_stage1() {
    static const std::vector<double> grid{2.0, 3.0, 5.0, 7.0, 10.0};
    return grid;
}

const std::vector<double>& production_slab_high_tau_theta_grid_stage1() {
    static const std::vector<double> grid{0.05, 0.1, 0.2, 0.3, 0.5};
    return grid;
}

const std::vector<double>& production_slab_high_tau_grid_stage2() {
    static const std::vector<double> grid = make_log_grid(1.0, 10.0, 50);
    return grid;
}

const std::vector<double>& production_slab_high_tau_theta_grid_stage2() {
    static const std::vector<double> grid = make_log_grid(0.02, 0.5, 80);
    return grid;
}

const std::vector<double>& diagnostic_high_tau_seed_tau_grid() {
    static const std::vector<double> grid{1.0e-2, 1.0e-1, 1.0, 5.0, 10.0};
    return grid;
}

const std::vector<double>& diagnostic_high_tau_seed_theta_grid() {
    static const std::vector<double> grid{0.02, 0.1, 0.5};
    return grid;
}

const std::vector<double>& diagnostic_high_tau_seed_photon_energies() {
    static const std::vector<double> grid{1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3};
    return grid;
}

const std::vector<double>& production_slab_seed_energy_tau_grid_stage1() {
    static const std::vector<double> grid = make_log_grid(1.0e-2, 1.0e1, 80);
    return grid;
}

const std::vector<double>& production_slab_seed_energy_theta_grid_stage1() {
    static const std::vector<double> grid = make_log_grid(0.02, 0.5, 80);
    return grid;
}

const std::vector<double>& production_slab_seed_energy_photon_grid_stage1() {
    static const std::vector<double> grid = make_log_grid(1.0e-5, 1.0e-3, 80);
    return grid;
}

const std::vector<double>& production_slab_seed_energy_tau_grid_stage2() {
    static const std::vector<double> grid = make_log_grid(1.0e-2, 2.0e1, 80);
    return grid;
}

const std::vector<double>& production_slab_seed_energy_theta_grid_stage2() {
    static const std::vector<double> grid = make_log_grid(1.0e-2, 1.0, 80);
    return grid;
}

const std::vector<double>& production_slab_seed_energy_photon_grid_stage2() {
    static const std::vector<double> grid = make_log_grid(1.0e-5, 1.0e-3, 20);
    return grid;
}

const std::vector<std::string>& production_slab_high_tau_injection_models() {
    static const std::vector<std::string> models{"beam", "lambert"};
    return models;
}

std::vector<std::string> selected_high_tau_injection_models(const ic::Config& cfg) {
    if (cfg.slab_injection_model == "both") {
        return production_slab_high_tau_injection_models();
    }
    return {cfg.slab_injection_model};
}

double downward_scattered_escape_fraction(const RunStats& stats) {
    return stats.events_completed > 0
               ? static_cast<double>(stats.escaped_down_scattered) /
                     static_cast<double>(stats.events_completed)
               : 0.0;
}

double mean_energy_amplification(const ic::Config& cfg, const RunStats& stats) {
    return cfg.incident_photon_energy > 0.0 && stats.scattered_events > 0
               ? (stats.sum_scattered_energy / static_cast<double>(stats.scattered_events)) /
                     cfg.incident_photon_energy
               : 0.0;
}

double mean_up_scattered_energy_mec2(const RunStats& stats) {
    return stats.escaped_up_scattered > 0
               ? stats.sum_up_scattered_energy / static_cast<double>(stats.escaped_up_scattered)
               : 0.0;
}

double mean_down_scattered_energy_mec2(const RunStats& stats) {
    return stats.escaped_down_scattered > 0
               ? stats.sum_down_scattered_energy / static_cast<double>(stats.escaped_down_scattered)
               : 0.0;
}

double up_scattered_hard_fraction(const RunStats& stats) {
    return stats.escaped_up_scattered > 0
               ? static_cast<double>(stats.up_scattered_hard_count) /
                     static_cast<double>(stats.escaped_up_scattered)
               : 0.0;
}

double down_scattered_hard_fraction(const RunStats& stats) {
    return stats.escaped_down_scattered > 0
               ? static_cast<double>(stats.down_scattered_hard_count) /
                     static_cast<double>(stats.escaped_down_scattered)
               : 0.0;
}

double mean_mu_up_scattered(const RunStats& stats) {
    return stats.escaped_up_scattered > 0
               ? stats.sum_mu_up_scattered / static_cast<double>(stats.escaped_up_scattered)
               : 0.0;
}

double mean_mu_down_scattered(const RunStats& stats) {
    return stats.escaped_down_scattered > 0
               ? stats.sum_mu_down_scattered / static_cast<double>(stats.escaped_down_scattered)
               : 0.0;
}

double safe_mean(double sum, std::uint64_t count) {
    return count > 0 ? sum / static_cast<double>(count) : 0.0;
}

double scatter_fraction(const RunStats& stats) {
    return stats.events_completed > 0
               ? static_cast<double>(stats.scattered_events) / static_cast<double>(stats.events_completed)
               : 0.0;
}

double mean_scatter_count(const RunStats& stats) {
    return safe_mean(static_cast<double>(stats.sum_scatter_count), stats.events_completed);
}

double max_scatter_termination_fraction(const RunStats& stats) {
    return stats.events_completed > 0
               ? static_cast<double>(stats.terminated_at_max_scatters) /
                     static_cast<double>(stats.events_completed)
               : 0.0;
}

std::uint64_t sum_counts(const std::vector<std::uint64_t>& counts) {
    std::uint64_t total = 0;
    for (std::uint64_t count : counts) {
        total += count;
    }
    return total;
}

double count_fraction(std::uint64_t count, std::uint64_t denominator) {
    return denominator > 0 ? static_cast<double>(count) / static_cast<double>(denominator) : 0.0;
}

double order_fraction(const std::vector<std::uint64_t>& counts,
                      std::size_t order,
                      std::uint64_t denominator) {
    return order < counts.size() ? count_fraction(counts[order], denominator) : 0.0;
}

double escape_probability_up_sum(const RunStats& stats) {
    return count_fraction(sum_counts(stats.escaped_up_by_order), stats.events_completed);
}

double escape_probability_down_sum(const RunStats& stats) {
    return count_fraction(sum_counts(stats.escaped_down_by_order), stats.events_completed);
}

double escape_probability_sum(const RunStats& stats) {
    return escape_probability_up_sum(stats) + escape_probability_down_sum(stats);
}

double probability_budget_closure(const RunStats& stats) {
    return escape_probability_sum(stats) + max_scatter_termination_fraction(stats);
}

double mean_scatter_count_escaped_only(const RunStats& stats) {
    std::uint64_t escaped = 0;
    std::uint64_t weighted = 0;
    for (std::size_t order = 0; order < stats.escaped_up_by_order.size(); ++order) {
        const std::uint64_t count =
            stats.escaped_up_by_order[order] + stats.escaped_down_by_order[order];
        escaped += count;
        weighted += static_cast<std::uint64_t>(order) * count;
    }
    return safe_mean(static_cast<double>(weighted), escaped);
}

double escaped_up_total_fraction(const RunStats& stats) {
    return stats.events_completed > 0
               ? static_cast<double>(stats.total_escaped_up) /
                     static_cast<double>(stats.events_completed)
               : 0.0;
}

double escaped_down_total_fraction(const RunStats& stats) {
    return stats.events_completed > 0
               ? static_cast<double>(stats.total_escaped_down) /
                     static_cast<double>(stats.events_completed)
               : 0.0;
}

double energy_hist_overflow_fraction(const RunStats& stats) {
    return stats.scattered_events > 0 ? stats.energy_hist.overflow() / static_cast<double>(stats.scattered_events)
                                      : 0.0;
}

double energy_hist_underflow_fraction(const RunStats& stats) {
    return stats.scattered_events > 0 ? stats.energy_hist.underflow() / static_cast<double>(stats.scattered_events)
                                      : 0.0;
}

double mean_scattered_energy_mec2(const RunStats& stats) {
    return safe_mean(stats.sum_scattered_energy, stats.scattered_events);
}

double mean_photon_energy_ratio_erf(const RunStats& stats) {
    return safe_mean(stats.sum_photon_energy_ratio_erf, stats.scattered_events);
}

double up_scattered_mu_hist_overflow_fraction(const RunStats& stats) {
    return stats.escaped_up_scattered > 0
               ? stats.mu_hist_up_scattered.overflow() / static_cast<double>(stats.escaped_up_scattered)
               : 0.0;
}

double down_scattered_mu_hist_overflow_fraction(const RunStats& stats) {
    return stats.escaped_down_scattered > 0
               ? stats.mu_hist_down_scattered.overflow() / static_cast<double>(stats.escaped_down_scattered)
               : 0.0;
}

bool slab_bookkeeping_consistent(const RunStats& stats) {
    const std::uint64_t total_accounted =
        stats.escaped_up_unscattered + stats.escaped_down_unscattered +
        stats.escaped_up_scattered + stats.escaped_down_scattered +
        stats.terminated_at_max_scatters;
    const std::uint64_t scattered_accounted =
        stats.escaped_up_scattered + stats.escaped_down_scattered +
        stats.terminated_at_max_scatters;
    const std::uint64_t escaped_accounted = stats.total_escaped_up + stats.total_escaped_down;
    const std::uint64_t escaped_by_order_accounted =
        sum_counts(stats.escaped_up_by_order) + sum_counts(stats.escaped_down_by_order);
    return total_accounted == stats.events_completed &&
           scattered_accounted == stats.scattered_events &&
           escaped_accounted ==
               stats.escaped_up_unscattered + stats.escaped_down_unscattered +
                   stats.escaped_up_scattered + stats.escaped_down_scattered &&
           escaped_by_order_accounted == escaped_accounted;
}

std::string format_number(double value) {
    std::ostringstream out;
    out << std::scientific << std::setprecision(4) << value;
    return out.str();
}

std::string format_optional_number(bool available, double value) {
    return available ? format_number(value) : "nan";
}

std::string format_order_index(std::size_t order) {
    std::ostringstream out;
    out << std::setw(3) << std::setfill('0') << order;
    return out.str();
}

std::vector<std::string> split_csv_tokens(const std::string& text) {
    std::vector<std::string> tokens;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        token.erase(token.begin(),
                    std::find_if(token.begin(), token.end(), [](unsigned char ch) {
                        return !std::isspace(ch);
                    }));
        token.erase(std::find_if(token.rbegin(), token.rend(), [](unsigned char ch) {
                        return !std::isspace(ch);
                    }).base(),
                    token.end());
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

std::vector<double> parse_double_list_or_default(const std::string& text,
                                                 const std::vector<double>& fallback) {
    if (text.empty()) {
        return fallback;
    }

    const std::vector<std::string> tokens = split_csv_tokens(text);
    if (tokens.empty()) {
        throw std::runtime_error("list option did not contain any values");
    }

    std::vector<double> values;
    values.reserve(tokens.size());
    for (const std::string& token : tokens) {
        values.push_back(std::stod(token));
    }
    return values;
}

std::vector<std::string> parse_string_list_or_default(const std::string& text,
                                                      const std::vector<std::string>& fallback) {
    if (text.empty()) {
        return fallback;
    }

    const std::vector<std::string> tokens = split_csv_tokens(text);
    if (tokens.empty()) {
        throw std::runtime_error("list option did not contain any values");
    }
    return tokens;
}

std::string sanitize_token(std::string text) {
    for (char& ch : text) {
        if (!std::isalnum(static_cast<unsigned char>(ch))) {
            ch = '_';
        }
    }

    while (!text.empty() && text.back() == '_') {
        text.pop_back();
    }
    if (text.empty()) {
        return "run";
    }
    return text;
}

std::string compact_number_token(double value) {
    std::ostringstream out;
    out << std::setprecision(6) << std::defaultfloat << value;
    std::string text = out.str();
    for (char& ch : text) {
        if (ch == '.') {
            ch = 'p';
        } else if (ch == '-') {
            ch = 'm';
        } else if (ch == '+') {
            ch = '_';
        }
    }
    return sanitize_token(text);
}

std::string make_run_tag(const ic::Config& cfg) {
    // Keep per-case output filenames short.  The full parameter record is still
    // written to *_config_snapshot.csv and *_run_summary.csv.  This matters for
    // order-resolved outputs such as *_energy_escape_hist_*_order_000.csv,
    // where a verbose tag can exceed macOS' per-file-name length limit.
    if (!cfg.run_label.empty()) {
        return sanitize_token(cfg.run_label);
    }

    const std::string transport_suffix =
        cfg.transport_cross_section == "thomson"
            ? ""
            : "_" + sanitize_token(cfg.transport_cross_section);
    return "run_" + sanitize_token(cfg.geometry_model) +
           "_" + sanitize_token(cfg.electron_model) +
           "_" + sanitize_token(cfg.seed_photon_model) +
           "_te" + sanitize_token(format_number(cfg.electron_kTe)) +
           "_tau" + sanitize_token(format_number(cfg.slab_optical_depth)) +
           "_" + sanitize_token(cfg.slab_injection_model) +
           transport_suffix;
}

std::string validation_tag(const ic::Config& cfg, const std::string& name) {
    const std::string prefix = cfg.run_label.empty() ? "" : sanitize_token(cfg.run_label) + "_";
    return prefix + sanitize_token(name);
}

ic::Config normalized_config(ic::Config cfg) {
    if (cfg.energy_max <= cfg.energy_min) {
        cfg.energy_max = ic::recommended_energy_max(cfg);
    }
    if ((cfg.energy_bin_spacing == "log" || cfg.energy_bin_spacing == "logarithmic") &&
        cfg.energy_min <= 0.0) {
        const double seed_reference_energy =
            (cfg.seed_photon_model == "blackbody" || cfg.seed_photon_model == "bb")
                ? ic::blackbody_temperature_ev_to_mec2(cfg.seed_temperature_eV)
                : cfg.incident_photon_energy;
        cfg.energy_min = std::max(1.0e-12, seed_reference_energy * 1.0e-2);
    }
    return cfg;
}

ic::TransportCrossSectionSettings transport_cross_section_settings(const ic::Config& cfg) {
    if (cfg.transport_cross_section == "thermal_kn" && cfg.electron_model != "thermal") {
        throw std::runtime_error("transport-cross-section=thermal_kn requires --electron-model thermal");
    }

    ic::TransportCrossSectionSettings settings;
    settings.mode = cfg.transport_cross_section;
    settings.electron_kTe_mec2 = cfg.electron_kTe;
    settings.thermal_kn_table_path = cfg.thermal_kn_table_path;
    return settings;
}

ic::ThermalKnTableGenerationSettings thermal_kn_generation_settings(const ic::Config& cfg) {
    ic::ThermalKnTableGenerationSettings settings;
    settings.energy_points = cfg.thermal_kn_energy_points;
    settings.theta_points = cfg.thermal_kn_theta_points;
    settings.energy_min = cfg.thermal_kn_energy_min;
    settings.energy_max = cfg.thermal_kn_energy_max;
    settings.theta_min = cfg.thermal_kn_theta_min;
    settings.theta_max = cfg.thermal_kn_theta_max;
    settings.integration.z_points = cfg.thermal_kn_z_points;
    settings.integration.mu_points = cfg.thermal_kn_mu_points;
    settings.integration.z_max = cfg.thermal_kn_z_max;
    return settings;
}

void apply_multi_scatter_validation_energy_range(ic::Config& cfg) {
    if (cfg.energy_max > cfg.energy_min) {
        return;
    }

    // The multi-scatter slab validation cases are designed to probe broadened
    // high-energy tails. Expand beyond the generic automatic range so these
    // validations primarily test transport behavior rather than histogram clipping.
    cfg.energy_max = 2.0 * ic::recommended_energy_max(cfg);
}

void write_key_value_csv(const std::string& path,
                         const std::vector<std::pair<std::string, std::string>>& rows) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open CSV output: " + path);
    }

    out << "key,value\n";
    for (const auto& row : rows) {
        out << row.first << ',' << row.second << "\n";
    }
}

void write_table_csv(const std::string& path,
                     const std::vector<std::string>& header,
                     const std::vector<std::vector<std::string>>& rows) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open CSV output: " + path);
    }

    for (std::size_t i = 0; i < header.size(); ++i) {
        if (i > 0) {
            out << ',';
        }
        out << header[i];
    }
    out << "\n";

    for (const auto& row : rows) {
        for (std::size_t i = 0; i < row.size(); ++i) {
            if (i > 0) {
                out << ',';
            }
            out << row[i];
        }
        out << "\n";
    }
}

void write_config_snapshot_csv(const std::string& path,
                               const ic::Config& cfg,
                               const std::string& tag) {
    write_key_value_csv(path,
                        {
                            {"run_tag", tag},
                            {"mode", cfg.mode},
                            {"run_label", cfg.run_label.empty() ? "none" : cfg.run_label},
                            {"num_events", std::to_string(cfg.num_events)},
                            {"seed", std::to_string(cfg.seed)},
                            {"incident_photon_energy_mec2", format_number(cfg.incident_photon_energy)},
                            {"seed_photon_model", cfg.seed_photon_model},
                            {"seed_temperature_eV", format_number(cfg.seed_temperature_eV)},
                            {"seed_temperature_mec2", format_number(ic::blackbody_temperature_ev_to_mec2(cfg.seed_temperature_eV))},
                            {"mean_blackbody_seed_energy_mec2", format_number(ic::mean_blackbody_photon_energy_mec2(cfg.seed_temperature_eV))},
                            {"geometry_model", cfg.geometry_model},
                            {"transport_cross_section", cfg.transport_cross_section},
                            {"thermal_kn_table_path", cfg.thermal_kn_table_path},
                            {"electron_model", cfg.electron_model},
                            {"electron_gamma", format_number(cfg.electron_gamma)},
                            {"electron_kTe_mec2", format_number(cfg.electron_kTe)},
                            {"slab_height", format_number(cfg.slab_height)},
                            {"slab_optical_depth", format_number(cfg.slab_optical_depth)},
                            {"slab_injection_model", cfg.slab_injection_model},
                            {"max_scatters", std::to_string(cfg.max_scatters)},
                            {"energy_bins", std::to_string(cfg.energy_bins)},
                            {"energy_bin_spacing", cfg.energy_bin_spacing},
                            {"mu_bins", std::to_string(cfg.mu_bins)},
                            {"energy_min", format_number(cfg.energy_min)},
                            {"energy_max", format_number(cfg.energy_max)},
                            {"output_dir", cfg.output_dir},
                            {"tau_list", cfg.tau_list.empty() ? "none" : cfg.tau_list},
                            {"injection_list", cfg.injection_list.empty() ? "none" : cfg.injection_list},
                        });
}

void maybe_warn_about_histogram_overflow(const RunStats& stats) {
    if (stats.scattered_events == 0) {
        return;
    }

    const double overflow_fraction = stats.energy_hist.overflow() /
                                     static_cast<double>(stats.scattered_events);
    if (overflow_fraction > kOverflowWarningThreshold) {
        std::cerr << "warning: energy histogram overflow_fraction=" << overflow_fraction
                  << " exceeds the warning threshold of " << kOverflowWarningThreshold << "\n";
    }
}

void maybe_warn_about_validation_overflow(const std::string& mode_name,
                                          const RunStats& stats) {
    const double overflow_fraction = energy_hist_overflow_fraction(stats);
    if (overflow_fraction > kValidationOverflowAcceptableThreshold) {
        std::cerr << "warning: validation mode " << mode_name
                  << " has energy_hist_overflow_fraction=" << overflow_fraction
                  << " above the acceptability threshold of "
                  << kValidationOverflowAcceptableThreshold << "\n";
    }
}

void write_summary_csv(const std::string& path,
                       const ic::Config& cfg,
                       const std::string& tag,
                       const RunStats& stats) {
    const double scattered_events = static_cast<double>(stats.scattered_events);
    const double mean_gamma = safe_mean(stats.sum_sampled_electron_gamma, stats.scattered_events);
    const double mean_gamma2 = safe_mean(stats.sum_sampled_electron_gamma2, stats.scattered_events);
    const double gamma_variance = std::max(0.0, mean_gamma2 - mean_gamma * mean_gamma);
    const double energy_hist_underflow_fraction = ::energy_hist_underflow_fraction(stats);
    const double energy_hist_overflow_fraction = ::energy_hist_overflow_fraction(stats);
    const double up_scattered_energy_hist_underflow_fraction =
        stats.escaped_up_scattered > 0 ? stats.energy_hist_up_scattered.underflow() /
                                             static_cast<double>(stats.escaped_up_scattered)
                                       : 0.0;
    const double up_scattered_energy_hist_overflow_fraction =
        stats.escaped_up_scattered > 0 ? stats.energy_hist_up_scattered.overflow() /
                                             static_cast<double>(stats.escaped_up_scattered)
                                       : 0.0;
    const double down_scattered_energy_hist_underflow_fraction =
        stats.escaped_down_scattered > 0 ? stats.energy_hist_down_scattered.underflow() /
                                               static_cast<double>(stats.escaped_down_scattered)
                                         : 0.0;
    const double down_scattered_energy_hist_overflow_fraction =
        stats.escaped_down_scattered > 0 ? stats.energy_hist_down_scattered.overflow() /
                                               static_cast<double>(stats.escaped_down_scattered)
                                         : 0.0;
    const double mu_hist_underflow_fraction =
        stats.scattered_events > 0 ? stats.mu_hist.underflow() / scattered_events : 0.0;
    const double mu_hist_overflow_fraction =
        stats.scattered_events > 0 ? stats.mu_hist.overflow() / scattered_events : 0.0;
    const double up_scattered_mu_hist_underflow_fraction =
        stats.escaped_up_scattered > 0 ? stats.mu_hist_up_scattered.underflow() /
                                             static_cast<double>(stats.escaped_up_scattered)
                                       : 0.0;
    const double up_scattered_mu_hist_overflow_fraction =
        stats.escaped_up_scattered > 0 ? stats.mu_hist_up_scattered.overflow() /
                                             static_cast<double>(stats.escaped_up_scattered)
                                       : 0.0;
    const double down_scattered_mu_hist_underflow_fraction =
        stats.escaped_down_scattered > 0 ? stats.mu_hist_down_scattered.underflow() /
                                               static_cast<double>(stats.escaped_down_scattered)
                                         : 0.0;
    const double down_scattered_mu_hist_overflow_fraction =
        stats.escaped_down_scattered > 0 ? stats.mu_hist_down_scattered.overflow() /
                                               static_cast<double>(stats.escaped_down_scattered)
                                         : 0.0;
    const double escaped_up_total_fraction = ::escaped_up_total_fraction(stats);
    const double escaped_down_total_fraction = ::escaped_down_total_fraction(stats);
    const double escape_probability_up_sum = ::escape_probability_up_sum(stats);
    const double escape_probability_down_sum = ::escape_probability_down_sum(stats);
    const double escape_probability_sum = ::escape_probability_sum(stats);
    const double termination_fraction = ::max_scatter_termination_fraction(stats);
    const double probability_budget_closure = ::probability_budget_closure(stats);

    write_key_value_csv(path,
                        {
                            {"run_tag", tag},
                            {"mode", cfg.mode},
                            {"events_requested", std::to_string(cfg.num_events)},
                            {"events_completed", std::to_string(stats.events_completed)},
                            {"scattered_events", std::to_string(stats.scattered_events)},
                            {"total_scatter_interactions", std::to_string(stats.total_scatter_interactions)},
                            {"scatter_fraction", format_number(::scatter_fraction(stats))},
                            {"seed", std::to_string(cfg.seed)},
                            {"incident_photon_energy_mec2", format_number(cfg.incident_photon_energy)},
                            {"seed_photon_model", cfg.seed_photon_model},
                            {"seed_temperature_eV", format_number(cfg.seed_temperature_eV)},
                            {"seed_temperature_mec2", format_number(ic::blackbody_temperature_ev_to_mec2(cfg.seed_temperature_eV))},
                            {"mean_blackbody_seed_energy_mec2", format_number(ic::mean_blackbody_photon_energy_mec2(cfg.seed_temperature_eV))},
                            {"geometry_model", cfg.geometry_model},
                            {"transport_cross_section", cfg.transport_cross_section},
                            {"thermal_kn_table_path", cfg.thermal_kn_table_path},
                            {"electron_model", cfg.electron_model},
                            {"electron_gamma", format_number(cfg.electron_gamma)},
                            {"electron_kTe_mec2", format_number(cfg.electron_kTe)},
                            {"slab_height", format_number(cfg.slab_height)},
                            {"slab_optical_depth", format_number(cfg.slab_optical_depth)},
                            {"slab_injection_model", cfg.slab_injection_model},
                            {"max_scatters", std::to_string(cfg.max_scatters)},
                            {"energy_bins", std::to_string(cfg.energy_bins)},
                            {"energy_bin_spacing", cfg.energy_bin_spacing},
                            {"mu_bins", std::to_string(cfg.mu_bins)},
                            {"energy_min", format_number(cfg.energy_min)},
                            {"energy_max", format_number(cfg.energy_max)},
                            {"total_escaped_up", std::to_string(stats.total_escaped_up)},
                            {"total_escaped_down", std::to_string(stats.total_escaped_down)},
                            {"escaped_up_unscattered", std::to_string(stats.escaped_up_unscattered)},
                            {"escaped_down_unscattered", std::to_string(stats.escaped_down_unscattered)},
                            {"escaped_up_scattered", std::to_string(stats.escaped_up_scattered)},
                            {"escaped_down_scattered", std::to_string(stats.escaped_down_scattered)},
                            {"terminated_at_max_scatters", std::to_string(stats.terminated_at_max_scatters)},
                            {"escaped_up_total_fraction", format_number(escaped_up_total_fraction)},
                            {"escaped_down_total_fraction", format_number(escaped_down_total_fraction)},
                            {"escaped_up_order_0_fraction", format_number(order_fraction(stats.escaped_up_by_order, 0, stats.events_completed))},
                            {"escaped_down_order_0_fraction", format_number(order_fraction(stats.escaped_down_by_order, 0, stats.events_completed))},
                            {"escaped_up_order_1_fraction", format_number(order_fraction(stats.escaped_up_by_order, 1, stats.events_completed))},
                            {"escaped_down_order_1_fraction", format_number(order_fraction(stats.escaped_down_by_order, 1, stats.events_completed))},
                            {"P0_up", format_number(order_fraction(stats.escaped_up_by_order, 0, stats.events_completed))},
                            {"P0_down", format_number(order_fraction(stats.escaped_down_by_order, 0, stats.events_completed))},
                            {"P1_up", format_number(order_fraction(stats.escaped_up_by_order, 1, stats.events_completed))},
                            {"P1_down", format_number(order_fraction(stats.escaped_down_by_order, 1, stats.events_completed))},
                            {"P2_up", format_number(order_fraction(stats.escaped_up_by_order, 2, stats.events_completed))},
                            {"P2_down", format_number(order_fraction(stats.escaped_down_by_order, 2, stats.events_completed))},
                            {"escape_probability_sum", format_number(escape_probability_sum)},
                            {"escape_probability_up_sum", format_number(escape_probability_up_sum)},
                            {"escape_probability_down_sum", format_number(escape_probability_down_sum)},
                            {"termination_fraction", format_number(termination_fraction)},
                            {"probability_budget_closure", format_number(probability_budget_closure)},
                            {"mean_scatter_count", format_number(::mean_scatter_count(stats))},
                            {"mean_scatter_count_escaped_only", format_number(::mean_scatter_count_escaped_only(stats))},
                            {"mean_scatter_count_all_histories", format_number(::mean_scatter_count(stats))},
                            {"max_scatter_termination_fraction", format_number(termination_fraction)},
                            {"slab_bookkeeping_consistent", slab_bookkeeping_consistent(stats) ? "true" : "false"},
                            {"mean_sampled_electron_gamma", format_number(mean_gamma)},
                            {"stddev_sampled_electron_gamma", format_number(std::sqrt(gamma_variance))},
                            {"min_sampled_electron_gamma", format_number(stats.scattered_events > 0 ? stats.min_sampled_electron_gamma : 0.0)},
                            {"max_sampled_electron_gamma", format_number(stats.scattered_events > 0 ? stats.max_sampled_electron_gamma : 0.0)},
                            {"mean_scattered_energy_mec2", format_number(mean_scattered_energy_mec2(stats))},
                            {"min_scattered_energy_mec2", format_number(stats.scattered_events > 0 ? stats.min_scattered_energy : 0.0)},
                            {"max_scattered_energy_mec2", format_number(stats.max_scattered_energy)},
                            {"mean_abs_energy_error", format_number(safe_mean(stats.sum_abs_energy_error, stats.scattered_events))},
                            {"max_abs_energy_error", format_number(stats.max_abs_energy_error)},
                            {"mean_mass_shell_error", format_number(safe_mean(stats.sum_mass_shell_error, stats.scattered_events))},
                            {"max_mass_shell_error", format_number(stats.max_mass_shell_error)},
                            {"mean_incoming_photon_energy_erf", format_number(safe_mean(stats.sum_incoming_photon_energy_erf, stats.scattered_events))},
                            {"mean_outgoing_photon_energy_erf", format_number(safe_mean(stats.sum_outgoing_photon_energy_erf, stats.scattered_events))},
                            {"mean_photon_energy_ratio_erf", format_number(mean_photon_energy_ratio_erf(stats))},
                            {"max_abs_recoil_indicator", format_number(stats.max_abs_recoil_indicator)},
                            {"energy_hist_in_range_total", format_number(stats.energy_hist.in_range_total())},
                            {"energy_hist_underflow", format_number(stats.energy_hist.underflow())},
                            {"energy_hist_overflow", format_number(stats.energy_hist.overflow())},
                            {"energy_hist_underflow_fraction", format_number(energy_hist_underflow_fraction)},
                            {"energy_hist_overflow_fraction", format_number(energy_hist_overflow_fraction)},
                            {"energy_hist_up_scattered_in_range_total", format_number(stats.energy_hist_up_scattered.in_range_total())},
                            {"energy_hist_up_scattered_underflow", format_number(stats.energy_hist_up_scattered.underflow())},
                            {"energy_hist_up_scattered_overflow", format_number(stats.energy_hist_up_scattered.overflow())},
                            {"energy_hist_up_scattered_underflow_fraction", format_number(up_scattered_energy_hist_underflow_fraction)},
                            {"energy_hist_up_scattered_overflow_fraction", format_number(up_scattered_energy_hist_overflow_fraction)},
                            {"energy_hist_down_scattered_in_range_total", format_number(stats.energy_hist_down_scattered.in_range_total())},
                            {"energy_hist_down_scattered_underflow", format_number(stats.energy_hist_down_scattered.underflow())},
                            {"energy_hist_down_scattered_overflow", format_number(stats.energy_hist_down_scattered.overflow())},
                            {"energy_hist_down_scattered_underflow_fraction", format_number(down_scattered_energy_hist_underflow_fraction)},
                            {"energy_hist_down_scattered_overflow_fraction", format_number(down_scattered_energy_hist_overflow_fraction)},
                            {"mu_hist_in_range_total", format_number(stats.mu_hist.in_range_total())},
                            {"mu_hist_underflow", format_number(stats.mu_hist.underflow())},
                            {"mu_hist_overflow", format_number(stats.mu_hist.overflow())},
                            {"mu_hist_underflow_fraction", format_number(mu_hist_underflow_fraction)},
                            {"mu_hist_overflow_fraction", format_number(mu_hist_overflow_fraction)},
                            {"mu_hist_up_scattered_in_range_total", format_number(stats.mu_hist_up_scattered.in_range_total())},
                            {"mu_hist_up_scattered_underflow", format_number(stats.mu_hist_up_scattered.underflow())},
                            {"mu_hist_up_scattered_overflow", format_number(stats.mu_hist_up_scattered.overflow())},
                            {"mu_hist_up_scattered_underflow_fraction", format_number(up_scattered_mu_hist_underflow_fraction)},
                            {"mu_hist_up_scattered_overflow_fraction", format_number(up_scattered_mu_hist_overflow_fraction)},
                            {"mu_hist_down_scattered_in_range_total", format_number(stats.mu_hist_down_scattered.in_range_total())},
                            {"mu_hist_down_scattered_underflow", format_number(stats.mu_hist_down_scattered.underflow())},
                            {"mu_hist_down_scattered_overflow", format_number(stats.mu_hist_down_scattered.overflow())},
                            {"mu_hist_down_scattered_underflow_fraction", format_number(down_scattered_mu_hist_underflow_fraction)},
                            {"mu_hist_down_scattered_overflow_fraction", format_number(down_scattered_mu_hist_overflow_fraction)},
                        });
}

void write_scatter_count_hist_csv(const std::string& path, const RunStats& stats) {
    std::vector<std::vector<std::string>> rows;
    rows.reserve(stats.scatter_count_hist.size());

    for (std::size_t scatter_count = 0; scatter_count < stats.scatter_count_hist.size(); ++scatter_count) {
        const std::uint64_t count = stats.scatter_count_hist[scatter_count];
        const double fraction =
            stats.events_completed > 0
                ? static_cast<double>(count) / static_cast<double>(stats.events_completed)
                : 0.0;
        rows.push_back({
            std::to_string(scatter_count),
            std::to_string(count),
            format_number(fraction),
        });
    }

    write_table_csv(path,
                    {"scatter_count", "count", "fraction_of_events"},
                    rows);
}

void write_escape_probability_by_order_csv(const std::string& path, const RunStats& stats) {
    std::vector<std::vector<std::string>> rows;
    rows.reserve(stats.escaped_up_by_order.size());

    double cumulative_up = 0.0;
    double cumulative_down = 0.0;
    for (std::size_t order = 0; order < stats.escaped_up_by_order.size(); ++order) {
        const std::uint64_t up_count = stats.escaped_up_by_order[order];
        const std::uint64_t down_count = stats.escaped_down_by_order[order];
        const std::uint64_t total_count = up_count + down_count;
        const double p_up = count_fraction(up_count, stats.events_completed);
        const double p_down = count_fraction(down_count, stats.events_completed);
        const double p_total = count_fraction(total_count, stats.events_completed);
        cumulative_up += p_up;
        cumulative_down += p_down;
        rows.push_back({
            std::to_string(order),
            std::to_string(up_count),
            std::to_string(down_count),
            std::to_string(total_count),
            format_number(p_up),
            format_number(p_down),
            format_number(p_total),
            format_number(cumulative_up),
            format_number(cumulative_down),
            format_number(cumulative_up + cumulative_down),
        });
    }

    write_table_csv(path,
                    {
                        "scatter_order",
                        "escaped_up_count",
                        "escaped_down_count",
                        "escaped_total_count",
                        "P_n_up",
                        "P_n_down",
                        "P_n_total",
                        "P_n_cumulative_up",
                        "P_n_cumulative_down",
                        "P_n_cumulative_total",
                    },
                    rows);
}

void write_by_order_histograms(const std::string& prefix,
                               const std::vector<ic::Histogram>& histograms,
                               const std::string& suffix) {
    for (std::size_t order = 0; order < histograms.size(); ++order) {
        histograms[order].write_csv(prefix + suffix + "_order_" + format_order_index(order) + ".csv");
    }
}

void write_normalized_histogram_csv(const std::string& path,
                                    const ic::Histogram& histogram,
                                    std::uint64_t events_completed,
                                    const std::string& left_name,
                                    const std::string& right_name,
                                    const std::string& center_name) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open normalized histogram output: " + path);
    }

    // dPn/dmu and dPn/dE are paper-facing probability densities. They are
    // normalized by the total number of emitted photon histories, not by the
    // number that escaped in this order/boundary channel.
    out << "bin_index," << left_name << ',' << right_name << ',' << center_name
        << ",count,bin_probability,bin_density\n";
    for (std::size_t i = 0; i < histogram.num_bins(); ++i) {
        const double left = histogram.low_edge_for(i);
        const double right = histogram.high_edge_for(i);
        const double width = right - left;
        const double count = histogram.count(i);
        const double probability =
            events_completed > 0 ? count / static_cast<double>(events_completed) : 0.0;
        const double density =
            events_completed > 0 && width > 0.0 ? count / (static_cast<double>(events_completed) * width) : 0.0;
        out << i << ',' << left << ',' << right << ',' << histogram.center_for(i) << ','
            << count << ',' << probability << ',' << density << "\n";
    }
}

void write_normalized_by_order_histograms(const std::string& prefix,
                                          const std::vector<ic::Histogram>& histograms,
                                          const std::string& suffix,
                                          std::uint64_t events_completed,
                                          const std::string& left_name,
                                          const std::string& right_name,
                                          const std::string& center_name) {
    for (std::size_t order = 0; order < histograms.size(); ++order) {
        write_normalized_histogram_csv(prefix + suffix + "_order_" + format_order_index(order) + ".csv",
                                       histograms[order],
                                       events_completed,
                                       left_name,
                                       right_name,
                                       center_name);
    }
}

void write_all_order_density_csv(const std::string& path,
                                 const std::vector<ic::Histogram>& histograms,
                                 std::uint64_t events_completed,
                                 const std::string& center_name) {
    if (histograms.empty()) {
        return;
    }

    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open all-order density output: " + path);
    }

    out << center_name;
    for (std::size_t order = 0; order < histograms.size(); ++order) {
        out << ",order_" << format_order_index(order);
    }
    out << "\n";

    const ic::Histogram& first = histograms.front();
    for (std::size_t bin = 0; bin < first.num_bins(); ++bin) {
        out << first.center_for(bin);
        for (const ic::Histogram& histogram : histograms) {
            const double width = histogram.high_edge_for(bin) - histogram.low_edge_for(bin);
            const double density =
                events_completed > 0 && width > 0.0
                    ? histogram.count(bin) / (static_cast<double>(events_completed) * width)
                    : 0.0;
            out << ',' << density;
        }
        out << "\n";
    }
}

void write_angle_resolved_spectra_csv(const std::string& path,
                                      const RunStats& stats) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open angle-resolved spectrum output: " + path);
    }

    out << "boundary,theta_label,theta_center_deg,theta_min_deg,theta_max_deg,"
           "mu_min,mu_max,bin_index,E_left_mec2,E_right_mec2,E_center_mec2,"
           "E_left_eV,E_right_eV,E_center_eV,count,bin_probability,"
           "dP_dE_mec2,dP_dE_eV,E_dP_dE\n";

    const double mec2_per_ev = ic::ev_to_mec2(1.0);
    const auto write_boundary = [&](const std::string& boundary,
                                    const std::vector<ic::Histogram>& histograms) {
        const auto& windows = angular_spectrum_windows();
        for (std::size_t iwin = 0; iwin < windows.size(); ++iwin) {
            const AngularSpectrumWindow& window = windows.at(iwin);
            const ic::Histogram& histogram = histograms.at(iwin);
            const double mu_min =
                std::cos(window.theta_max_deg * ic::constants::pi / 180.0);
            const double mu_max =
                std::cos(window.theta_min_deg * ic::constants::pi / 180.0);

            for (std::size_t ibin = 0; ibin < histogram.num_bins(); ++ibin) {
                const double left = histogram.low_edge_for(ibin);
                const double right = histogram.high_edge_for(ibin);
                const double center = histogram.center_for(ibin);
                const double width = right - left;
                const double left_ev = left / mec2_per_ev;
                const double right_ev = right / mec2_per_ev;
                const double center_ev = center / mec2_per_ev;
                const double width_ev = right_ev - left_ev;
                const double count = histogram.count(ibin);
                const double probability =
                    stats.events_completed > 0
                        ? count / static_cast<double>(stats.events_completed)
                        : 0.0;
                const double density_mec2 =
                    stats.events_completed > 0 && width > 0.0
                        ? count / (static_cast<double>(stats.events_completed) * width)
                        : 0.0;
                const double density_ev =
                    stats.events_completed > 0 && width_ev > 0.0
                        ? count / (static_cast<double>(stats.events_completed) * width_ev)
                        : 0.0;

                out << boundary << ','
                    << window.label << ','
                    << format_number(window.theta_center_deg) << ','
                    << format_number(window.theta_min_deg) << ','
                    << format_number(window.theta_max_deg) << ','
                    << format_number(mu_min) << ','
                    << format_number(mu_max) << ','
                    << ibin << ','
                    << format_number(left) << ','
                    << format_number(right) << ','
                    << format_number(center) << ','
                    << format_number(left_ev) << ','
                    << format_number(right_ev) << ','
                    << format_number(center_ev) << ','
                    << format_number(count) << ','
                    << format_number(probability) << ','
                    << format_number(density_mec2) << ','
                    << format_number(density_ev) << ','
                    << format_number(center_ev * density_ev) << "\n";
            }
        }
    };

    write_boundary("up", stats.angular_energy_hist_up);
    write_boundary("down", stats.angular_energy_hist_down);
}

std::vector<std::string> make_production_slab_thermal_row(const CompletedRun& run) {
    const bool bookkeeping_consistent = slab_bookkeeping_consistent(run.stats);
    const double overflow_fraction = energy_hist_overflow_fraction(run.stats);
    const bool overflow_acceptable =
        overflow_fraction <= kValidationOverflowAcceptableThreshold;
    const double amplification = mean_energy_amplification(run.cfg, run.stats);
    const double downward_scattered_fraction =
        downward_scattered_escape_fraction(run.stats);
    const bool up_channel_available = run.stats.escaped_up_scattered > 0;
    const bool down_channel_available = run.stats.escaped_down_scattered > 0;
    const double up_mean_energy = mean_up_scattered_energy_mec2(run.stats);
    const double down_mean_energy = mean_down_scattered_energy_mec2(run.stats);
    const double up_mean_mu = mean_mu_up_scattered(run.stats);
    const double down_mean_mu = mean_mu_down_scattered(run.stats);
    const bool mu_asymmetry_available = up_channel_available && down_channel_available;
    const double mu_asymmetry = up_mean_mu - down_mean_mu;

    return {
        format_number(run.cfg.slab_optical_depth),
        run.cfg.slab_injection_model,
        format_number(run.cfg.electron_kTe),
        std::to_string(run.cfg.max_scatters),
        run.tag,
        std::to_string(run.stats.events_completed),
        std::to_string(run.stats.scattered_events),
        format_number(scatter_fraction(run.stats)),
        format_number(escaped_up_total_fraction(run.stats)),
        format_number(escaped_down_total_fraction(run.stats)),
        std::to_string(run.stats.escaped_up_scattered),
        std::to_string(run.stats.escaped_down_scattered),
        format_number(mean_scatter_count(run.stats)),
        format_number(mean_scattered_energy_mec2(run.stats)),
        format_number(mean_photon_energy_ratio_erf(run.stats)),
        format_number(amplification),
        format_number(downward_scattered_fraction),
        format_optional_number(up_channel_available, up_mean_energy),
        format_optional_number(down_channel_available, down_mean_energy),
        format_optional_number(up_channel_available, up_scattered_hard_fraction(run.stats)),
        format_optional_number(down_channel_available, down_scattered_hard_fraction(run.stats)),
        format_optional_number(up_channel_available, up_mean_mu),
        format_optional_number(down_channel_available, down_mean_mu),
        format_optional_number(mu_asymmetry_available, mu_asymmetry),
        bookkeeping_consistent ? "true" : "false",
        format_number(overflow_fraction),
        overflow_acceptable ? "true" : "false",
    };
}

std::vector<std::string> make_production_slab_seed_energy_row(const CompletedRun& run) {
    std::vector<std::string> row = make_production_slab_thermal_row(run);
    row.insert(row.begin() + 3, format_number(run.cfg.incident_photon_energy));
    return row;
}

void maybe_warn_about_production_slab_thermal_case(const CompletedRun& run) {
    const bool bookkeeping_consistent = slab_bookkeeping_consistent(run.stats);
    const double overflow_fraction = energy_hist_overflow_fraction(run.stats);
    const bool overflow_acceptable =
        overflow_fraction <= kValidationOverflowAcceptableThreshold;
    const bool up_channel_available = run.stats.escaped_up_scattered > 0;
    const bool down_channel_available = run.stats.escaped_down_scattered > 0;

    if (!bookkeeping_consistent) {
        std::cerr << "warning: production slab thermal case bookkeeping failed for tau="
                  << run.cfg.slab_optical_depth
                  << " injection_model=" << run.cfg.slab_injection_model
                  << " theta_e=" << run.cfg.electron_kTe << "\n";
    }
    if (!overflow_acceptable) {
        std::cerr << "warning: production slab thermal case overflow_fraction="
                  << overflow_fraction << " exceeds the acceptability threshold of "
                  << kValidationOverflowAcceptableThreshold
                  << " for tau=" << run.cfg.slab_optical_depth
                  << " injection_model=" << run.cfg.slab_injection_model
                  << " theta_e=" << run.cfg.electron_kTe << "\n";
    }
    if (!up_channel_available) {
        std::cerr << "warning: production slab thermal case has empty upward scattered escape channel for tau="
                  << run.cfg.slab_optical_depth
                  << " injection_model=" << run.cfg.slab_injection_model
                  << " theta_e=" << run.cfg.electron_kTe << "\n";
    }
    if (!down_channel_available) {
        std::cerr << "warning: production slab thermal case has empty downward scattered escape channel for tau="
                  << run.cfg.slab_optical_depth
                  << " injection_model=" << run.cfg.slab_injection_model
                  << " theta_e=" << run.cfg.electron_kTe << "\n";
    }
}

std::vector<std::vector<std::string>> run_production_slab_thermal_grid_rows(
    const ic::Config& base_cfg,
    const std::vector<double>& tau_values,
    const std::vector<std::string>& injection_models,
    const std::vector<double>& theta_values,
    const std::string& mode_name) {
    std::vector<std::vector<std::string>> rows;
    rows.reserve(tau_values.size() * injection_models.size() * theta_values.size());

    for (double tau : tau_values) {
        for (const std::string& injection_model : injection_models) {
            for (double theta_e : theta_values) {
                ic::Config cfg = base_cfg;
                cfg.mode = mode_name;
                // Keep bulk production filenames compact. The combined table is
                // the main scientific index, and long user labels can push file
                // names over filesystem limits on the cluster.
                cfg.run_label.clear();
                cfg.geometry_model = "slab";
                cfg.slab_height = 1.0;
                cfg.slab_optical_depth = tau;
                cfg.slab_injection_model = injection_model;
                cfg.electron_model = "thermal";
                cfg.electron_kTe = theta_e;
                cfg.max_scatters = base_cfg.max_scatters;
                apply_multi_scatter_validation_energy_range(cfg);

                const CompletedRun run = run_and_record(cfg);
                rows.push_back(make_production_slab_thermal_row(run));
                maybe_warn_about_production_slab_thermal_case(run);
            }
        }
    }

    return rows;
}

void write_production_slab_thermal_table(const std::string& path,
                                         const std::vector<std::vector<std::string>>& rows) {
    write_table_csv(path, production_slab_thermal_table_header(), rows);
    std::cout << "Wrote: " << path << "\n";
}

std::vector<std::vector<std::string>> run_production_slab_seed_energy_grid_rows(
    const ic::Config& base_cfg,
    const std::vector<double>& tau_values,
    const std::vector<std::string>& injection_models,
    const std::vector<double>& theta_values,
    const std::vector<double>& photon_energies,
    const std::string& mode_name) {
    std::vector<std::vector<std::string>> rows;
    rows.reserve(tau_values.size() * injection_models.size() * theta_values.size() * photon_energies.size());

    for (double tau : tau_values) {
        for (const std::string& injection_model : injection_models) {
            for (double photon_energy : photon_energies) {
                for (double theta_e : theta_values) {
                    ic::Config cfg = base_cfg;
                    cfg.mode = mode_name;
                    cfg.run_label.clear();
                    cfg.geometry_model = "slab";
                    cfg.slab_height = 1.0;
                    cfg.slab_optical_depth = tau;
                    cfg.slab_injection_model = injection_model;
                    cfg.incident_photon_energy = photon_energy;
                    cfg.electron_model = "thermal";
                    cfg.electron_kTe = theta_e;
                    cfg.max_scatters = base_cfg.max_scatters;
                    apply_multi_scatter_validation_energy_range(cfg);

                    const CompletedRun run = run_and_record(cfg);
                    rows.push_back(make_production_slab_seed_energy_row(run));
                    maybe_warn_about_production_slab_thermal_case(run);
                }
            }
        }
    }

    return rows;
}

void write_production_slab_seed_energy_table(const std::string& path,
                                             const std::vector<std::vector<std::string>>& rows) {
    write_table_csv(path, production_slab_seed_energy_table_header(), rows);
    std::cout << "Wrote: " << path << "\n";
}

ic::Electron sample_electron(const ic::Config& cfg, std::mt19937_64& rng) {
    if (cfg.electron_model == "monoenergetic") {
        return ic::sample_isotropic_monoenergetic_electron(cfg.electron_gamma, rng);
    }

    if (cfg.electron_model == "thermal") {
        return ic::sample_isotropic_maxwell_juttner_electron(ic::ThermalElectronParams{cfg.electron_kTe}, rng);
    }

    throw std::runtime_error("unsupported electron model: " + cfg.electron_model);
}

ic::Electron sample_electron_for_scatter(const ic::Config& cfg,
                                         const ic::Photon& photon,
                                         std::mt19937_64& rng) {
    if (cfg.electron_model == "thermal") {
        return ic::sample_conditioned_thermal_electron_for_scatter(
            photon,
            ic::ThermalElectronParams{cfg.electron_kTe},
            rng);
    }

    return sample_electron(cfg, rng);
}

void record_escape_count(RunStats& stats, bool scattered, ic::SlabBoundaryFace face) {
    if (scattered) {
        if (face == ic::SlabBoundaryFace::upper) {
            ++stats.escaped_up_scattered;
            ++stats.total_escaped_up;
        } else if (face == ic::SlabBoundaryFace::lower) {
            ++stats.escaped_down_scattered;
            ++stats.total_escaped_down;
        }
        return;
    }

    if (face == ic::SlabBoundaryFace::upper) {
        ++stats.escaped_up_unscattered;
        ++stats.total_escaped_up;
    } else if (face == ic::SlabBoundaryFace::lower) {
        ++stats.escaped_down_unscattered;
        ++stats.total_escaped_down;
    }
}

void record_completion_stats(RunStats& stats, std::size_t scatter_count) {
    stats.sum_scatter_count += static_cast<std::uint64_t>(scatter_count);
    const std::size_t bin = std::min(scatter_count, stats.scatter_count_hist.size() - 1);
    ++stats.scatter_count_hist[bin];
    ++stats.events_completed;
}

void record_transport_termination(RunStats& stats, ic::SlabTransportTermination termination) {
    if (termination == ic::SlabTransportTermination::hit_max_scatters) {
        ++stats.terminated_at_max_scatters;
    }
}

void record_angle_resolved_escape_spectrum(RunStats& stats,
                                           ic::SlabBoundaryFace escape_face,
                                           double mu_escape,
                                           double photon_energy) {
    const double clamped_mu = std::clamp(mu_escape, 0.0, 1.0);
    const double theta_deg =
        std::acos(clamped_mu) * 180.0 / ic::constants::pi;
    const auto& windows = angular_spectrum_windows();

    for (std::size_t i = 0; i < windows.size(); ++i) {
        const AngularSpectrumWindow& window = windows[i];
        if (theta_deg < window.theta_min_deg || theta_deg > window.theta_max_deg) {
            continue;
        }
        if (escape_face == ic::SlabBoundaryFace::upper) {
            stats.angular_energy_hist_up.at(i).fill(photon_energy);
        } else if (escape_face == ic::SlabBoundaryFace::lower) {
            stats.angular_energy_hist_down.at(i).fill(photon_energy);
        }
    }
}

void record_escape_by_order(RunStats& stats,
                            const ic::SlabTransportResult& transport,
                            std::size_t scatter_count) {
    if (!transport.escaped()) {
        return;
    }
    if (scatter_count >= stats.escaped_up_by_order.size()) {
        throw std::runtime_error("scatter order exceeded by-order statistics size");
    }

    double mu_escape = 0.0;
    if (transport.escape_face == ic::SlabBoundaryFace::upper) {
        mu_escape = transport.final_photon.direction.normalized().z;
    } else if (transport.escape_face == ic::SlabBoundaryFace::lower) {
        mu_escape = -transport.final_photon.direction.normalized().z;
    } else {
        return;
    }

    // Histogram::fill uses half-open bins [min, max), so the exact endpoint
    // mu_escape=1 is nudged into the last bin after the physical [0, 1] clamp.
    const double clamped_mu_escape = std::clamp(mu_escape, 0.0, 1.0);
    const double histogram_mu_escape =
        clamped_mu_escape >= 1.0 ? std::nextafter(1.0, 0.0) : clamped_mu_escape;

    if (transport.escape_face == ic::SlabBoundaryFace::upper) {
        ++stats.escaped_up_by_order[scatter_count];
        stats.mu_escape_hist_up_by_order[scatter_count].fill(histogram_mu_escape);
        stats.energy_escape_hist_up_by_order[scatter_count].fill(transport.final_photon.energy);
    } else if (transport.escape_face == ic::SlabBoundaryFace::lower) {
        ++stats.escaped_down_by_order[scatter_count];
        stats.mu_escape_hist_down_by_order[scatter_count].fill(histogram_mu_escape);
        stats.energy_escape_hist_down_by_order[scatter_count].fill(transport.final_photon.energy);
    }
    record_angle_resolved_escape_spectrum(stats,
                                          transport.escape_face,
                                          clamped_mu_escape,
                                          transport.final_photon.energy);
}

void record_scatter_observables(const ic::Config& cfg,
                                RunStats& stats,
                                const ic::Electron& electron,
                                const ic::ScatterResult& result,
                                ic::SlabBoundaryFace escape_face) {
    stats.energy_hist.fill(result.scattered_photon.energy);
    stats.mu_hist.fill(result.scattering_mu_lab);
    if (escape_face == ic::SlabBoundaryFace::upper) {
        stats.energy_hist_up_scattered.fill(result.scattered_photon.energy);
        stats.mu_hist_up_scattered.fill(result.scattering_mu_lab);
        stats.sum_up_scattered_energy += result.scattered_photon.energy;
        stats.sum_mu_up_scattered += result.scattering_mu_lab;
        if (result.scattered_photon.energy > 2.0 * cfg.incident_photon_energy) {
            ++stats.up_scattered_hard_count;
        }
    } else if (escape_face == ic::SlabBoundaryFace::lower) {
        stats.energy_hist_down_scattered.fill(result.scattered_photon.energy);
        stats.mu_hist_down_scattered.fill(result.scattering_mu_lab);
        stats.sum_down_scattered_energy += result.scattered_photon.energy;
        stats.sum_mu_down_scattered += result.scattering_mu_lab;
        if (result.scattered_photon.energy > 2.0 * cfg.incident_photon_energy) {
            ++stats.down_scattered_hard_count;
        }
    }

    const double abs_energy_error = std::abs(result.energy_error);
    stats.sum_abs_energy_error += abs_energy_error;
    stats.max_abs_energy_error = std::max(stats.max_abs_energy_error, abs_energy_error);
    stats.sum_mass_shell_error += result.mass_shell_error;
    stats.max_mass_shell_error = std::max(stats.max_mass_shell_error, result.mass_shell_error);
    stats.sum_scattered_energy += result.scattered_photon.energy;
    stats.min_scattered_energy = std::min(stats.min_scattered_energy, result.scattered_photon.energy);
    stats.max_scattered_energy = std::max(stats.max_scattered_energy, result.scattered_photon.energy);
    stats.sum_incoming_photon_energy_erf += result.incoming_photon_energy_erf;
    stats.sum_outgoing_photon_energy_erf += result.outgoing_photon_energy_erf;
    stats.sum_photon_energy_ratio_erf += result.photon_energy_ratio_erf;
    stats.max_abs_recoil_indicator = std::max(stats.max_abs_recoil_indicator,
                                              std::abs(result.photon_energy_ratio_erf - 1.0));
    stats.sum_sampled_electron_gamma += electron.gamma;
    stats.sum_sampled_electron_gamma2 += electron.gamma * electron.gamma;
    stats.min_sampled_electron_gamma = std::min(stats.min_sampled_electron_gamma, electron.gamma);
    stats.max_sampled_electron_gamma = std::max(stats.max_sampled_electron_gamma, electron.gamma);
}

RunStats execute_run(const ic::Config& cfg) {
    RunStats stats(cfg);
    std::mt19937_64 rng(cfg.seed);
    const ic::Slab slab{0.0, cfg.slab_height};
    const ic::TransportCrossSectionSettings transport_settings =
        cfg.geometry_model == "slab" ? transport_cross_section_settings(cfg)
                                      : ic::TransportCrossSectionSettings{};

    for (std::uint64_t i = 0; i < cfg.num_events; ++i) {
        const double seed_photon_energy =
            ic::sample_seed_photon_energy_mec2(cfg.seed_photon_model,
                                               cfg.incident_photon_energy,
                                               cfg.seed_temperature_eV,
                                               rng);
        if (cfg.geometry_model == "slab") {
            const ic::SlabInjectionSample injection =
                ic::sample_slab_injection(slab,
                                          seed_photon_energy,
                                          cfg.slab_injection_model,
                                          rng);
            const ic::SlabTransportResult transport =
                ic::transport_photon_through_slab(slab,
                                                  cfg.slab_optical_depth,
                                                  injection.position,
                                                  injection.photon,
                                                  cfg.max_scatters,
                                                  rng,
                                                  [&cfg](const ic::Photon& scattering_photon,
                                                         std::mt19937_64& event_rng) {
                                                      return sample_electron_for_scatter(
                                                          cfg,
                                                          scattering_photon,
                                                          event_rng);
                                                  },
                                                  transport_settings);

            const std::size_t scatter_count = transport.scatter_events.size();
            stats.total_scatter_interactions += static_cast<std::uint64_t>(scatter_count);

            if (!transport.success) {
                record_completion_stats(stats, scatter_count);
                continue;
            }

            if (scatter_count > 0) {
                ++stats.scattered_events;
                const ic::ScatterEvent& last_scatter = transport.scatter_events.back();
                record_scatter_observables(cfg,
                                           stats,
                                           last_scatter.electron,
                                           last_scatter.result,
                                           transport.escape_face);
            }

            if (transport.escaped()) {
                record_escape_count(stats, scatter_count > 0, transport.escape_face);
                record_escape_by_order(stats, transport, scatter_count);
            }
            record_transport_termination(stats, transport.termination);
            record_completion_stats(stats, scatter_count);
            continue;
        }

        const ic::Photon photon = ic::make_incident_monoenergetic_photon(seed_photon_energy);
        const ic::Electron electron = sample_electron_for_scatter(cfg, photon, rng);
        const ic::ScatterResult result = ic::scatter_single_kn(photon, electron, rng);
        if (!result.success) {
            record_completion_stats(stats, 0);
            continue;
        }

        ++stats.scattered_events;
        ++stats.total_scatter_interactions;
        record_scatter_observables(cfg, stats, electron, result, ic::SlabBoundaryFace::none);
        record_completion_stats(stats, 1);
    }

    return stats;
}

CompletedRun run_and_record(ic::Config cfg) {
    cfg = normalized_config(std::move(cfg));
    std::filesystem::create_directories(cfg.output_dir);

    RunStats stats = execute_run(cfg);
    const std::string tag = make_run_tag(cfg);
    const std::string prefix = cfg.output_dir + "/" + tag;

    const std::string energy_path = prefix + "_eh.csv";
    const std::string energy_up_path = prefix + "_ehu.csv";
    const std::string energy_down_path = prefix + "_ehd.csv";
    const std::string mu_path = prefix + "_mh.csv";
    const std::string mu_up_path = prefix + "_mhu.csv";
    const std::string mu_down_path = prefix + "_mhd.csv";
    const std::string scatter_count_path = prefix + "_sc.csv";
    const std::string escape_probability_by_order_path =
        prefix + "_Pn.csv";
    const std::string angle_resolved_spectra_path =
        prefix + "_angspec.csv";
    const std::string summary_path = prefix + "_sum.csv";
    const std::string config_path = prefix + "_cfg.csv";

    stats.energy_hist.write_csv(energy_path);
    stats.energy_hist_up_scattered.write_csv(energy_up_path);
    stats.energy_hist_down_scattered.write_csv(energy_down_path);
    stats.mu_hist.write_csv(mu_path);
    stats.mu_hist_up_scattered.write_csv(mu_up_path);
    stats.mu_hist_down_scattered.write_csv(mu_down_path);
    write_scatter_count_hist_csv(scatter_count_path, stats);
    write_escape_probability_by_order_csv(escape_probability_by_order_path, stats);
    write_by_order_histograms(prefix, stats.mu_escape_hist_up_by_order, "_muu");
    write_by_order_histograms(prefix, stats.mu_escape_hist_down_by_order, "_mud");
    write_by_order_histograms(prefix, stats.energy_escape_hist_up_by_order, "_eu");
    write_by_order_histograms(prefix, stats.energy_escape_hist_down_by_order, "_ed");
    write_normalized_by_order_histograms(prefix,
                                         stats.mu_escape_hist_up_by_order,
                                         "_du",
                                         stats.events_completed,
                                         "mu_left",
                                         "mu_right",
                                         "mu_center");
    write_normalized_by_order_histograms(prefix,
                                         stats.mu_escape_hist_down_by_order,
                                         "_dd",
                                         stats.events_completed,
                                         "mu_left",
                                         "mu_right",
                                         "mu_center");
    write_normalized_by_order_histograms(prefix,
                                         stats.energy_escape_hist_up_by_order,
                                         "_Eu",
                                         stats.events_completed,
                                         "E_left",
                                         "E_right",
                                         "E_center");
    write_normalized_by_order_histograms(prefix,
                                         stats.energy_escape_hist_down_by_order,
                                         "_Ed",
                                         stats.events_completed,
                                         "E_left",
                                         "E_right",
                                         "E_center");
    write_all_order_density_csv(prefix + "_du_all.csv",
                                stats.mu_escape_hist_up_by_order,
                                stats.events_completed,
                                "mu_center");
    write_all_order_density_csv(prefix + "_dd_all.csv",
                                stats.mu_escape_hist_down_by_order,
                                stats.events_completed,
                                "mu_center");
    write_all_order_density_csv(prefix + "_Eu_all.csv",
                                stats.energy_escape_hist_up_by_order,
                                stats.events_completed,
                                "E_center");
    write_all_order_density_csv(prefix + "_Ed_all.csv",
                                stats.energy_escape_hist_down_by_order,
                                stats.events_completed,
                                "E_center");
    write_angle_resolved_spectra_csv(angle_resolved_spectra_path, stats);
    write_summary_csv(summary_path, cfg, tag, stats);
    write_config_snapshot_csv(config_path, cfg, tag);
    maybe_warn_about_histogram_overflow(stats);

    std::cout << "Wrote: " << energy_path << "\n";
    std::cout << "Wrote: " << energy_up_path << "\n";
    std::cout << "Wrote: " << energy_down_path << "\n";
    std::cout << "Wrote: " << mu_path << "\n";
    std::cout << "Wrote: " << mu_up_path << "\n";
    std::cout << "Wrote: " << mu_down_path << "\n";
    std::cout << "Wrote: " << scatter_count_path << "\n";
    std::cout << "Wrote: " << escape_probability_by_order_path << "\n";
    std::cout << "Wrote: " << prefix << "_mu{u,d}_order_*.csv\n";
    std::cout << "Wrote: " << prefix << "_E{u,d}_order_*.csv\n";
    std::cout << "Wrote: " << prefix << "_d{u,d}_order_*.csv\n";
    std::cout << "Wrote: " << prefix << "_E{u,d}_order_*.csv\n";
    std::cout << "Wrote: " << prefix << "_d{u,d}_all.csv\n";
    std::cout << "Wrote: " << prefix << "_E{u,d}_all.csv\n";
    std::cout << "Wrote: " << angle_resolved_spectra_path << "\n";
    std::cout << "Wrote: " << summary_path << "\n";
    std::cout << "Wrote: " << config_path << "\n";

    return CompletedRun{std::move(cfg), tag, std::move(stats)};
}

void write_validation_report(const std::string& path,
                             const std::vector<std::pair<std::string, std::string>>& rows) {
    write_key_value_csv(path, rows);
    std::cout << "Wrote: " << path << "\n";
}

void run_validation_thomson(const ic::Config& base_cfg) {
    ic::Config cfg = base_cfg;
    cfg.mode = "validate-thomson";
    cfg.run_label = base_cfg.run_label.empty() ? "thomson" : base_cfg.run_label + "_thomson";
    cfg.incident_photon_energy = 1.0e-6;
    cfg.electron_gamma = 10.0;

    const CompletedRun run = run_and_record(cfg);
    const double mean_energy = safe_mean(run.stats.sum_scattered_energy, run.stats.scattered_events);
    const double thomson_scale = 4.0 * run.cfg.electron_gamma * run.cfg.electron_gamma *
                                 run.cfg.incident_photon_energy;
    const std::string report_path = run.cfg.output_dir + "/" +
                                    validation_tag(base_cfg, "validate_thomson_report") + ".csv";

    // In the Thomson limit, the cleanest diagnostic is small recoil in the
    // electron rest frame: epsilon'_out / epsilon'_in should remain close to 1.
    // This is more direct than comparing only to the lab-frame 4 gamma^2 epsilon
    // scaling, which mixes kinematics, anisotropy, and boosting effects.
    write_validation_report(report_path,
                            {
                                {"validation_case", "thomson_like_low_energy"},
                                {"source_run_tag", run.tag},
                                {"mean_scattered_energy_mec2", format_number(mean_energy)},
                                {"thomson_scale_estimate_mec2", format_number(thomson_scale)},
                                {"mean_to_thomson_scale_ratio", format_number(mean_energy / thomson_scale)},
                                {"mean_incoming_photon_energy_erf", format_number(safe_mean(run.stats.sum_incoming_photon_energy_erf, run.stats.scattered_events))},
                                {"mean_outgoing_photon_energy_erf", format_number(safe_mean(run.stats.sum_outgoing_photon_energy_erf, run.stats.scattered_events))},
                                {"mean_erf_energy_ratio", format_number(safe_mean(run.stats.sum_photon_energy_ratio_erf, run.stats.scattered_events))},
                                {"max_abs_erf_recoil_indicator", format_number(run.stats.max_abs_recoil_indicator)},
                                {"mean_abs_energy_error", format_number(safe_mean(run.stats.sum_abs_energy_error, run.stats.scattered_events))},
                            });
}

void run_validation_kn(const ic::Config& base_cfg) {
    ic::Config low_cfg = base_cfg;
    low_cfg.mode = "validate-kn-low";
    low_cfg.run_label = base_cfg.run_label.empty() ? "kn_low" : base_cfg.run_label + "_kn_low";
    low_cfg.incident_photon_energy = 1.0e-6;
    low_cfg.electron_gamma = 10.0;

    ic::Config high_cfg = base_cfg;
    high_cfg.mode = "validate-kn-high";
    high_cfg.run_label = base_cfg.run_label.empty() ? "kn_high" : base_cfg.run_label + "_kn_high";
    high_cfg.incident_photon_energy = 5.0e-2;
    high_cfg.electron_gamma = 10.0;

    const CompletedRun low_run = run_and_record(low_cfg);
    const CompletedRun high_run = run_and_record(high_cfg);

    const double low_mean_energy = safe_mean(low_run.stats.sum_scattered_energy, low_run.stats.scattered_events);
    const double high_mean_energy = safe_mean(high_run.stats.sum_scattered_energy, high_run.stats.scattered_events);
    const double low_thomson_scale = 4.0 * low_run.cfg.electron_gamma * low_run.cfg.electron_gamma *
                                     low_run.cfg.incident_photon_energy;
    const double high_thomson_scale = 4.0 * high_run.cfg.electron_gamma * high_run.cfg.electron_gamma *
                                      high_run.cfg.incident_photon_energy;
    const double low_ratio = low_mean_energy / low_thomson_scale;
    const double high_ratio = high_mean_energy / high_thomson_scale;
    const std::string report_path = base_cfg.output_dir + "/" +
                                    validation_tag(base_cfg, "validate_kn_trend_report") + ".csv";

    write_validation_report(report_path,
                            {
                                {"validation_case", "klein_nishina_suppression_trend"},
                                {"low_energy_run_tag", low_run.tag},
                                {"high_energy_run_tag", high_run.tag},
                                {"low_energy_mean_scattered_mec2", format_number(low_mean_energy)},
                                {"high_energy_mean_scattered_mec2", format_number(high_mean_energy)},
                                {"low_energy_thomson_ratio", format_number(low_ratio)},
                                {"high_energy_thomson_ratio", format_number(high_ratio)},
                                {"suppression_observed", high_ratio < low_ratio ? "true" : "false"},
                            });
}

void run_validation_conservation(const ic::Config& base_cfg) {
    ic::Config cfg = base_cfg;
    cfg.mode = "validate-conservation";
    cfg.run_label = base_cfg.run_label.empty() ? "conservation" : base_cfg.run_label + "_conservation";

    const CompletedRun run = run_and_record(cfg);
    const std::string report_path = run.cfg.output_dir + "/" +
                                    validation_tag(base_cfg, "validate_conservation_report") + ".csv";

    write_validation_report(report_path,
                            {
                                {"validation_case", "energy_mass_shell_conservation"},
                                {"source_run_tag", run.tag},
                                {"mean_abs_energy_error", format_number(safe_mean(run.stats.sum_abs_energy_error, run.stats.scattered_events))},
                                {"max_abs_energy_error", format_number(run.stats.max_abs_energy_error)},
                                {"mean_mass_shell_error", format_number(safe_mean(run.stats.sum_mass_shell_error, run.stats.scattered_events))},
                                {"max_mass_shell_error", format_number(run.stats.max_mass_shell_error)},
                            });
}

void run_validation_thermal(const ic::Config& base_cfg) {
    ic::Config cfg = base_cfg;
    cfg.mode = "validate-thermal";
    cfg.run_label = base_cfg.run_label.empty() ? "thermal" : base_cfg.run_label + "_thermal";
    cfg.electron_model = "thermal";
    cfg.electron_kTe = 0.2;
    cfg.incident_photon_energy = 1.0e-3;

    const CompletedRun run = run_and_record(cfg);
    const double mean_gamma = safe_mean(run.stats.sum_sampled_electron_gamma, run.stats.scattered_events);
    const double mean_gamma2 = safe_mean(run.stats.sum_sampled_electron_gamma2, run.stats.scattered_events);
    const double gamma_stddev = std::sqrt(std::max(0.0, mean_gamma2 - mean_gamma * mean_gamma));
    const std::string report_path = run.cfg.output_dir + "/" +
                                    validation_tag(base_cfg, "validate_thermal_report") + ".csv";

    write_validation_report(report_path,
                            {
                                {"validation_case", "thermal_electron_single_scatter"},
                                {"source_run_tag", run.tag},
                                {"electron_model", run.cfg.electron_model},
                                {"electron_kTe_mec2", format_number(run.cfg.electron_kTe)},
                                {"mean_sampled_electron_gamma", format_number(mean_gamma)},
                                {"stddev_sampled_electron_gamma", format_number(gamma_stddev)},
                                {"min_sampled_electron_gamma", format_number(run.stats.min_sampled_electron_gamma)},
                                {"max_sampled_electron_gamma", format_number(run.stats.max_sampled_electron_gamma)},
                                {"mean_scattered_energy_mec2", format_number(safe_mean(run.stats.sum_scattered_energy, run.stats.scattered_events))},
                                {"mean_erf_energy_ratio", format_number(safe_mean(run.stats.sum_photon_energy_ratio_erf, run.stats.scattered_events))},
                            });
}

void run_validation_thermal_sweep(const ic::Config& base_cfg) {
    const std::vector<double> theta_values{0.02, 0.05, 0.1, 0.2, 0.5};
    std::vector<std::vector<std::string>> rows;
    rows.reserve(theta_values.size());

    double previous_mean_scattered_energy = -1.0;
    bool monotonic_mean_scattered_energy = true;

    for (double theta_e : theta_values) {
        ic::Config cfg = base_cfg;
        cfg.mode = "validate-thermal-sweep";
        cfg.run_label = base_cfg.run_label.empty()
                            ? "thermal_sweep_theta_" + sanitize_token(format_number(theta_e))
                            : base_cfg.run_label + "_thermal_sweep_theta_" + sanitize_token(format_number(theta_e));
        cfg.electron_model = "thermal";
        cfg.electron_kTe = theta_e;
        cfg.incident_photon_energy = kThermalSweepIncidentPhotonEnergy;

        const CompletedRun run = run_and_record(cfg);
        const double mean_gamma = safe_mean(run.stats.sum_sampled_electron_gamma, run.stats.scattered_events);
        const double mean_gamma2 = safe_mean(run.stats.sum_sampled_electron_gamma2, run.stats.scattered_events);
        const double gamma_stddev = std::sqrt(std::max(0.0, mean_gamma2 - mean_gamma * mean_gamma));
        const double mean_scattered_energy = safe_mean(run.stats.sum_scattered_energy, run.stats.scattered_events);
        const double scattered_energy_gain = mean_scattered_energy / run.cfg.incident_photon_energy;
        const double mean_photon_energy_ratio_erf =
            safe_mean(run.stats.sum_photon_energy_ratio_erf, run.stats.scattered_events);
        const double scattered_events = static_cast<double>(run.stats.scattered_events);
        const double overflow_fraction =
            run.stats.scattered_events > 0 ? run.stats.energy_hist.overflow() / scattered_events : 0.0;
        const double underflow_fraction =
            run.stats.scattered_events > 0 ? run.stats.energy_hist.underflow() / scattered_events : 0.0;
        const bool monotonic_vs_previous =
            previous_mean_scattered_energy < 0.0 || mean_scattered_energy >= previous_mean_scattered_energy;

        monotonic_mean_scattered_energy = monotonic_mean_scattered_energy && monotonic_vs_previous;
        previous_mean_scattered_energy = mean_scattered_energy;

        rows.push_back(
            {
                format_number(theta_e),
                run.tag,
                format_number(mean_gamma),
                format_number(gamma_stddev),
                format_number(mean_scattered_energy),
                format_number(scattered_energy_gain),
                format_number(mean_photon_energy_ratio_erf),
                format_number(overflow_fraction),
                format_number(underflow_fraction),
                monotonic_vs_previous ? "true" : "false",
            });
    }

    const std::string table_path = base_cfg.output_dir + "/" +
                                   validation_tag(base_cfg, "validate_thermal_sweep_table") + ".csv";
    write_table_csv(table_path,
                    {
                        "theta_e_mec2",
                        "source_run_tag",
                        "mean_sampled_electron_gamma",
                        "stddev_sampled_electron_gamma",
                        "mean_scattered_energy_mec2",
                        "mean_scattered_energy_over_incident",
                        "mean_photon_energy_ratio_erf",
                        "energy_hist_overflow_fraction",
                        "energy_hist_underflow_fraction",
                        "monotonic_vs_previous_theta",
                    },
                    rows);
    std::cout << "Wrote: " << table_path << "\n";

    const std::string checks_path = base_cfg.output_dir + "/" +
                                    validation_tag(base_cfg, "validate_thermal_sweep_checks") + ".csv";
    write_validation_report(checks_path,
                            {
                                {"validation_case", "thermal_electron_sweep"},
                                {"incident_photon_energy_mec2", format_number(kThermalSweepIncidentPhotonEnergy)},
                                {"mean_scattered_energy_monotonic", monotonic_mean_scattered_energy ? "true" : "false"},
                            });

    if (!monotonic_mean_scattered_energy) {
        std::cerr << "warning: thermal sweep did not show a monotonic increase in mean scattered energy\n";
    }
}

void run_validation_slab_case(const ic::Config& base_cfg,
                              const std::string& mode_name,
                              const std::string& default_label,
                              const std::string& injection_model,
                              double slab_tau,
                              std::uint64_t num_events,
                              bool require_scatter) {
    ic::Config cfg = base_cfg;
    cfg.mode = mode_name;
    cfg.run_label = base_cfg.run_label.empty() ? default_label : base_cfg.run_label + "_" + default_label;
    cfg.geometry_model = "slab";
    cfg.slab_height = 1.0;
    cfg.slab_optical_depth = slab_tau;
    cfg.slab_injection_model = injection_model;
    cfg.max_scatters = 1;
    cfg.num_events = num_events;

    const CompletedRun run = run_and_record(cfg);
    const bool bookkeeping_consistent = slab_bookkeeping_consistent(run.stats);
    const bool scattered_events_positive = run.stats.scattered_events > 0;
    const double escaped_up_total_fraction = ::escaped_up_total_fraction(run.stats);
    const double escaped_down_total_fraction = ::escaped_down_total_fraction(run.stats);
    const double energy_hist_overflow_fraction = ::energy_hist_overflow_fraction(run.stats);
    const double energy_hist_underflow_fraction = ::energy_hist_underflow_fraction(run.stats);

    const std::string report_path = run.cfg.output_dir + "/" +
                                    validation_tag(base_cfg, sanitize_token(mode_name) + "_report") + ".csv";
    write_validation_report(report_path,
                            {
                                {"validation_case", mode_name},
                                {"source_run_tag", run.tag},
                                {"slab_injection_model", run.cfg.slab_injection_model},
                                {"events_completed", std::to_string(run.stats.events_completed)},
                                {"scattered_events", std::to_string(run.stats.scattered_events)},
                                {"scatter_fraction", format_number(::scatter_fraction(run.stats))},
                                {"escaped_up_unscattered", std::to_string(run.stats.escaped_up_unscattered)},
                                {"escaped_down_unscattered", std::to_string(run.stats.escaped_down_unscattered)},
                                {"escaped_up_scattered", std::to_string(run.stats.escaped_up_scattered)},
                                {"escaped_down_scattered", std::to_string(run.stats.escaped_down_scattered)},
                                {"escaped_up_total_fraction", format_number(escaped_up_total_fraction)},
                                {"escaped_down_total_fraction", format_number(escaped_down_total_fraction)},
                                {"mu_hist_up_scattered_in_range_total", format_number(run.stats.mu_hist_up_scattered.in_range_total())},
                                {"mu_hist_up_scattered_underflow_fraction",
                                 format_number(run.stats.escaped_up_scattered > 0
                                                   ? run.stats.mu_hist_up_scattered.underflow() /
                                                         static_cast<double>(run.stats.escaped_up_scattered)
                                                   : 0.0)},
                                {"mu_hist_up_scattered_overflow_fraction",
                                 format_number(run.stats.escaped_up_scattered > 0
                                                   ? run.stats.mu_hist_up_scattered.overflow() /
                                                         static_cast<double>(run.stats.escaped_up_scattered)
                                                   : 0.0)},
                                {"mu_hist_down_scattered_in_range_total", format_number(run.stats.mu_hist_down_scattered.in_range_total())},
                                {"mu_hist_down_scattered_underflow_fraction",
                                 format_number(run.stats.escaped_down_scattered > 0
                                                   ? run.stats.mu_hist_down_scattered.underflow() /
                                                         static_cast<double>(run.stats.escaped_down_scattered)
                                                   : 0.0)},
                                {"mu_hist_down_scattered_overflow_fraction",
                                 format_number(run.stats.escaped_down_scattered > 0
                                                   ? run.stats.mu_hist_down_scattered.overflow() /
                                                         static_cast<double>(run.stats.escaped_down_scattered)
                                                   : 0.0)},
                                {"energy_hist_overflow_fraction", format_number(energy_hist_overflow_fraction)},
                                {"energy_hist_underflow_fraction", format_number(energy_hist_underflow_fraction)},
                                {"bookkeeping_consistent", bookkeeping_consistent ? "true" : "false"},
                                {"scattered_events_positive", scattered_events_positive ? "true" : "false"},
                                {"required_scatter_condition_met",
                                 (!require_scatter || scattered_events_positive) ? "true" : "false"},
                            });

    if (!bookkeeping_consistent) {
        std::cerr << "warning: slab validation bookkeeping failed for mode " << mode_name << "\n";
    }
    if (require_scatter && !scattered_events_positive) {
        std::cerr << "warning: slab validation mode " << mode_name
                  << " produced zero scattered events; consider increasing events or tau\n";
    }
}

void run_validation_slab_paper(const ic::Config& base_cfg) {
    const std::uint64_t validation_events = std::max<std::uint64_t>(base_cfg.num_events, 20000);

    ic::Config beam_cfg = base_cfg;
    beam_cfg.mode = "validate-slab-paper-beam-thin";
    beam_cfg.run_label = base_cfg.run_label.empty()
                             ? "paper_beam_thin"
                             : base_cfg.run_label + "_paper_beam_thin";
    beam_cfg.geometry_model = "slab";
    beam_cfg.transport_cross_section = "thomson";
    beam_cfg.slab_height = 1.0;
    beam_cfg.slab_optical_depth = kPaperValidationThinTau;
    beam_cfg.slab_injection_model = "beam";
    beam_cfg.max_scatters = 8;
    beam_cfg.num_events = validation_events;

    ic::Config lambert_cfg = base_cfg;
    lambert_cfg.mode = "validate-slab-paper-lambert-thin";
    lambert_cfg.run_label = base_cfg.run_label.empty()
                                ? "paper_lambert_thin"
                                : base_cfg.run_label + "_paper_lambert_thin";
    lambert_cfg.geometry_model = "slab";
    lambert_cfg.transport_cross_section = "thomson";
    lambert_cfg.slab_height = 1.0;
    lambert_cfg.slab_optical_depth = kPaperValidationThinTau;
    lambert_cfg.slab_injection_model = "lambert";
    lambert_cfg.max_scatters = 8;
    lambert_cfg.num_events = validation_events;

    ic::Config internal_cfg = base_cfg;
    internal_cfg.mode = "validate-slab-paper-internal-iso";
    internal_cfg.run_label = base_cfg.run_label.empty()
                                ? "paper_internal_iso"
                                : base_cfg.run_label + "_paper_internal_iso";
    internal_cfg.geometry_model = "slab";
    internal_cfg.transport_cross_section = "thomson";
    internal_cfg.slab_height = 1.0;
    internal_cfg.slab_optical_depth = 0.1;
    internal_cfg.slab_injection_model = "internal_iso";
    internal_cfg.max_scatters = 8;
    internal_cfg.num_events = validation_events;

    const CompletedRun beam_run = run_and_record(beam_cfg);
    const CompletedRun lambert_run = run_and_record(lambert_cfg);
    const CompletedRun internal_run = run_and_record(internal_cfg);

    const double expected_beam_p0_up = std::exp(-beam_run.cfg.slab_optical_depth);
    const double beam_p0_up =
        order_fraction(beam_run.stats.escaped_up_by_order, 0, beam_run.stats.events_completed);
    const double beam_p0_down =
        order_fraction(beam_run.stats.escaped_down_by_order, 0, beam_run.stats.events_completed);
    const double beam_high_order =
        1.0 - order_fraction(beam_run.stats.escaped_up_by_order, 0, beam_run.stats.events_completed) -
        order_fraction(beam_run.stats.escaped_down_by_order, 0, beam_run.stats.events_completed);
    const double lambert_p0_down =
        order_fraction(lambert_run.stats.escaped_down_by_order, 0, lambert_run.stats.events_completed);
    const double internal_p0_up =
        order_fraction(internal_run.stats.escaped_up_by_order, 0, internal_run.stats.events_completed);
    const double internal_p0_down =
        order_fraction(internal_run.stats.escaped_down_by_order, 0, internal_run.stats.events_completed);
    const double internal_p0_difference = std::abs(internal_p0_up - internal_p0_down);

    const bool beam_p0_up_ok =
        std::abs(beam_p0_up - expected_beam_p0_up) <= kPaperValidationThinBeamP0Tolerance;
    const bool beam_p0_down_ok = beam_p0_down <= kPaperValidationSmallProbabilityTolerance;
    const bool beam_high_order_ok = beam_high_order <= kPaperValidationSmallProbabilityTolerance;
    const bool lambert_p0_down_ok =
        lambert_p0_down <= kPaperValidationSmallProbabilityTolerance;
    const bool internal_symmetry_ok =
        internal_p0_difference <= kPaperValidationInternalSymmetryTolerance;
    const bool beam_closure_ok =
        std::abs(probability_budget_closure(beam_run.stats) - 1.0) <= kPaperValidationClosureTolerance;
    const bool lambert_closure_ok =
        std::abs(probability_budget_closure(lambert_run.stats) - 1.0) <= kPaperValidationClosureTolerance;
    const bool internal_closure_ok =
        std::abs(probability_budget_closure(internal_run.stats) - 1.0) <= kPaperValidationClosureTolerance;

    const auto relative_error = [](double measured, double reference) {
        return reference != 0.0 ? std::abs(measured - reference) / std::abs(reference)
                                : std::abs(measured - reference);
    };

    std::vector<std::vector<std::string>> check_rows;
    check_rows.push_back({
        "thin_beam_P0_up",
        beam_p0_up_ok ? "true" : "false",
        format_number(beam_p0_up),
        format_number(expected_beam_p0_up),
        format_number(relative_error(beam_p0_up, expected_beam_p0_up)),
        "normal beam uses mu_b=1, so P0_up=exp(-tau)",
    });
    check_rows.push_back({
        "thin_lambert_P0_down",
        lambert_p0_down_ok ? "true" : "false",
        format_number(lambert_p0_down),
        format_number(0.0),
        format_number(relative_error(lambert_p0_down, 0.0)),
        "lower-boundary Lambert injection cannot escape downward without scattering",
    });
    check_rows.push_back({
        "internal_iso_P0_symmetry",
        internal_symmetry_ok ? "true" : "false",
        format_number(internal_p0_difference),
        format_number(0.0),
        format_number(relative_error(internal_p0_difference, 0.0)),
        "checks abs(P0_up-P0_down); P0_up=" + format_number(internal_p0_up) +
            " P0_down=" + format_number(internal_p0_down),
    });
    check_rows.push_back({
        "probability_budget_beam_thin",
        beam_closure_ok ? "true" : "false",
        format_number(probability_budget_closure(beam_run.stats)),
        format_number(1.0),
        format_number(relative_error(probability_budget_closure(beam_run.stats), 1.0)),
        "sum_n(Pn_up+Pn_down)+termination_fraction",
    });
    check_rows.push_back({
        "probability_budget_lambert_thin",
        lambert_closure_ok ? "true" : "false",
        format_number(probability_budget_closure(lambert_run.stats)),
        format_number(1.0),
        format_number(relative_error(probability_budget_closure(lambert_run.stats), 1.0)),
        "sum_n(Pn_up+Pn_down)+termination_fraction",
    });
    check_rows.push_back({
        "probability_budget_internal_iso",
        internal_closure_ok ? "true" : "false",
        format_number(probability_budget_closure(internal_run.stats)),
        format_number(1.0),
        format_number(relative_error(probability_budget_closure(internal_run.stats), 1.0)),
        "sum_n(Pn_up+Pn_down)+termination_fraction",
    });

    const std::string checks_path = base_cfg.output_dir + "/" +
                                    validation_tag(base_cfg, "validate_slab_paper_validation_checks") + ".csv";
    write_table_csv(checks_path,
                    {"test_name", "passed", "measured_value", "reference_value", "relative_error", "notes"},
                    check_rows);
    std::cout << "Wrote: " << checks_path << "\n";

    const std::string report_path = base_cfg.output_dir + "/" +
                                    validation_tag(base_cfg, "validate_slab_paper_report") + ".csv";
    write_validation_report(report_path,
                            {
                                {"validation_case", "slab_paper_by_order_outputs"},
                                {"beam_thin_run_tag", beam_run.tag},
                                {"lambert_thin_run_tag", lambert_run.tag},
                                {"internal_iso_run_tag", internal_run.tag},
                                {"thin_tau", format_number(kPaperValidationThinTau)},
                                {"beam_thin_expected_P0_up_exp_minus_tau", format_number(expected_beam_p0_up)},
                                {"beam_thin_P0_up", format_number(beam_p0_up)},
                                {"beam_thin_P0_down", format_number(beam_p0_down)},
                                {"beam_thin_nonzero_scatter_or_termination_fraction", format_number(beam_high_order)},
                                {"lambert_thin_P0_down", format_number(lambert_p0_down)},
                                {"internal_iso_P0_up", format_number(internal_p0_up)},
                                {"internal_iso_P0_down", format_number(internal_p0_down)},
                                {"internal_iso_abs_P0_difference", format_number(internal_p0_difference)},
                                {"beam_thin_P0_up_close_to_exp_minus_tau", beam_p0_up_ok ? "true" : "false"},
                                {"beam_thin_P0_down_small", beam_p0_down_ok ? "true" : "false"},
                                {"beam_thin_higher_order_small", beam_high_order_ok ? "true" : "false"},
                                {"lambert_thin_P0_down_small", lambert_p0_down_ok ? "true" : "false"},
                                {"internal_iso_P0_symmetry_ok", internal_symmetry_ok ? "true" : "false"},
                                {"beam_probability_budget_closure_ok", beam_closure_ok ? "true" : "false"},
                                {"lambert_probability_budget_closure_ok", lambert_closure_ok ? "true" : "false"},
                                {"internal_iso_probability_budget_closure_ok", internal_closure_ok ? "true" : "false"},
                            });

    if (!beam_p0_up_ok || !beam_p0_down_ok || !beam_high_order_ok ||
        !lambert_p0_down_ok || !internal_symmetry_ok ||
        !beam_closure_ok || !lambert_closure_ok || !internal_closure_ok) {
        std::cerr << "warning: validate-slab-paper reported at least one failed check\n";
    }
}

void run_sweep_slab_paper(const ic::Config& base_cfg) {
    const std::vector<double> tau_values =
        parse_double_list_or_default(base_cfg.tau_list, {0.1, 0.3, 1.0, 3.0, 5.0});
    const std::vector<std::string> injection_models =
        parse_string_list_or_default(base_cfg.injection_list, {"beam", "lambert", "internal_iso"});

    std::vector<std::vector<std::string>> rows;
    rows.reserve(tau_values.size() * injection_models.size());

    for (double tau : tau_values) {
        if (tau < 0.0) {
            throw std::runtime_error("sweep-slab-paper tau values must be >= 0");
        }

        for (const std::string& injection_model : injection_models) {
            if (!ic::is_supported_slab_injection_model(injection_model)) {
                throw std::runtime_error("unsupported sweep slab injection model: " + injection_model);
            }

            const std::string case_id =
                "slab_tau_" + compact_number_token(tau) +
                "_" + sanitize_token(injection_model) +
                "_" + sanitize_token(base_cfg.transport_cross_section);

            ic::Config cfg = base_cfg;
            cfg.mode = "sweep-slab-paper-case";
            cfg.run_label = base_cfg.run_label.empty()
                                ? case_id
                                : sanitize_token(base_cfg.run_label) + "_" + case_id;
            cfg.geometry_model = "slab";
            cfg.slab_height = 1.0;
            cfg.slab_optical_depth = tau;
            cfg.slab_injection_model = injection_model;
            cfg.transport_cross_section = base_cfg.transport_cross_section;

            const CompletedRun run = run_and_record(cfg);
            rows.push_back({
                case_id,
                format_number(tau),
                injection_model,
                run.cfg.transport_cross_section,
                std::to_string(run.stats.events_completed),
                run.tag,
                format_number(order_fraction(run.stats.escaped_up_by_order, 0, run.stats.events_completed)),
                format_number(order_fraction(run.stats.escaped_down_by_order, 0, run.stats.events_completed)),
                format_number(order_fraction(run.stats.escaped_up_by_order, 1, run.stats.events_completed)),
                format_number(order_fraction(run.stats.escaped_down_by_order, 1, run.stats.events_completed)),
                format_number(escape_probability_sum(run.stats)),
                format_number(max_scatter_termination_fraction(run.stats)),
                format_number(probability_budget_closure(run.stats)),
            });
        }
    }

    const std::string summary_prefix =
        base_cfg.run_label.empty()
            ? "sweep_slab_paper"
            : sanitize_token(base_cfg.run_label) + "_sweep_slab_paper";
    const std::string table_path = base_cfg.output_dir + "/" + summary_prefix + "_sweep_summary.csv";
    write_table_csv(table_path,
                    {
                        "case_id",
                        "tau",
                        "injection_mode",
                        "transport_mode",
                        "events",
                        "source_run_tag",
                        "P0_up",
                        "P0_down",
                        "P1_up",
                        "P1_down",
                        "escape_probability_sum",
                        "termination_fraction",
                        "probability_budget_closure",
                    },
                    rows);
    std::cout << "Wrote: " << table_path << "\n";
}

void run_validation_slab_thin(const ic::Config& base_cfg) {
    run_validation_slab_case(base_cfg,
                             "validate-slab-thin",
                             "slab_thin",
                             base_cfg.slab_injection_model,
                             1.0e-4,
                             2000,
                             false);
}

void run_validation_slab_moderate(const ic::Config& base_cfg) {
    const std::vector<std::string> injection_models{"beam", "lambert"};
    std::vector<std::vector<std::string>> rows;
    rows.reserve(injection_models.size());

    for (const std::string& injection_model : injection_models) {
        ic::Config cfg = base_cfg;
        cfg.mode = "validate-slab-moderate";
        cfg.run_label = base_cfg.run_label.empty()
                            ? "slab_moderate_" + injection_model
                            : base_cfg.run_label + "_slab_moderate_" + injection_model;
        cfg.geometry_model = "slab";
        cfg.slab_height = 1.0;
        cfg.slab_optical_depth = 0.5;
        cfg.slab_injection_model = injection_model;
        cfg.max_scatters = 1;
        cfg.num_events = 4000;

        const CompletedRun run = run_and_record(cfg);
        const bool bookkeeping_consistent = slab_bookkeeping_consistent(run.stats);
        rows.push_back({
            run.cfg.slab_injection_model,
            run.tag,
            format_number(::scatter_fraction(run.stats)),
            format_number(::escaped_up_total_fraction(run.stats)),
            format_number(::escaped_down_total_fraction(run.stats)),
            bookkeeping_consistent ? "true" : "false",
            format_number(run.stats.mu_hist_up_scattered.in_range_total()),
            format_number(run.stats.escaped_up_scattered > 0
                              ? run.stats.mu_hist_up_scattered.underflow() /
                                    static_cast<double>(run.stats.escaped_up_scattered)
                              : 0.0),
            format_number(run.stats.escaped_up_scattered > 0
                              ? run.stats.mu_hist_up_scattered.overflow() /
                                    static_cast<double>(run.stats.escaped_up_scattered)
                              : 0.0),
            format_number(run.stats.mu_hist_down_scattered.in_range_total()),
            format_number(run.stats.escaped_down_scattered > 0
                              ? run.stats.mu_hist_down_scattered.underflow() /
                                    static_cast<double>(run.stats.escaped_down_scattered)
                              : 0.0),
            format_number(run.stats.escaped_down_scattered > 0
                              ? run.stats.mu_hist_down_scattered.overflow() /
                                    static_cast<double>(run.stats.escaped_down_scattered)
                              : 0.0),
            run.stats.scattered_events > 0 ? "true" : "false",
        });
    }

    const std::string table_path = base_cfg.output_dir + "/" +
                                   validation_tag(base_cfg, "validate_slab_moderate_table") + ".csv";
    write_table_csv(table_path,
                    {
                        "slab_injection_model",
                        "source_run_tag",
                        "scatter_fraction",
                        "escaped_up_total_fraction",
                        "escaped_down_total_fraction",
                        "bookkeeping_consistent",
                        "mu_hist_up_scattered_in_range_total",
                        "mu_hist_up_scattered_underflow_fraction",
                        "mu_hist_up_scattered_overflow_fraction",
                        "mu_hist_down_scattered_in_range_total",
                        "mu_hist_down_scattered_underflow_fraction",
                        "mu_hist_down_scattered_overflow_fraction",
                        "scattered_events_positive",
                    },
                    rows);
    std::cout << "Wrote: " << table_path << "\n";
}

void run_validation_slab_tau_sweep(const ic::Config& base_cfg) {
    const std::vector<std::string> injection_models{"beam", "lambert"};
    const std::vector<double> tau_values{1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1, 2.0e-1, 5.0e-1, 1.0};
    std::vector<std::vector<std::string>> rows;
    rows.reserve(injection_models.size() * tau_values.size());

    bool beam_small_tau_matches_transmission = true;
    bool beam_scatter_fraction_monotonic = true;
    bool lambert_scatter_fraction_monotonic = true;
    double previous_beam_scatter_fraction = -1.0;
    double previous_lambert_scatter_fraction = -1.0;

    for (const std::string& injection_model : injection_models) {
        double* previous_scatter_fraction =
            injection_model == "beam" ? &previous_beam_scatter_fraction : &previous_lambert_scatter_fraction;
        bool* monotonic_ok =
            injection_model == "beam" ? &beam_scatter_fraction_monotonic : &lambert_scatter_fraction_monotonic;

        for (double tau : tau_values) {
            ic::Config cfg = base_cfg;
            cfg.mode = "validate-slab-tau-sweep";
            cfg.run_label = base_cfg.run_label.empty()
                                ? "slab_tau_" + injection_model + "_" + sanitize_token(format_number(tau))
                                : base_cfg.run_label + "_slab_tau_" + injection_model + "_" +
                                      sanitize_token(format_number(tau));
            cfg.geometry_model = "slab";
            cfg.slab_height = 1.0;
            cfg.slab_optical_depth = tau;
            cfg.slab_injection_model = injection_model;
            cfg.max_scatters = 1;
            cfg.num_events = 6000;

            const CompletedRun run = run_and_record(cfg);
            const double current_scatter_fraction = scatter_fraction(run.stats);
            const double current_escaped_up_total_fraction = escaped_up_total_fraction(run.stats);
            const double current_escaped_down_total_fraction = escaped_down_total_fraction(run.stats);
            const bool bookkeeping_consistent = slab_bookkeeping_consistent(run.stats);
            const bool monotonic_vs_previous =
                *previous_scatter_fraction < 0.0 || current_scatter_fraction >= *previous_scatter_fraction;
            *monotonic_ok = *monotonic_ok && monotonic_vs_previous;
            *previous_scatter_fraction = current_scatter_fraction;

            if (injection_model == "beam" && tau <= 1.0e-2) {
                const double expected_transmission = std::exp(-tau);
                beam_small_tau_matches_transmission =
                    beam_small_tau_matches_transmission &&
                    std::abs(current_escaped_up_total_fraction - expected_transmission) <=
                        kBeamThinTransmissionTolerance;
            }

            rows.push_back({
                format_number(tau),
                injection_model,
                run.tag,
                format_number(current_scatter_fraction),
                format_number(current_escaped_up_total_fraction),
                format_number(current_escaped_down_total_fraction),
                std::to_string(run.stats.escaped_up_unscattered),
                std::to_string(run.stats.escaped_down_unscattered),
                std::to_string(run.stats.escaped_up_scattered),
                std::to_string(run.stats.escaped_down_scattered),
                bookkeeping_consistent ? "true" : "false",
                format_number(energy_hist_overflow_fraction(run.stats)),
                format_number(up_scattered_mu_hist_overflow_fraction(run.stats)),
                format_number(down_scattered_mu_hist_overflow_fraction(run.stats)),
                monotonic_vs_previous ? "true" : "false",
            });
        }
    }

    const std::string table_path = base_cfg.output_dir + "/" +
                                   validation_tag(base_cfg, "validate_slab_tau_sweep_table") + ".csv";
    write_table_csv(table_path,
                    {
                        "slab_tau",
                        "slab_injection_model",
                        "source_run_tag",
                        "scatter_fraction",
                        "escaped_up_total_fraction",
                        "escaped_down_total_fraction",
                        "escaped_up_unscattered",
                        "escaped_down_unscattered",
                        "escaped_up_scattered",
                        "escaped_down_scattered",
                        "bookkeeping_consistent",
                        "energy_hist_overflow_fraction",
                        "mu_hist_up_scattered_overflow_fraction",
                        "mu_hist_down_scattered_overflow_fraction",
                        "scatter_fraction_monotonic_vs_previous_tau",
                    },
                    rows);
    std::cout << "Wrote: " << table_path << "\n";

    const std::string checks_path = base_cfg.output_dir + "/" +
                                    validation_tag(base_cfg, "validate_slab_tau_sweep_checks") + ".csv";
    write_validation_report(checks_path,
                            {
                                {"validation_case", "slab_optical_depth_sweep"},
                                {"beam_small_tau_transmission_matches_exp_minus_tau",
                                 beam_small_tau_matches_transmission ? "true" : "false"},
                                {"beam_scatter_fraction_monotonic",
                                 beam_scatter_fraction_monotonic ? "true" : "false"},
                                {"lambert_scatter_fraction_monotonic",
                                 lambert_scatter_fraction_monotonic ? "true" : "false"},
                            });

    if (!beam_small_tau_matches_transmission) {
        std::cerr << "warning: beam small-tau transmission deviates from exp(-tau) beyond tolerance\n";
    }
    if (!beam_scatter_fraction_monotonic || !lambert_scatter_fraction_monotonic) {
        std::cerr << "warning: slab tau sweep did not show monotonic scatter fraction growth for all injection models\n";
    }
}

void run_validation_slab_electron_sweep(const ic::Config& base_cfg) {
    const std::vector<std::string> injection_models{"beam", "lambert"};
    const std::vector<double> tau_values{1.0e-1, 5.0e-1};
    const std::vector<double> theta_values{0.02, 0.05, 0.1, 0.2, 0.5};
    std::vector<std::vector<std::string>> rows;
    std::vector<std::vector<std::string>> checks_rows;
    rows.reserve(injection_models.size() * tau_values.size() * (1 + theta_values.size()));
    checks_rows.reserve(injection_models.size() * tau_values.size());

    for (const std::string& injection_model : injection_models) {
        for (double tau : tau_values) {
            bool thermal_mean_energy_monotonic = true;
            double previous_thermal_mean_scattered_energy = -1.0;

            {
                ic::Config cfg = base_cfg;
                cfg.mode = "validate-slab-electron-sweep";
                cfg.run_label.clear();
                cfg.geometry_model = "slab";
                cfg.slab_height = 1.0;
                cfg.slab_optical_depth = tau;
                cfg.slab_injection_model = injection_model;
                cfg.max_scatters = 1;
                cfg.electron_model = "monoenergetic";
                cfg.electron_gamma = 10.0;
                cfg.num_events = 4000;

                const CompletedRun run = run_and_record(cfg);
                rows.push_back({
                    format_number(tau),
                    run.cfg.slab_injection_model,
                    run.cfg.electron_model,
                    format_number(run.cfg.electron_gamma),
                    "na",
                    run.tag,
                    format_number(scatter_fraction(run.stats)),
                    format_number(escaped_up_total_fraction(run.stats)),
                    format_number(escaped_down_total_fraction(run.stats)),
                    std::to_string(run.stats.escaped_up_unscattered),
                    std::to_string(run.stats.escaped_down_unscattered),
                    std::to_string(run.stats.escaped_up_scattered),
                    std::to_string(run.stats.escaped_down_scattered),
                    format_number(mean_scattered_energy_mec2(run.stats)),
                    format_number(mean_photon_energy_ratio_erf(run.stats)),
                    slab_bookkeeping_consistent(run.stats) ? "true" : "false",
                    format_number(energy_hist_overflow_fraction(run.stats)),
                    format_number(up_scattered_mu_hist_overflow_fraction(run.stats)),
                    format_number(down_scattered_mu_hist_overflow_fraction(run.stats)),
                    "na",
                });
            }

            for (double theta_e : theta_values) {
                ic::Config cfg = base_cfg;
                cfg.mode = "validate-slab-electron-sweep";
                cfg.run_label.clear();
                cfg.geometry_model = "slab";
                cfg.slab_height = 1.0;
                cfg.slab_optical_depth = tau;
                cfg.slab_injection_model = injection_model;
                cfg.max_scatters = 1;
                cfg.electron_model = "thermal";
                cfg.electron_kTe = theta_e;
                cfg.num_events = 4000;

                const CompletedRun run = run_and_record(cfg);
                const double current_mean_scattered_energy = mean_scattered_energy_mec2(run.stats);
                const bool monotonic_vs_previous =
                    previous_thermal_mean_scattered_energy < 0.0 ||
                    current_mean_scattered_energy >= previous_thermal_mean_scattered_energy;
                thermal_mean_energy_monotonic = thermal_mean_energy_monotonic && monotonic_vs_previous;
                previous_thermal_mean_scattered_energy = current_mean_scattered_energy;

                rows.push_back({
                    format_number(tau),
                    run.cfg.slab_injection_model,
                    run.cfg.electron_model,
                    "na",
                    format_number(run.cfg.electron_kTe),
                    run.tag,
                    format_number(scatter_fraction(run.stats)),
                    format_number(escaped_up_total_fraction(run.stats)),
                    format_number(escaped_down_total_fraction(run.stats)),
                    std::to_string(run.stats.escaped_up_unscattered),
                    std::to_string(run.stats.escaped_down_unscattered),
                    std::to_string(run.stats.escaped_up_scattered),
                    std::to_string(run.stats.escaped_down_scattered),
                    format_number(current_mean_scattered_energy),
                    format_number(mean_photon_energy_ratio_erf(run.stats)),
                    slab_bookkeeping_consistent(run.stats) ? "true" : "false",
                    format_number(energy_hist_overflow_fraction(run.stats)),
                    format_number(up_scattered_mu_hist_overflow_fraction(run.stats)),
                    format_number(down_scattered_mu_hist_overflow_fraction(run.stats)),
                    monotonic_vs_previous ? "true" : "false",
                });
            }

            checks_rows.push_back({
                format_number(tau),
                injection_model,
                thermal_mean_energy_monotonic ? "true" : "false",
            });

            if (!thermal_mean_energy_monotonic) {
                std::cerr << "warning: slab electron sweep did not show monotonic thermal mean scattered energy for injection_model="
                          << injection_model << " tau=" << tau << "\n";
            }
        }
    }

    const std::string table_path = base_cfg.output_dir + "/" +
                                   validation_tag(base_cfg, "validate_slab_electron_sweep_table") + ".csv";
    write_table_csv(table_path,
                    {
                        "slab_tau",
                        "slab_injection_model",
                        "electron_model",
                        "electron_gamma",
                        "electron_kTe_mec2",
                        "source_run_tag",
                        "scatter_fraction",
                        "escaped_up_total_fraction",
                        "escaped_down_total_fraction",
                        "escaped_up_unscattered",
                        "escaped_down_unscattered",
                        "escaped_up_scattered",
                        "escaped_down_scattered",
                        "mean_scattered_energy_mec2",
                        "mean_photon_energy_ratio_erf",
                        "bookkeeping_consistent",
                        "energy_hist_overflow_fraction",
                        "mu_hist_up_scattered_overflow_fraction",
                        "mu_hist_down_scattered_overflow_fraction",
                        "thermal_mean_energy_monotonic_vs_previous_theta",
                    },
                    rows);
    std::cout << "Wrote: " << table_path << "\n";

    const std::string checks_path = base_cfg.output_dir + "/" +
                                    validation_tag(base_cfg, "validate_slab_electron_sweep_checks") + ".csv";
    write_table_csv(checks_path,
                    {
                        "slab_tau",
                        "slab_injection_model",
                        "thermal_mean_scattered_energy_monotonic",
                    },
                    checks_rows);
    std::cout << "Wrote: " << checks_path << "\n";
}

void run_validation_slab_multi_case(const ic::Config& base_cfg,
                                    const std::string& mode_name,
                                    const std::string& default_label,
                                    double slab_tau,
                                    std::size_t max_scatters,
                                    std::uint64_t num_events) {
    ic::Config cfg = base_cfg;
    cfg.mode = mode_name;
    cfg.run_label = base_cfg.run_label.empty() ? default_label : base_cfg.run_label + "_" + default_label;
    cfg.geometry_model = "slab";
    cfg.slab_height = 1.0;
    cfg.slab_optical_depth = slab_tau;
    cfg.max_scatters = max_scatters;
    cfg.num_events = num_events;
    apply_multi_scatter_validation_energy_range(cfg);

    const CompletedRun run = run_and_record(cfg);
    const bool bookkeeping_consistent = slab_bookkeeping_consistent(run.stats);
    const bool multiple_scatterings_observed =
        run.stats.total_scatter_interactions > run.stats.scattered_events;
    const double overflow_fraction = energy_hist_overflow_fraction(run.stats);
    const bool overflow_acceptable =
        overflow_fraction <= kValidationOverflowAcceptableThreshold;
    const std::string report_path = run.cfg.output_dir + "/" +
                                    validation_tag(base_cfg, sanitize_token(mode_name) + "_report") + ".csv";

    write_validation_report(report_path,
                            {
                                {"validation_case", mode_name},
                                {"source_run_tag", run.tag},
                                {"slab_injection_model", run.cfg.slab_injection_model},
                                {"slab_tau", format_number(run.cfg.slab_optical_depth)},
                                {"max_scatters", std::to_string(run.cfg.max_scatters)},
                                {"events_completed", std::to_string(run.stats.events_completed)},
                                {"scattered_events", std::to_string(run.stats.scattered_events)},
                                {"total_scatter_interactions", std::to_string(run.stats.total_scatter_interactions)},
                                {"scatter_fraction", format_number(scatter_fraction(run.stats))},
                                {"escaped_up_total_fraction", format_number(escaped_up_total_fraction(run.stats))},
                                {"escaped_down_total_fraction", format_number(escaped_down_total_fraction(run.stats))},
                                {"terminated_at_max_scatters", std::to_string(run.stats.terminated_at_max_scatters)},
                                {"max_scatter_termination_fraction", format_number(max_scatter_termination_fraction(run.stats))},
                                {"mean_scatter_count", format_number(mean_scatter_count(run.stats))},
                                {"energy_max", format_number(run.cfg.energy_max)},
                                {"energy_hist_overflow_fraction", format_number(overflow_fraction)},
                                {"energy_hist_overflow_acceptable_threshold",
                                 format_number(kValidationOverflowAcceptableThreshold)},
                                {"energy_hist_overflow_acceptable", overflow_acceptable ? "true" : "false"},
                                {"bookkeeping_consistent", bookkeeping_consistent ? "true" : "false"},
                                {"multiple_scatterings_observed", multiple_scatterings_observed ? "true" : "false"},
                            });

    if (!bookkeeping_consistent) {
        std::cerr << "warning: multi-scatter slab validation bookkeeping failed for mode "
                  << mode_name << "\n";
    }
    maybe_warn_about_validation_overflow(mode_name, run.stats);
}

void run_validation_slab_multi_thin(const ic::Config& base_cfg) {
    run_validation_slab_multi_case(base_cfg,
                                   "validate-slab-multi-thin",
                                   "slab_multi_thin",
                                   1.0e-2,
                                   std::max<std::size_t>(base_cfg.max_scatters, 16),
                                   4000);
}

void run_validation_slab_multi_moderate(const ic::Config& base_cfg) {
    run_validation_slab_multi_case(base_cfg,
                                   "validate-slab-multi-moderate",
                                   "slab_multi_moderate",
                                   1.0,
                                   std::max<std::size_t>(base_cfg.max_scatters, 16),
                                   6000);
}

void run_validation_slab_multi_convergence(const ic::Config& base_cfg) {
    struct InjectionConvergenceSummary {
        std::string injection_model;
        bool termination_fraction_nonincreasing = true;
        bool stabilization_emerging = false;
        bool observable_differences_shrink = false;
        bool reference_differences_shrink = false;
        bool max_scatters_16_close_to_reference = false;
        double termination_fraction_at_2 = 0.0;
        double late_difference_sum = 0.0;
        double reference_difference_sum_at_1 = 0.0;
        double reference_difference_sum_at_16 = 0.0;
    };

    const std::vector<double> tau_values{1.0e-1, 1.0};
    const std::vector<std::string> injection_models{"beam", "lambert"};
    const std::vector<std::size_t> max_scatters_values{1, 2, 4, 8, 16, 32};
    std::vector<std::vector<std::string>> rows;
    std::vector<std::vector<std::string>> difference_rows;
    std::vector<std::vector<std::string>> reference_difference_rows;
    std::vector<std::vector<std::string>> checks_rows;
    std::vector<std::vector<std::string>> reference_checks_rows;
    std::vector<std::vector<std::string>> comparison_rows;
    rows.reserve(tau_values.size() * injection_models.size() * max_scatters_values.size());
    difference_rows.reserve(tau_values.size() * injection_models.size() * (max_scatters_values.size() - 1));
    reference_difference_rows.reserve(tau_values.size() * injection_models.size() * (max_scatters_values.size() - 1));
    checks_rows.reserve(tau_values.size() * injection_models.size());
    reference_checks_rows.reserve(tau_values.size() * injection_models.size());
    comparison_rows.reserve(tau_values.size());

    for (double tau : tau_values) {
        std::vector<InjectionConvergenceSummary> injection_summaries;
        injection_summaries.reserve(injection_models.size());

        for (const std::string& injection_model : injection_models) {
            double previous_termination_fraction = -1.0;
            bool termination_fraction_nonincreasing = true;
            bool stabilization_emerging = false;
            bool observable_differences_shrink = false;
            bool reference_differences_shrink = false;
            bool max_scatters_16_close_to_reference = false;

            double mean_scatter_count_16 = -1.0;
            double mean_scatter_count_32 = -1.0;
            double escaped_up_fraction_16 = -1.0;
            double escaped_up_fraction_32 = -1.0;
            double escaped_down_fraction_16 = -1.0;
            double escaped_down_fraction_32 = -1.0;
            double mean_scattered_energy_16 = -1.0;
            double mean_scattered_energy_32 = -1.0;

            double termination_fraction_at_2 = 0.0;
            double early_difference_sum = -1.0;
            double late_difference_sum = -1.0;
            double previous_reference_difference_sum = -1.0;
            double reference_difference_sum_at_1 = -1.0;
            double reference_difference_sum_at_16 = -1.0;
            std::vector<ConvergenceSnapshot> snapshots;
            snapshots.reserve(max_scatters_values.size());

            for (std::size_t max_scatters : max_scatters_values) {
                ic::Config cfg = base_cfg;
                cfg.mode = "validate-slab-multi-convergence";
                cfg.run_label.clear();
                cfg.geometry_model = "slab";
                cfg.slab_height = 1.0;
                cfg.slab_optical_depth = tau;
                cfg.slab_injection_model = injection_model;
                cfg.max_scatters = max_scatters;
                cfg.num_events = tau < 0.5 ? 5000 : 7000;
                apply_multi_scatter_validation_energy_range(cfg);

                const CompletedRun run = run_and_record(cfg);
                ConvergenceSnapshot snapshot;
                snapshot.tau = tau;
                snapshot.injection_model = injection_model;
                snapshot.max_scatters = max_scatters;
                snapshot.source_run_tag = run.tag;
                snapshot.events_completed = run.stats.events_completed;
                snapshot.scattered_events = run.stats.scattered_events;
                snapshot.total_scatter_interactions = run.stats.total_scatter_interactions;
                snapshot.scatter_fraction = scatter_fraction(run.stats);
                snapshot.escaped_up_total_fraction = escaped_up_total_fraction(run.stats);
                snapshot.escaped_down_total_fraction = escaped_down_total_fraction(run.stats);
                snapshot.mean_scatter_count = mean_scatter_count(run.stats);
                snapshot.terminated_at_max_scatters = run.stats.terminated_at_max_scatters;
                snapshot.max_scatter_termination_fraction = max_scatter_termination_fraction(run.stats);
                snapshot.mean_scattered_energy_mec2 = mean_scattered_energy_mec2(run.stats);
                snapshot.mean_energy_amplification = mean_energy_amplification(run.cfg, run.stats);
                snapshot.energy_hist_overflow_fraction = energy_hist_overflow_fraction(run.stats);
                snapshot.bookkeeping_consistent = slab_bookkeeping_consistent(run.stats);

                // The max_scatters=1 branch intentionally reproduces legacy
                // single-scatter behavior and therefore does not use the "hit max
                // scatters" termination path. Start the truncation check from
                // max_scatters>=2, where iterative transport is active.
                if (max_scatters > 1 && previous_termination_fraction >= 0.0) {
                    termination_fraction_nonincreasing =
                        termination_fraction_nonincreasing &&
                        snapshot.max_scatter_termination_fraction <= previous_termination_fraction + 1.0e-12;
                }
                if (max_scatters > 1) {
                    previous_termination_fraction = snapshot.max_scatter_termination_fraction;
                }
                if (max_scatters == 2) {
                    termination_fraction_at_2 = snapshot.max_scatter_termination_fraction;
                }
                if (max_scatters == 16) {
                    mean_scatter_count_16 = snapshot.mean_scatter_count;
                    escaped_up_fraction_16 = snapshot.escaped_up_total_fraction;
                    escaped_down_fraction_16 = snapshot.escaped_down_total_fraction;
                    mean_scattered_energy_16 = snapshot.mean_scattered_energy_mec2;
                } else if (max_scatters == 32) {
                    mean_scatter_count_32 = snapshot.mean_scatter_count;
                    escaped_up_fraction_32 = snapshot.escaped_up_total_fraction;
                    escaped_down_fraction_32 = snapshot.escaped_down_total_fraction;
                    mean_scattered_energy_32 = snapshot.mean_scattered_energy_mec2;
                }

                rows.push_back({
                    format_number(tau),
                    injection_model,
                    std::to_string(max_scatters),
                    snapshot.source_run_tag,
                    std::to_string(snapshot.events_completed),
                    std::to_string(snapshot.scattered_events),
                    std::to_string(snapshot.total_scatter_interactions),
                    format_number(snapshot.scatter_fraction),
                    format_number(snapshot.escaped_up_total_fraction),
                    format_number(snapshot.escaped_down_total_fraction),
                    format_number(snapshot.mean_scatter_count),
                    std::to_string(snapshot.terminated_at_max_scatters),
                    format_number(snapshot.max_scatter_termination_fraction),
                    format_number(snapshot.mean_scattered_energy_mec2),
                    format_number(snapshot.energy_hist_overflow_fraction),
                    snapshot.bookkeeping_consistent ? "true" : "false",
                });
                snapshots.push_back(snapshot);
            }

            for (std::size_t i = 0; i + 1 < snapshots.size(); ++i) {
                const ConvergenceSnapshot& lhs = snapshots[i];
                const ConvergenceSnapshot& rhs = snapshots[i + 1];
                const double delta_mean_scatter_count =
                    std::abs(rhs.mean_scatter_count - lhs.mean_scatter_count);
                const double delta_escaped_up =
                    std::abs(rhs.escaped_up_total_fraction - lhs.escaped_up_total_fraction);
                const double delta_escaped_down =
                    std::abs(rhs.escaped_down_total_fraction - lhs.escaped_down_total_fraction);
                const double delta_mean_scattered_energy =
                    std::abs(rhs.mean_scattered_energy_mec2 - lhs.mean_scattered_energy_mec2);
                const double difference_sum = convergence_difference_sum(lhs, rhs);

                if (lhs.max_scatters == 2 && rhs.max_scatters == 4) {
                    early_difference_sum = difference_sum;
                } else if (lhs.max_scatters == 16 && rhs.max_scatters == 32) {
                    late_difference_sum = difference_sum;
                }

                difference_rows.push_back({
                    format_number(tau),
                    injection_model,
                    std::to_string(lhs.max_scatters) + "->" + std::to_string(rhs.max_scatters),
                    format_number(delta_mean_scatter_count),
                    format_number(delta_escaped_up),
                    format_number(delta_escaped_down),
                    format_number(delta_mean_scattered_energy),
                    format_number(difference_sum),
                });
            }

            const ConvergenceSnapshot& reference_snapshot = snapshots.back();
            for (const ConvergenceSnapshot& snapshot : snapshots) {
                if (snapshot.max_scatters == reference_snapshot.max_scatters) {
                    continue;
                }

                const double delta_mean_scatter_count_vs_ref =
                    std::abs(reference_snapshot.mean_scatter_count - snapshot.mean_scatter_count);
                const double delta_escaped_up_vs_ref =
                    std::abs(reference_snapshot.escaped_up_total_fraction - snapshot.escaped_up_total_fraction);
                const double delta_escaped_down_vs_ref =
                    std::abs(reference_snapshot.escaped_down_total_fraction - snapshot.escaped_down_total_fraction);
                const double delta_mean_scattered_energy_vs_ref =
                    std::abs(reference_snapshot.mean_scattered_energy_mec2 - snapshot.mean_scattered_energy_mec2);
                const double reference_difference_sum =
                    convergence_difference_sum_vs_reference(snapshot, reference_snapshot);

                if (previous_reference_difference_sum >= 0.0) {
                    reference_differences_shrink =
                        reference_differences_shrink &&
                        reference_difference_sum <= previous_reference_difference_sum + 1.0e-12;
                } else {
                    reference_differences_shrink = true;
                }
                previous_reference_difference_sum = reference_difference_sum;

                if (snapshot.max_scatters == 1) {
                    reference_difference_sum_at_1 = reference_difference_sum;
                } else if (snapshot.max_scatters == 16) {
                    reference_difference_sum_at_16 = reference_difference_sum;
                }

                reference_difference_rows.push_back({
                    format_number(tau),
                    injection_model,
                    std::to_string(snapshot.max_scatters) + "->" +
                        std::to_string(reference_snapshot.max_scatters),
                    format_number(delta_mean_scatter_count_vs_ref),
                    format_number(delta_escaped_up_vs_ref),
                    format_number(delta_escaped_down_vs_ref),
                    format_number(delta_mean_scattered_energy_vs_ref),
                    format_number(reference_difference_sum),
                });
            }

            stabilization_emerging =
                mean_scatter_count_16 >= 0.0 &&
                mean_scatter_count_32 >= 0.0 &&
                std::abs(mean_scatter_count_32 - mean_scatter_count_16) <=
                    kConvergenceMeanScatterCountTolerance &&
                std::abs(escaped_up_fraction_32 - escaped_up_fraction_16) <=
                    kConvergenceEscapeFractionTolerance &&
                std::abs(escaped_down_fraction_32 - escaped_down_fraction_16) <=
                    kConvergenceEscapeFractionTolerance;
            observable_differences_shrink =
                early_difference_sum >= 0.0 &&
                late_difference_sum >= 0.0 &&
                late_difference_sum <= early_difference_sum + 1.0e-12;
            max_scatters_16_close_to_reference =
                mean_scatter_count_16 >= 0.0 &&
                mean_scatter_count_32 >= 0.0 &&
                mean_scattered_energy_16 >= 0.0 &&
                mean_scattered_energy_32 >= 0.0 &&
                std::abs(mean_scatter_count_32 - mean_scatter_count_16) <=
                    kConvergenceMeanScatterCountTolerance &&
                std::abs(escaped_up_fraction_32 - escaped_up_fraction_16) <=
                    kConvergenceEscapeFractionTolerance &&
                std::abs(escaped_down_fraction_32 - escaped_down_fraction_16) <=
                    kConvergenceEscapeFractionTolerance &&
                std::abs(mean_scattered_energy_32 - mean_scattered_energy_16) <=
                    kConvergenceMeanScatteredEnergyTolerance;

            checks_rows.push_back({
                format_number(tau),
                injection_model,
                termination_fraction_nonincreasing ? "true" : "false",
                stabilization_emerging ? "true" : "false",
                observable_differences_shrink ? "true" : "false",
                format_number(termination_fraction_at_2),
                format_number(early_difference_sum >= 0.0 ? early_difference_sum : 0.0),
                format_number(late_difference_sum >= 0.0 ? late_difference_sum : 0.0),
                format_number(mean_scatter_count_16),
                format_number(mean_scatter_count_32),
                format_number(escaped_up_fraction_16),
                format_number(escaped_up_fraction_32),
                format_number(escaped_down_fraction_16),
                format_number(escaped_down_fraction_32),
            });
            reference_checks_rows.push_back({
                format_number(tau),
                injection_model,
                reference_differences_shrink ? "true" : "false",
                max_scatters_16_close_to_reference ? "true" : "false",
                format_number(reference_difference_sum_at_1 >= 0.0 ? reference_difference_sum_at_1 : 0.0),
                format_number(reference_difference_sum_at_16 >= 0.0 ? reference_difference_sum_at_16 : 0.0),
                format_number(mean_scatter_count_16),
                format_number(mean_scatter_count_32),
                format_number(mean_scattered_energy_16),
                format_number(mean_scattered_energy_32),
            });

            injection_summaries.push_back({
                injection_model,
                termination_fraction_nonincreasing,
                stabilization_emerging,
                observable_differences_shrink,
                reference_differences_shrink,
                max_scatters_16_close_to_reference,
                termination_fraction_at_2,
                late_difference_sum >= 0.0 ? late_difference_sum : 0.0,
                reference_difference_sum_at_1 >= 0.0 ? reference_difference_sum_at_1 : 0.0,
                reference_difference_sum_at_16 >= 0.0 ? reference_difference_sum_at_16 : 0.0,
            });

            if (!termination_fraction_nonincreasing) {
                std::cerr << "warning: multi-scatter convergence sweep has non-monotonic max-scatter termination fraction at tau="
                          << tau << " injection_model=" << injection_model << "\n";
            }
            if (!stabilization_emerging) {
                std::cerr << "warning: multi-scatter convergence sweep has not stabilized by max_scatters=32 at tau="
                          << tau << " injection_model=" << injection_model << "\n";
            }
            if (!observable_differences_shrink) {
                std::cerr << "warning: multi-scatter convergence differences are not shrinking at tau="
                          << tau << " injection_model=" << injection_model << "\n";
            }
            if (!reference_differences_shrink) {
                std::cerr << "warning: reference-case truncation differences are not shrinking at tau="
                          << tau << " injection_model=" << injection_model << "\n";
            }
            if (!max_scatters_16_close_to_reference) {
                std::cerr << "warning: max_scatters=16 is not yet close to the reference case at tau="
                          << tau << " injection_model=" << injection_model << "\n";
            }
        }

        const InjectionConvergenceSummary* beam_summary = nullptr;
        const InjectionConvergenceSummary* lambert_summary = nullptr;
        for (const InjectionConvergenceSummary& summary : injection_summaries) {
            if (summary.injection_model == "beam") {
                beam_summary = &summary;
            } else if (summary.injection_model == "lambert") {
                lambert_summary = &summary;
            }
        }
        if (beam_summary != nullptr && lambert_summary != nullptr) {
            const bool lambert_termination_at_2_ge_beam =
                lambert_summary->termination_fraction_at_2 >= beam_summary->termination_fraction_at_2;
            const bool lambert_late_difference_ge_beam =
                lambert_summary->late_difference_sum >= beam_summary->late_difference_sum;
            const bool lambert_reference_difference_at_16_ge_beam =
                lambert_summary->reference_difference_sum_at_16 >= beam_summary->reference_difference_sum_at_16;
            comparison_rows.push_back({
                format_number(tau),
                beam_summary->termination_fraction_nonincreasing ? "true" : "false",
                lambert_summary->termination_fraction_nonincreasing ? "true" : "false",
                beam_summary->stabilization_emerging ? "true" : "false",
                lambert_summary->stabilization_emerging ? "true" : "false",
                beam_summary->observable_differences_shrink ? "true" : "false",
                lambert_summary->observable_differences_shrink ? "true" : "false",
                beam_summary->reference_differences_shrink ? "true" : "false",
                lambert_summary->reference_differences_shrink ? "true" : "false",
                beam_summary->max_scatters_16_close_to_reference ? "true" : "false",
                lambert_summary->max_scatters_16_close_to_reference ? "true" : "false",
                format_number(beam_summary->termination_fraction_at_2),
                format_number(lambert_summary->termination_fraction_at_2),
                lambert_termination_at_2_ge_beam ? "true" : "false",
                format_number(beam_summary->late_difference_sum),
                format_number(lambert_summary->late_difference_sum),
                lambert_late_difference_ge_beam ? "true" : "false",
                format_number(beam_summary->reference_difference_sum_at_16),
                format_number(lambert_summary->reference_difference_sum_at_16),
                lambert_reference_difference_at_16_ge_beam ? "true" : "false",
            });
        }
    }

    const std::string table_path = base_cfg.output_dir + "/" +
                                   validation_tag(base_cfg, "validate_slab_multi_convergence_table") + ".csv";
    write_table_csv(table_path,
                    {
                        "slab_tau",
                        "slab_injection_model",
                        "max_scatters",
                        "source_run_tag",
                        "events_completed",
                        "scattered_events",
                        "total_scatter_interactions",
                        "scatter_fraction",
                        "escaped_up_total_fraction",
                        "escaped_down_total_fraction",
                        "mean_scatter_count",
                        "terminated_at_max_scatters",
                        "max_scatter_termination_fraction",
                        "mean_scattered_energy_mec2",
                        "energy_hist_overflow_fraction",
                        "bookkeeping_consistent",
                    },
                    rows);
    std::cout << "Wrote: " << table_path << "\n";

    const std::string differences_path = base_cfg.output_dir + "/" +
                                         validation_tag(base_cfg, "validate_slab_multi_convergence_differences") + ".csv";
    write_table_csv(differences_path,
                    {
                        "slab_tau",
                        "slab_injection_model",
                        "max_scatters_pair",
                        "delta_mean_scatter_count",
                        "delta_escaped_up_total_fraction",
                        "delta_escaped_down_total_fraction",
                        "delta_mean_scattered_energy_mec2",
                        "difference_sum",
                    },
                    difference_rows);
    std::cout << "Wrote: " << differences_path << "\n";

    const std::string reference_differences_path = base_cfg.output_dir + "/" +
                                                   validation_tag(base_cfg, "validate_slab_multi_convergence_reference_differences") + ".csv";
    write_table_csv(reference_differences_path,
                    {
                        "slab_tau",
                        "slab_injection_model",
                        "max_scatters_vs_reference",
                        "delta_mean_scatter_count_vs_ref",
                        "delta_escaped_up_total_fraction_vs_ref",
                        "delta_escaped_down_total_fraction_vs_ref",
                        "delta_mean_scattered_energy_mec2_vs_ref",
                        "difference_sum_vs_ref",
                    },
                    reference_difference_rows);
    std::cout << "Wrote: " << reference_differences_path << "\n";

    const std::string checks_path = base_cfg.output_dir + "/" +
                                    validation_tag(base_cfg, "validate_slab_multi_convergence_checks") + ".csv";
    write_table_csv(checks_path,
                    {
                        "slab_tau",
                        "slab_injection_model",
                        "termination_fraction_nonincreasing",
                        "stabilization_emerging_by_32",
                        "observable_differences_shrink",
                        "termination_fraction_at_2",
                        "difference_sum_2_to_4",
                        "difference_sum_16_to_32",
                        "mean_scatter_count_16",
                        "mean_scatter_count_32",
                        "escaped_up_total_fraction_16",
                        "escaped_up_total_fraction_32",
                        "escaped_down_total_fraction_16",
                        "escaped_down_total_fraction_32",
                    },
                    checks_rows);
    std::cout << "Wrote: " << checks_path << "\n";

    const std::string reference_checks_path = base_cfg.output_dir + "/" +
                                              validation_tag(base_cfg, "validate_slab_multi_convergence_reference_checks") + ".csv";
    write_table_csv(reference_checks_path,
                    {
                        "slab_tau",
                        "slab_injection_model",
                        "reference_differences_shrink_toward_32",
                        "max_scatters_16_close_to_reference",
                        "difference_sum_1_to_32",
                        "difference_sum_16_to_32",
                        "mean_scatter_count_16",
                        "mean_scatter_count_32",
                        "mean_scattered_energy_mec2_16",
                        "mean_scattered_energy_mec2_32",
                    },
                    reference_checks_rows);
    std::cout << "Wrote: " << reference_checks_path << "\n";

    const std::string comparison_path = base_cfg.output_dir + "/" +
                                        validation_tag(base_cfg, "validate_slab_multi_convergence_comparison_checks") + ".csv";
    write_table_csv(comparison_path,
                    {
                        "slab_tau",
                        "beam_termination_fraction_nonincreasing",
                        "lambert_termination_fraction_nonincreasing",
                        "beam_stabilization_emerging_by_32",
                        "lambert_stabilization_emerging_by_32",
                        "beam_observable_differences_shrink",
                        "lambert_observable_differences_shrink",
                        "beam_reference_differences_shrink",
                        "lambert_reference_differences_shrink",
                        "beam_max_scatters_16_close_to_reference",
                        "lambert_max_scatters_16_close_to_reference",
                        "beam_termination_fraction_at_2",
                        "lambert_termination_fraction_at_2",
                        "lambert_termination_fraction_at_2_ge_beam",
                        "beam_difference_sum_16_to_32",
                        "lambert_difference_sum_16_to_32",
                        "lambert_difference_sum_16_to_32_ge_beam",
                        "beam_difference_sum_vs_ref_16_to_32",
                        "lambert_difference_sum_vs_ref_16_to_32",
                        "lambert_difference_sum_vs_ref_16_to_32_ge_beam",
                    },
                    comparison_rows);
    std::cout << "Wrote: " << comparison_path << "\n";
}

void run_production_slab_thermal_sweep(const ic::Config& base_cfg) {
    const std::string table_path = base_cfg.output_dir + "/" +
                                   validation_tag(base_cfg, "production_slab_thermal_sweep_table") + ".csv";
    write_production_slab_thermal_table(
        table_path,
        run_production_slab_thermal_grid_rows(base_cfg,
                                              production_slab_thermal_tau_grid_stage1(),
                                              production_slab_thermal_injection_models(),
                                              production_slab_thermal_theta_grid_stage1(),
                                              "production-slab-thermal-sweep"));
}

void run_validation_slab_high_tau_convergence(const ic::Config& base_cfg) {
    struct HighTauCaseSummary {
        double tau = 0.0;
        double theta_e = 0.0;
        bool termination_fraction_nonincreasing = true;
        bool mean_scatter_count_stabilized_by_1024 = false;
        bool escape_fractions_stabilized_by_1024 = false;
        bool mean_scattered_energy_stabilized_by_1024 = false;
        bool bookkeeping_consistent_at_1024 = false;
        bool controlled_at_max_tested = false;
        bool controlled_at_256 = false;
        double termination_fraction_at_256 = 0.0;
        double termination_fraction_at_1024 = 0.0;
        double difference_sum_256_to_1024 = 0.0;
        double difference_sum_512_to_1024 = 0.0;
    };

    const std::vector<double> tau_values{2.0, 5.0, 10.0};
    const std::vector<double> theta_values{0.1, 0.5};
    const std::vector<std::size_t> max_scatters_values{16, 32, 64, 128, 256, 512, 1024};
    std::vector<std::vector<std::string>> rows;
    std::vector<std::vector<std::string>> checks_rows;
    std::vector<std::vector<std::string>> control_rows;
    rows.reserve(tau_values.size() * theta_values.size() * max_scatters_values.size());
    checks_rows.reserve(tau_values.size() * theta_values.size());
    control_rows.reserve(tau_values.size() * theta_values.size());

    for (double tau : tau_values) {
        for (double theta_e : theta_values) {
            std::vector<ConvergenceSnapshot> snapshots;
            snapshots.reserve(max_scatters_values.size());
            double previous_termination_fraction = -1.0;
            bool termination_fraction_nonincreasing = true;

            for (std::size_t max_scatters : max_scatters_values) {
                ic::Config cfg = base_cfg;
                cfg.mode = "validate-slab-high-tau-convergence";
                cfg.run_label.clear();
                cfg.geometry_model = "slab";
                cfg.slab_height = 1.0;
                cfg.slab_optical_depth = tau;
                cfg.slab_injection_model = "beam";
                cfg.electron_model = "thermal";
                cfg.electron_kTe = theta_e;
                cfg.max_scatters = max_scatters;
                apply_multi_scatter_validation_energy_range(cfg);

                const CompletedRun run = run_and_record(cfg);
                ConvergenceSnapshot snapshot;
                snapshot.tau = tau;
                snapshot.theta_e = theta_e;
                snapshot.injection_model = "beam";
                snapshot.max_scatters = max_scatters;
                snapshot.source_run_tag = run.tag;
                snapshot.events_completed = run.stats.events_completed;
                snapshot.scattered_events = run.stats.scattered_events;
                snapshot.total_scatter_interactions = run.stats.total_scatter_interactions;
                snapshot.scatter_fraction = scatter_fraction(run.stats);
                snapshot.escaped_up_total_fraction = escaped_up_total_fraction(run.stats);
                snapshot.escaped_down_total_fraction = escaped_down_total_fraction(run.stats);
                snapshot.mean_scatter_count = mean_scatter_count(run.stats);
                snapshot.terminated_at_max_scatters = run.stats.terminated_at_max_scatters;
                snapshot.max_scatter_termination_fraction = max_scatter_termination_fraction(run.stats);
                snapshot.mean_scattered_energy_mec2 = mean_scattered_energy_mec2(run.stats);
                snapshot.mean_energy_amplification = mean_energy_amplification(run.cfg, run.stats);
                snapshot.energy_hist_overflow_fraction = energy_hist_overflow_fraction(run.stats);
                snapshot.bookkeeping_consistent = slab_bookkeeping_consistent(run.stats);

                if (previous_termination_fraction >= 0.0) {
                    termination_fraction_nonincreasing =
                        termination_fraction_nonincreasing &&
                        snapshot.max_scatter_termination_fraction <= previous_termination_fraction + 1.0e-12;
                }
                previous_termination_fraction = snapshot.max_scatter_termination_fraction;

                rows.push_back({
                    format_number(tau),
                    format_number(theta_e),
                    std::to_string(max_scatters),
                    snapshot.source_run_tag,
                    std::to_string(snapshot.events_completed),
                    std::to_string(snapshot.scattered_events),
                    std::to_string(snapshot.total_scatter_interactions),
                    format_number(snapshot.mean_scatter_count),
                    std::to_string(snapshot.terminated_at_max_scatters),
                    format_number(snapshot.max_scatter_termination_fraction),
                    format_number(snapshot.escaped_up_total_fraction),
                    format_number(snapshot.escaped_down_total_fraction),
                    format_number(snapshot.mean_scattered_energy_mec2),
                    format_number(snapshot.mean_energy_amplification),
                    format_number(snapshot.energy_hist_overflow_fraction),
                    snapshot.bookkeeping_consistent ? "true" : "false",
                });
                snapshots.push_back(snapshot);
            }

            const ConvergenceSnapshot& at_256 = snapshots[4];
            const ConvergenceSnapshot& at_512 = snapshots[5];
            const ConvergenceSnapshot& at_1024 = snapshots[6];
            const bool mean_scatter_count_stabilized_by_1024 =
                approximately_stable(at_512.mean_scatter_count,
                                     at_1024.mean_scatter_count,
                                     kHighTauConvergenceMeanScatterCountAbsTolerance,
                                     kHighTauConvergenceMeanScatterCountRelTolerance);
            const bool escape_fractions_stabilized_by_1024 =
                std::abs(at_512.escaped_up_total_fraction - at_1024.escaped_up_total_fraction) <=
                    kHighTauConvergenceEscapeFractionTolerance &&
                std::abs(at_512.escaped_down_total_fraction - at_1024.escaped_down_total_fraction) <=
                    kHighTauConvergenceEscapeFractionTolerance;
            const bool mean_scattered_energy_stabilized_by_1024 =
                approximately_stable(at_512.mean_scattered_energy_mec2,
                                     at_1024.mean_scattered_energy_mec2,
                                     kHighTauConvergenceMeanScatteredEnergyAbsTolerance,
                                     kHighTauConvergenceMeanScatteredEnergyRelTolerance);
            const bool bookkeeping_consistent_at_1024 = at_1024.bookkeeping_consistent;
            const bool controlled_at_max_tested =
                termination_fraction_nonincreasing &&
                mean_scatter_count_stabilized_by_1024 &&
                escape_fractions_stabilized_by_1024 &&
                mean_scattered_energy_stabilized_by_1024 &&
                bookkeeping_consistent_at_1024 &&
                at_1024.max_scatter_termination_fraction <=
                    kHighTauControlledTerminationFractionThreshold;

            const bool controlled_at_256 =
                approximately_stable(at_256.mean_scatter_count,
                                     at_1024.mean_scatter_count,
                                     kHighTauConvergenceMeanScatterCountAbsTolerance,
                                     kHighTauConvergenceMeanScatterCountRelTolerance) &&
                std::abs(at_256.escaped_up_total_fraction - at_1024.escaped_up_total_fraction) <=
                    kHighTauConvergenceEscapeFractionTolerance &&
                std::abs(at_256.escaped_down_total_fraction - at_1024.escaped_down_total_fraction) <=
                    kHighTauConvergenceEscapeFractionTolerance &&
                approximately_stable(at_256.mean_scattered_energy_mec2,
                                     at_1024.mean_scattered_energy_mec2,
                                     kHighTauConvergenceMeanScatteredEnergyAbsTolerance,
                                     kHighTauConvergenceMeanScatteredEnergyRelTolerance) &&
                at_256.max_scatter_termination_fraction <=
                    kHighTauControlledTerminationFractionThreshold;

            const double difference_sum_256_to_1024 =
                convergence_difference_sum_vs_reference(at_256, at_1024);
            const double difference_sum_512_to_1024 =
                convergence_difference_sum_vs_reference(at_512, at_1024);

            checks_rows.push_back({
                format_number(tau),
                format_number(theta_e),
                termination_fraction_nonincreasing ? "true" : "false",
                mean_scatter_count_stabilized_by_1024 ? "true" : "false",
                escape_fractions_stabilized_by_1024 ? "true" : "false",
                mean_scattered_energy_stabilized_by_1024 ? "true" : "false",
                bookkeeping_consistent_at_1024 ? "true" : "false",
                format_number(at_512.mean_scatter_count),
                format_number(at_1024.mean_scatter_count),
                format_number(at_512.escaped_up_total_fraction),
                format_number(at_1024.escaped_up_total_fraction),
                format_number(at_512.escaped_down_total_fraction),
                format_number(at_1024.escaped_down_total_fraction),
                format_number(difference_sum_512_to_1024),
            });

            control_rows.push_back({
                format_number(tau),
                format_number(theta_e),
                std::to_string(at_1024.max_scatters),
                format_number(at_256.max_scatter_termination_fraction),
                format_number(at_1024.max_scatter_termination_fraction),
                format_number(difference_sum_256_to_1024),
                format_number(difference_sum_512_to_1024),
                controlled_at_256 ? "true" : "false",
                controlled_at_max_tested ? "true" : "false",
            });

            if (!controlled_at_max_tested) {
                std::cerr << "warning: high-tau convergence case not yet controlled at max_scatters="
                          << at_1024.max_scatters << " for tau=" << tau
                          << " theta_e=" << theta_e << "\n";
            }
        }
    }

    const std::string table_path = base_cfg.output_dir + "/" +
                                   validation_tag(base_cfg, "validate_slab_high_tau_convergence_table") + ".csv";
    write_table_csv(table_path,
                    {
                        "slab_tau",
                        "electron_kTe_mec2",
                        "max_scatters",
                        "source_run_tag",
                        "events_completed",
                        "scattered_events",
                        "total_scatter_interactions",
                        "mean_scatter_count",
                        "terminated_at_max_scatters",
                        "max_scatter_termination_fraction",
                        "escaped_up_total_fraction",
                        "escaped_down_total_fraction",
                        "mean_scattered_energy_mec2",
                        "mean_energy_amplification",
                        "energy_hist_overflow_fraction",
                        "bookkeeping_consistent",
                    },
                    rows);
    std::cout << "Wrote: " << table_path << "\n";

    const std::string checks_path = base_cfg.output_dir + "/" +
                                    validation_tag(base_cfg, "validate_slab_high_tau_convergence_checks") + ".csv";
    write_table_csv(checks_path,
                    {
                        "slab_tau",
                        "electron_kTe_mec2",
                        "termination_fraction_nonincreasing",
                        "mean_scatter_count_stabilized_by_1024",
                        "escape_fractions_stabilized_by_1024",
                        "mean_scattered_energy_stabilized_by_1024",
                        "bookkeeping_consistent_at_1024",
                        "mean_scatter_count_512",
                        "mean_scatter_count_1024",
                        "escaped_up_total_fraction_512",
                        "escaped_up_total_fraction_1024",
                        "escaped_down_total_fraction_512",
                        "escaped_down_total_fraction_1024",
                        "difference_sum_512_to_1024",
                    },
                    checks_rows);
    std::cout << "Wrote: " << checks_path << "\n";

    const std::string control_path = base_cfg.output_dir + "/" +
                                     validation_tag(base_cfg, "validate_slab_high_tau_convergence_control_summary") + ".csv";
    write_table_csv(control_path,
                    {
                        "slab_tau",
                        "electron_kTe_mec2",
                        "max_tested_max_scatters",
                        "termination_fraction_at_256",
                        "termination_fraction_at_1024",
                        "difference_sum_256_to_1024",
                        "difference_sum_512_to_1024",
                        "appears_controlled_by_256",
                        "controlled_at_max_tested",
                    },
                    control_rows);
    std::cout << "Wrote: " << control_path << "\n";
}

void run_validation_slab_high_tau_seed_energy(const ic::Config& base_cfg) {
    const std::vector<double>& tau_values = diagnostic_high_tau_seed_tau_grid();
    const std::vector<double>& theta_values = diagnostic_high_tau_seed_theta_grid();
    const std::vector<double>& photon_energies = diagnostic_high_tau_seed_photon_energies();
    std::vector<std::vector<std::string>> rows;
    std::vector<std::vector<std::string>> summary_rows;
    rows.reserve(tau_values.size() * theta_values.size() * photon_energies.size());
    summary_rows.reserve(tau_values.size() * theta_values.size());

    for (double tau : tau_values) {
        for (double theta_e : theta_values) {
            double min_mean_scattered_energy = std::numeric_limits<double>::infinity();
            double max_mean_scattered_energy = 0.0;
            double sum_mean_scattered_energy = 0.0;
            double amplification_at_min_seed = 0.0;
            double amplification_at_max_seed = 0.0;

            for (std::size_t i = 0; i < photon_energies.size(); ++i) {
                const double photon_energy = photon_energies[i];
                ic::Config cfg = base_cfg;
                cfg.mode = "validate-slab-high-tau-seed-energy";
                cfg.run_label.clear();
                cfg.geometry_model = "slab";
                cfg.slab_height = 1.0;
                cfg.slab_optical_depth = tau;
                cfg.slab_injection_model = "beam";
                cfg.electron_model = "thermal";
                cfg.electron_kTe = theta_e;
                cfg.incident_photon_energy = photon_energy;
                if (cfg.max_scatters == ic::Config{}.max_scatters) {
                    cfg.max_scatters = 256;
                }
                apply_multi_scatter_validation_energy_range(cfg);

                const CompletedRun run = run_and_record(cfg);
                const double mean_scattered_energy = mean_scattered_energy_mec2(run.stats);
                const double amplification = mean_energy_amplification(run.cfg, run.stats);
                const double overflow_fraction = energy_hist_overflow_fraction(run.stats);
                const bool bookkeeping_consistent = slab_bookkeeping_consistent(run.stats);

                rows.push_back({
                    format_number(tau),
                    format_number(theta_e),
                    format_number(photon_energy),
                    std::to_string(cfg.max_scatters),
                    run.tag,
                    std::to_string(run.stats.events_completed),
                    format_number(mean_scattered_energy),
                    format_number(amplification),
                    format_number(mean_scatter_count(run.stats)),
                    format_number(scatter_fraction(run.stats)),
                    format_number(escaped_up_total_fraction(run.stats)),
                    format_number(escaped_down_total_fraction(run.stats)),
                    bookkeeping_consistent ? "true" : "false",
                    format_number(overflow_fraction),
                });

                min_mean_scattered_energy = std::min(min_mean_scattered_energy, mean_scattered_energy);
                max_mean_scattered_energy = std::max(max_mean_scattered_energy, mean_scattered_energy);
                sum_mean_scattered_energy += mean_scattered_energy;
                if (i == 0) {
                    amplification_at_min_seed = amplification;
                }
                if (i + 1 == photon_energies.size()) {
                    amplification_at_max_seed = amplification;
                }

                if (!bookkeeping_consistent) {
                    std::cerr << "warning: high-tau seed-energy case bookkeeping failed for tau="
                              << tau << " theta_e=" << theta_e
                              << " photon_energy=" << photon_energy << "\n";
                }
                if (overflow_fraction > kValidationOverflowAcceptableThreshold) {
                    std::cerr << "warning: high-tau seed-energy overflow_fraction="
                              << overflow_fraction << " exceeds the acceptability threshold of "
                              << kValidationOverflowAcceptableThreshold
                              << " for tau=" << tau
                              << " theta_e=" << theta_e
                              << " photon_energy=" << photon_energy << "\n";
                }
            }

            const double mean_over_seeds =
                sum_mean_scattered_energy / static_cast<double>(photon_energies.size());
            const double relative_span =
                mean_over_seeds > 0.0
                    ? (max_mean_scattered_energy - min_mean_scattered_energy) / mean_over_seeds
                    : 0.0;
            const bool weakly_seed_dependent =
                relative_span <= kHighTauSeedEnergyWeakDependenceThreshold;

            summary_rows.push_back({
                format_number(tau),
                format_number(theta_e),
                format_number(photon_energies.front()),
                format_number(photon_energies.back()),
                format_number(min_mean_scattered_energy),
                format_number(max_mean_scattered_energy),
                format_number(relative_span),
                format_number(amplification_at_min_seed),
                format_number(amplification_at_max_seed),
                weakly_seed_dependent ? "true" : "false",
            });

            if (tau == tau_values.back() && !weakly_seed_dependent) {
                std::cerr << "warning: highest-tau seed-energy case still shows strong seed dependence for theta_e="
                          << theta_e << " with relative span " << relative_span << "\n";
            }
        }
    }

    const std::string table_path = base_cfg.output_dir + "/" +
                                   validation_tag(base_cfg, "validate_slab_high_tau_seed_energy_table") + ".csv";
    write_table_csv(table_path,
                    {
                        "slab_tau",
                        "electron_kTe_mec2",
                        "incident_photon_energy_mec2",
                        "max_scatters",
                        "source_run_tag",
                        "events_completed",
                        "mean_scattered_energy_mec2",
                        "mean_energy_amplification",
                        "mean_scatter_count",
                        "scatter_fraction",
                        "escaped_up_total_fraction",
                        "escaped_down_total_fraction",
                        "bookkeeping_consistent",
                        "energy_hist_overflow_fraction",
                    },
                    rows);
    std::cout << "Wrote: " << table_path << "\n";

    const std::string checks_path = base_cfg.output_dir + "/" +
                                    validation_tag(base_cfg, "validate_slab_high_tau_seed_energy_checks") + ".csv";
    write_table_csv(checks_path,
                    {
                        "slab_tau",
                        "electron_kTe_mec2",
                        "incident_photon_energy_min_mec2",
                        "incident_photon_energy_max_mec2",
                        "min_mean_scattered_energy_mec2",
                        "max_mean_scattered_energy_mec2",
                        "relative_span_mean_scattered_energy",
                        "mean_energy_amplification_at_min_seed",
                        "mean_energy_amplification_at_max_seed",
                        "mean_scattered_energy_weakly_seed_dependent",
                    },
                    summary_rows);
    std::cout << "Wrote: " << checks_path << "\n";
}

void run_generate_thermal_kn_transport_table(const ic::Config& base_cfg) {
    std::filesystem::create_directories(base_cfg.output_dir);
    const ic::ThermalKnTableGenerationSettings settings =
        thermal_kn_generation_settings(base_cfg);
    const ic::ThermalKnTransportTable table =
        ic::generate_thermal_kn_transport_table(settings);
    ic::write_thermal_kn_transport_table_hdf5(base_cfg.thermal_kn_table_path, table);

    const std::string report_path =
        base_cfg.output_dir + "/" +
        validation_tag(base_cfg, "generate_thermal_kn_transport_table_report") + ".csv";
    write_validation_report(report_path,
                            {
                                {"thermal_kn_table_path", base_cfg.thermal_kn_table_path},
                                {"energy_points", std::to_string(settings.energy_points)},
                                {"theta_points", std::to_string(settings.theta_points)},
                                {"energy_min_mec2", format_number(settings.energy_min)},
                                {"energy_max_mec2", format_number(settings.energy_max)},
                                {"theta_min_mec2", format_number(settings.theta_min)},
                                {"theta_max_mec2", format_number(settings.theta_max)},
                                {"seed_photon_model_default", base_cfg.seed_photon_model},
                                {"seed_temperature_eV_default", format_number(base_cfg.seed_temperature_eV)},
                                {"seed_temperature_mec2_default", format_number(ic::blackbody_temperature_ev_to_mec2(base_cfg.seed_temperature_eV))},
                                {"mean_blackbody_seed_energy_mec2_default", format_number(ic::mean_blackbody_photon_energy_mec2(base_cfg.seed_temperature_eV))},
                                {"z_points", std::to_string(settings.integration.z_points)},
                                {"mu_points", std::to_string(settings.integration.mu_points)},
                                {"z_max", format_number(settings.integration.z_max)},
                            });
    std::cout << "Wrote: " << base_cfg.thermal_kn_table_path << "\n";
}

void run_validate_thermal_kn_transport_table(const ic::Config& base_cfg) {
    std::filesystem::create_directories(base_cfg.output_dir);
    const ic::ThermalKnTransportTable table =
        ic::load_thermal_kn_transport_table_hdf5(base_cfg.thermal_kn_table_path);

    double low_energy_max_abs_deviation = 0.0;
    for (double theta_e : table.theta_grid) {
        low_energy_max_abs_deviation = std::max(
            low_energy_max_abs_deviation,
            std::abs(ic::interpolate_thermal_kn_transport_ratio(table, table.energy_grid.front(), theta_e) - 1.0));
    }

    double low_theta_mean_abs_difference = 0.0;
    double low_theta_max_abs_difference = 0.0;
    for (double energy : table.energy_grid) {
        const double thermal_ratio =
            ic::interpolate_thermal_kn_transport_ratio(table, energy, table.theta_grid.front());
        const double stationary_ratio = ic::sigma_kn_total_over_sigma_t(energy);
        const double abs_difference = std::abs(thermal_ratio - stationary_ratio);
        low_theta_mean_abs_difference += abs_difference;
        low_theta_max_abs_difference = std::max(low_theta_max_abs_difference, abs_difference);
    }
    low_theta_mean_abs_difference /= static_cast<double>(table.energy_grid.size());

    std::size_t high_energy_monotonic_failures = 0;
    const std::size_t high_energy_start = (3 * table.energy_grid.size()) / 4;
    for (std::size_t itheta = 0; itheta < table.theta_size(); ++itheta) {
        double previous_ratio =
            std::exp(table.log_ratio_table[itheta * table.energy_size() + high_energy_start]);
        for (std::size_t ienergy = high_energy_start + 1; ienergy < table.energy_size(); ++ienergy) {
            const double current_ratio =
                std::exp(table.log_ratio_table[itheta * table.energy_size() + ienergy]);
            if (current_ratio > previous_ratio + 1.0e-8) {
                ++high_energy_monotonic_failures;
            }
            previous_ratio = current_ratio;
        }
    }

    double max_adjacent_energy_jump = 0.0;
    double max_adjacent_theta_jump = 0.0;
    double max_second_difference = 0.0;
    for (std::size_t itheta = 0; itheta < table.theta_size(); ++itheta) {
        for (std::size_t ienergy = 1; ienergy < table.energy_size(); ++ienergy) {
            const double lhs = table.log_ratio_table[itheta * table.energy_size() + ienergy - 1];
            const double rhs = table.log_ratio_table[itheta * table.energy_size() + ienergy];
            max_adjacent_energy_jump = std::max(max_adjacent_energy_jump, std::abs(rhs - lhs));
        }
        for (std::size_t ienergy = 1; ienergy + 1 < table.energy_size(); ++ienergy) {
            const double prev = table.log_ratio_table[itheta * table.energy_size() + ienergy - 1];
            const double curr = table.log_ratio_table[itheta * table.energy_size() + ienergy];
            const double next = table.log_ratio_table[itheta * table.energy_size() + ienergy + 1];
            max_second_difference = std::max(max_second_difference, std::abs(next - 2.0 * curr + prev));
        }
    }
    for (std::size_t itheta = 1; itheta < table.theta_size(); ++itheta) {
        for (std::size_t ienergy = 0; ienergy < table.energy_size(); ++ienergy) {
            const double lhs = table.log_ratio_table[(itheta - 1) * table.energy_size() + ienergy];
            const double rhs = table.log_ratio_table[itheta * table.energy_size() + ienergy];
            max_adjacent_theta_jump = std::max(max_adjacent_theta_jump, std::abs(rhs - lhs));
        }
    }

    const auto [min_it, max_it] =
        std::minmax_element(table.log_ratio_table.begin(), table.log_ratio_table.end());
    const double table_min_ratio = std::exp(*min_it);
    const double table_max_ratio = std::exp(*max_it);
    const bool low_energy_ok = low_energy_max_abs_deviation <= kThermalKnLowEnergyTolerance;
    const bool low_theta_ok =
        low_theta_mean_abs_difference <= kThermalKnLowThetaMeanAbsTolerance &&
        low_theta_max_abs_difference <= kThermalKnLowThetaMaxAbsTolerance;
    const bool high_energy_monotonic_ok = high_energy_monotonic_failures == 0;
    const bool smoothness_ok =
        max_second_difference <= kThermalKnSmoothnessSecondDiffThreshold;

    const std::string report_path =
        base_cfg.output_dir + "/" +
        validation_tag(base_cfg, "validate_thermal_kn_transport_table_report") + ".csv";
    write_validation_report(report_path,
                            {
                                {"thermal_kn_table_path", base_cfg.thermal_kn_table_path},
                                {"energy_points", std::to_string(table.energy_size())},
                                {"theta_points", std::to_string(table.theta_size())},
                                {"low_energy_max_abs_deviation_from_unity", format_number(low_energy_max_abs_deviation)},
                                {"low_energy_limit_ok", low_energy_ok ? "true" : "false"},
                                {"low_theta_mean_abs_difference_to_stationary_kn", format_number(low_theta_mean_abs_difference)},
                                {"low_theta_max_abs_difference_to_stationary_kn", format_number(low_theta_max_abs_difference)},
                                {"low_theta_limit_ok", low_theta_ok ? "true" : "false"},
                                {"high_energy_monotonic_failures", std::to_string(high_energy_monotonic_failures)},
                                {"high_energy_monotonic_ok", high_energy_monotonic_ok ? "true" : "false"},
                                {"max_adjacent_log_ratio_jump_energy", format_number(max_adjacent_energy_jump)},
                                {"max_adjacent_log_ratio_jump_theta", format_number(max_adjacent_theta_jump)},
                                {"max_second_difference_log_ratio", format_number(max_second_difference)},
                                {"smoothness_ok", smoothness_ok ? "true" : "false"},
                                {"table_min_ratio", format_number(table_min_ratio)},
                                {"table_max_ratio", format_number(table_max_ratio)},
                            });
}

void run_validation_slab_transport_cross_section(const ic::Config& base_cfg) {
    struct TransportSnapshot {
        double tau = 0.0;
        double theta_e = 0.0;
        double photon_energy = 0.0;
        std::string transport_cross_section;
        std::string source_run_tag;
        std::uint64_t events_completed = 0;
        std::uint64_t scattered_events = 0;
        std::uint64_t terminated_at_max_scatters = 0;
        double mean_scatter_count = 0.0;
        double scatter_fraction = 0.0;
        double mean_scattered_energy_mec2 = 0.0;
        double mean_energy_amplification = 0.0;
        double escaped_up_total_fraction = 0.0;
        double escaped_down_total_fraction = 0.0;
        double max_scatter_termination_fraction = 0.0;
        bool bookkeeping_consistent = false;
        double energy_hist_overflow_fraction = 0.0;
    };

    const std::vector<double> tau_values{0.5, 1.0, 5.0, 10.0};
    const std::vector<double> theta_values{0.1, 0.5};
    const std::vector<double> photon_energies{1.0e-5, 1.0e-4, 1.0e-3};
    const std::vector<std::string> transport_modes{"thomson", "kn", "thermal_kn"};
    std::vector<std::vector<std::string>> rows;
    std::vector<std::vector<std::string>> difference_rows;
    rows.reserve(tau_values.size() * theta_values.size() * photon_energies.size() * transport_modes.size());
    difference_rows.reserve(tau_values.size() * theta_values.size() * photon_energies.size());

    for (double tau : tau_values) {
        for (double theta_e : theta_values) {
            for (double photon_energy : photon_energies) {
                std::vector<TransportSnapshot> snapshots;
                snapshots.reserve(transport_modes.size());

                for (const std::string& transport_mode : transport_modes) {
                    ic::Config cfg = base_cfg;
                    cfg.mode = "validate-slab-transport-cross-section";
                    cfg.run_label.clear();
                    cfg.geometry_model = "slab";
                    cfg.slab_height = 1.0;
                    cfg.slab_optical_depth = tau;
                    cfg.slab_injection_model = "beam";
                    cfg.electron_model = "thermal";
                    cfg.electron_kTe = theta_e;
                    cfg.incident_photon_energy = photon_energy;
                    cfg.transport_cross_section = transport_mode;
                    if (cfg.max_scatters == ic::Config{}.max_scatters) {
                        cfg.max_scatters = 256;
                    }
                    apply_multi_scatter_validation_energy_range(cfg);

                    const CompletedRun run = run_and_record(cfg);
                    const bool bookkeeping_consistent = slab_bookkeeping_consistent(run.stats);
                    const double overflow_fraction = energy_hist_overflow_fraction(run.stats);

                    rows.push_back({
                        format_number(tau),
                        format_number(theta_e),
                        format_number(photon_energy),
                        transport_mode,
                        std::to_string(run.cfg.max_scatters),
                        run.tag,
                        std::to_string(run.stats.events_completed),
                        std::to_string(run.stats.scattered_events),
                        format_number(mean_scatter_count(run.stats)),
                        format_number(scatter_fraction(run.stats)),
                        format_number(mean_scattered_energy_mec2(run.stats)),
                        format_number(mean_energy_amplification(run.cfg, run.stats)),
                        format_number(escaped_up_total_fraction(run.stats)),
                        format_number(escaped_down_total_fraction(run.stats)),
                        std::to_string(run.stats.terminated_at_max_scatters),
                        format_number(max_scatter_termination_fraction(run.stats)),
                        bookkeeping_consistent ? "true" : "false",
                        format_number(overflow_fraction),
                    });

                    snapshots.push_back({
                        tau,
                        theta_e,
                        photon_energy,
                        transport_mode,
                        run.tag,
                        run.stats.events_completed,
                        run.stats.scattered_events,
                        run.stats.terminated_at_max_scatters,
                        mean_scatter_count(run.stats),
                        scatter_fraction(run.stats),
                        mean_scattered_energy_mec2(run.stats),
                        mean_energy_amplification(run.cfg, run.stats),
                        escaped_up_total_fraction(run.stats),
                        escaped_down_total_fraction(run.stats),
                        max_scatter_termination_fraction(run.stats),
                        bookkeeping_consistent,
                        overflow_fraction,
                    });

                    if (!bookkeeping_consistent) {
                        std::cerr << "warning: transport cross section validation bookkeeping failed for tau="
                                  << tau << " theta_e=" << theta_e
                                  << " photon_energy=" << photon_energy
                                  << " transport=" << transport_mode << "\n";
                    }
                    maybe_warn_about_validation_overflow(cfg.mode, run.stats);
                }

                const TransportSnapshot& thomson = snapshots[0];
                const TransportSnapshot& kn = snapshots[1];
                const TransportSnapshot& thermal_kn = snapshots[2];
                const double mean_scattered_energy_difference =
                    kn.mean_scattered_energy_mec2 - thomson.mean_scattered_energy_mec2;
                const double relative_energy_difference =
                    thomson.mean_scattered_energy_mec2 > 0.0
                        ? std::abs(mean_scattered_energy_difference) /
                              thomson.mean_scattered_energy_mec2
                        : 0.0;
                const bool weak_high_tau_seed_dependence_difference =
                    tau >= 5.0 && relative_energy_difference <= kKnTransportWeakDifferenceThreshold;

                difference_rows.push_back({
                    format_number(tau),
                    format_number(theta_e),
                    format_number(photon_energy),
                    thomson.source_run_tag,
                    kn.source_run_tag,
                    thermal_kn.source_run_tag,
                    format_number(kn.mean_scatter_count - thomson.mean_scatter_count),
                    format_number(mean_scattered_energy_difference),
                    format_number(kn.mean_energy_amplification - thomson.mean_energy_amplification),
                    format_number(kn.escaped_up_total_fraction - thomson.escaped_up_total_fraction),
                    format_number(kn.escaped_down_total_fraction - thomson.escaped_down_total_fraction),
                    format_number(thermal_kn.mean_scatter_count - kn.mean_scatter_count),
                    format_number(thermal_kn.mean_scattered_energy_mec2 - kn.mean_scattered_energy_mec2),
                    format_number(thermal_kn.mean_energy_amplification - kn.mean_energy_amplification),
                    format_number(thermal_kn.escaped_up_total_fraction - kn.escaped_up_total_fraction),
                    format_number(thermal_kn.escaped_down_total_fraction - kn.escaped_down_total_fraction),
                    weak_high_tau_seed_dependence_difference ? "true" : "false",
                });
            }
        }
    }

    const std::string table_path = base_cfg.output_dir + "/" +
                                   validation_tag(base_cfg, "validate_slab_transport_cross_section_table") + ".csv";
    write_table_csv(table_path,
                    {
                        "slab_tau",
                        "electron_kTe_mec2",
                        "incident_photon_energy_mec2",
                        "transport_cross_section",
                        "max_scatters",
                        "source_run_tag",
                        "events_completed",
                        "scattered_events",
                        "mean_scatter_count",
                        "scatter_fraction",
                        "mean_scattered_energy_mec2",
                        "mean_energy_amplification",
                        "escaped_up_total_fraction",
                        "escaped_down_total_fraction",
                        "terminated_at_max_scatters",
                        "max_scatter_termination_fraction",
                        "bookkeeping_consistent",
                        "energy_hist_overflow_fraction",
                    },
                    rows);
    std::cout << "Wrote: " << table_path << "\n";

    const std::string difference_path =
        base_cfg.output_dir + "/" +
        validation_tag(base_cfg, "validate_slab_transport_cross_section_differences") + ".csv";
    write_table_csv(difference_path,
                    {
                        "slab_tau",
                        "electron_kTe_mec2",
                        "incident_photon_energy_mec2",
                        "thomson_source_run_tag",
                        "kn_source_run_tag",
                        "thermal_kn_source_run_tag",
                        "kn_minus_thomson_mean_scatter_count",
                        "kn_minus_thomson_mean_scattered_energy_mec2",
                        "kn_minus_thomson_mean_energy_amplification",
                        "kn_minus_thomson_escaped_up_total_fraction",
                        "kn_minus_thomson_escaped_down_total_fraction",
                        "thermal_kn_minus_kn_mean_scatter_count",
                        "thermal_kn_minus_kn_mean_scattered_energy_mec2",
                        "thermal_kn_minus_kn_mean_energy_amplification",
                        "thermal_kn_minus_kn_escaped_up_total_fraction",
                        "thermal_kn_minus_kn_escaped_down_total_fraction",
                        "high_tau_mean_energy_difference_is_weak",
                    },
                    difference_rows);
    std::cout << "Wrote: " << difference_path << "\n";
}

void run_production_slab_thermal_case(const ic::Config& base_cfg) {
    ic::Config cfg = base_cfg;
    cfg.mode = "production-slab-thermal-case";
    cfg.geometry_model = "slab";
    cfg.slab_height = 1.0;
    cfg.electron_model = "thermal";
    apply_multi_scatter_validation_energy_range(cfg);

    const CompletedRun run = run_and_record(cfg);
    maybe_warn_about_production_slab_thermal_case(run);

    const std::string table_path =
        cfg.output_dir + "/" + run.tag + "_production_slab_thermal_case_table.csv";
    write_production_slab_thermal_table(table_path, {make_production_slab_thermal_row(run)});
}

void run_production_slab_thermal_refined_sweep(const ic::Config& base_cfg) {
    const std::string table_path =
        base_cfg.output_dir + "/" +
        validation_tag(base_cfg, "production_slab_thermal_refined_sweep_table") + ".csv";
    write_production_slab_thermal_table(
        table_path,
        run_production_slab_thermal_grid_rows(base_cfg,
                                              production_slab_thermal_tau_grid_stage2(),
                                              production_slab_thermal_injection_models(),
                                              production_slab_thermal_theta_grid_stage2(),
                                              "production-slab-thermal-refined-sweep"));
}

void run_production_slab_thermal_refined_split(const ic::Config& base_cfg) {
    ic::Config cfg = base_cfg;
    cfg.mode = "production-slab-thermal-refined-split";
    cfg.geometry_model = "slab";
    cfg.slab_height = 1.0;
    cfg.electron_model = "thermal";

    const std::string prefix =
        cfg.run_label.empty() ? "refined_split" : sanitize_token(cfg.run_label);
    const std::string table_path =
        cfg.output_dir + "/" + prefix + "_production_slab_thermal_refined_split_table.csv";
    write_production_slab_thermal_table(
        table_path,
        run_production_slab_thermal_grid_rows(cfg,
                                              {cfg.slab_optical_depth},
                                              {cfg.slab_injection_model},
                                              production_slab_thermal_theta_grid_stage2(),
                                              "production-slab-thermal-refined-split"));
}

void run_production_slab_thermal_expanded_sweep(const ic::Config& base_cfg) {
    const std::string table_path =
        base_cfg.output_dir + "/" +
        validation_tag(base_cfg, "production_slab_thermal_expanded_sweep_table") + ".csv";
    write_production_slab_thermal_table(
        table_path,
        run_production_slab_thermal_grid_rows(base_cfg,
                                              production_slab_thermal_tau_grid_stage3(),
                                              production_slab_thermal_injection_models(),
                                              production_slab_thermal_theta_grid_stage3(),
                                              "production-slab-thermal-expanded-sweep"));
}

void run_production_slab_thermal_expanded_split(const ic::Config& base_cfg) {
    ic::Config cfg = base_cfg;
    cfg.mode = "production-slab-thermal-expanded-split";
    cfg.geometry_model = "slab";
    cfg.slab_height = 1.0;
    cfg.electron_model = "thermal";

    const std::string prefix =
        cfg.run_label.empty() ? "expanded_split" : sanitize_token(cfg.run_label);
    const std::string table_path =
        cfg.output_dir + "/" + prefix + "_production_slab_thermal_expanded_split_table.csv";
    write_production_slab_thermal_table(
        table_path,
        run_production_slab_thermal_grid_rows(cfg,
                                              {cfg.slab_optical_depth},
                                              {cfg.slab_injection_model},
                                              production_slab_thermal_theta_grid_stage3(),
                                              "production-slab-thermal-expanded-split"));
}

void run_production_slab_high_tau_sweep(const ic::Config& base_cfg) {
    ic::Config cfg = base_cfg;
    if (cfg.max_scatters == ic::Config{}.max_scatters) {
        // The generic config default remains tuned for low-to-moderate optical
        // depth. Promote the first high-tau production sweep to the validated
        // high-depth default unless the caller explicitly overrides it.
        cfg.max_scatters = 256;
    }

    const std::string table_path =
        cfg.output_dir + "/" +
        validation_tag(cfg, "production_slab_high_tau_sweep_table") + ".csv";
    write_production_slab_thermal_table(
        table_path,
        run_production_slab_thermal_grid_rows(cfg,
                                              production_slab_high_tau_grid_stage1(),
                                              selected_high_tau_injection_models(cfg),
                                              production_slab_high_tau_theta_grid_stage1(),
                                              "production-slab-high-tau-sweep"));
}

void run_production_slab_high_tau_dense_sweep(const ic::Config& base_cfg) {
    ic::Config cfg = base_cfg;
    if (cfg.max_scatters == ic::Config{}.max_scatters) {
        cfg.max_scatters = 256;
    }

    const std::string table_path =
        cfg.output_dir + "/" +
        validation_tag(cfg, "production_slab_high_tau_dense_sweep_table") + ".csv";
    write_production_slab_thermal_table(
        table_path,
        run_production_slab_thermal_grid_rows(cfg,
                                              production_slab_high_tau_grid_stage2(),
                                              selected_high_tau_injection_models(cfg),
                                              production_slab_high_tau_theta_grid_stage2(),
                                              "production-slab-high-tau-dense-sweep"));
}

void run_production_slab_high_tau_dense_split(const ic::Config& base_cfg) {
    ic::Config cfg = base_cfg;
    cfg.mode = "production-slab-high-tau-dense-split";
    cfg.geometry_model = "slab";
    cfg.slab_height = 1.0;
    cfg.electron_model = "thermal";
    if (cfg.max_scatters == ic::Config{}.max_scatters) {
        cfg.max_scatters = 256;
    }

    const std::string prefix =
        cfg.run_label.empty() ? "high_tau_dense_split" : sanitize_token(cfg.run_label);
    const std::string table_path =
        cfg.output_dir + "/" + prefix + "_production_slab_high_tau_dense_split_table.csv";
    write_production_slab_thermal_table(
        table_path,
        run_production_slab_thermal_grid_rows(cfg,
                                              {cfg.slab_optical_depth},
                                              {cfg.slab_injection_model},
                                              production_slab_high_tau_theta_grid_stage2(),
                                              "production-slab-high-tau-dense-split"));
}

void run_production_slab_seed_energy_dense_sweep(const ic::Config& base_cfg) {
    ic::Config cfg = base_cfg;
    if (cfg.max_scatters == ic::Config{}.max_scatters) {
        cfg.max_scatters = 256;
    }

    const std::string table_path =
        cfg.output_dir + "/" +
        validation_tag(cfg, "production_slab_seed_energy_dense_sweep_table") + ".csv";
    write_production_slab_seed_energy_table(
        table_path,
        run_production_slab_seed_energy_grid_rows(cfg,
                                                  production_slab_seed_energy_tau_grid_stage1(),
                                                  {cfg.slab_injection_model},
                                                  production_slab_seed_energy_theta_grid_stage1(),
                                                  production_slab_seed_energy_photon_grid_stage1(),
                                                  "production-slab-seed-energy-dense-sweep"));
}

void run_production_slab_seed_energy_dense_split(const ic::Config& base_cfg) {
    ic::Config cfg = base_cfg;
    cfg.mode = "production-slab-seed-energy-dense-split";
    cfg.geometry_model = "slab";
    cfg.slab_height = 1.0;
    cfg.electron_model = "thermal";
    if (cfg.max_scatters == ic::Config{}.max_scatters) {
        cfg.max_scatters = 256;
    }

    const std::string prefix =
        cfg.run_label.empty() ? "seed_energy_dense_split" : sanitize_token(cfg.run_label);
    const std::string table_path =
        cfg.output_dir + "/" + prefix + "_production_slab_seed_energy_dense_split_table.csv";
    write_production_slab_seed_energy_table(
        table_path,
        run_production_slab_seed_energy_grid_rows(cfg,
                                                  {cfg.slab_optical_depth},
                                                  {cfg.slab_injection_model},
                                                  production_slab_seed_energy_theta_grid_stage1(),
                                                  {cfg.incident_photon_energy},
                                                  "production-slab-seed-energy-dense-split"));
}

void run_production_slab_seed_energy_broad_sweep(const ic::Config& base_cfg) {
    ic::Config cfg = base_cfg;
    if (cfg.max_scatters == ic::Config{}.max_scatters) {
        cfg.max_scatters = 256;
    }

    const std::string table_path =
        cfg.output_dir + "/" +
        validation_tag(cfg, "production_slab_seed_energy_broad_sweep_table") + ".csv";
    write_production_slab_seed_energy_table(
        table_path,
        run_production_slab_seed_energy_grid_rows(cfg,
                                                  production_slab_seed_energy_tau_grid_stage2(),
                                                  {cfg.slab_injection_model},
                                                  production_slab_seed_energy_theta_grid_stage2(),
                                                  production_slab_seed_energy_photon_grid_stage2(),
                                                  "production-slab-seed-energy-broad-sweep"));
}

void run_production_slab_seed_energy_broad_split(const ic::Config& base_cfg) {
    ic::Config cfg = base_cfg;
    cfg.mode = "production-slab-seed-energy-broad-split";
    cfg.geometry_model = "slab";
    cfg.slab_height = 1.0;
    cfg.electron_model = "thermal";
    if (cfg.max_scatters == ic::Config{}.max_scatters) {
        cfg.max_scatters = 256;
    }

    const std::string prefix =
        cfg.run_label.empty() ? "seed_energy_broad_split" : sanitize_token(cfg.run_label);
    const std::string table_path =
        cfg.output_dir + "/" + prefix + "_production_slab_seed_energy_broad_split_table.csv";
    write_production_slab_seed_energy_table(
        table_path,
        run_production_slab_seed_energy_grid_rows(cfg,
                                                  {cfg.slab_optical_depth},
                                                  {cfg.slab_injection_model},
                                                  production_slab_seed_energy_theta_grid_stage2(),
                                                  {cfg.incident_photon_energy},
                                                  "production-slab-seed-energy-broad-split"));
}

void run_validation_mode(const ic::Config& cfg) {
    if (cfg.mode == "generate-thermal-kn-transport-table") {
        run_generate_thermal_kn_transport_table(cfg);
    } else if (cfg.mode == "validate-thermal-kn-transport-table") {
        run_validate_thermal_kn_transport_table(cfg);
    } else if (cfg.mode == "production-slab-thermal-case") {
        run_production_slab_thermal_case(cfg);
    } else if (cfg.mode == "production-slab-high-tau-sweep") {
        run_production_slab_high_tau_sweep(cfg);
    } else if (cfg.mode == "production-slab-seed-energy-dense-split") {
        run_production_slab_seed_energy_dense_split(cfg);
    } else if (cfg.mode == "production-slab-seed-energy-dense-sweep") {
        run_production_slab_seed_energy_dense_sweep(cfg);
    } else if (cfg.mode == "production-slab-seed-energy-broad-split") {
        run_production_slab_seed_energy_broad_split(cfg);
    } else if (cfg.mode == "production-slab-seed-energy-broad-sweep") {
        run_production_slab_seed_energy_broad_sweep(cfg);
    } else if (cfg.mode == "production-slab-high-tau-dense-split") {
        run_production_slab_high_tau_dense_split(cfg);
    } else if (cfg.mode == "production-slab-high-tau-dense-sweep") {
        run_production_slab_high_tau_dense_sweep(cfg);
    } else if (cfg.mode == "production-slab-thermal-refined-split") {
        run_production_slab_thermal_refined_split(cfg);
    } else if (cfg.mode == "production-slab-thermal-refined-sweep") {
        run_production_slab_thermal_refined_sweep(cfg);
    } else if (cfg.mode == "production-slab-thermal-expanded-split") {
        run_production_slab_thermal_expanded_split(cfg);
    } else if (cfg.mode == "production-slab-thermal-expanded-sweep") {
        run_production_slab_thermal_expanded_sweep(cfg);
    } else if (cfg.mode == "production-slab-thermal-sweep") {
        run_production_slab_thermal_sweep(cfg);
    } else if (cfg.mode == "sweep-slab-paper") {
        run_sweep_slab_paper(cfg);
    } else if (cfg.mode == "validate-thomson") {
        run_validation_thomson(cfg);
    } else if (cfg.mode == "validate-kn") {
        run_validation_kn(cfg);
    } else if (cfg.mode == "validate-conservation") {
        run_validation_conservation(cfg);
    } else if (cfg.mode == "validate-thermal") {
        run_validation_thermal(cfg);
    } else if (cfg.mode == "validate-thermal-sweep") {
        run_validation_thermal_sweep(cfg);
    } else if (cfg.mode == "validate-slab-thin") {
        run_validation_slab_thin(cfg);
    } else if (cfg.mode == "validate-slab-moderate") {
        run_validation_slab_moderate(cfg);
    } else if (cfg.mode == "validate-slab-tau-sweep") {
        run_validation_slab_tau_sweep(cfg);
    } else if (cfg.mode == "validate-slab-electron-sweep") {
        run_validation_slab_electron_sweep(cfg);
    } else if (cfg.mode == "validate-slab-multi-thin") {
        run_validation_slab_multi_thin(cfg);
    } else if (cfg.mode == "validate-slab-multi-moderate") {
        run_validation_slab_multi_moderate(cfg);
    } else if (cfg.mode == "validate-slab-multi-convergence") {
        run_validation_slab_multi_convergence(cfg);
    } else if (cfg.mode == "validate-slab-high-tau-convergence") {
        run_validation_slab_high_tau_convergence(cfg);
    } else if (cfg.mode == "validate-slab-high-tau-seed-energy") {
        run_validation_slab_high_tau_seed_energy(cfg);
    } else if (cfg.mode == "validate-slab-transport-cross-section") {
        run_validation_slab_transport_cross_section(cfg);
    } else if (cfg.mode == "validate-slab-paper") {
        run_validation_slab_paper(cfg);
    } else if (cfg.mode == "validate-all") {
        run_validation_thomson(cfg);
        run_validation_kn(cfg);
        run_validation_conservation(cfg);
        run_validation_thermal(cfg);
        run_validation_thermal_sweep(cfg);
        run_validation_slab_thin(cfg);
        run_validation_slab_moderate(cfg);
        run_validation_slab_tau_sweep(cfg);
        run_validation_slab_electron_sweep(cfg);
        run_validation_slab_multi_thin(cfg);
        run_validation_slab_multi_moderate(cfg);
        run_validation_slab_multi_convergence(cfg);
        run_validation_slab_paper(cfg);
        run_production_slab_thermal_sweep(cfg);
    } else {
        throw std::runtime_error("unknown mode: " + cfg.mode);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const ic::Config cfg = ic::parse_config(argc, argv);

        if (cfg.mode == "run") {
            run_and_record(cfg);
        } else {
            run_validation_mode(cfg);
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
