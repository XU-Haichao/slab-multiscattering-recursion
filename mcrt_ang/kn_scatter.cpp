#include "kn_scatter.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "constants.hpp"
#include "kinematics.hpp"

namespace ic {
namespace {

constexpr double kKnAcceptanceEnvelope = 2.0;
constexpr double kKnEnvelopeTolerance = 1.0e-12;

double sample_kn_mu(double epsilon_in, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    while (true) {
        // Propose mu uniformly on [-1, 1], then accept using rejection sampling
        // against the reduced unpolarized Klein-Nishina angular kernel.
        const double mu = 2.0 * unit(rng) - 1.0;
        const double ratio = 1.0 / (1.0 + epsilon_in * (1.0 - mu));
        const double weight = ratio * ratio * (ratio + 1.0 / ratio - (1.0 - mu * mu));

        // For mu in [-1, 1], the reduced kernel is bounded by 2. This guard
        // makes the assumption explicit and fails loudly if the envelope is ever
        // violated by a future code change or numerical issue.
        if (weight > kKnAcceptanceEnvelope + kKnEnvelopeTolerance) {
            throw std::runtime_error("Klein-Nishina acceptance weight exceeded the assumed envelope");
        }

        if (unit(rng) * kKnAcceptanceEnvelope <= weight) {
            return mu;
        }
    }
}

}  // namespace

ScatterResult scatter_single_kn(const Photon& photon,
                                const Electron& electron,
                                std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    const FourVector k_in_lab = photon_four_vector(photon);
    const FourVector p_in_lab = electron_four_vector(electron);
    const Vec3 beta_vec = electron.beta_vector();

    const FourVector k_in_erf = boost_to_rest_frame(k_in_lab, beta_vec);
    const double epsilon_in = k_in_erf.t;
    const Vec3 incoming_dir_erf = k_in_erf.x.normalized();

    const double mu_erf = sample_kn_mu(epsilon_in, rng);
    const double phi_erf = constants::two_pi * unit(rng);
    const double epsilon_out = epsilon_in / (1.0 + epsilon_in * (1.0 - mu_erf));
    const Vec3 outgoing_dir_erf = direction_from_axis(incoming_dir_erf, mu_erf, phi_erf);

    const FourVector k_out_erf{epsilon_out, outgoing_dir_erf * epsilon_out};
    const FourVector k_out_lab = boost_from_rest_frame(k_out_erf, beta_vec);
    const FourVector p_out_lab = p_in_lab + k_in_lab - k_out_lab;

    ScatterResult result;
    result.scattered_photon = photon_from_four_vector(k_out_lab);
    result.outgoing_electron_gamma = p_out_lab.t;
    result.initial_total_energy = p_in_lab.t + k_in_lab.t;
    result.final_total_energy = p_out_lab.t + k_out_lab.t;
    result.energy_error = result.final_total_energy - result.initial_total_energy;
    result.outgoing_electron_mass2 = minkowski_mass_squared(p_out_lab);
    result.mass_shell_error = std::abs(result.outgoing_electron_mass2 - 1.0);
    result.scattering_mu_lab = std::clamp(dot(photon.direction.normalized(),
                                              result.scattered_photon.direction.normalized()),
                                          -1.0,
                                          1.0);
    result.incoming_photon_energy_erf = epsilon_in;
    result.outgoing_photon_energy_erf = epsilon_out;
    result.photon_energy_ratio_erf = epsilon_out / epsilon_in;
    result.success = true;
    return result;
}

double sigma_kn_total_over_sigma_t(double photon_energy_mec2) {
    if (!(photon_energy_mec2 > 0.0)) {
        return 1.0;
    }

    const double x = photon_energy_mec2;
    if (x < 1.0e-3) {
        const double x2 = x * x;
        return std::clamp(1.0 - 2.0 * x + 5.2 * x2, 0.0, 1.0);
    }

    if (x > 1.0e3) {
        const double asymptotic = (3.0 / (8.0 * x)) * (std::log(2.0 * x) + 0.5);
        return std::clamp(asymptotic, 0.0, 1.0);
    }

    const double one_plus_2x = 1.0 + 2.0 * x;
    const double log_term = std::log1p(2.0 * x);
    const double x2 = x * x;
    const double x3 = x2 * x;

    const double term1 =
        ((1.0 + x) / x3) * ((2.0 * x * (1.0 + x) / one_plus_2x) - log_term);
    const double term2 = log_term / (2.0 * x);
    const double term3 = (1.0 + 3.0 * x) / (one_plus_2x * one_plus_2x);
    const double ratio = 0.75 * (term1 + term2 - term3);
    return std::clamp(ratio, 0.0, 1.0);
}

}  // namespace ic
