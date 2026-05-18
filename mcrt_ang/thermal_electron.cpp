#include "thermal_electron.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "kn_scatter.hpp"

namespace ic {
namespace {

struct MaxwellJuttnerTable {
    double theta_e = -1.0;
    std::vector<double> z_grid;
    std::vector<double> cdf;
};

double mj_pdf_in_scaled_variable(double theta_e, double z) {
    const double gamma = 1.0 + theta_e * z;
    const double beta2 = 1.0 - 1.0 / (gamma * gamma);
    const double beta = std::sqrt(std::max(0.0, beta2));

    // For gamma = 1 + theta_e z, the Maxwell-Juttner density becomes
    // proportional to gamma^2 beta exp(-z) up to a constant factor.
    return gamma * gamma * beta * std::exp(-z);
}

void build_table(MaxwellJuttnerTable& table, double theta_e) {
    constexpr std::size_t kNumGridPoints = 8192;
    constexpr double kScaledUpper = 60.0;

    table.theta_e = theta_e;
    table.z_grid.resize(kNumGridPoints + 1);
    table.cdf.resize(kNumGridPoints + 1);

    const double dz = kScaledUpper / static_cast<double>(kNumGridPoints);
    table.z_grid[0] = 0.0;
    table.cdf[0] = 0.0;

    double cumulative = 0.0;
    for (std::size_t i = 1; i <= kNumGridPoints; ++i) {
        const double z_prev = dz * static_cast<double>(i - 1);
        const double z_curr = dz * static_cast<double>(i);
        const double pdf_prev = mj_pdf_in_scaled_variable(theta_e, z_prev);
        const double pdf_curr = mj_pdf_in_scaled_variable(theta_e, z_curr);
        cumulative += 0.5 * (pdf_prev + pdf_curr) * dz;
        table.z_grid[i] = z_curr;
        table.cdf[i] = cumulative;
    }

    if (!(cumulative > 0.0)) {
        throw std::runtime_error("failed to normalize Maxwell-Juttner sampler");
    }

    for (double& value : table.cdf) {
        value /= cumulative;
    }
    table.cdf.back() = 1.0;
}

const MaxwellJuttnerTable& get_table(double theta_e) {
    static thread_local MaxwellJuttnerTable table;
    if (table.theta_e != theta_e) {
        build_table(table, theta_e);
    }
    return table;
}

double sample_gamma_from_table(double theta_e, std::mt19937_64& rng) {
    const MaxwellJuttnerTable& table = get_table(theta_e);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const double u = unit(rng);

    auto upper = std::lower_bound(table.cdf.begin(), table.cdf.end(), u);
    if (upper == table.cdf.begin()) {
        return 1.0;
    }

    const std::size_t i = static_cast<std::size_t>(upper - table.cdf.begin());
    const double cdf0 = table.cdf[i - 1];
    const double cdf1 = table.cdf[i];
    const double z0 = table.z_grid[i - 1];
    const double z1 = table.z_grid[i];
    const double frac = (cdf1 > cdf0) ? (u - cdf0) / (cdf1 - cdf0) : 0.0;
    const double z = z0 + frac * (z1 - z0);
    return 1.0 + theta_e * z;
}

}  // namespace

Electron sample_isotropic_maxwell_juttner_electron(const ThermalElectronParams& params,
                                                   std::mt19937_64& rng) {
    if (params.theta_e <= 0.0) {
        throw std::runtime_error("thermal electron theta_e must be > 0");
    }

    Electron electron;
    electron.gamma = sample_gamma_from_table(params.theta_e, rng);
    electron.direction = sample_isotropic_direction(rng);
    return electron;
}

Electron sample_conditioned_thermal_electron_for_scatter(const Photon& photon,
                                                         const ThermalElectronParams& params,
                                                         std::mt19937_64& rng) {
    if (params.theta_e <= 0.0) {
        throw std::runtime_error("thermal electron theta_e must be > 0");
    }

    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const Vec3 photon_direction = photon.direction.normalized();

    while (true) {
        Electron electron = sample_isotropic_maxwell_juttner_electron(params, rng);
        const double beta = electron.beta();
        const double mu = dot(electron.direction.normalized(), photon_direction);
        const double relative_factor = 1.0 - beta * mu;
        const double epsilon_erf = electron.gamma * photon.energy * relative_factor;
        const double weight = relative_factor * sigma_kn_total_over_sigma_t(epsilon_erf);

        // The collision-conditioned kernel is bounded by
        // (1 - beta * mu) < 2 and sigma_KN / sigma_T <= 1.
        if (unit(rng) * 2.0 <= weight) {
            return electron;
        }
    }
}

}  // namespace ic
