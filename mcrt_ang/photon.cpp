#include "photon.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace ic {
namespace {

constexpr double kElectronRestEnergyEv = 510998.95000;
constexpr double kPlanckPhotonMeanX = 2.701178034186;
constexpr double kPlanckXMax = 60.0;
constexpr std::size_t kPlanckCdfPoints = 20000;

double planck_photon_number_density_x(double x) {
    if (x <= 0.0) {
        return 0.0;
    }
    if (x < 1.0e-5) {
        // x^2 / (exp(x)-1) = x - x^2/2 + O(x^3)
        return x - 0.5 * x * x;
    }
    return x * x / std::expm1(x);
}

struct PlanckPhotonNumberCdf {
    std::vector<double> x_grid;
    std::vector<double> cdf;

    PlanckPhotonNumberCdf() {
        x_grid.resize(kPlanckCdfPoints + 1);
        cdf.resize(kPlanckCdfPoints + 1);

        const double dx = kPlanckXMax / static_cast<double>(kPlanckCdfPoints);
        double cumulative = 0.0;
        x_grid[0] = 0.0;
        cdf[0] = 0.0;
        double previous_density = planck_photon_number_density_x(0.0);
        for (std::size_t i = 1; i <= kPlanckCdfPoints; ++i) {
            const double x = dx * static_cast<double>(i);
            const double density = planck_photon_number_density_x(x);
            cumulative += 0.5 * (previous_density + density) * dx;
            x_grid[i] = x;
            cdf[i] = cumulative;
            previous_density = density;
        }

        if (!(cumulative > 0.0) || !std::isfinite(cumulative)) {
            throw std::runtime_error("failed to build blackbody seed photon CDF");
        }
        for (double& value : cdf) {
            value /= cumulative;
        }
        cdf.back() = 1.0;
    }

    double sample(std::mt19937_64& rng) const {
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        const double u = unit(rng);
        auto upper = std::lower_bound(cdf.begin(), cdf.end(), u);
        if (upper == cdf.begin()) {
            return x_grid.front();
        }
        if (upper == cdf.end()) {
            return x_grid.back();
        }
        const std::size_t i = static_cast<std::size_t>(upper - cdf.begin());
        const double cdf0 = cdf[i - 1];
        const double cdf1 = cdf[i];
        const double x0 = x_grid[i - 1];
        const double x1 = x_grid[i];
        const double frac = (cdf1 > cdf0) ? (u - cdf0) / (cdf1 - cdf0) : 0.0;
        return x0 + frac * (x1 - x0);
    }
};

const PlanckPhotonNumberCdf& planck_photon_number_cdf() {
    static const PlanckPhotonNumberCdf cdf;
    return cdf;
}

}  // namespace

Photon make_incident_monoenergetic_photon(double energy, const Vec3& direction) {
    Photon photon;
    photon.energy = energy;
    photon.direction = direction.normalized();
    return photon;
}

double ev_to_mec2(double energy_ev) {
    return energy_ev / kElectronRestEnergyEv;
}

double blackbody_temperature_ev_to_mec2(double temperature_ev) {
    if (!(temperature_ev > 0.0)) {
        throw std::runtime_error("blackbody seed temperature must be > 0 eV");
    }
    return ev_to_mec2(temperature_ev);
}

double mean_blackbody_photon_energy_mec2(double temperature_ev) {
    return kPlanckPhotonMeanX * blackbody_temperature_ev_to_mec2(temperature_ev);
}

double sample_blackbody_photon_energy_mec2(double temperature_ev, std::mt19937_64& rng) {
    const double theta_seed = blackbody_temperature_ev_to_mec2(temperature_ev);
    // Photon histories are sampled from the blackbody photon-number spectrum,
    // p(x) dx proportional to x^2/(exp(x)-1) dx, x = E/(kT).
    const double x = planck_photon_number_cdf().sample(rng);
    const double energy = theta_seed * x;
    return std::max(energy, std::numeric_limits<double>::min());
}

double sample_seed_photon_energy_mec2(const std::string& seed_photon_model,
                                      double monoenergetic_energy_mec2,
                                      double seed_temperature_ev,
                                      std::mt19937_64& rng) {
    if (seed_photon_model == "monoenergetic" || seed_photon_model == "mono") {
        if (!(monoenergetic_energy_mec2 > 0.0)) {
            throw std::runtime_error("monoenergetic seed photon energy must be > 0");
        }
        return monoenergetic_energy_mec2;
    }
    if (seed_photon_model == "blackbody" || seed_photon_model == "bb") {
        return sample_blackbody_photon_energy_mec2(seed_temperature_ev, rng);
    }
    throw std::runtime_error("unsupported seed photon model: " + seed_photon_model);
}

}  // namespace ic
