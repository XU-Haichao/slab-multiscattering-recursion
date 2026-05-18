#include "thermal_kn_transport.hpp"

#include <H5Cpp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "kn_scatter.hpp"

namespace ic {
namespace {

constexpr const char* kThermalKnGeneratorVersion = "thermal_kn_transport_v1";
constexpr double kInterpolationFloor = 1.0e-300;

std::vector<double> make_log_grid(double min_value, double max_value, std::size_t count) {
    if (!(min_value > 0.0) || !(max_value > min_value) || count < 2) {
        throw std::runtime_error("invalid thermal_kn log-grid settings");
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

double mj_scaled_weight(double theta_e, double z) {
    const double gamma = 1.0 + theta_e * z;
    const double beta2 = std::max(0.0, 1.0 - 1.0 / (gamma * gamma));
    const double beta = std::sqrt(beta2);
    return gamma * gamma * beta * std::exp(-z);
}

std::string current_timestamp_utc() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc_tm{};
#if defined(_WIN32)
    gmtime_s(&utc_tm, &time);
#else
    gmtime_r(&time, &utc_tm);
#endif
    char buffer[64];
    if (std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc_tm) == 0) {
        return "unknown";
    }
    return buffer;
}

std::size_t lower_index_clamped(const std::vector<double>& grid, double value) {
    if (grid.size() < 2) {
        throw std::runtime_error("thermal_kn interpolation grid must have at least two points");
    }
    if (value <= grid.front()) {
        return 0;
    }
    if (value >= grid.back()) {
        return grid.size() - 2;
    }
    auto upper = std::upper_bound(grid.begin(), grid.end(), value);
    return static_cast<std::size_t>((upper - grid.begin()) - 1);
}

double interpolation_fraction(double lower, double upper, double value) {
    if (!(upper > lower)) {
        return 0.0;
    }
    return std::clamp((value - lower) / (upper - lower), 0.0, 1.0);
}

std::size_t table_index(const ThermalKnTransportTable& table,
                        std::size_t theta_index,
                        std::size_t energy_index) {
    return theta_index * table.energy_size() + energy_index;
}

void write_string_attribute(H5::H5Object& object,
                            const std::string& name,
                            const std::string& value) {
    H5::StrType type(H5::PredType::C_S1, value.size());
    H5::DataSpace scalar(H5S_SCALAR);
    H5::Attribute attribute = object.createAttribute(name, type, scalar);
    attribute.write(type, value);
}

std::string read_string_attribute(const H5::H5Object& object, const std::string& name) {
    H5::Attribute attribute = object.openAttribute(name);
    H5::StrType type = attribute.getStrType();
    std::string value;
    attribute.read(type, value);
    return value;
}

template <typename T>
void write_scalar_attribute(H5::H5Object& object,
                            const std::string& name,
                            const H5::PredType& type,
                            const T& value) {
    H5::DataSpace scalar(H5S_SCALAR);
    H5::Attribute attribute = object.createAttribute(name, type, scalar);
    attribute.write(type, &value);
}

template <typename T>
T read_scalar_attribute(const H5::H5Object& object,
                        const std::string& name,
                        const H5::PredType& type) {
    H5::Attribute attribute = object.openAttribute(name);
    T value{};
    attribute.read(type, &value);
    return value;
}

}  // namespace

double compute_thermal_kn_transport_ratio(double photon_energy_mec2,
                                          double theta_e,
                                          const ThermalKnIntegrationSettings& settings) {
    if (!(photon_energy_mec2 > 0.0)) {
        throw std::runtime_error("thermal_kn photon energy must be > 0");
    }
    if (!(theta_e > 0.0)) {
        throw std::runtime_error("thermal_kn theta_e must be > 0");
    }
    if (settings.z_points == 0 || settings.mu_points == 0 || !(settings.z_max > 0.0)) {
        throw std::runtime_error("invalid thermal_kn integration settings");
    }

    const double dz = settings.z_max / static_cast<double>(settings.z_points);
    const double dmu = 2.0 / static_cast<double>(settings.mu_points);
    double numerator = 0.0;
    double denominator = 0.0;

    for (std::size_t iz = 0; iz <= settings.z_points; ++iz) {
        const double z = dz * static_cast<double>(iz);
        const double gamma = 1.0 + theta_e * z;
        const double beta2 = std::max(0.0, 1.0 - 1.0 / (gamma * gamma));
        const double beta = std::sqrt(beta2);
        const double gamma_weight = (iz == 0 || iz == settings.z_points) ? 0.5 : 1.0;
        const double mj_weight = mj_scaled_weight(theta_e, z);

        double mu_average = 0.0;
        for (std::size_t imu = 0; imu < settings.mu_points; ++imu) {
            const double mu = -1.0 + (static_cast<double>(imu) + 0.5) * dmu;
            const double relative_factor = 1.0 - beta * mu;
            const double epsilon_erf = gamma * photon_energy_mec2 * relative_factor;
            mu_average += relative_factor * sigma_kn_total_over_sigma_t(epsilon_erf);
        }
        mu_average /= static_cast<double>(settings.mu_points);

        numerator += gamma_weight * mj_weight * mu_average;
        denominator += gamma_weight * mj_weight;
    }

    numerator *= dz;
    denominator *= dz;
    if (!(denominator > 0.0)) {
        throw std::runtime_error("failed to normalize thermal_kn attenuation ratio");
    }

    return std::clamp(numerator / denominator, 0.0, 1.0);
}

ThermalKnTransportTable generate_thermal_kn_transport_table(
    const ThermalKnTableGenerationSettings& settings) {
    ThermalKnTransportTable table;
    table.energy_grid = make_log_grid(settings.energy_min, settings.energy_max, settings.energy_points);
    table.theta_grid = make_log_grid(settings.theta_min, settings.theta_max, settings.theta_points);
    table.log_ratio_table.resize(table.energy_size() * table.theta_size());
    table.integration = settings.integration;
    table.generator_version = kThermalKnGeneratorVersion;
    table.physics_definition_note =
        "R_th(E,theta_e)=< (1-beta*mu) sigma_KN(gamma E (1-beta mu))/sigma_T > over isotropic Maxwell-Juttner electrons";
    table.creation_timestamp = current_timestamp_utc();

    for (std::size_t itheta = 0; itheta < table.theta_size(); ++itheta) {
        const double theta_e = table.theta_grid[itheta];
        for (std::size_t ienergy = 0; ienergy < table.energy_size(); ++ienergy) {
            const double energy = table.energy_grid[ienergy];
            const double ratio = compute_thermal_kn_transport_ratio(energy, theta_e, settings.integration);
            table.log_ratio_table[table_index(table, itheta, ienergy)] =
                std::log(std::max(ratio, kInterpolationFloor));
        }
    }

    return table;
}

void write_thermal_kn_transport_table_hdf5(const std::string& path,
                                           const ThermalKnTransportTable& table) {
    if (table.energy_size() < 2 || table.theta_size() < 2) {
        throw std::runtime_error("thermal_kn table must have at least 2x2 grid points");
    }

    std::filesystem::path out_path(path);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    H5::H5File file(path, H5F_ACC_TRUNC);

    {
        const hsize_t dims[1]{table.energy_size()};
        H5::DataSpace space(1, dims);
        H5::DataSet dataset = file.createDataSet("energy_grid", H5::PredType::NATIVE_DOUBLE, space);
        dataset.write(table.energy_grid.data(), H5::PredType::NATIVE_DOUBLE);
    }
    {
        const hsize_t dims[1]{table.theta_size()};
        H5::DataSpace space(1, dims);
        H5::DataSet dataset = file.createDataSet("theta_grid", H5::PredType::NATIVE_DOUBLE, space);
        dataset.write(table.theta_grid.data(), H5::PredType::NATIVE_DOUBLE);
    }
    {
        const hsize_t dims[2]{table.theta_size(), table.energy_size()};
        H5::DataSpace space(2, dims);
        H5::DataSet dataset = file.createDataSet("log_ratio_table", H5::PredType::NATIVE_DOUBLE, space);
        dataset.write(table.log_ratio_table.data(), H5::PredType::NATIVE_DOUBLE);
    }

    write_string_attribute(file, "generator_version", table.generator_version);
    write_string_attribute(file, "physics_definition_note", table.physics_definition_note);
    write_string_attribute(file, "creation_timestamp", table.creation_timestamp);
    write_scalar_attribute(file,
                           "integration_z_points",
                           H5::PredType::NATIVE_ULLONG,
                           static_cast<unsigned long long>(table.integration.z_points));
    write_scalar_attribute(file,
                           "integration_mu_points",
                           H5::PredType::NATIVE_ULLONG,
                           static_cast<unsigned long long>(table.integration.mu_points));
    write_scalar_attribute(file, "integration_z_max", H5::PredType::NATIVE_DOUBLE, table.integration.z_max);
}

ThermalKnTransportTable load_thermal_kn_transport_table_hdf5(const std::string& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("thermal_kn table file not found: " + path);
    }

    ThermalKnTransportTable table;
    H5::H5File file(path, H5F_ACC_RDONLY);

    {
        H5::DataSet dataset = file.openDataSet("energy_grid");
        H5::DataSpace space = dataset.getSpace();
        hsize_t dims[1]{0};
        space.getSimpleExtentDims(dims);
        table.energy_grid.resize(static_cast<std::size_t>(dims[0]));
        dataset.read(table.energy_grid.data(), H5::PredType::NATIVE_DOUBLE);
    }
    {
        H5::DataSet dataset = file.openDataSet("theta_grid");
        H5::DataSpace space = dataset.getSpace();
        hsize_t dims[1]{0};
        space.getSimpleExtentDims(dims);
        table.theta_grid.resize(static_cast<std::size_t>(dims[0]));
        dataset.read(table.theta_grid.data(), H5::PredType::NATIVE_DOUBLE);
    }
    {
        H5::DataSet dataset = file.openDataSet("log_ratio_table");
        H5::DataSpace space = dataset.getSpace();
        hsize_t dims[2]{0, 0};
        space.getSimpleExtentDims(dims);
        if (dims[0] != table.theta_grid.size() || dims[1] != table.energy_grid.size()) {
            throw std::runtime_error("thermal_kn table dimensions do not match grid axes");
        }
        table.log_ratio_table.resize(static_cast<std::size_t>(dims[0] * dims[1]));
        dataset.read(table.log_ratio_table.data(), H5::PredType::NATIVE_DOUBLE);
    }

    table.generator_version = read_string_attribute(file, "generator_version");
    table.physics_definition_note = read_string_attribute(file, "physics_definition_note");
    table.creation_timestamp = read_string_attribute(file, "creation_timestamp");
    table.integration.z_points = static_cast<std::size_t>(
        read_scalar_attribute<unsigned long long>(file, "integration_z_points", H5::PredType::NATIVE_ULLONG));
    table.integration.mu_points = static_cast<std::size_t>(
        read_scalar_attribute<unsigned long long>(file, "integration_mu_points", H5::PredType::NATIVE_ULLONG));
    table.integration.z_max =
        read_scalar_attribute<double>(file, "integration_z_max", H5::PredType::NATIVE_DOUBLE);

    return table;
}

double interpolate_thermal_kn_transport_ratio(const ThermalKnTransportTable& table,
                                              double photon_energy_mec2,
                                              double theta_e) {
    if (table.energy_size() < 2 || table.theta_size() < 2) {
        throw std::runtime_error("thermal_kn table must have at least 2x2 grid points for interpolation");
    }
    if (!(photon_energy_mec2 > 0.0) || !(theta_e > 0.0)) {
        throw std::runtime_error("thermal_kn interpolation arguments must be > 0");
    }

    const double clamped_energy =
        std::clamp(photon_energy_mec2, table.energy_grid.front(), table.energy_grid.back());
    const double clamped_theta =
        std::clamp(theta_e, table.theta_grid.front(), table.theta_grid.back());

    const std::size_t ienergy = lower_index_clamped(table.energy_grid, clamped_energy);
    const std::size_t itheta = lower_index_clamped(table.theta_grid, clamped_theta);

    const double log_e0 = std::log(table.energy_grid[ienergy]);
    const double log_e1 = std::log(table.energy_grid[ienergy + 1]);
    const double log_t0 = std::log(table.theta_grid[itheta]);
    const double log_t1 = std::log(table.theta_grid[itheta + 1]);
    const double fe = interpolation_fraction(log_e0, log_e1, std::log(clamped_energy));
    const double ft = interpolation_fraction(log_t0, log_t1, std::log(clamped_theta));

    const double q00 = table.log_ratio_table[table_index(table, itheta, ienergy)];
    const double q01 = table.log_ratio_table[table_index(table, itheta, ienergy + 1)];
    const double q10 = table.log_ratio_table[table_index(table, itheta + 1, ienergy)];
    const double q11 = table.log_ratio_table[table_index(table, itheta + 1, ienergy + 1)];

    const double interpolated_log_ratio =
        (1.0 - ft) * ((1.0 - fe) * q00 + fe * q01) +
        ft * ((1.0 - fe) * q10 + fe * q11);
    return std::exp(interpolated_log_ratio);
}

const ThermalKnTransportTable& cached_thermal_kn_transport_table(const std::string& path) {
    static thread_local std::string cached_path;
    static thread_local ThermalKnTransportTable cached_table;
    if (cached_path != path) {
        cached_table = load_thermal_kn_transport_table_hdf5(path);
        cached_path = path;
    }
    return cached_table;
}

}  // namespace ic
