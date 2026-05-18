#include "histogram.hpp"

#include <cmath>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace ic {

HistogramSpacing histogram_spacing_from_string(const std::string& value) {
    if (value == "linear") {
        return HistogramSpacing::linear;
    }
    if (value == "log" || value == "logarithmic") {
        return HistogramSpacing::logarithmic;
    }
    throw std::runtime_error("unsupported histogram spacing: " + value);
}

std::string histogram_spacing_to_string(HistogramSpacing spacing) {
    switch (spacing) {
        case HistogramSpacing::linear:
            return "linear";
        case HistogramSpacing::logarithmic:
            return "log";
    }
    throw std::runtime_error("unsupported histogram spacing enum");
}

Histogram::Histogram(std::size_t num_bins,
                     double min_value,
                     double max_value,
                     HistogramSpacing spacing)
    : min_value_(min_value), max_value_(max_value), spacing_(spacing), counts_(num_bins, 0.0) {
    if (num_bins == 0) {
        throw std::runtime_error("histogram requires at least one bin");
    }
    if (!(max_value > min_value)) {
        throw std::runtime_error("histogram max must be larger than min");
    }

    if (spacing_ == HistogramSpacing::linear) {
        bin_width_ = (max_value_ - min_value_) / static_cast<double>(counts_.size());
        return;
    }

    if (!(min_value_ > 0.0)) {
        throw std::runtime_error("logarithmic histogram requires min_value > 0");
    }
    log_min_value_ = std::log(min_value_);
    log_bin_width_ =
        (std::log(max_value_) - log_min_value_) / static_cast<double>(counts_.size());
}

void Histogram::fill(double value, double weight) {
    if (value < min_value_) {
        underflow_ += weight;
        return;
    }
    if (value >= max_value_) {
        overflow_ += weight;
        return;
    }
    counts_.at(index_for(value)) += weight;
}

void Histogram::write_csv(const std::string& path) const {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open histogram output: " + path);
    }

    out << "bin_index,low_edge,high_edge,center,count\n";
    for (std::size_t i = 0; i < counts_.size(); ++i) {
        const double low = low_edge_for(i);
        const double high = high_edge_for(i);
        out << i << ',' << low << ',' << high << ',' << center_for(i) << ',' << counts_[i] << "\n";
    }
    out << "underflow,,,," << underflow_ << "\n";
    out << "overflow,,,," << overflow_ << "\n";
}

double Histogram::in_range_total() const {
    return std::accumulate(counts_.begin(), counts_.end(), 0.0);
}

std::size_t Histogram::index_for(double value) const {
    if (spacing_ == HistogramSpacing::linear) {
        const double offset = (value - min_value_) / bin_width_;
        std::size_t index = static_cast<std::size_t>(offset);
        if (index >= counts_.size()) {
            index = counts_.size() - 1;
        }
        return index;
    }

    const double offset = (std::log(value) - log_min_value_) / log_bin_width_;
    std::size_t index = static_cast<std::size_t>(offset);
    if (index >= counts_.size()) {
        index = counts_.size() - 1;
    }
    return index;
}

double Histogram::low_edge_for(std::size_t index) const {
    if (spacing_ == HistogramSpacing::linear) {
        return min_value_ + static_cast<double>(index) * bin_width_;
    }
    return std::exp(log_min_value_ + static_cast<double>(index) * log_bin_width_);
}

double Histogram::high_edge_for(std::size_t index) const {
    if (spacing_ == HistogramSpacing::linear) {
        return low_edge_for(index) + bin_width_;
    }
    return std::exp(log_min_value_ + static_cast<double>(index + 1) * log_bin_width_);
}

double Histogram::center_for(std::size_t index) const {
    const double low = low_edge_for(index);
    const double high = high_edge_for(index);
    return spacing_ == HistogramSpacing::linear ? 0.5 * (low + high) : std::sqrt(low * high);
}

}  // namespace ic
