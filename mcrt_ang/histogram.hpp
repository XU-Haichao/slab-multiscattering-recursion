#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace ic {

enum class HistogramSpacing {
    linear,
    logarithmic,
};

HistogramSpacing histogram_spacing_from_string(const std::string& value);
std::string histogram_spacing_to_string(HistogramSpacing spacing);

class Histogram {
public:
    Histogram(std::size_t num_bins,
              double min_value,
              double max_value,
              HistogramSpacing spacing = HistogramSpacing::linear);

    void fill(double value, double weight = 1.0);
    void write_csv(const std::string& path) const;

    [[nodiscard]] double in_range_total() const;
    [[nodiscard]] double underflow() const { return underflow_; }
    [[nodiscard]] double overflow() const { return overflow_; }
    [[nodiscard]] std::size_t num_bins() const { return counts_.size(); }
    [[nodiscard]] double count(std::size_t index) const { return counts_.at(index); }
    [[nodiscard]] double low_edge_for(std::size_t index) const;
    [[nodiscard]] double high_edge_for(std::size_t index) const;
    [[nodiscard]] double center_for(std::size_t index) const;

private:
    [[nodiscard]] std::size_t index_for(double value) const;

    double min_value_ = 0.0;
    double max_value_ = 0.0;
    double bin_width_ = 0.0;
    double log_min_value_ = 0.0;
    double log_bin_width_ = 0.0;
    HistogramSpacing spacing_ = HistogramSpacing::linear;
    std::vector<double> counts_;
    double underflow_ = 0.0;
    double overflow_ = 0.0;
};

}  // namespace ic
