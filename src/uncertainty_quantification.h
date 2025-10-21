#pragma once

#include <vector>
#include <random>
#include <functional>
#include <memory>
#include <map>
#include <string>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <thread>
#include <future>
#include <mutex>

namespace physgrad {

struct UncertaintyParameter {
    std::string name;
    double mean;
    double std_dev;
    double min_value;
    double max_value;

    enum DistributionType {
        NORMAL,
        UNIFORM,
        LOG_NORMAL,
        BETA
    } distribution;

    UncertaintyParameter(const std::string& n, double m, double s,
                        double min_val = -1e6, double max_val = 1e6,
                        DistributionType dist = NORMAL)
        : name(n), mean(m), std_dev(s), min_value(min_val), max_value(max_val), distribution(dist) {}
};

class PriorDistribution {
public:
    virtual ~PriorDistribution() = default;
    virtual double logProbability(const std::vector<double>& params) const = 0;
    virtual std::vector<double> sample(std::mt19937& gen) const = 0;
    virtual size_t dimension() const = 0;
};

class GaussianPrior : public PriorDistribution {
private:
    std::vector<double> mean_;
    std::vector<double> std_dev_;

public:
    GaussianPrior(const std::vector<double>& mean, const std::vector<double>& std_dev)
        : mean_(mean), std_dev_(std_dev) {
        if (mean.size() != std_dev.size()) {
            throw std::invalid_argument("Mean and std_dev vectors must have same size");
        }
    }

    double logProbability(const std::vector<double>& params) const override {
        if (params.size() != mean_.size()) return -std::numeric_limits<double>::infinity();

        double log_prob = 0.0;
        for (size_t i = 0; i < params.size(); ++i) {
            double z = (params[i] - mean_[i]) / std_dev_[i];
            log_prob -= 0.5 * z * z + std::log(std_dev_[i] * std::sqrt(2.0 * M_PI));
        }
        return log_prob;
    }

    std::vector<double> sample(std::mt19937& gen) const override {
        std::vector<double> sample(mean_.size());
        for (size_t i = 0; i < mean_.size(); ++i) {
            std::normal_distribution<double> dist(mean_[i], std_dev_[i]);
            sample[i] = dist(gen);
        }
        return sample;
    }

    size_t dimension() const override { return mean_.size(); }
};

class UniformPrior : public PriorDistribution {
private:
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;

public:
    UniformPrior(const std::vector<double>& lower, const std::vector<double>& upper)
        : lower_bounds_(lower), upper_bounds_(upper) {
        if (lower.size() != upper.size()) {
            throw std::invalid_argument("Lower and upper bounds must have same size");
        }
    }

    double logProbability(const std::vector<double>& params) const override {
        if (params.size() != lower_bounds_.size()) return -std::numeric_limits<double>::infinity();

        double log_prob = 0.0;
        for (size_t i = 0; i < params.size(); ++i) {
            if (params[i] < lower_bounds_[i] || params[i] > upper_bounds_[i]) {
                return -std::numeric_limits<double>::infinity();
            }
            log_prob -= std::log(upper_bounds_[i] - lower_bounds_[i]);
        }
        return log_prob;
    }

    std::vector<double> sample(std::mt19937& gen) const override {
        std::vector<double> sample(lower_bounds_.size());
        for (size_t i = 0; i < lower_bounds_.size(); ++i) {
            std::uniform_real_distribution<double> dist(lower_bounds_[i], upper_bounds_[i]);
            sample[i] = dist(gen);
        }
        return sample;
    }

    size_t dimension() const override { return lower_bounds_.size(); }
};

struct MonteCarloSample {
    std::vector<double> parameters;
    std::vector<double> outputs;
    double log_likelihood;
    double weight;

    MonteCarloSample(const std::vector<double>& params, const std::vector<double>& out,
                    double log_like = 0.0, double w = 1.0)
        : parameters(params), outputs(out), log_likelihood(log_like), weight(w) {}
};

struct UncertaintyStatistics {
    std::vector<double> mean;
    std::vector<double> variance;
    std::vector<double> std_dev;
    std::vector<double> percentile_5;
    std::vector<double> percentile_95;
    std::vector<double> median;
    std::vector<std::vector<double>> covariance_matrix;

    double total_variance() const {
        double total = 0.0;
        for (double var : variance) total += var;
        return total;
    }

    std::vector<double> confidence_interval_95() const {
        std::vector<double> ci_width(mean.size());
        for (size_t i = 0; i < mean.size(); ++i) {
            ci_width[i] = percentile_95[i] - percentile_5[i];
        }
        return ci_width;
    }
};

class BayesianInference {
private:
    std::shared_ptr<PriorDistribution> prior_;
    std::function<double(const std::vector<double>&, const std::vector<double>&)> likelihood_func_;
    std::vector<MonteCarloSample> samples_;
    std::mt19937 generator_;
    mutable std::mutex samples_mutex_;

public:
    BayesianInference(std::shared_ptr<PriorDistribution> prior,
                     std::function<double(const std::vector<double>&, const std::vector<double>&)> likelihood)
        : prior_(prior), likelihood_func_(likelihood), generator_(std::random_device{}()) {}

    void setLikelihoodFunction(std::function<double(const std::vector<double>&, const std::vector<double>&)> func) {
        likelihood_func_ = func;
    }

    std::vector<MonteCarloSample> metropolisHastings(
        const std::vector<double>& observed_data,
        size_t num_samples,
        double step_size = 0.1,
        size_t burn_in = 1000) {

        std::vector<MonteCarloSample> chain;
        chain.reserve(num_samples + burn_in);

        auto current_params = prior_->sample(generator_);
        double current_log_posterior = computeLogPosterior(current_params, observed_data);

        size_t accepted = 0;
        std::normal_distribution<double> proposal_dist(0.0, step_size);

        for (size_t i = 0; i < num_samples + burn_in; ++i) {
            auto proposed_params = current_params;
            for (auto& param : proposed_params) {
                param += proposal_dist(generator_);
            }

            double proposed_log_posterior = computeLogPosterior(proposed_params, observed_data);
            double acceptance_ratio = std::exp(proposed_log_posterior - current_log_posterior);

            std::uniform_real_distribution<double> uniform(0.0, 1.0);
            if (uniform(generator_) < acceptance_ratio) {
                current_params = proposed_params;
                current_log_posterior = proposed_log_posterior;
                accepted++;
            }

            if (i >= burn_in) {
                auto outputs = evaluateModel(current_params);
                chain.emplace_back(current_params, outputs, current_log_posterior);
            }
        }

        std::cout << "MCMC acceptance rate: " << (double)accepted / (num_samples + burn_in) << std::endl;
        return chain;
    }

    std::vector<MonteCarloSample> importanceSampling(
        const std::vector<double>& observed_data,
        size_t num_samples) {

        std::vector<MonteCarloSample> samples;
        samples.reserve(num_samples);

        for (size_t i = 0; i < num_samples; ++i) {
            auto params = prior_->sample(generator_);
            auto outputs = evaluateModel(params);
            double log_likelihood = likelihood_func_(outputs, observed_data);
            double weight = std::exp(log_likelihood);

            samples.emplace_back(params, outputs, log_likelihood, weight);
        }

        double total_weight = 0.0;
        for (const auto& sample : samples) {
            total_weight += sample.weight;
        }

        for (auto& sample : samples) {
            sample.weight /= total_weight;
        }

        return samples;
    }

    UncertaintyStatistics computeStatistics(const std::vector<MonteCarloSample>& samples) const {
        if (samples.empty()) throw std::runtime_error("No samples provided");

        size_t output_dim = samples[0].outputs.size();
        UncertaintyStatistics stats;

        stats.mean.resize(output_dim, 0.0);
        stats.variance.resize(output_dim, 0.0);
        stats.std_dev.resize(output_dim, 0.0);
        stats.percentile_5.resize(output_dim);
        stats.percentile_95.resize(output_dim);
        stats.median.resize(output_dim);
        stats.covariance_matrix.resize(output_dim, std::vector<double>(output_dim, 0.0));

        double total_weight = 0.0;
        for (const auto& sample : samples) {
            total_weight += sample.weight;
        }

        for (const auto& sample : samples) {
            double norm_weight = sample.weight / total_weight;
            for (size_t i = 0; i < output_dim; ++i) {
                stats.mean[i] += sample.outputs[i] * norm_weight;
            }
        }

        for (const auto& sample : samples) {
            double norm_weight = sample.weight / total_weight;
            for (size_t i = 0; i < output_dim; ++i) {
                double diff = sample.outputs[i] - stats.mean[i];
                stats.variance[i] += diff * diff * norm_weight;

                for (size_t j = 0; j < output_dim; ++j) {
                    double diff_j = sample.outputs[j] - stats.mean[j];
                    stats.covariance_matrix[i][j] += diff * diff_j * norm_weight;
                }
            }
        }

        for (size_t i = 0; i < output_dim; ++i) {
            stats.std_dev[i] = std::sqrt(stats.variance[i]);

            std::vector<double> values;
            for (const auto& sample : samples) {
                values.push_back(sample.outputs[i]);
            }
            std::sort(values.begin(), values.end());

            stats.percentile_5[i] = values[static_cast<size_t>(0.05 * values.size())];
            stats.percentile_95[i] = values[static_cast<size_t>(0.95 * values.size())];
            stats.median[i] = values[values.size() / 2];
        }

        return stats;
    }

    std::vector<MonteCarloSample> parallelSampling(
        const std::vector<double>& observed_data,
        size_t total_samples,
        size_t num_threads = std::thread::hardware_concurrency()) {

        if (num_threads == 0) num_threads = 1;

        std::vector<std::future<std::vector<MonteCarloSample>>> futures;
        size_t samples_per_thread = total_samples / num_threads;
        size_t remaining_samples = total_samples % num_threads;

        for (size_t t = 0; t < num_threads; ++t) {
            size_t thread_samples = samples_per_thread + (t < remaining_samples ? 1 : 0);

            futures.push_back(std::async(std::launch::async, [this, observed_data, thread_samples]() {
                return this->importanceSampling(observed_data, thread_samples);
            }));
        }

        std::vector<MonteCarloSample> all_samples;
        for (auto& future : futures) {
            auto thread_samples = future.get();
            all_samples.insert(all_samples.end(), thread_samples.begin(), thread_samples.end());
        }

        return all_samples;
    }

private:
    double computeLogPosterior(const std::vector<double>& params, const std::vector<double>& observed_data) {
        double log_prior = prior_->logProbability(params);
        if (std::isinf(log_prior)) return log_prior;

        auto model_outputs = evaluateModel(params);
        double log_likelihood = likelihood_func_(model_outputs, observed_data);

        return log_prior + log_likelihood;
    }

    std::vector<double> evaluateModel(const std::vector<double>& params) {
        if (params.size() < 2) return {0.0, 0.0};

        double k = params[0]; // spring constant
        double m = params[1]; // mass

        if (m <= 0.0 || k <= 0.0) return {0.0, 0.0};

        double omega = std::sqrt(k / m);
        double period = 2.0 * M_PI / omega;

        return {omega, period};
    }
};

class SensitivityAnalysis {
private:
    std::function<std::vector<double>(const std::vector<double>&)> model_func_;
    std::vector<UncertaintyParameter> parameters_;

public:
    SensitivityAnalysis(std::function<std::vector<double>(const std::vector<double>&)> model,
                       const std::vector<UncertaintyParameter>& params)
        : model_func_(model), parameters_(params) {}

    struct SobolIndices {
        std::vector<double> first_order;
        std::vector<double> total_order;
        std::vector<std::vector<double>> second_order;

        SobolIndices(size_t num_params) {
            first_order.resize(num_params, 0.0);
            total_order.resize(num_params, 0.0);
            second_order.resize(num_params, std::vector<double>(num_params, 0.0));
        }
    };

    SobolIndices computeSobolIndices(size_t num_samples = 10000) {
        SobolIndices indices(parameters_.size());
        std::mt19937 gen(std::random_device{}());

        std::vector<std::vector<double>> sample_matrix_a = generateSamples(num_samples, gen);
        std::vector<std::vector<double>> sample_matrix_b = generateSamples(num_samples, gen);

        std::vector<std::vector<double>> outputs_a = evaluateModelBatch(sample_matrix_a);
        std::vector<std::vector<double>> outputs_b = evaluateModelBatch(sample_matrix_b);

        size_t output_dim = outputs_a[0].size();

        for (size_t out_idx = 0; out_idx < output_dim; ++out_idx) {
            double mean_a = computeMean(outputs_a, out_idx);
            double mean_b = computeMean(outputs_b, out_idx);
            double variance_total = computeVariance(outputs_a, out_idx);

            if (variance_total <= 1e-12) {
                // Handle case where variance is too small
                for (size_t param_idx = 0; param_idx < parameters_.size(); ++param_idx) {
                    indices.first_order[param_idx] = 0.0;
                    indices.total_order[param_idx] = 0.0;
                }
                continue;
            }

            for (size_t param_idx = 0; param_idx < parameters_.size(); ++param_idx) {
                auto sample_matrix_c = sample_matrix_b;
                for (size_t i = 0; i < num_samples; ++i) {
                    sample_matrix_c[i][param_idx] = sample_matrix_a[i][param_idx];
                }

                std::vector<std::vector<double>> outputs_c = evaluateModelBatch(sample_matrix_c);

                // Improved Sobol index calculation
                double sum_ab = 0.0, sum_bc = 0.0;
                for (size_t i = 0; i < num_samples; ++i) {
                    sum_ab += outputs_a[i][out_idx] * outputs_b[i][out_idx];
                    sum_bc += outputs_b[i][out_idx] * outputs_c[i][out_idx];
                }

                double f0_squared = mean_a * mean_b;
                double variance_conditional = (sum_bc / num_samples - f0_squared);
                double variance_complementary = variance_total - (sum_ab / num_samples - f0_squared);

                indices.first_order[param_idx] = std::max(0.0, std::min(1.0, variance_conditional / variance_total));
                indices.total_order[param_idx] = std::max(0.0, std::min(1.0, variance_complementary / variance_total));
            }
        }

        return indices;
    }

private:
    std::vector<std::vector<double>> generateSamples(size_t num_samples, std::mt19937& gen) {
        std::vector<std::vector<double>> samples(num_samples, std::vector<double>(parameters_.size()));

        for (size_t i = 0; i < num_samples; ++i) {
            for (size_t j = 0; j < parameters_.size(); ++j) {
                const auto& param = parameters_[j];

                switch (param.distribution) {
                    case UncertaintyParameter::NORMAL: {
                        std::normal_distribution<double> dist(param.mean, param.std_dev);
                        samples[i][j] = std::clamp(dist(gen), param.min_value, param.max_value);
                        break;
                    }
                    case UncertaintyParameter::UNIFORM: {
                        std::uniform_real_distribution<double> dist(param.min_value, param.max_value);
                        samples[i][j] = dist(gen);
                        break;
                    }
                    case UncertaintyParameter::LOG_NORMAL: {
                        std::lognormal_distribution<double> dist(param.mean, param.std_dev);
                        samples[i][j] = std::clamp(dist(gen), param.min_value, param.max_value);
                        break;
                    }
                    default:
                        samples[i][j] = param.mean;
                }
            }
        }

        return samples;
    }

    std::vector<std::vector<double>> evaluateModelBatch(const std::vector<std::vector<double>>& samples) {
        std::vector<std::vector<double>> outputs;
        outputs.reserve(samples.size());

        for (const auto& sample : samples) {
            outputs.push_back(model_func_(sample));
        }

        return outputs;
    }

    double computeMean(const std::vector<std::vector<double>>& outputs, size_t output_index) {
        double mean = 0.0;
        for (const auto& output : outputs) {
            mean += output[output_index];
        }
        return mean / outputs.size();
    }

    double computeVariance(const std::vector<std::vector<double>>& outputs, size_t output_index) {
        double mean = computeMean(outputs, output_index);

        double variance = 0.0;
        for (const auto& output : outputs) {
            double diff = output[output_index] - mean;
            variance += diff * diff;
        }
        variance /= outputs.size();

        return variance;
    }
};

class UncertaintyPropagation {
private:
    std::function<std::vector<double>(const std::vector<double>&)> model_func_;
    std::vector<UncertaintyParameter> input_uncertainties_;

public:
    UncertaintyPropagation(std::function<std::vector<double>(const std::vector<double>&)> model,
                          const std::vector<UncertaintyParameter>& uncertainties)
        : model_func_(model), input_uncertainties_(uncertainties) {}

    UncertaintyStatistics monteCarloAnalysis(size_t num_samples = 10000) {
        std::mt19937 gen(std::random_device{}());
        std::vector<std::vector<double>> outputs;
        outputs.reserve(num_samples);

        for (size_t i = 0; i < num_samples; ++i) {
            std::vector<double> input_sample(input_uncertainties_.size());

            for (size_t j = 0; j < input_uncertainties_.size(); ++j) {
                const auto& param = input_uncertainties_[j];

                switch (param.distribution) {
                    case UncertaintyParameter::NORMAL: {
                        std::normal_distribution<double> dist(param.mean, param.std_dev);
                        input_sample[j] = std::clamp(dist(gen), param.min_value, param.max_value);
                        break;
                    }
                    case UncertaintyParameter::UNIFORM: {
                        std::uniform_real_distribution<double> dist(param.min_value, param.max_value);
                        input_sample[j] = dist(gen);
                        break;
                    }
                    case UncertaintyParameter::LOG_NORMAL: {
                        std::lognormal_distribution<double> dist(param.mean, param.std_dev);
                        input_sample[j] = std::clamp(dist(gen), param.min_value, param.max_value);
                        break;
                    }
                    default:
                        input_sample[j] = param.mean;
                }
            }

            auto result = model_func_(input_sample);
            outputs.push_back(result);
        }

        if (outputs.empty() || outputs[0].empty()) {
            throw std::runtime_error("Model function produced no valid outputs");
        }

        size_t output_dim = outputs[0].size();
        UncertaintyStatistics stats;

        stats.mean.resize(output_dim, 0.0);
        stats.variance.resize(output_dim, 0.0);
        stats.std_dev.resize(output_dim, 0.0);
        stats.percentile_5.resize(output_dim);
        stats.percentile_95.resize(output_dim);
        stats.median.resize(output_dim);
        stats.covariance_matrix.resize(output_dim, std::vector<double>(output_dim, 0.0));

        // Compute means
        for (const auto& output : outputs) {
            for (size_t i = 0; i < output_dim; ++i) {
                stats.mean[i] += output[i];
            }
        }
        for (size_t i = 0; i < output_dim; ++i) {
            stats.mean[i] /= num_samples;
        }

        // Compute variances and covariances
        for (const auto& output : outputs) {
            for (size_t i = 0; i < output_dim; ++i) {
                double diff_i = output[i] - stats.mean[i];
                stats.variance[i] += diff_i * diff_i;

                for (size_t j = 0; j < output_dim; ++j) {
                    double diff_j = output[j] - stats.mean[j];
                    stats.covariance_matrix[i][j] += diff_i * diff_j;
                }
            }
        }

        for (size_t i = 0; i < output_dim; ++i) {
            stats.variance[i] /= num_samples;
            stats.std_dev[i] = std::sqrt(stats.variance[i]);

            for (size_t j = 0; j < output_dim; ++j) {
                stats.covariance_matrix[i][j] /= num_samples;
            }
        }

        // Compute percentiles
        for (size_t i = 0; i < output_dim; ++i) {
            std::vector<double> values;
            for (const auto& output : outputs) {
                values.push_back(output[i]);
            }
            std::sort(values.begin(), values.end());

            stats.percentile_5[i] = values[static_cast<size_t>(0.05 * values.size())];
            stats.percentile_95[i] = values[static_cast<size_t>(0.95 * values.size())];
            stats.median[i] = values[values.size() / 2];
        }

        return stats;
    }

    std::map<std::string, double> computeParameterContributions(size_t num_samples = 1000) {
        std::map<std::string, double> contributions;

        auto baseline_output = model_func_(getMeanParameters());
        double baseline_variance = computeBaselineVariance(num_samples);

        for (size_t param_idx = 0; param_idx < input_uncertainties_.size(); ++param_idx) {
            double param_variance = computeParameterVariance(param_idx, num_samples);
            double contribution = param_variance / baseline_variance;
            contributions[input_uncertainties_[param_idx].name] = contribution;
        }

        return contributions;
    }

private:
    std::vector<double> getMeanParameters() const {
        std::vector<double> means;
        for (const auto& param : input_uncertainties_) {
            means.push_back(param.mean);
        }
        return means;
    }

    double computeBaselineVariance(size_t num_samples) {
        std::mt19937 gen(std::random_device{}());
        std::vector<double> outputs;

        for (size_t i = 0; i < num_samples; ++i) {
            std::vector<double> sample = getMeanParameters();
            for (size_t j = 0; j < input_uncertainties_.size(); ++j) {
                const auto& param = input_uncertainties_[j];
                std::normal_distribution<double> dist(param.mean, param.std_dev);
                sample[j] = dist(gen);
            }

            auto result = model_func_(sample);
            outputs.push_back(result[0]);
        }

        double mean = 0.0;
        for (double val : outputs) mean += val;
        mean /= outputs.size();

        double variance = 0.0;
        for (double val : outputs) {
            variance += (val - mean) * (val - mean);
        }
        return variance / outputs.size();
    }

    double computeParameterVariance(size_t param_index, size_t num_samples) {
        std::mt19937 gen(std::random_device{}());
        std::vector<double> outputs;

        for (size_t i = 0; i < num_samples; ++i) {
            std::vector<double> sample = getMeanParameters();

            const auto& param = input_uncertainties_[param_index];
            std::normal_distribution<double> dist(param.mean, param.std_dev);
            sample[param_index] = dist(gen);

            auto result = model_func_(sample);
            outputs.push_back(result[0]);
        }

        double mean = 0.0;
        for (double val : outputs) mean += val;
        mean /= outputs.size();

        double variance = 0.0;
        for (double val : outputs) {
            variance += (val - mean) * (val - mean);
        }
        return variance / outputs.size();
    }
};

} // namespace physgrad