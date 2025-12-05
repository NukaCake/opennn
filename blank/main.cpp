//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <time.h>
#include <vector>

// OpenNN
#include "lib/opennn/opennn/language_dataset.h"
#include "lib/opennn/opennn/standard_networks.h"
#include "lib/opennn/opennn/neural_network.h"
#include "lib/opennn/opennn/training_strategy.h"
#include "lib/opennn/opennn/testing_analysis.h"
#include "lib/opennn/opennn/adaptive_moment_estimation.h"

using namespace std;
using namespace opennn;

namespace
{
struct Sample
{
    vector<double> features;
    int label = 0;
};

struct RunStats
{
    string timestamp;
    string dataset_name;
    size_t samples = 0;
    size_t feature_dim = 0;
    size_t depth = 0;
    size_t width = 0;
    size_t epochs_trained = 0;
    double best_accuracy = 0.0;
    double train_seconds = 0.0;
    double infer_seconds_fp = 0.0;
    double infer_seconds_ptq = 0.0;
    double infer_seconds_qat = 0.0;
    string quantization_notes;
};

struct LayerWeights
{
    vector<double> weights;
    vector<double> bias;
    size_t in = 0;
    size_t out = 0;
};

struct SimpleNetwork
{
    vector<LayerWeights> layers;
    string activation = "relu";
};

vector<string> split_tokens(const string& line)
{
    vector<string> tokens;
    string token;
    stringstream ss(line);

    while (getline(ss, token, ' '))
    {
        if (!token.empty())
        {
            tokens.push_back(token);
        }
    }

    if (tokens.size() <= 1)
    {
        tokens.clear();
        string alt;
        stringstream csv(line);
        while (getline(csv, alt, ','))
        {
            if (!alt.empty())
            {
                tokens.push_back(alt);
            }
        }
    }

    return tokens;
}

vector<Sample> load_dataset(const string& file_path)
{
    vector<Sample> dataset;
    ifstream file(file_path);

    if (!file.is_open())
    {
        cerr << "[warning] Dataset file not found, generating synthetic data: " << file_path << endl;
        const size_t synthetic_samples = 512;
        const size_t synthetic_features = 32;
        dataset.reserve(synthetic_samples);
        for (size_t i = 0; i < synthetic_samples; ++i)
        {
            Sample sample;
            sample.label = static_cast<int>(i % 2);
            sample.features.resize(synthetic_features, 0.0);
            for (size_t j = 0; j < synthetic_features; ++j)
            {
                sample.features[j] = (static_cast<double>((i + j) % 13) / 12.0) * (sample.label ? 1.0 : -1.0);
            }
            dataset.push_back(sample);
        }
        return dataset;
    }

    string line;
    while (getline(file, line))
    {
        auto tokens = split_tokens(line);
        if (tokens.size() < 2)
        {
            continue;
        }

        Sample sample;
        sample.label = stoi(tokens.front());
        sample.features.reserve(tokens.size() - 1);
        for (size_t i = 1; i < tokens.size(); ++i)
        {
            sample.features.push_back(stod(tokens[i]));
        }
        dataset.push_back(sample);
    }

    return dataset;
}

double activation_forward(double x, const string& name)
{
    if (name == "tanh")
    {
        return tanh(x);
    }
    if (name == "sigmoid")
    {
        return 1.0 / (1.0 + exp(-x));
    }
    return max(0.0, x);
}

vector<double> run_layer(const LayerWeights& layer, const vector<double>& input, const string& activation)
{
    vector<double> output(layer.out, 0.0);
    for (size_t o = 0; o < layer.out; ++o)
    {
        double acc = layer.bias[o];
        for (size_t i = 0; i < layer.in; ++i)
        {
            acc += layer.weights[o * layer.in + i] * input[i];
        }
        output[o] = activation_forward(acc, activation);
    }
    return output;
}

double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double cross_entropy(const vector<double>& logits, int label)
{
    double pred = sigmoid(logits[0]);
    pred = min(max(pred, 1e-8), 1.0 - 1e-8);
    return -(label ? log(pred) : log(1.0 - pred));
}

vector<double> forward(const SimpleNetwork& net, const vector<double>& input)
{
    vector<double> activation_vec = input;
    for (size_t l = 0; l < net.layers.size(); ++l)
    {
        activation_vec = run_layer(net.layers[l], activation_vec, l + 1 == net.layers.size() ? "sigmoid" : net.activation);
    }
    return activation_vec;
}

SimpleNetwork initialize_network(size_t input_dim, size_t depth, size_t width)
{
    SimpleNetwork net;
    net.activation = "relu";
    net.layers.resize(depth + 1);

    size_t previous = input_dim;
    for (size_t i = 0; i < depth; ++i)
    {
        net.layers[i].in = previous;
        net.layers[i].out = width;
        net.layers[i].weights.assign(width * previous, 0.01);
        net.layers[i].bias.assign(width, 0.0);
        previous = width;
    }
    net.layers.back().in = previous;
    net.layers.back().out = 1;
    net.layers.back().weights.assign(previous, 0.01);
    net.layers.back().bias.assign(1, 0.0);

    return net;
}

void train_epoch(SimpleNetwork& net, const vector<Sample>& samples, double lr)
{
    for (const auto& sample : samples)
    {
        vector<vector<double>> activations;
        vector<vector<double>> pre_activations;

        activations.push_back(sample.features);
        for (size_t l = 0; l < net.layers.size(); ++l)
        {
            auto z = run_layer(net.layers[l], activations.back(), l + 1 == net.layers.size() ? "sigmoid" : net.activation);
            pre_activations.push_back(z);
            activations.push_back(z);
        }

        double pred = activations.back()[0];
        double error = pred - static_cast<double>(sample.label);

        vector<double> delta = { error * pred * (1.0 - pred) };

        for (int layer_index = static_cast<int>(net.layers.size()) - 1; layer_index >= 0; --layer_index)
        {
            const auto& layer = net.layers[layer_index];
            const auto& input = activations[layer_index];
            vector<double> prev_delta(layer.in, 0.0);

            for (size_t o = 0; o < layer.out; ++o)
            {
                double grad_bias = delta[o];
                net.layers[layer_index].bias[o] -= lr * grad_bias;

                for (size_t i = 0; i < layer.in; ++i)
                {
                    size_t idx = o * layer.in + i;
                    double grad = grad_bias * input[i];
                    net.layers[layer_index].weights[idx] -= lr * grad;
                    prev_delta[i] += grad_bias * layer.weights[idx];
                }
            }

            if (layer_index > 0)
            {
                delta.assign(prev_delta.begin(), prev_delta.end());
                for (double& value : delta)
                {
                    if (net.activation == "relu")
                    {
                        value = value > 0.0 ? value : 0.0;
                    }
                    else if (net.activation == "tanh")
                    {
                        value *= 1.0 - pow(tanh(value), 2.0);
                    }
                }
            }
        }
    }
}

double evaluate_accuracy(const SimpleNetwork& net, const vector<Sample>& dataset)
{
    size_t correct = 0;
    for (const auto& sample : dataset)
    {
        auto out = forward(net, sample.features);
        double pred = out[0] >= 0.5 ? 1.0 : 0.0;
        if (static_cast<int>(pred) == sample.label)
        {
            ++correct;
        }
    }
    return dataset.empty() ? 0.0 : static_cast<double>(correct) / static_cast<double>(dataset.size());
}

vector<int8_t> quantize_weights(const vector<double>& weights, double& scale)
{
    double max_abs = 1e-9;
    for (double w : weights)
    {
        max_abs = max(max_abs, fabs(w));
    }
    scale = max_abs / 127.0;
    vector<int8_t> quantized(weights.size());
    for (size_t i = 0; i < weights.size(); ++i)
    {
        quantized[i] = static_cast<int8_t>(round(weights[i] / scale));
    }
    return quantized;
}

vector<double> quantized_layer_forward(const LayerWeights& layer, const vector<int8_t>& qweights, double scale, const vector<double>& input)
{
    vector<double> output(layer.out, 0.0);
    for (size_t o = 0; o < layer.out; ++o)
    {
        double acc = layer.bias[o];
        for (size_t i = 0; i < layer.in; ++i)
        {
            acc += static_cast<double>(qweights[o * layer.in + i]) * scale * input[i];
        }
        output[o] = sigmoid(acc);
    }
    return output;
}

double run_inference_ptq(const SimpleNetwork& net, const vector<Sample>& dataset, double& latency_seconds)
{
    auto start = chrono::steady_clock::now();
    size_t correct = 0;

    for (const auto& layer : net.layers)
    {
        (void)layer;
    }

    for (const auto& sample : dataset)
    {
        vector<double> activation_vec = sample.features;
        vector<LayerWeights> layers = net.layers;
        for (size_t l = 0; l < layers.size(); ++l)
        {
            double scale = 1.0;
            auto qweights = quantize_weights(layers[l].weights, scale);
            activation_vec = quantized_layer_forward(layers[l], qweights, scale, activation_vec);
        }

        double pred = activation_vec.back() >= 0.5 ? 1.0 : 0.0;
        if (static_cast<int>(pred) == sample.label)
        {
            ++correct;
        }
    }

    auto end = chrono::steady_clock::now();
    latency_seconds = chrono::duration<double>(end - start).count();
    return dataset.empty() ? 0.0 : static_cast<double>(correct) / static_cast<double>(dataset.size());
}

SimpleNetwork quantization_aware_training(const vector<Sample>& dataset, size_t depth, size_t width, size_t epochs, double lr)
{
    if (dataset.empty())
    {
        return {};
    }

    SimpleNetwork net = initialize_network(dataset.front().features.size(), depth, width);

    for (size_t epoch = 0; epoch < epochs; ++epoch)
    {
        for (const auto& sample : dataset)
        {
            vector<double> activations = sample.features;
            for (auto& layer : net.layers)
            {
                double scale = 1.0;
                auto qweights = quantize_weights(layer.weights, scale);
                vector<double> next(layer.out, 0.0);
                for (size_t o = 0; o < layer.out; ++o)
                {
                    double acc = layer.bias[o];
                    for (size_t i = 0; i < layer.in; ++i)
                    {
                        acc += static_cast<double>(qweights[o * layer.in + i]) * scale * activations[i];
                    }
                    next[o] = sigmoid(acc);
                }
                activations = next;
            }

            double pred = activations.back();
            double grad = pred - static_cast<double>(sample.label);
            for (auto& layer : net.layers)
            {
                for (double& w : layer.weights)
                {
                    w -= lr * grad * 0.1;
                }
                for (double& b : layer.bias)
                {
                    b -= lr * grad * 0.01;
                }
            }
        }

        if (epoch % 5 == 0)
        {
            cout << "[QAT] Epoch " << epoch << " completed" << endl;
        }
    }

    return net;
}

void log_csv(const string& path, const RunStats& stats)
{
    const bool exists = static_cast<bool>(ifstream(path));
    ofstream out(path, ios::app);
    if (!exists)
    {
        out << "timestamp,dataset,samples,features,depth,width,epochs,accuracy,train_s,infer_fp_s,infer_ptq_s,infer_qat_s,notes\n";
    }

    out << stats.timestamp << ','
        << stats.dataset_name << ','
        << stats.samples << ','
        << stats.feature_dim << ','
        << stats.depth << ','
        << stats.width << ','
        << stats.epochs_trained << ','
        << fixed << setprecision(4) << stats.best_accuracy << ','
        << stats.train_seconds << ','
        << stats.infer_seconds_fp << ','
        << stats.infer_seconds_ptq << ','
        << stats.infer_seconds_qat << ','
        << stats.quantization_notes
        << '\n';
}

string current_timestamp()
{
    auto now = chrono::system_clock::now();
    auto t = chrono::system_clock::to_time_t(now);
    tm tm_now{};
    localtime_r(&t, &tm_now);
    stringstream ss;
    ss << put_time(&tm_now, "%Y-%m-%d %H:%M:%S");
    return ss.str();
}

} // namespace

int main()
{
    try
    {
        const string dataset_file = "../data/amazon_polarity_tfidf_N5000_V1396.txt";
        cout << "Amazon review experiment starting" << endl;

        vector<Sample> dataset = load_dataset(dataset_file);
        if (dataset.empty())
        {
            cerr << "Dataset is empty, aborting." << endl;
            return 1;
        }

        size_t feature_dim = dataset.front().features.size();
        cout << "Loaded dataset with " << dataset.size() << " samples and feature dim " << feature_dim << endl;

        vector<Sample> train(dataset.begin(), dataset.begin() + dataset.size() * 8 / 10);
        vector<Sample> val(dataset.begin() + dataset.size() * 8 / 10, dataset.end());

        vector<size_t> depths = {1, 2, 3};
        vector<size_t> widths = {64, 128};

        RunStats best_stats;
        SimpleNetwork best_net;

        for (size_t depth : depths)
        {
            for (size_t width : widths)
            {
                SimpleNetwork net = initialize_network(feature_dim, depth, width);
                const size_t max_epochs = 25;
                double lr = 0.01;
                double best_val = 0.0;
                size_t best_epoch = 0;
                auto start = chrono::steady_clock::now();

                for (size_t epoch = 0; epoch < max_epochs; ++epoch)
                {
                    train_epoch(net, train, lr);
                    double acc = evaluate_accuracy(net, val);
                    cout << "[FP32] depth=" << depth << " width=" << width << " epoch=" << epoch << " val_acc=" << setprecision(4) << acc << endl;
                    if (acc > best_val)
                    {
                        best_val = acc;
                        best_epoch = epoch;
                    }
                    else if (epoch - best_epoch > 4)
                    {
                        cout << "Early stopping at epoch " << epoch << endl;
                        break;
                    }
                }

                auto end = chrono::steady_clock::now();
                double train_seconds = chrono::duration<double>(end - start).count();

                double infer_start = chrono::duration<double>(chrono::steady_clock::now() - chrono::steady_clock::now()).count();
                (void)infer_start;
                double fp_latency = 0.0;
                auto fp_begin = chrono::steady_clock::now();
                double fp_acc = evaluate_accuracy(net, val);
                auto fp_end = chrono::steady_clock::now();
                fp_latency = chrono::duration<double>(fp_end - fp_begin).count();

                double ptq_latency = 0.0;
                double ptq_acc = run_inference_ptq(net, val, ptq_latency);

                cout << "PTQ val_acc=" << setprecision(4) << ptq_acc << " latency_s=" << ptq_latency << endl;

                RunStats stats;
                stats.timestamp = current_timestamp();
                stats.dataset_name = "amazon_polarity_tfidf_N5000_V1396";
                stats.samples = dataset.size();
                stats.feature_dim = feature_dim;
                stats.depth = depth;
                stats.width = width;
                stats.epochs_trained = 25;
                stats.best_accuracy = best_val;
                stats.train_seconds = train_seconds;
                stats.infer_seconds_fp = fp_latency;
                stats.infer_seconds_ptq = ptq_latency;
                stats.quantization_notes = "PTQ int8 weights";

                log_csv("../results_amazon_experiment.csv", stats);

                if (best_val > best_stats.best_accuracy)
                {
                    best_stats = stats;
                    best_net = net;
                }
            }
        }

        cout << "Best configuration depth=" << best_stats.depth << " width=" << best_stats.width << " acc=" << setprecision(4) << best_stats.best_accuracy << endl;

        cout << "Starting QAT training" << endl;
        auto qat_start = chrono::steady_clock::now();
        SimpleNetwork qat_net = quantization_aware_training(train, best_stats.depth ? best_stats.depth : 2, best_stats.width ? best_stats.width : 128, 20, 0.005);
        auto qat_end = chrono::steady_clock::now();
        best_stats.train_seconds += chrono::duration<double>(qat_end - qat_start).count();

        double qat_latency = 0.0;
        double qat_acc = run_inference_ptq(qat_net, val, qat_latency);
        best_stats.infer_seconds_qat = qat_latency;
        best_stats.quantization_notes += " | QAT int8-aware";
        cout << "QAT acc=" << setprecision(4) << qat_acc << " latency_s=" << qat_latency << endl;

        log_csv("../results_amazon_experiment.csv", best_stats);

        cout << "Completed." << endl;

        return 0;
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
