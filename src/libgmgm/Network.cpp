/*
    This code is part of gmgm, copyright (C) 2020 Junhee Yoo

    gmgm is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gmgm is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gmgm.  If not, see <http://www.gnu.org/licenses/>.
*/
/*
    This source originated from Leela Zero (http://github.com/leela-zero/leela-zero
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors
*/

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <boost/utility.hpp>
#include <boost/format.hpp>
#include <boost/spirit/home/x3.hpp>
#include <Eigen/Dense>

#include "globals.h"

#include "half/half.hpp"
#include "zlib.h"

#include "Network.h"
#include "CPUPipe.h"
#include "Board.h"

namespace x3 = boost::spirit::x3;
using gmgm::globals::myprintf;

#ifndef USE_BLAS
// Eigen helpers
template <typename T>
using EigenVectorMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
#endif

template<class container>
void process_bn_var(container& weights) {
    constexpr float epsilon = 1e-5f;
    for (auto&& w : weights) {
        w = 1.0f / std::sqrt(w + epsilon);
    }
}

std::vector<float> Network::winograd_transform_f(const std::vector<float>& f,
                                                 const int outputs,
                                                 const int channels) {
    // F(4x4, 3x3) Winograd filter transformation
    // transpose(G.dot(f).dot(G.transpose()))
    // U matrix is transposed for better memory layout in SGEMM
    auto U = std::vector<float>(WINOGRAD_TILE * outputs * channels);
    const auto G = std::array<float, 3 * WINOGRAD_ALPHA>
                    { 1.0f,        0.0f,      0.0f,
                      -2.0f/3.0f, -SQ2/3.0f, -1.0f/3.0f,
                      -2.0f/3.0f,  SQ2/3.0f, -1.0f/3.0f,
                      1.0f/6.0f,   SQ2/6.0f,  1.0f/3.0f,
                      1.0f/6.0f,  -SQ2/6.0f,  1.0f/3.0f,
                      0.0f,        0.0f,      1.0f};

    auto temp = std::array<float, 3 * WINOGRAD_ALPHA>{};

    constexpr auto max_buffersize = 8;
    auto buffersize = max_buffersize;

    if (outputs % buffersize != 0) {
        buffersize = 1;
    }

    std::array<float, max_buffersize * WINOGRAD_ALPHA * WINOGRAD_ALPHA> buffer;

    for (auto c = 0; c < channels; c++) {
        for (auto o_b = 0; o_b < outputs/buffersize; o_b++) {
            for (auto bufferline = 0; bufferline < buffersize; bufferline++) {
                const auto o = o_b * buffersize + bufferline;

                for (auto i = 0; i < WINOGRAD_ALPHA; i++) {
                    for (auto j = 0; j < 3; j++) {
                        auto acc = 0.0f;
                        for (auto k = 0; k < 3; k++) {
                            acc += G[i*3 + k] * f[o*channels*9 + c*9 + k*3 + j];
                        }
                        temp[i*3 + j] = acc;
                    }
                }

                for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
                    for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                        auto acc = 0.0f;
                        for (auto k = 0; k < 3; k++) {
                            acc += temp[xi*3 + k] * G[nu*3 + k];
                        }
                        buffer[(xi * WINOGRAD_ALPHA + nu) * buffersize + bufferline] = acc;
                    }
                }
            }
            for (auto i = 0; i < WINOGRAD_ALPHA * WINOGRAD_ALPHA; i++) {
                for (auto entry = 0; entry < buffersize; entry++) {
                    const auto o = o_b * buffersize + entry;
                    U[i * outputs * channels
                      + c * outputs
                      + o] =
                    buffer[buffersize * i + entry];
                }
            }
        }
    }

    return U;
}

std::pair<int, int> Network::load_v1_network(std::istream& wtfile) {
    // Count size of the network
    myprintf("Detecting residual layers...");
    // We are version 1 or 2
    myprintf("v%d...", 1);

    // First line was the version number
    auto linecount = size_t{1};
    auto channels = 0;
    auto line = std::string{};
    while (std::getline(wtfile, line)) {
        auto iss = std::stringstream{line};
        // Third line of parameters are the convolution layer biases,
        // so this tells us the amount of channels in the residual layers.
        // We are assuming all layers have the same amount of filters.
        if (linecount == 2) {
            auto count = std::distance(std::istream_iterator<std::string>(iss),
                                       std::istream_iterator<std::string>());
            myprintf("%lu channels...", count);
            channels = count;
        }
        linecount++;
    }
    // 1 format id, 1 input layer (4 x weights), 14 ending weights,
    // the rest are residuals, every residual has 8 x weight lines
    auto residual_blocks = linecount - (1 + 4 + 14);

    if (residual_blocks % 8 != 0) {
        throw std::runtime_error("Inconsistent number of weights in the file.");
        return {0, 0};
    }
    residual_blocks /= 8;
    myprintf("%lu blocks.\n", residual_blocks);

    // Re-read file and process
    wtfile.clear();
    wtfile.seekg(0, std::ios::beg);

    // Get the file format id out of the way
    std::getline(wtfile, line);

    const auto plain_conv_layers = 1 + (residual_blocks * 2);
    const auto plain_conv_wts = plain_conv_layers * 4;
    linecount = 0;
    while (std::getline(wtfile, line)) {
        std::vector<float> weights;
        auto it_line = line.cbegin();
        const auto ok = phrase_parse(it_line, line.cend(),
                                     *x3::float_, x3::space, weights);
        if (!ok || it_line != line.cend()) {
            myprintf("\nFailed to parse weight file. Error on line %lu.\n",
                    linecount + 2); //+1 from version line, +1 from 0-indexing
            throw std::runtime_error("Invalid weight file format");
            return {0, 0};
        }
        if (linecount < plain_conv_wts) {
            if (linecount % 4 == 0) {
                m_fwd_weights->m_conv_weights.emplace_back(weights);
            } else if (linecount % 4 == 1) {
                // Redundant in our model, but they encode the
                // number of outputs so we have to read them in.
                m_fwd_weights->m_conv_biases.emplace_back(weights);
            } else if (linecount % 4 == 2) {
                m_fwd_weights->m_batchnorm_means.emplace_back(weights);
            } else if (linecount % 4 == 3) {
                process_bn_var(weights);
                m_fwd_weights->m_batchnorm_stddevs.emplace_back(weights);

                m_fwd_weights->m_squeeze_1.push_back({});
                m_fwd_weights->m_squeeze_2.push_back({});
            }
        } else {
            switch (linecount - plain_conv_wts) {
                case  0: m_fwd_weights->m_conv_pol_w = std::move(weights); break;
                case  1: m_fwd_weights->m_conv_pol_b = std::move(weights); break;
                case  2: m_fwd_weights->m_bn_pol_w1 = std::move(weights); break;
                case  3: m_fwd_weights->m_bn_pol_w2 = std::move(weights); break;
                case  4: /*if (weights.size() != OUTPUTS_POLICY
                                               * POTENTIAL_MOVES) {
                             myprintf("The weights file is not for %dx%d boards.\n",
                                      gmgm::BOARD_H, gmgm::BOARD_W);
                             return {0, 0};
                         }*/
                         m_fwd_weights->m_ip_pol_w = std::move(weights); break;
                case  5: m_fwd_weights->m_ip_pol_b = std::move(weights); break; 
                case  6: m_fwd_weights->m_conv_val_w = std::move(weights); break;
                case  7: m_fwd_weights->m_conv_val_b = std::move(weights); break;
                case  8: m_fwd_weights->m_bn_val_w1 = std::move(weights); break;
                case  9: m_fwd_weights->m_bn_val_w2 = std::move(weights); break;
                case 10: m_fwd_weights->m_ip_val_w = std::move(weights); break;
                case 11: m_fwd_weights->m_ip_val_b = std::move(weights); break;
                case 12: assert(m_ip2_val_w.size() == weights.size());
                         std::copy(cbegin(weights), cend(weights),
                                   begin(m_ip2_val_w)); break;
                case 13: assert(m_ip2_val_b.size() == weights.size());
                         std::copy(cbegin(weights), cend(weights),
                                   begin(m_ip2_val_b)); break;
                default:
                        break;
            }
        }
        linecount++;
    }
    process_bn_var(m_fwd_weights->m_bn_pol_w2);
    process_bn_var(m_fwd_weights->m_bn_val_w2);

    return {channels, static_cast<int>(residual_blocks)};
}


// SENET
std::pair<int, int> Network::load_v5_network(std::istream& wtfile) {
    // Count size of the network
    myprintf("Detecting residual layers...");
    myprintf("v%d...", 5);

    // First line was the version number
    auto linecount = size_t{1};
    auto channels = 0;
    auto line = std::string{};
    while (std::getline(wtfile, line)) {
        auto iss = std::stringstream{line};
        // Third line of parameters are the convolution layer biases,
        // so this tells us the amount of channels in the residual layers.
        // We are assuming all layers have the same amount of filters.
        if (linecount == 2) {
            auto count = std::distance(std::istream_iterator<std::string>(iss),
                                       std::istream_iterator<std::string>());
            myprintf("%lu channels...", count);
            channels = count;
        }
        linecount++;
    }
    // 1 format id, 1 input layer (4 x weights), 14 ending weights,
    // the rest are residuals, every residual has 8 x weight lines + 2 SE layers
    auto residual_blocks = linecount - (1 + 4 + 14);

    if (residual_blocks % 10 != 0) {
        throw std::runtime_error("Inconsistent number of weights in the file.");
        return {0, 0};
    }
    residual_blocks /= 10;
    myprintf("%lu blocks.\n", residual_blocks);

    // Re-read file and process
    wtfile.clear();
    wtfile.seekg(0, std::ios::beg);

    // Get the file format id out of the way
    std::getline(wtfile, line);

    const auto plain_conv_layers = 1 + (residual_blocks * 2);
    const auto plain_conv_wts = plain_conv_layers * 4 + residual_blocks * 2;

    linecount = 0;
    auto residual_index = 0;
    while (std::getline(wtfile, line)) {
        std::vector<float> weights;
        auto it_line = line.cbegin();
        const auto ok = phrase_parse(it_line, line.cend(),
                                     *x3::float_, x3::space, weights);
        if (!ok || it_line != line.cend()) {
            throw std::runtime_error("Invalid weight file format");
            return {0, 0};
        }
        if (linecount < plain_conv_wts) {
            if (residual_index % 6 == 0) {
                m_fwd_weights->m_conv_weights.emplace_back(weights);
                residual_index++;
            } else if (residual_index % 6 == 1) {
                // Redundant in our model, but they encode the
                // number of outputs so we have to read them in.
                m_fwd_weights->m_conv_biases.emplace_back(weights);
                residual_index++;
            } else if (residual_index % 6 == 2) {
                m_fwd_weights->m_batchnorm_means.emplace_back(weights);
                residual_index++;
            } else if (residual_index % 6 == 3) {
                process_bn_var(weights);
                m_fwd_weights->m_batchnorm_stddevs.emplace_back(weights);
                residual_index++;
                auto residual_num = residual_index / 6;
                if(residual_num == 0 || residual_num % 2 == 1) {
                    m_fwd_weights->m_squeeze_1.push_back({});
                    m_fwd_weights->m_squeeze_2.push_back({});
                    residual_index += 2;
                }
            } else if(residual_index % 6 == 4) {
                m_fwd_weights->m_squeeze_1.emplace_back(weights);
                residual_index++;
            } else if(residual_index % 6 == 5) {
                m_fwd_weights->m_squeeze_2.emplace_back(weights);
                residual_index++;
            }
        } else {
            switch (linecount - plain_conv_wts) {
                case  0: m_fwd_weights->m_conv_pol_w = std::move(weights); break;
                case  1: m_fwd_weights->m_conv_pol_b = std::move(weights); break;
                case  2: m_fwd_weights->m_bn_pol_w1 = std::move(weights); break;
                case  3: m_fwd_weights->m_bn_pol_w2 = std::move(weights); break;
                case  4: /*if (weights.size() != OUTPUTS_POLICY
                                               * POTENTIAL_MOVES) {
                             myprintf("The weights file is not for %dx%d boards.\n",
                                      gmgm::BOARD_H, gmgm::BOARD_W);
                             return {0, 0};
                         }*/
                         m_fwd_weights->m_ip_pol_w = std::move(weights); break;
                case  5: m_fwd_weights->m_ip_pol_b = std::move(weights); break; 
                case  6: m_fwd_weights->m_conv_val_w = std::move(weights); break;
                case  7: m_fwd_weights->m_conv_val_b = std::move(weights); break;
                case  8: m_fwd_weights->m_bn_val_w1 = std::move(weights); break;
                case  9: m_fwd_weights->m_bn_val_w2 = std::move(weights); break;
                case 10: m_fwd_weights->m_ip_val_w = std::move(weights); break;
                case 11: m_fwd_weights->m_ip_val_b = std::move(weights); break;
                case 12: assert(m_ip2_val_w.size() == weights.size());
                         std::copy(cbegin(weights), cend(weights),
                                   begin(m_ip2_val_w)); break;
                case 13: assert(m_ip2_val_b.size() == weights.size());
                         std::copy(cbegin(weights), cend(weights),
                                   begin(m_ip2_val_b)); break;
                default:
                        break;
            }
        }
        linecount++;
    }
    process_bn_var(m_fwd_weights->m_bn_pol_w2);
    process_bn_var(m_fwd_weights->m_bn_val_w2);

    assert(plain_conv_layers == m_fwd_weights->m_squeeze_1.size());
    assert(plain_conv_layers == m_fwd_weights->m_squeeze_2.size());
    assert(plain_conv_layers == m_fwd_weights->m_conv_weights.size());
    assert(plain_conv_layers == m_fwd_weights->m_batchnorm_stddevs.size());
    return {channels, static_cast<int>(residual_blocks)};
}

std::pair<int, int> Network::load_network_file(const std::string& filename) {
    // gzopen supports both gz and non-gz files, will decompress
    // or just read directly as needed.
    auto gzhandle = gzopen(filename.c_str(), "rb");
    if (gzhandle == nullptr) {
        throw std::runtime_error("Could not open weights file");
        return {0, 0};
    }
    // Stream the gz file in to a memory buffer stream.
    auto buffer = std::stringstream{};
    constexpr auto chunkBufferSize = 64 * 1024;
    std::vector<char> chunkBuffer(chunkBufferSize);
    while (true) {
        auto bytesRead = gzread(gzhandle, chunkBuffer.data(), chunkBufferSize);
        if (bytesRead == 0) break;
        if (bytesRead < 0) {
            gzclose(gzhandle);
            throw std::runtime_error("Failed to decompress or read file");
            return {0, 0};
        }
        assert(bytesRead <= chunkBufferSize);
        buffer.write(chunkBuffer.data(), bytesRead);
    }
    gzclose(gzhandle);

    // Read format version
    auto line = std::string{};
    auto format_version = -1;
    if (std::getline(buffer, line)) {
        auto iss = std::stringstream{line};
        // First line is the file format version id
        iss >> format_version;
        if (iss.fail() || (format_version != 1 && format_version != 5)) {
            throw std::runtime_error("Weights file is the wrong version");
            return {0, 0};
        } else if (format_version == 1) {
            return load_v1_network(buffer);
        } else if (format_version == 5) {
            return load_v5_network(buffer);
        }
    }
    return {0, 0};
}

std::unique_ptr<ForwardPipe>&& Network::init_net(int channels,
    std::unique_ptr<ForwardPipe>&& pipe) {

    pipe->initialize(channels);
    pipe->push_weights(WINOGRAD_ALPHA, INPUT_CHANNELS, channels, m_fwd_weights);

    return std::move(pipe);
}

void Network::initialize(const std::string & weightsfile) {
    myprintf("BLAS Core: built-in Eigen %d.%d.%d library.\n",
             EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);

    m_fwd_weights = std::make_shared<ForwardPipeWeights>();

    // Load network from file
    size_t channels, residual_blocks;
    std::tie(channels, residual_blocks) = load_network_file(weightsfile);
    if (channels == 0) {
        throw std::runtime_error("Could not load net");
    }

    auto weight_index = size_t{0};
    // Input convolution
    // Winograd transform convolution weights
    m_fwd_weights->m_conv_weights[weight_index] =
        winograd_transform_f(m_fwd_weights->m_conv_weights[weight_index],
                             channels, INPUT_CHANNELS);
    weight_index++;

    // Residual block convolutions
    for (auto i = size_t{0}; i < residual_blocks * 2; i++) {
        m_fwd_weights->m_conv_weights[weight_index] =
            winograd_transform_f(m_fwd_weights->m_conv_weights[weight_index],
                                 channels, channels);
        weight_index++;
    }

    // Biases are not calculated and are typically zero but some networks might
    // still have non-zero biases.
    // Move biases to batchnorm means to make the output match without having
    // to separately add the biases.
    auto bias_size = m_fwd_weights->m_conv_biases.size();
    for (auto i = size_t{0}; i < bias_size; i++) {
        auto means_size = m_fwd_weights->m_batchnorm_means[i].size();
        for (auto j = size_t{0}; j < means_size; j++) {
            m_fwd_weights->m_batchnorm_means[i][j] -= m_fwd_weights->m_conv_biases[i][j];
            m_fwd_weights->m_conv_biases[i][j] = 0.0f;
        }
    }

    for (auto i = size_t{0}; i < m_fwd_weights->m_bn_val_w1.size(); i++) {
        m_fwd_weights->m_bn_val_w1[i] -= m_fwd_weights->m_conv_val_b[i];
        m_fwd_weights->m_conv_val_b[i] = 0.0f;
    }

    for (auto i = size_t{0}; i < m_fwd_weights->m_bn_pol_w1.size(); i++) {
        m_fwd_weights->m_bn_pol_w1[i] -= m_fwd_weights->m_conv_pol_b[i];
        m_fwd_weights->m_conv_pol_b[i] = 0.0f;
    }

    if (false) {
        myprintf("Initializing CPU-only evaluation.\n");
        m_forward = init_net(channels, std::make_unique<CPUPipe>());
    } else {
        myprintf("Initializing GPU evaluation.\n");
        m_forward_cpu = init_net(channels, std::make_unique<CPUPipe>());
        m_forward = init_net(channels, std::make_unique<OpenCLScheduler<half_float::half>>());
        // m_forward = init_net(channels, std::make_unique<OpenCLScheduler<float>>());
    }

    // Need to estimate size before clearing up the pipe.
    get_estimated_size();
    m_fwd_weights.reset();
}

template<unsigned int inputs,
         unsigned int outputs,
         bool ReLU,
         size_t W>
std::vector<float> innerproduct(const std::vector<float>& input,
                                const std::array<float, W>& weights,
                                const std::array<float, outputs>& biases) {
    std::vector<float> output(outputs);

    EigenVectorMap<float> y(output.data(), outputs);
    y.noalias() =
        ConstEigenMatrixMap<float>(weights.data(),
                                   inputs,
                                   outputs).transpose()
        * ConstEigenVectorMap<float>(input.data(), inputs);
    const auto lambda_ReLU = [](const auto val) { return (val > 0.0f) ?
                                                          val : 0.0f; };
    for (unsigned int o = 0; o < outputs; o++) {
        auto val = biases[o] + output[o];
        if (ReLU) {
            val = lambda_ReLU(val);
        }
        output[o] = val;
    }

    return output;
}

template <size_t spatial_size>
void batchnorm(const size_t channels,
               std::vector<float>& data,
               const float* const means,
               const float* const stddivs,
               const float* const eltwise = nullptr) {
    const auto lambda_ReLU = [](const auto val) { return (val > 0.0f) ?
                                                          val : 0.0f; };
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto mean = means[c];
        const auto scale_stddiv = stddivs[c];
        const auto arr = &data[c * spatial_size];

        if (eltwise == nullptr) {
            // Classical BN
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = lambda_ReLU(scale_stddiv * (arr[b] - mean));
            }
        } else {
            // BN + residual add
            const auto res = &eltwise[c * spatial_size];
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = lambda_ReLU((scale_stddiv * (arr[b] - mean)) + res[b]);
            }
        }
    }
}

void Network::compare_net_outputs(const gmgm::PositionEval::RawResult& data,
                                  const gmgm::PositionEval::RawResult& ref) {
    // Calculates L2-norm between data and ref.
    constexpr auto max_error = 0.05f;

    auto error = 0.0f;

    for(auto x=0u; x<data.first.size(); x++) {
        const auto diff = data.first.at(x) - ref.first.at(x);
        error += diff * diff;
    }

    const auto diff_value = data.second - ref.second;
    error += diff_value * diff_value;

    error = std::sqrt(error);

    if (error > max_error || std::isnan(error)) {
        if(error_passed_threshold++ > 10) {
            printf("Error in OpenCL calculation: Update your device's OpenCL drivers "
                   "or reduce the amount of games played simultaneously.\n");
            throw std::runtime_error("OpenCL self-check mismatch.");
        }
    } else {
        while (true) {
            auto x = error_passed_threshold.load();
            if(x > 0) {
                auto success = error_passed_threshold.compare_exchange_weak(x, x-1);
                if(success) break;
            } else {
                break;
            }
        }
    }
}

std::vector<float> softmax(const std::vector<float>& input,
                           const float temperature = 1.0f) {
    auto output = std::vector<float>{};
    output.reserve(input.size());

    const auto alpha = *std::max_element(cbegin(input), cend(input));
    auto denom = 0.0f;

    for (const auto in_val : input) {
        auto val = std::exp((in_val - alpha) / temperature);
        denom += val;
        output.push_back(val);
    }

    for (auto& out : output) {
        out /= denom;
    }

    return output;
}

std::shared_ptr<gmgm::EvalResult> Network::evaluate_raw(gmgm::Board & state) {
    const auto d = extract_input_features(state);
    std::vector<float> input_data (66 * gmgm::BOARD_W * gmgm::BOARD_H, 0.0f);
    for(size_t i=0; i<input_data.size(); i++) {
        auto boardsize = (gmgm::BOARD_W * gmgm::BOARD_H);
        input_data[i] = d.features[i/boardsize][i%boardsize];
    }

    bool run_selfcheck = (m_forward_cpu != nullptr && m_randsource() % 10000 == 0);
    auto rawout = evaluate_raw(input_data);
    if (run_selfcheck) {
        auto rawout_cpu = __evaluate_raw(input_data, true);
        compare_net_outputs(*rawout, *rawout_cpu);
    }

    auto raw_to_result = [&state](auto & rawout) {
        auto ret = std::make_shared<gmgm::EvalResult>();
        auto lm = state.get_legal_moves();
        ret->value = rawout->second;
        ret->policy.reserve(lm.size());
        for(auto m : lm) {
            auto p = state.get_piece_on(m.yx_from);
            p = p % 16;
            ret->policy.emplace_back(m, rawout->first[
                    p * gmgm::BOARD_W * gmgm::BOARD_H
                    + (m.yx_to/10) * gmgm::BOARD_W + m.yx_to%10
                ]
            );
        }
        return ret;
    };

    return raw_to_result(rawout);
}

std::shared_ptr<gmgm::PositionEval::RawResult> Network::__evaluate_raw(const std::vector<float> & input_data, bool selfcheck) {
    std::vector<float> policy_data(OUTPUTS_POLICY);
    std::vector<float> value_data(OUTPUTS_VALUE);

    if (selfcheck) {
        assert(m_forward_cpu != nullptr);
        m_forward_cpu->forward(input_data, policy_data, value_data);
    } else {
        m_forward->forward(input_data, policy_data, value_data);
    }

    // plane 32~48 from input is 'legal moves'.
    // if illegal move (zero), subtract 1000 from policy_data before applying softmax
    // so that we can filter out any noise from invalid moves
    auto p = cbegin(input_data) + 32 * NUM_INTERSECTIONS;
    for(auto i=0u; i<policy_data.size(); i++) {
        if(*p++ < 0.5) {
            policy_data[i] -= 1000.0f;
        }
    }

    const auto outputs = softmax(policy_data, 1.0f);

    // relu
    for(auto i=0u; i<value_data.size(); i++) {
        value_data[i] = value_data[i] > 0 ? value_data[i] : 0;
    }

    const auto winrate_out =
        innerproduct<OUTPUTS_VALUE, 1, false>(value_data, m_ip2_val_w, m_ip2_val_b);

    const auto winrate = std::tanh(winrate_out[0]);

    auto ret = std::make_shared<gmgm::PositionEval::RawResult>();
    ret->first = std::move(outputs);
    ret->second = winrate;
    return ret;
}

std::shared_ptr<gmgm::PositionEval::RawResult> Network::evaluate_raw(const std::vector<float> & input_data) {
    return __evaluate_raw(input_data);
}

size_t Network::get_estimated_size() {
    if (estimated_size != 0) {
        return estimated_size;
    }
    auto result = size_t{0};

    const auto lambda_vector_size =  [](const std::vector<std::vector<float>> &v) {
        auto result = size_t{0};
        for (auto it = begin(v); it != end(v); ++it) {
            result += it->size() * sizeof(float);
        }
        return result;
    };

    result += lambda_vector_size(m_fwd_weights->m_conv_weights);
    result += lambda_vector_size(m_fwd_weights->m_conv_biases);
    result += lambda_vector_size(m_fwd_weights->m_batchnorm_means);
    result += lambda_vector_size(m_fwd_weights->m_batchnorm_stddevs);

    result += m_fwd_weights->m_conv_pol_w.size() * sizeof(float);
    result += m_fwd_weights->m_conv_pol_b.size() * sizeof(float);

    return estimated_size = result;
}

