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


#ifndef NETWORK_H_INCLUDED
#define NETWORK_H_INCLUDED

#include <deque>
#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <random>

#include "PositionEval.h"
#include "Board.h"
#include "ForwardPipe.h"
#include "OpenCLScheduler.h"

// Winograd filter transformation changes 3x3 filters to M + 3 - 1
constexpr auto WINOGRAD_M = 4;
constexpr auto WINOGRAD_ALPHA = WINOGRAD_M + 3 - 1;
constexpr auto WINOGRAD_WTILES = gmgm::BOARD_W / WINOGRAD_M + (gmgm::BOARD_W % WINOGRAD_M != 0);
constexpr auto WINOGRAD_TILE = WINOGRAD_ALPHA * WINOGRAD_ALPHA;
constexpr auto WINOGRAD_P = WINOGRAD_WTILES * WINOGRAD_WTILES;
constexpr auto SQ2 = 1.4142135623730951f; // Square root of 2
constexpr auto NUM_INTERSECTIONS = gmgm::BOARD_W * gmgm::BOARD_H;
constexpr auto POTENTIAL_MOVES = NUM_INTERSECTIONS * 16;

class Network : public gmgm::PositionEval {
    using ForwardPipeWeights = ForwardPipe::ForwardPipeWeights;
private:
    std::shared_ptr<gmgm::PositionEval::RawResult> __evaluate_raw(const std::vector<float> & v, bool selfcheck = false);
public:
    using PolicyVertexPair = std::pair<float,int>;

    virtual std::shared_ptr<gmgm::EvalResult> evaluate_raw(gmgm::Board & b);
    virtual std::shared_ptr<gmgm::PositionEval::RawResult> evaluate_raw(const std::vector<float> & v);

    static constexpr auto INPUT_CHANNELS = 66;
    static constexpr auto OUTPUTS_POLICY = 16*NUM_INTERSECTIONS;
    static constexpr auto OUTPUTS_VALUE = 256;
    std::atomic<int> error_passed_threshold{0};

    void initialize(const std::string & weightsfile);


    static std::vector<float> gather_features(const gmgm::Board & state);

    size_t get_estimated_size();
    size_t get_estimated_cache_size();

    virtual ~Network() {}
private:
    std::pair<int, int> load_v1_network(std::istream& wtfile);
    std::pair<int, int> load_v5_network(std::istream& wtfile);
    std::pair<int, int> load_network_file(const std::string& filename);

    static std::vector<float> winograd_transform_f(const std::vector<float>& f,
                                                   const int outputs, const int channels);
    static std::vector<float> zeropad_U(const std::vector<float>& U,
                                        const int outputs, const int channels,
                                        const int outputs_pad, const int channels_pad);
    static void winograd_transform_in(const std::vector<float>& in,
                                      std::vector<float>& V,
                                      const int C);
    static void winograd_transform_out(const std::vector<float>& M,
                                       std::vector<float>& Y,
                                       const int K);
    static void winograd_convolve3(const int outputs,
                                   const std::vector<float>& input,
                                   const std::vector<float>& U,
                                   std::vector<float>& V,
                                   std::vector<float>& M,
                                   std::vector<float>& output);
    static void winograd_sgemm(const std::vector<float>& U,
                               const std::vector<float>& V,
                               std::vector<float>& M, const int C, const int K);
    std::unique_ptr<ForwardPipe>&& init_net(int channels,
                                            std::unique_ptr<ForwardPipe>&& pipe);
    std::unique_ptr<ForwardPipe> m_forward;
    void compare_net_outputs(const gmgm::PositionEval::RawResult& data, const gmgm::PositionEval::RawResult& ref);
    std::unique_ptr<ForwardPipe> m_forward_cpu;
    std::mt19937 m_randsource{1111};

    size_t estimated_size{0};

    // Residual tower
    std::shared_ptr<ForwardPipeWeights> m_fwd_weights;

    // Value head remainder that isn't in m_fwd_weights
    std::array<float, OUTPUTS_VALUE> m_ip2_val_w;
    std::array<float, 1> m_ip2_val_b;
};
#endif
