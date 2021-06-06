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


#include <Eigen/Dense>

#include "CPUPipe.h"
#include "Network.h"
#include "Im2Col.h"
#include "Board.h"

#ifndef USE_BLAS
// Eigen helpers
template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
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

void CPUPipe::initialize(int channels) {
    m_input_channels = channels;
}

void CPUPipe::winograd_transform_in(const std::vector<float>& in,
                                    std::vector<float>& V,
                                    const int C) {
    constexpr auto W = gmgm::BOARD_W;
    constexpr auto H = gmgm::BOARD_H;
    constexpr auto WTILES = WINOGRAD_WTILES;
    constexpr auto P = WINOGRAD_P;

    constexpr auto Wpad = 2 + WINOGRAD_M * WTILES;

    constexpr auto buffersize = 32;

    std::array<std::array<float, Wpad>, Wpad> in_pad{0.0f};

    std::array<float, buffersize * WINOGRAD_ALPHA * WINOGRAD_ALPHA> buffer;
    auto buffer_offset = 0;
    auto buffer_entries = 0;


    // multiple vector [i0..i5] by Bt and produce [o0..o5]
    // const auto Bt = std::array<float, WINOGRAD_TILE>
    //           {1.0f,  0.0f,     -5.0f/2.0f,  0.0f,      1.0f, 0.0f,
    //            0.0f, -SQ2,      -2.0f,       SQ2/2.0f,  1.0f, 0.0f,
    //            0.0f,  SQ2,      -2.0f,      -SQ2/2.0f,  1.0f, 0.0f,
    //            0.0f, -SQ2/2.0f, -1.0f/2.0f,  SQ2,       1.0f, 0.0f,
    //            0.0f,  SQ2/2.0f, -1.0f/2.0f, -SQ2,       1.0f, 0.0f,
    //            0.0f,  1.0f,      0.0f,      -5.0f/2.0f, 0.0f, 1.0f};
    auto multiply_bt = [](
        float & o0, float & o1, float & o2, float & o3, float & o4, float & o5,
        float i0, float i1, float i2, float i3, float i4, float i5
    ) {
        auto i3m1 = i1 * -SQ2 + i3 * (SQ2 / 2.0f);
        auto i4m2 = i2 * -2.0f + i4 * 1.0f;

        o0 = i0 + i2 * (-5.0f/2.0f) + i4;
        o1 = i3m1 + i4m2;
        o2 = -i3m1 + i4m2;

        auto i3m1_2 = i3 * (SQ2) + i1 * (-SQ2/2.0f);
        auto i4m2_2 = i2 * (-1.0f/2.0f) + i4;

        o3 = i3m1_2 + i4m2_2;
        o4 = -i3m1_2 + i4m2_2;

        o5 = i1 + i3 * (-5.0f/2.0f) + i5;
    };

    for (auto ch = 0; ch < C; ch++) {
        for (auto yin = 0; yin < H; yin++) {
            for (auto xin = 0; xin < W; xin++) {
                in_pad[yin + 1][xin + 1] = in[ch*(W*H) + yin*W + xin];
            }
        }
        for (auto block_y = 0; block_y < WTILES; block_y++) {
            // Tiles overlap by 2
            const auto yin = WINOGRAD_M * block_y;
            for (auto block_x = 0; block_x < WTILES; block_x++) {
                const auto xin = WINOGRAD_M * block_x;
#define DECL_T1(XX) \
                float T1_##XX##_0, T1_##XX##_1, T1_##XX##_2, T1_##XX##_3, T1_##XX##_4, T1_##XX##_5;
                DECL_T1(0)
                DECL_T1(1)
                DECL_T1(2)
                DECL_T1(3)
                DECL_T1(4)
                DECL_T1(5)

                // Calculates transpose(B).x.B
#define MULTIPLY_BT(XX) \
                multiply_bt( \
                    T1_0_##XX, T1_1_##XX, T1_2_##XX, T1_3_##XX, T1_4_##XX, T1_5_##XX, \
                    in_pad[yin + 0][xin + XX], \
                    in_pad[yin + 1][xin + XX], \
                    in_pad[yin + 2][xin + XX], \
                    in_pad[yin + 3][xin + XX], \
                    in_pad[yin + 4][xin + XX], \
                    in_pad[yin + 5][xin + XX] \
                );
                MULTIPLY_BT(0)
                MULTIPLY_BT(1)
                MULTIPLY_BT(2)
                MULTIPLY_BT(3)
                MULTIPLY_BT(4)
                MULTIPLY_BT(5)

#define MULTIPLY_B(XX) \
                multiply_bt( \
                    buffer[buffersize * (XX * WINOGRAD_ALPHA + 0) + buffer_entries], \
                    buffer[buffersize * (XX * WINOGRAD_ALPHA + 1) + buffer_entries], \
                    buffer[buffersize * (XX * WINOGRAD_ALPHA + 2) + buffer_entries], \
                    buffer[buffersize * (XX * WINOGRAD_ALPHA + 3) + buffer_entries], \
                    buffer[buffersize * (XX * WINOGRAD_ALPHA + 4) + buffer_entries], \
                    buffer[buffersize * (XX * WINOGRAD_ALPHA + 5) + buffer_entries], \
                    T1_##XX##_0, T1_##XX##_1, T1_##XX##_2, T1_##XX##_3, T1_##XX##_4, T1_##XX##_5 \
                );
                MULTIPLY_B(0)
                MULTIPLY_B(1)
                MULTIPLY_B(2)
                MULTIPLY_B(3)
                MULTIPLY_B(4)
                MULTIPLY_B(5)

                if (buffer_entries == 0) {
                    buffer_offset = ch * P + block_y * WTILES + block_x;
                }
                buffer_entries++;

                if (buffer_entries >= buffersize ||
                    (ch == C - 1 && block_x == WTILES - 1 && block_y == WTILES - 1)) {

                    for (auto i = 0; i < WINOGRAD_ALPHA * WINOGRAD_ALPHA; i++) {
                        for (auto entry = 0; entry < buffer_entries; entry++) {
                            V[i*C*P + buffer_offset + entry] = buffer[i*buffersize + entry];
                        }
                    }
                    buffer_entries = 0;
                }
            }
        }
    }
}

void CPUPipe::winograd_sgemm(const std::vector<float>& U,
                             const std::vector<float>& V,
                             std::vector<float>& M,
                             const int C, const int K) {
    constexpr auto P = WINOGRAD_P;

    for (auto b = 0; b < WINOGRAD_TILE; b++) {
        const auto offset_u = b * K * C;
        const auto offset_v = b * C * P;
        const auto offset_m = b * K * P;
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    K, P, C,
                    1.0f,
                    &U[offset_u], K,
                    &V[offset_v], P,
                    0.0f,
                    &M[offset_m], P);
#else
        auto C_mat = EigenMatrixMap<float>(M.data() + offset_m, P, K);
        C_mat.noalias() =
           ConstEigenMatrixMap<float>(V.data() + offset_v, P, C)
            * ConstEigenMatrixMap<float>(U.data() + offset_u, K, C).transpose();
#endif
    }
}

void CPUPipe::winograd_transform_out(const std::vector<float>& M,
                                     std::vector<float>& Y,
                                     const int K) {
    constexpr auto W = gmgm::BOARD_W;
    constexpr auto H = gmgm::BOARD_H;
    constexpr auto WTILES = WINOGRAD_WTILES;
    constexpr auto P = WINOGRAD_P;

    // multiple vector [i0..i5] by At and produce [o0..o3]
    // const auto At = std::array<float, WINOGRAD_ALPHA * WINOGRAD_M>
    //       {1.0f, 1.0f,      1.0f,       1.0f,      1.0f,     0.0f,
    //        0.0f, SQ2/2.0f, -SQ2/2.0f,   SQ2,      -SQ2,      0.0f,
    //        0.0f, 1.0f/2.0f, 1.0f/2.0f,  2.0f,      2.0f,     0.0f,
    //        0.0f, SQ2/4.0f, -SQ2/4.0f,   2.0f*SQ2, -2.0f*SQ2, 1.0f};
    auto multiply_at = [](
        float & o0, float & o1, float & o2, float & o3,
        float i0, float i1, float i2, float i3, float i4, float i5
    ) {
        auto t1p2 = (i1 + i2) * (1.0f / 2.0f);
        auto t1m2 = (i1 - i2) * (SQ2/4.0f);
        auto t3p4 = i3 + i4;
        auto t3m4 = (i3 - i4) * (SQ2);

        o0 = i0 + t1p2 + t1p2 + t3p4;
        o1 = t1m2 + t1m2 + t3m4;
        o2 = t1p2 + t3p4 + t3p4;
        o3 = t1m2 + t3m4 + t3m4 + i5;
    };

    for (auto k = 0; k < K; k++) {
        for (auto block_x = 0; block_x < WTILES; block_x++) {
            const auto x = WINOGRAD_M * block_x;
            for (auto block_y = 0; block_y < WTILES; block_y++) {
                const auto y = WINOGRAD_M * block_y;

                const auto b = block_y * WTILES + block_x;
                using WinogradTile =
                    std::array<std::array<float, WINOGRAD_ALPHA>, WINOGRAD_ALPHA>;
                WinogradTile temp_m;
                for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
                    for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                        temp_m[xi][nu] =
                            M[(xi*WINOGRAD_ALPHA + nu)*K*P + k*P + b];
                    }
                }
                std::array<std::array<float, WINOGRAD_ALPHA>, WINOGRAD_M> temp;
                std::array<std::array<float, WINOGRAD_M>, WINOGRAD_M> o;

                // Calculates transpose(A).temp_m.A
                for (auto j = 0; j < WINOGRAD_ALPHA; j++){
                    multiply_at(
                        temp[0][j], temp[1][j], temp[2][j], temp[3][j],
                        temp_m[0][j], temp_m[1][j], temp_m[2][j], temp_m[3][j], temp_m[4][j], temp_m[5][j]
                    );
                }

                for (auto i = 0; i < WINOGRAD_M; i++){
                    multiply_at(
                        o[i][0], o[i][1], o[i][2], o[i][3],
                        temp[i][0], temp[i][1], temp[i][2], temp[i][3], temp[i][4], temp[i][5]
                    );
                }

                const auto y_ind = k * H * W + y * W + x;
                for (auto i = 0; i < WINOGRAD_M; i++) {
                    for (auto j = 0; j < WINOGRAD_M; j++) {
                        if (y + i < H && x + j < W) {
                            Y[y_ind + i * W + j] = o[i][j];
                        }
                    }
                }
            }
        }
    }
}

void CPUPipe::winograd_convolve3(const int outputs,
                                 const std::vector<float>& input,
                                 const std::vector<float>& U,
                                 std::vector<float>& V,
                                 std::vector<float>& M,
                                 std::vector<float>& output) {

    constexpr unsigned int filter_len = WINOGRAD_ALPHA * WINOGRAD_ALPHA;
    const auto input_channels = U.size() / (outputs * filter_len);

    winograd_transform_in(input, V, input_channels);
    winograd_sgemm(U, V, M, input_channels, outputs);
    winograd_transform_out(M, output, outputs);
}

template<unsigned int filter_size>
void convolve(const size_t outputs,
              const std::vector<float>& input,
              const std::vector<float>& weights,
              const std::vector<float>& biases,
              std::vector<float>& output) {
    // The size of the board is defined at compile time
    constexpr unsigned int width = gmgm::BOARD_W;
    constexpr unsigned int height = gmgm::BOARD_H;
    constexpr auto num_intersections = width * height;
    constexpr auto filter_len = filter_size * filter_size;
    const auto input_channels = weights.size() / (biases.size() * filter_len);
    const auto filter_dim = filter_len * input_channels;
    assert(outputs * num_intersections == output.size());

    std::vector<float> col(filter_dim * width * height);
    im2col<filter_size>(input_channels, input, col);

    // Weight shape (output, input, filter_size, filter_size)
    // 96 18 3 3
    // C←αAB + βC
    // outputs[96,19x19] = weights[96,18x3x3] x col[18x3x3,19x19]
    // M Number of rows in matrices A and C.
    // N Number of columns in matrices B and C.
    // K Number of columns in matrix A; number of rows in matrix B.
    // lda The size of the first dimention of matrix A; if you are
    // passing a matrix A[m][n], the value should be m.
    //    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                ldb, beta, C, N);
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                // M        N            K
                outputs, num_intersections, filter_dim,
                1.0f, &weights[0], filter_dim,
                &col[0], num_intersections,
                0.0f, &output[0], num_intersections);
#else
    auto C_mat = EigenMatrixMap<float>(output.data(),
                                       num_intersections, outputs);
    C_mat.noalias() =
        ConstEigenMatrixMap<float>(col.data(), num_intersections, filter_dim)
        * ConstEigenMatrixMap<float>(weights.data(), filter_dim, outputs);
#endif

    for (unsigned int o = 0; o < outputs; o++) {
        for (unsigned int b = 0; b < num_intersections; b++) {
            output[(o * num_intersections) + b] += biases[o];
        }
    }
}

template<unsigned int inputs,
         unsigned int outputs,
         bool ReLU>
std::vector<float> innerproduct(const std::vector<float>& input,
                                const std::vector<float>& weights,
                                const std::vector<float>& biases) {
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
void relu(const size_t channels, std::vector<float>& data) {
    const auto lambda_ReLU = [](const auto val) { return (val > 0.0f) ?
                                                          val : 0.0f; };
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto arr = &data[c * spatial_size];
        for (auto b = size_t{0}; b < spatial_size; b++) {
            arr[b] = lambda_ReLU(arr[b]);
        }
    }
}

template <size_t spatial_size>
void sigmoid(const size_t channels, std::vector<float>& data) {
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto arr = &data[c * spatial_size];
        for (auto b = size_t{0}; b < spatial_size; b++) {
            arr[b] = 1.0f / (1.0f + exp(-arr[b]));
        }
    }
}

template <size_t spatial_size>
void eltwise_add(const size_t channels,
               std::vector<float>& data,
               const float* const eltwise) {
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto arr = &data[c * spatial_size];
        const auto res = &eltwise[c * spatial_size];
        for (auto b = size_t{0}; b < spatial_size; b++) {
            arr[b] = arr[b] + res[b];
        }
    }
}

template <size_t spatial_size>
void channel_scale(const size_t channels,
               std::vector<float>& data,
               const float* const scale) {
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto arr = &data[c * spatial_size];
        for (auto b = size_t{0}; b < spatial_size; b++) {
            arr[b] = arr[b] * scale[c];
        }
    }
}

template <size_t spatial_size>
std::vector<float> channel_average(
               const size_t channels,
               std::vector<float>& data) {
    std::vector<float> ret;
    ret.reserve(channels);
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto arr = &data[c * spatial_size];
        float sum=0;
        for (auto b = size_t{0}; b < spatial_size; b++) {
            sum += arr[b];
        }
        ret.push_back(sum / spatial_size);
    }
    return ret;
}



template <size_t spatial_size>
void batchnorm(const size_t channels,
               std::vector<float>& data,
               const float* const means,
               const float* const stddevs) {
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto mean = means[c];
        const auto scale_stddev = stddevs[c];
        const auto arr = &data[c * spatial_size];

        for (auto b = size_t{0}; b < spatial_size; b++) {
            arr[b] = scale_stddev * (arr[b] - mean);
        }
    }
}

void CPUPipe::forward(const std::vector<float>& input,
                      std::vector<float>& output_pol,
                      std::vector<float>& output_val) {
    // Input convolution
    constexpr auto P = WINOGRAD_P;
    // Calculate output channels
    const auto output_channels = m_input_channels;
    // input_channels is the maximum number of input channels of any
    // convolution. Residual blocks are identical, but the first convolution
    // might be bigger when the network has very few filters
    const auto input_channels = std::max(static_cast<size_t>(output_channels),
                                         static_cast<size_t>(Network::INPUT_CHANNELS));
    auto conv_out = std::vector<float>(output_channels * NUM_INTERSECTIONS);

    auto V = std::vector<float>(WINOGRAD_TILE * input_channels * P);
    auto M = std::vector<float>(WINOGRAD_TILE * output_channels * P);

    winograd_convolve3(output_channels, input, m_weights->m_conv_weights[0], V, M, conv_out);
    batchnorm<NUM_INTERSECTIONS>(output_channels, conv_out,
                                 m_weights->m_batchnorm_means[0].data(),
                                 m_weights->m_batchnorm_stddevs[0].data());
    relu<NUM_INTERSECTIONS>(output_channels, conv_out);

    // Residual tower
    auto conv_in = std::vector<float>(output_channels * NUM_INTERSECTIONS);
    auto res = std::vector<float>(output_channels * NUM_INTERSECTIONS);
    for (auto i = size_t{1}; i < m_weights->m_conv_weights.size(); i += 2) {
        auto output_channels = m_input_channels;
        std::swap(conv_out, conv_in);
        winograd_convolve3(output_channels, conv_in,
                           m_weights->m_conv_weights[i], V, M, conv_out);
        batchnorm<NUM_INTERSECTIONS>(output_channels, conv_out,
                                     m_weights->m_batchnorm_means[i].data(),
                                     m_weights->m_batchnorm_stddevs[i].data());
        relu<NUM_INTERSECTIONS>(output_channels, conv_out);
        assert(m_weights->m_squeeze_1[i].size() == 0);

        std::swap(conv_in, res);
        std::swap(conv_out, conv_in);
        winograd_convolve3(output_channels, conv_in,
                           m_weights->m_conv_weights[i + 1], V, M, conv_out);
        batchnorm<NUM_INTERSECTIONS>(output_channels, conv_out,
                                     m_weights->m_batchnorm_means[i + 1].data(),
                                     m_weights->m_batchnorm_stddevs[i + 1].data());
        if(m_weights->m_squeeze_1[i + 1].size() > 0) {
            auto w = m_weights->m_squeeze_1[i+1].data();
            auto avg = channel_average<NUM_INTERSECTIONS>(output_channels, conv_out);
            std::vector<float> mid;
            mid.resize(output_channels/8);
            for(auto y=0; y<output_channels/8; y++) {
                float f = 0.0f;
                for(auto x=0; x<output_channels; x++) {
                    f += avg[x] * w[y * output_channels + x];
                }
                mid[y] = f;
            }
            relu<1>(output_channels/8, mid);
            auto w2 = m_weights->m_squeeze_2[i+1].data();
            std::vector<float> end;
            end.resize(output_channels);
            for(auto y=0; y<output_channels; y++) {
                float f = 0.0f;
                for(auto x=0; x<output_channels/8; x++) {
                    f += mid[x] * w2[y * (output_channels/8) + x];
                }
                end[y] = f;
            }
            sigmoid<1>(output_channels, end);

            channel_scale<NUM_INTERSECTIONS>(output_channels, conv_out, end.data());
        }
        eltwise_add<NUM_INTERSECTIONS>(output_channels, conv_out, res.data());
        relu<NUM_INTERSECTIONS>(output_channels, conv_out);
    }
    std::vector<float> policy_data(16*NUM_INTERSECTIONS);
    std::vector<float> value_data(1*NUM_INTERSECTIONS);
    convolve<1>(16, conv_out, m_conv_pol_w, m_conv_pol_b, policy_data);
    convolve<1>(1, conv_out, m_conv_val_w, m_conv_val_b, value_data);

    batchnorm<NUM_INTERSECTIONS>(16, policy_data,
        m_weights->m_bn_pol_w1.data(), m_weights->m_bn_pol_w2.data());
    relu<NUM_INTERSECTIONS>(16, policy_data);
    output_pol =
        innerproduct<16 * NUM_INTERSECTIONS, POTENTIAL_MOVES, false>(
            policy_data, m_weights->m_ip_pol_w, m_weights->m_ip_pol_b);

    // Now get the value
    batchnorm<NUM_INTERSECTIONS>(1, value_data,
        m_weights->m_bn_val_w1.data(), m_weights->m_bn_val_w2.data());
    relu<NUM_INTERSECTIONS>(1, value_data);
    output_val =
        innerproduct<1 * NUM_INTERSECTIONS, Network::OUTPUTS_VALUE, true>(
            value_data, m_weights->m_ip_val_w, m_weights->m_ip_val_b);

}

void CPUPipe::push_weights(unsigned int /*filter_size*/,
                           unsigned int /*channels*/,
                           unsigned int outputs,
                           std::shared_ptr<const ForwardPipeWeights> weights) {

    m_weights = weights;

    // Output head convolutions
    m_conv_pol_w = weights->m_conv_pol_w;
    m_conv_pol_b.resize(m_conv_pol_w.size() / outputs, 0.0f);
    m_conv_val_w = weights->m_conv_val_w;
    m_conv_val_b.resize(m_conv_val_w.size() / outputs, 0.0f);
}

