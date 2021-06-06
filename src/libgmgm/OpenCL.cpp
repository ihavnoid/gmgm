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

#include <cassert>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <iterator>
#include <limits>
#include <stdexcept>

#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "half/half.hpp"

#include "globals.h"
#include "OpenCL.h"
#include "Network.h"
#include "Tuner.h"

using gmgm::globals::myprintf;

static bool cfg_tune_only = false;

static size_t ceilMultiple(size_t a, size_t b) {
    if (a % b == 0) {
        return a;
    }

    auto ret = a + (b - a % b);
    return ret;
}

template <typename net_t> static std::string getClArgs();

template <> std::string getClArgs<float>() {
    return
        "-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-denorms-are-zero";
}


template <> std::string getClArgs<half_float::half>() {
    return
        "-DUSE_HALF "
        "-cl-mad-enable -cl-fast-relaxed-math -cl-no-signed-zeros -cl-denorms-are-zero";
}

const std::string sourceCode_common =
    #include "kernels/common.opencl"
;

static const std::string sourceCode_tensorcore_test =
    #include "kernels/tensorcore_test.opencl"
;

static const std::string sourceCode_config = 
    "\n#define BOARD_W " + std::to_string(gmgm::BOARD_W) +
    "\n#define BOARD_H " + std::to_string(gmgm::BOARD_H) +
    "\n#define NUM_INTERSECTIONS " + std::to_string(NUM_INTERSECTIONS) +
    "\n#define WINOGRAD_M " + std::to_string(WINOGRAD_M) +
    "\n#define WINOGRAD_ALPHA " + std::to_string(WINOGRAD_ALPHA) +
    "\n#define WTILES " + std::to_string(WINOGRAD_WTILES) +
    "\n";

static const std::string sourceCode_convolve1 =
    #include "kernels/convolve1.opencl"
;

static const std::string sourceCode_convolve3 =
    #include "kernels/convolve3.opencl"
;

const std::string sourceCode_sgemm =
    "#if TCE == 1\n" // Enable tensorcore
    #include "kernels/clblast/hgemm_tensorcore.opencl"
    "\n#else\n" // Use clblast
    #include "kernels/clblast/xgemm_part1.opencl"
    #include "kernels/clblast/xgemm_part2.opencl"
    #include "kernels/clblast/xgemm_part3.opencl"
    #include "kernels/clblast/xgemm_batched.opencl"
    "\n#endif\n"
;

template <typename net_t>
void OpenCL<net_t>::ensure_context_initialized(OpenCLContext &opencl_context) {
    if (!opencl_context.m_is_initialized) {
        // Make kernels
        opencl_context.m_convolve1_kernel =
            cl::Kernel(m_program, "convolve1");
        opencl_context.m_merge_kernel =
            cl::Kernel(m_program, "merge");
        opencl_context.m_fc_kernel =
            cl::Kernel(m_program, "fully_connected");
        opencl_context.m_in_transform_kernel =
            cl::Kernel(m_program, "in_transform");
        opencl_context.m_sgemm_kernel =
            cl::Kernel(m_program, "XgemmBatched");
        opencl_context.m_out_transform_bn_kernel =
            cl::Kernel(m_program, "out_transform_fused_bn");
        opencl_context.m_out_transform_bn_in_kernel =
            cl::Kernel(m_program, "out_transform_fused_bn_in");
        opencl_context.m_out_transform_kernel =
            cl::Kernel(m_program, "out_transform");
        opencl_context.m_fused_bn_res_scale_kernel =
            cl::Kernel(m_program, "fused_bn_res_scale");
        opencl_context.m_commandqueue =
            cl::CommandQueue(m_context, m_device);
        opencl_context.m_is_initialized = true;
    }
}

template <typename net_t>
void OpenCL_Network<net_t>::add_weights(size_t layer,
                                 size_t size,
                                 const net_t * weights) {
    if (layer >= m_layers.size()) {
        m_layers.push_back(Layer());
    }

    auto weightSize = size * sizeof(net_t);

    auto queue = cl::CommandQueue(getOpenCL().m_context, getOpenCL().m_device);
    auto buffer = cl::Buffer(
        m_opencl.m_context,
        CL_MEM_READ_ONLY,
        weightSize,
        nullptr
    );
    queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, weightSize, const_cast<net_t*>(weights));
    m_layers.back().weights.push_back(std::move(buffer));
}
template <typename net_t>
void OpenCL_Network<net_t>::push_convolve(unsigned int filter_size,
                            unsigned int channels,
                            unsigned int outputs,
                            const std::vector<net_t>& conv_weights,
                            const std::vector<net_t>& bn_pol_means,
                            const std::vector<net_t>& bn_pol_stddevs,
                            const std::vector<net_t>& fc_weights,
                            const std::vector<net_t>& fc_biases) {
    (void)filter_size;
    assert(filter_size == 1);

    size_t layer = get_layer_count();
    push_weights(layer, conv_weights);
    push_weights(layer, bn_pol_means);
    push_weights(layer, bn_pol_stddevs);
    push_weights(layer, fc_weights);
    push_weights(layer, fc_biases);
    m_layers[layer].is_final_conv = true;
    m_layers[layer].outputs = outputs;
    m_layers[layer].mids = fc_weights.size() / outputs / NUM_INTERSECTIONS;
    m_layers[layer].channels = channels;
#if 0
    myprintf(" --- %d %d %d %d %d %d\n",
        bn_pol_means.size(),
        bn_pol_stddevs.size(),
        fc_weights.size(),
        fc_biases.size(),
        m_layers[layer].outputs,
        m_layers[layer].mids
    );
#endif
}


template <typename net_t>
void OpenCL_Network<net_t>::forward(const std::vector<float>& input,
                             std::vector<float>& output_pol,
                             std::vector<float>& output_val,
                             OpenCLContext & opencl_context,
                             const int batch_size) {
    constexpr auto tiles = WINOGRAD_P;
    constexpr auto one_plane = NUM_INTERSECTIONS * sizeof(float);
    const auto finalSize_pol = m_layers[m_layers.size()-2].outputs * sizeof(float);
    const auto finalSize_val = m_layers.back().outputs * sizeof(float);
    const auto midSize_pol = m_layers[m_layers.size()-2].mids * one_plane;
    const auto midSize_val = m_layers.back().mids * one_plane;

    m_opencl.ensure_context_initialized(opencl_context);

    if (!opencl_context.m_buffers_allocated) {
        auto max_channels = unsigned{0};
        for (const auto& layer : m_layers) {
            max_channels = std::max(max_channels,
                                    std::max(layer.channels, layer.outputs));
        }

        const auto mwg = m_opencl.m_sgemm_tuners.mwg;
        const auto nwg = m_opencl.m_sgemm_tuners.nwg;
        const auto vwm = m_opencl.m_sgemm_tuners.vwm;
        const auto vwn = m_opencl.m_sgemm_tuners.vwn;

        const auto m_ceil = ceilMultiple(ceilMultiple(max_channels, mwg), vwm);
        const auto n_ceil = ceilMultiple(ceilMultiple(tiles, nwg), vwn);

        const auto alloc_inSize =
            getOpenCL().m_batch_size * NUM_INTERSECTIONS * max_channels * sizeof(net_t);
        const auto alloc_vm_size =
            getOpenCL().m_batch_size * WINOGRAD_TILE * m_ceil * n_ceil * sizeof(net_t);

        auto v_zeros = std::vector<net_t>(alloc_vm_size);

        opencl_context.m_inBuffer = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_READ_WRITE, alloc_inSize);
        opencl_context.m_tempBuffer = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_READ_WRITE, alloc_inSize);
        opencl_context.m_inBuffer2 = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_READ_WRITE, alloc_inSize);
        opencl_context.m_VBuffer = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_COPY_HOST_PTR,
            alloc_vm_size, v_zeros.data(), nullptr);
        opencl_context.m_MBuffer = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, alloc_vm_size);

        opencl_context.m_midBuffer_pol = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, getOpenCL().m_batch_size * midSize_pol);
        opencl_context.m_midBuffer_val = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, getOpenCL().m_batch_size * midSize_val);


        opencl_context.m_pinnedOutBuffer_pol = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, getOpenCL().m_batch_size * finalSize_pol);
        opencl_context.m_pinnedOutBuffer_val = cl::Buffer(
            m_opencl.m_context,
            CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, getOpenCL().m_batch_size * finalSize_val);

        opencl_context.m_buffers_allocated = true;
    }

    cl::Buffer & inBuffer = opencl_context.m_inBuffer;
    cl::Buffer & tempBuffer = opencl_context.m_tempBuffer;
    cl::Buffer & inBuffer2 = opencl_context.m_inBuffer2;
    cl::Buffer & VBuffer = opencl_context.m_VBuffer;
    cl::Buffer & MBuffer = opencl_context.m_MBuffer;
    cl::CommandQueue & queue = opencl_context.m_commandqueue;

    std::vector<net_t> net_t_input(input.size());
    std::copy(begin(input), end(input), begin(net_t_input));

    const auto inSize = sizeof(net_t) * input.size();
    queue.enqueueWriteBuffer(inBuffer, CL_FALSE, 0, inSize, net_t_input.data());

    // Fused in_out transformation kernel is slower with big batch_sizes than
    // calling out and in transformations separately.
    // This condition could be tunable in future.
    auto use_inout = false;

    auto skip_in_trans = false;
    for (auto iter = cbegin(m_layers); iter != cend(m_layers); iter++) {
        const auto& layer = *iter;
        const auto niter = std::next(iter);

        if (layer.is_input_convolution) {
            assert(niter != cend(m_layers));
            auto conv_weights = begin(layer.weights);
            auto bn_weights = begin(layer.weights) + 1;
            auto skip_next_in_trans = false;
            if (niter->is_residual_block) {
                skip_next_in_trans = use_inout;
            }

            convolve3(opencl_context,
                     layer.channels,
                     layer.outputs,
                     inBuffer,
                     inBuffer,
                     tempBuffer,
                     VBuffer,
                     MBuffer,
                     conv_weights,
                     nullptr,
                     bn_weights,
                     skip_in_trans, skip_next_in_trans, true,
                     batch_size, nullptr, nullptr);

            skip_in_trans = skip_next_in_trans;
        } else if (layer.is_residual_block) {
            assert(layer.channels == layer.outputs);
            assert(niter != cend(m_layers));
            auto conv1_weights = begin(layer.weights);
            auto bn1_weights   = begin(layer.weights) + 1;
            auto conv2_weights = begin(layer.weights) + 3;
            auto bn2_weights   = begin(layer.weights) + 4;
            convolve3(opencl_context,
                      layer.channels,
                      layer.outputs,
                      inBuffer,
                      inBuffer2,
                      tempBuffer,
                      VBuffer,
                      MBuffer,
                      conv1_weights,
                      nullptr,
                      bn1_weights,
                      skip_in_trans, use_inout, false,
                      batch_size, nullptr, nullptr);

            auto skip_next_in_trans = false;
            if (niter->is_residual_block) {
                skip_next_in_trans = use_inout;
            }

            const cl::Buffer * sq1 = nullptr;
            const cl::Buffer * sq2 = nullptr;
            if(layer.has_squeeze_layer) {
                sq1 = &(layer.weights[6]);
                sq2 = &(layer.weights[7]);
            }
            convolve3(opencl_context,
                      layer.channels,
                      layer.outputs,
                      inBuffer2,
                      inBuffer,
                      tempBuffer,
                      VBuffer,
                      MBuffer,
                      conv2_weights,
                      &inBuffer,
                      bn2_weights,
                      use_inout, skip_next_in_trans, true,
                      batch_size,
                      sq1, sq2
            );
            skip_in_trans = skip_next_in_trans;
        } else {
            assert(layer.is_final_conv);

            cl::Buffer mid_buffer;
            cl::Buffer out_buffer;
            if (niter == cend(m_layers)) {
                out_buffer = opencl_context.m_pinnedOutBuffer_val;
                mid_buffer = opencl_context.m_midBuffer_val;
            } else {
                out_buffer = opencl_context.m_pinnedOutBuffer_pol;
                mid_buffer = opencl_context.m_midBuffer_pol;
            }

            convolve1(opencl_context, layer.channels,
                    layer.mids,
                    layer.outputs,
                    inBuffer,
                    mid_buffer,
                    out_buffer,
                    VBuffer,
                    begin(layer.weights),
                    batch_size);
        }
    }


    void * pinnedOutBufferHost_pol;
    void * pinnedOutBufferHost_val;

    try {
        pinnedOutBufferHost_pol = queue.enqueueMapBuffer(
            opencl_context.m_pinnedOutBuffer_pol, CL_FALSE,
            CL_MAP_READ, 0, batch_size * finalSize_pol);
        pinnedOutBufferHost_val = queue.enqueueMapBuffer(
            opencl_context.m_pinnedOutBuffer_val, CL_FALSE,
            CL_MAP_READ, 0, batch_size * finalSize_val);
    } catch (const cl::Error &e) {
        std::cerr << "Error in enqueueMapBuffer: " << e.what() << ": "
            << e.err() << std::endl;
        throw;
    }

    {
        // Finish call is usually a busy wait. When using multiple threads
        // use the lock to avoid busy waiting with all threads.
        auto start = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::microseconds(sleeptime.load()));
        queue.finish();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        auto next_sleeptime = static_cast<int>( sleeptime.load() * 0.5 + (1 + duration.count() * 1.0 / 5 * 1000000));
        next_sleeptime = std::min(next_sleeptime, 1000000);
        sleeptime = next_sleeptime;
    }

    auto polptr = static_cast<float*>(pinnedOutBufferHost_pol);
    auto valptr = static_cast<float*>(pinnedOutBufferHost_val);
    std::copy(polptr, polptr + output_pol.size(), begin(output_pol));
    std::copy(valptr, valptr + output_val.size(), begin(output_val));

    queue.enqueueUnmapMemObject(opencl_context.m_pinnedOutBuffer_pol,
            pinnedOutBufferHost_pol);
    queue.enqueueUnmapMemObject(opencl_context.m_pinnedOutBuffer_val,
            pinnedOutBufferHost_val);

}

template <typename net_t>
void OpenCL_Network<net_t>::convolve3(OpenCLContext & opencl_context,
                              int channels, int outputs,
                              cl::Buffer& bufferIn,
                              cl::Buffer& bufferOut,
                              cl::Buffer& bufferTemp,
                              cl::Buffer& bufferV,
                              cl::Buffer& bufferM,
                              weight_slice_t weights,
                              cl::Buffer* bufferResidual,
                              weight_slice_t bn_weights,
                              bool skip_in_transform,
                              bool fuse_in_transform,
                              bool store_inout,
                              int batch_size,
                              const cl::Buffer* sq1,
                              const cl::Buffer* sq2) {

    cl::Kernel & in_transform_kernel = opencl_context.m_in_transform_kernel;
    cl::Kernel & sgemm_kernel = opencl_context.m_sgemm_kernel;
    cl::Kernel & out_transform_bn_kernel =
        opencl_context.m_out_transform_bn_kernel;
    cl::Kernel & out_transform_bn_in_kernel =
        opencl_context.m_out_transform_bn_in_kernel;
    cl::Kernel & out_transform_kernel =
        opencl_context.m_out_transform_kernel;
    cl::Kernel & fused_bn_res_scale_kernel =
        opencl_context.m_fused_bn_res_scale_kernel;

    auto mwg = m_opencl.m_sgemm_tuners.mwg;
    auto nwg = m_opencl.m_sgemm_tuners.nwg;
    auto kwg = m_opencl.m_sgemm_tuners.kwg;
    auto vwm = m_opencl.m_sgemm_tuners.vwm;
    auto vwn = m_opencl.m_sgemm_tuners.vwn;
    auto mdimc = m_opencl.m_sgemm_tuners.mdimc;
    auto ndimc = m_opencl.m_sgemm_tuners.ndimc;
    auto tce = m_opencl.m_sgemm_tuners.tce;
    auto mdima = m_opencl.m_sgemm_tuners.mdima;
    auto ndimb = m_opencl.m_sgemm_tuners.ndimb;

    auto wavefront_size = m_opencl.m_wavefront_size;

    assert(mwg != 0);
    assert(nwg != 0);
    assert(kwg != 0);
    assert(mdimc != 0);
    assert(ndimc != 0);
    assert(vwm != 0);
    assert(vwn != 0);
    assert(wavefront_size != 0);

    constexpr auto tiles = WINOGRAD_P;

    auto wgs = ceilMultiple(batch_size * tiles, wavefront_size);
    auto wgs_single = ceilMultiple(tiles, wavefront_size);

    auto m_ceil = int(ceilMultiple(ceilMultiple(outputs, mwg), vwm));
    auto n_ceil = int(ceilMultiple(ceilMultiple(batch_size * tiles, nwg), vwn));
    auto k_ceil = int(ceilMultiple(ceilMultiple(channels, kwg), vwm));

    cl::CommandQueue & queue = opencl_context.m_commandqueue;

    if (!skip_in_transform) {
        try {
            in_transform_kernel.setArg(0, bufferIn);
            in_transform_kernel.setArg(1, bufferV);
            in_transform_kernel.setArg(2, channels);
            in_transform_kernel.setArg(3, k_ceil);
            in_transform_kernel.setArg(4, n_ceil);
            in_transform_kernel.setArg(5, batch_size);

            queue.enqueueNDRangeKernel(in_transform_kernel, cl::NullRange,
                                       cl::NDRange(wgs, channels));
        } catch (const cl::Error &e) {
            std::cerr << "Error in convolve3/in: " << e.what() << ": "
                << e.err() << std::endl;
            throw;
        }
    }

    try {
        sgemm_kernel.setArg(0, m_ceil);
        sgemm_kernel.setArg(1, n_ceil);
        sgemm_kernel.setArg(2, k_ceil);
        sgemm_kernel.setArg(3, weights[0]);
        sgemm_kernel.setArg(4, bufferV);
        sgemm_kernel.setArg(5, bufferM);

        cl::NDRange local_sgemm = {mdimc, ndimc, 1};

        cl::NDRange size_sgemm = {(m_ceil * mdimc) / mwg,
                                  (n_ceil * ndimc) / nwg,
                                  cl::size_type(WINOGRAD_TILE)};

        // tensorcore implementation uses a different dimension
        if (tce) {
            local_sgemm = {32 * mdimc/mdima, ndimc/ndimb, 1};
            size_sgemm = {32 * m_ceil / mdima * mdimc / mwg,
                          n_ceil / ndimb * ndimc / nwg,
                          cl::size_type(WINOGRAD_TILE)};
        }
        queue.enqueueNDRangeKernel(sgemm_kernel, cl::NullRange,
                                   size_sgemm, local_sgemm);
    } catch (const cl::Error &e) {
        std::cerr << "Error in convolve3/sgemm: " << e.what() << ": "
            << e.err() << std::endl;
        throw;
    }

    try {
        if (fuse_in_transform) {
            // no fuse_in + squeeze support
            // we don't need it
            assert(sq1 == nullptr);
            assert(sq2 == nullptr);

            // TODO : Eventually this might also be something tuneable?
            // Needs to match OUTIN_KWG in kernel
            constexpr auto dim_size = 2;
            out_transform_bn_in_kernel.setArg(0, bufferM);
            if (store_inout) {
                out_transform_bn_in_kernel.setArg(1, bufferOut);
            } else {
                out_transform_bn_in_kernel.setArg(1, nullptr);
            }
            out_transform_bn_in_kernel.setArg(2, bufferV);
            out_transform_bn_in_kernel.setArg(3, outputs);
            out_transform_bn_in_kernel.setArg(4, m_ceil);
            out_transform_bn_in_kernel.setArg(5, n_ceil);
            // k_ceil of the next convolution
            auto k_ceil2 = int(ceilMultiple(ceilMultiple(outputs, kwg), vwm));
            out_transform_bn_in_kernel.setArg(6, k_ceil2);
            if (bufferResidual) {
                out_transform_bn_in_kernel.setArg(7, *bufferResidual);
            } else {
                out_transform_bn_in_kernel.setArg(7, nullptr);
            }
            out_transform_bn_in_kernel.setArg(8, bn_weights[0]);
            out_transform_bn_in_kernel.setArg(9, bn_weights[1]);

            queue.enqueueNDRangeKernel(out_transform_bn_in_kernel,
                                       cl::NullRange,
                                       cl::NDRange(outputs, wgs_single, batch_size),
                                       cl::NDRange(dim_size, wgs_single, 1));

        } else if(sq1 == nullptr || sq2 == nullptr) {
            // no squeeze layer
            out_transform_bn_kernel.setArg(0, bufferM);
            out_transform_bn_kernel.setArg(1, bufferOut);
            out_transform_bn_kernel.setArg(2, outputs);
            out_transform_bn_kernel.setArg(3, m_ceil);
            out_transform_bn_kernel.setArg(4, n_ceil);
            out_transform_bn_kernel.setArg(5, batch_size);
            if (bufferResidual) {
                out_transform_bn_kernel.setArg(6, *bufferResidual);
            } else {
                out_transform_bn_kernel.setArg(6, nullptr);
            }
            out_transform_bn_kernel.setArg(7, bn_weights[0]);
            out_transform_bn_kernel.setArg(8, bn_weights[1]);

            // Needs to match OUT_KWG, OUT_BWG in the kernel.
            // This could be tuned.
            cl::NDRange local_out = {32, 2};

            cl::NDRange global_out = {ceilMultiple(outputs, local_out[0]),
                                      ceilMultiple(tiles * batch_size, local_out[1])};

            queue.enqueueNDRangeKernel(out_transform_bn_kernel, cl::NullRange,
                                       global_out,
                                       local_out);
        } else {
            out_transform_kernel.setArg(0, bufferM);
            out_transform_kernel.setArg(1, bufferTemp);
            out_transform_kernel.setArg(2, outputs);
            out_transform_kernel.setArg(3, m_ceil);
            out_transform_kernel.setArg(4, n_ceil);
            out_transform_kernel.setArg(5, batch_size);
            out_transform_kernel.setArg(6, bn_weights[0]);
            out_transform_kernel.setArg(7, bn_weights[1]);

            // Needs to match OUT_KWG, OUT_BWG in the kernel.
            // This could be tuned.
            cl::NDRange local_out = {32, 2};

            cl::NDRange global_out = {ceilMultiple(outputs, local_out[0]),
                                      ceilMultiple(tiles * batch_size, local_out[1])};

            queue.enqueueNDRangeKernel(out_transform_kernel, cl::NullRange,
                                       global_out,
                                       local_out);

            fused_bn_res_scale_kernel.setArg(0, bufferTemp);
            fused_bn_res_scale_kernel.setArg(1, bufferOut);
            if (bufferResidual) {
                fused_bn_res_scale_kernel.setArg(2, *bufferResidual);
            } else {
                // unsupported
                assert(false);
            }
            fused_bn_res_scale_kernel.setArg(3, *sq1);
            fused_bn_res_scale_kernel.setArg(4, *sq2);
            local_out = {static_cast<size_t>(channels), 1};
            global_out = {static_cast<size_t>(channels * batch_size), 1};
            queue.enqueueNDRangeKernel(fused_bn_res_scale_kernel, cl::NullRange,
                                       global_out,
                                       local_out);
        }
    } catch (const cl::Error &e) {
        std::cerr << "Error in convolve3/out: " << e.what() << ": "
            << e.err() << std::endl;
        throw;
    }
}

template <typename net_t>
void OpenCL_Network<net_t>::convolve1(OpenCLContext & opencl_context,
                              int channels, int mids, int outputs,
                              cl::Buffer& bufferInput,
                              cl::Buffer& bufferMid,
                              cl::Buffer& bufferOutput,
                              cl::Buffer& bufferMerge,
                              weight_slice_t weights,
                              int batch_size) {
    // The size of the board is defined at compile time
    constexpr int width = gmgm::BOARD_W;
    constexpr int boardsize = NUM_INTERSECTIONS;
    constexpr int rowTiles = gmgm::BOARD_H;

    // Input channel grouping in multiples of 8
    constexpr int channelGroup = 8;
    constexpr int channelShift = 3;
    constexpr int rowGroup = 1;
    size_t outputGroup = std::min(mids, 32);

    auto m_convolve_kernel = &opencl_context.m_convolve1_kernel;

#ifndef NDEBUG
    // Total output size after reducing
    size_t outSize = boardsize * mids * sizeof(float);

    // Produce channel * output planes and merge them at the end
    size_t mergeSize = (channels >> channelShift) * outSize;
    assert(mergeSize <= bufferMerge.getInfo<CL_MEM_SIZE>());
#endif

    // Copy the rows locally
    size_t stripSize = width * sizeof(float);

    int rowBuffer = std::min<int>(channelGroup, 7);
    size_t rowSize = channelGroup * outputGroup * rowBuffer * sizeof(float);

    cl::CommandQueue & queue = opencl_context.m_commandqueue;

    try {
        m_convolve_kernel->setArg(0, bufferInput);
        m_convolve_kernel->setArg(1, bufferMerge);
        m_convolve_kernel->setArg(2, weights[0]);
        m_convolve_kernel->setArg(3, cl::Local(stripSize * channelGroup * rowGroup));
        m_convolve_kernel->setArg(4, cl::Local(rowSize));

        queue.enqueueNDRangeKernel(
            *m_convolve_kernel, cl::NullRange,
            cl::NDRange(channels, mids, batch_size * rowTiles),
            cl::NDRange(channelGroup, outputGroup, rowGroup));
    } catch (const cl::Error &e) {
        std::cerr << "Error in convolve1: " << e.what() << ": "
                  << e.err() << std::endl;
        throw;
    }

    cl::Kernel & merge_kernel = opencl_context.m_merge_kernel;
    assert(channels % (1 << channelShift) == 0);

    try {
        merge_kernel.setArg(0, bufferMerge);
        merge_kernel.setArg(1, bufferMid);
        merge_kernel.setArg(2, channels >> channelShift);
        merge_kernel.setArg(3, weights[1]);
        merge_kernel.setArg(4, weights[2]);

        queue.enqueueNDRangeKernel(
            merge_kernel, cl::NullRange,
            cl::NDRange(mids, boardsize, batch_size),
            cl::NDRange(std::min(8, mids), gmgm::BOARD_H, 1));
    } catch (const cl::Error &e) {
        std::cerr << "Error in merge: " << e.what() << ": "
                  << e.err() << std::endl;
        throw;
    }

    cl::Kernel & fc_kernel = opencl_context.m_fc_kernel;
    try {
        fc_kernel.setArg(0, bufferMid);
        fc_kernel.setArg(1, bufferOutput);
        fc_kernel.setArg(2, weights[3]);
        fc_kernel.setArg(3, weights[4]);
        fc_kernel.setArg(4, mids * boardsize);

        queue.enqueueNDRangeKernel(
            fc_kernel, cl::NullRange,
            cl::NDRange(outputs, batch_size, mids),
            cl::NDRange(1, 1, mids)
        );
    } catch (const cl::Error &e) {
        std::cerr << "Error in fc: " << e.what() << ": "
                  << e.err() << std::endl;
        throw;
    }
}

template<class T>
static std::string opencl_dev_type_to_string(T type) {
    if (type == CL_DEVICE_TYPE_CPU) {
        return "CPU";
    } else if (type == CL_DEVICE_TYPE_GPU) {
        return "GPU";
    } else if (type == CL_DEVICE_TYPE_ACCELERATOR) {
        return "Accelerator";
    } else {
        return "Unknown";
    }
}

static std::string trim(std::string trim_me) {
    boost::algorithm::trim(trim_me);
    return trim_me;
}

template <typename net_t>
void OpenCL<net_t>::process_tuners(std::string tuners) {
    std::string buf;
    std::stringstream ss(tuners);
    std::size_t found;

    auto mwg = false;
    auto nwg = false;
    auto kwg = false;
    auto ndimc = false;
    auto mdimc = false;
    auto mdima = false;
    auto ndimb = false;
    auto vwm = false;
    auto vwn = false;
    auto tce = false;

    while (ss >> buf) {
        found = buf.find("=");
        if (found == std::string::npos) {
            std::cerr << "Invalid tuner string: " << tuners << std::endl;
            std::exit(-1);
        }
        std::string name = buf.substr(0, found);
        auto value = std::stoi(buf.substr(found + 1, std::string::npos));
        if (name == "-DMWG") {
            m_sgemm_tuners.mwg = value;
            mwg = true;
        }
        if (name == "-DNWG") {
            m_sgemm_tuners.nwg = value;
            nwg = true;
        }
        if (name == "-DKWG") {
            m_sgemm_tuners.kwg = value;
            kwg = true;
        }
        if (name == "-DMDIMA") {
            m_sgemm_tuners.mdima = value;
            mdima = true;
        }
        if (name == "-DNDIMB") {
            m_sgemm_tuners.ndimb = value;
            ndimb = true;
        }
        if (name == "-DMDIMC") {
            m_sgemm_tuners.mdimc = value;
            mdimc = true;
        }
        if (name == "-DNDIMC") {
            m_sgemm_tuners.ndimc = value;
            ndimc = true;
        }
        if (name == "-DVWM") {
            m_sgemm_tuners.vwm = value;
            vwm = true;
        }
        if (name == "-DVWN") {
            m_sgemm_tuners.vwn = value;
            vwn = true;
        }
        if (name == "-DTCE") {
            m_sgemm_tuners.tce = value;
            tce = true;
        }
    }
    if (!mwg || !nwg || !kwg || !mdimc || !ndimc || !vwm || !vwn || !mdima || !ndimb) {
        std::cerr << "Missing tuner parameters";
        if (!mwg) {
            std::cerr << " MWG";
        }
        if (!nwg) {
            std::cerr << " NWG";
        }
        if (!kwg) {
            std::cerr << " KWG";
        }
        if (!mdima) {
            std::cerr << " MDIMA";
        }
        if (!ndimb) {
            std::cerr << " NDIMB";
        }
        if (!mdimc) {
            std::cerr << " MDIMC";
        }
        if (!ndimc) {
            std::cerr << " NDIMC";
        }
        if (!vwm) {
            std::cerr << " VWM";
        }
        if (!vwn) {
            std::cerr << " VWN";
        }
        if (!tce) {
            std::cerr << " VWN";
        }
        std::cerr << std::endl;
        std::exit(-1);
    }
}

template <typename net_t>
std::vector<size_t> OpenCL<net_t>::get_sgemm_tuners() {
    std::vector<size_t> tuners;

    tuners.emplace_back(m_sgemm_tuners.mwg);
    tuners.emplace_back(m_sgemm_tuners.nwg);
    tuners.emplace_back(m_sgemm_tuners.kwg);
    tuners.emplace_back(m_sgemm_tuners.vwm);
    tuners.emplace_back(m_sgemm_tuners.vwn);
    tuners.emplace_back(m_sgemm_tuners.mdimc);
    tuners.emplace_back(m_sgemm_tuners.ndimc);

    return tuners;
}

template <typename net_t>
OpenCL<net_t>::OpenCL(int gpu, bool silent) {
    std::vector<cl::Platform> platforms;
    try {
        cl::Platform::get(&platforms);
    } catch (const cl::Error &e) {
        myprintf("OpenCL: %s\n", e.what());
        throw;
    }

    auto best_version = 0.0f;
    cl::Platform best_platform;
    cl::Device best_device;
    std::string best_vendor;
    auto best_score = 0;
    auto found_device = false;
    auto found_preferred = false;
    auto id = 0;

    if (!silent) {
        myprintf("Detected %lu OpenCL platforms.\n", platforms.size());
    }

    for (const auto &p : platforms) {
        std::string platvers = p.getInfo<CL_PLATFORM_VERSION>();
        if (!silent) {
            std::string platprof = p.getInfo<CL_PLATFORM_PROFILE>();
            std::string platname = p.getInfo<CL_PLATFORM_NAME>();
            std::string platvend = p.getInfo<CL_PLATFORM_VENDOR>();
            myprintf("Platform version: %s\n", platvers.c_str());;
            myprintf("Platform profile: %s\n", platprof.c_str());
            myprintf("Platform name:    %s\n", platname.c_str());
            myprintf("Platform vendor:  %s\n", platvend.c_str());
        }

        std::istringstream versstream(platvers);
        std::string tmp;
        float opencl_version;
        versstream >> tmp >> opencl_version;

        std::vector<cl::Device> devices;
        try {
            p.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        } catch (const cl::Error &e) {
            myprintf("Error getting device(s): %s: %d\n", e.what(), e.err());
            devices.clear();
        }
        for (auto& d : devices) {
            if (!silent) {
                myprintf("Device ID:     %d\n", id);
                myprintf("Device name:   %s\n",
                         trim(d.getInfo<CL_DEVICE_NAME>()).c_str());
                myprintf("Device type:   %s\n",
                         opencl_dev_type_to_string(
                             d.getInfo<CL_DEVICE_TYPE>()).c_str());
                myprintf("Device vendor: %s\n",
                          d.getInfo<CL_DEVICE_VENDOR>().c_str());
                myprintf("Device driver: %s\n",
                          d.getInfo<CL_DRIVER_VERSION>().c_str());
                myprintf("Device speed:  %u MHz\n",
                          d.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>());
                myprintf("Device cores:  %u CU\n",
                          d.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>());
            }

            // assign score, try to find best device
            int this_score = 0;
            std::string this_vendor = d.getInfo<CL_DEVICE_VENDOR>();
            this_score += 1000 * boost::icontains(this_vendor, "advanced micro devices");
            this_score += 1000 * boost::icontains(this_vendor, "amd");
            this_score += 1000 * boost::icontains(this_vendor, "nvidia");
            this_score +=  500 * boost::icontains(this_vendor, "intel");
            this_score +=  100 * (d.getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU);
            this_score +=  opencl_version * 10;
            if (!silent) {
                myprintf("Device score:  %d\n", this_score);
            }

            bool preferred = (gpu == id);

            if (((this_score > best_score)
                 && (d.getInfo<CL_DEVICE_TYPE>() != CL_DEVICE_TYPE_CPU))
                || preferred) {
                if (preferred) {
                    found_preferred = true;
                }
                best_version = opencl_version;
                best_platform = p;
                best_device = d;
                best_vendor = this_vendor;
                if (preferred) {
                    best_score =
                        std::numeric_limits<decltype(best_score)>::max();
                } else {
                    best_score = this_score;
                }
                found_device = true;
            }
            id++;
        }
    }

    if (!found_device) {
        throw std::runtime_error("No suitable OpenCL device found.");
    }
    if (!found_preferred && gpu >= 0) {
        throw std::runtime_error("No more preferred OpenCL device found");
    }

    myprintf("Selected platform: %s\n",
        best_platform.getInfo<CL_PLATFORM_NAME>().c_str());
    myprintf("Selected device: %s\n",
        trim(best_device.getInfo<CL_DEVICE_NAME>()).c_str());
    myprintf("with OpenCL %2.1f capability.\n", best_version);

    cl::Context context;
    try {
        context = cl::Context(best_device);
    } catch (const cl::Error &e) {
        myprintf("Error creating OpenCL context: %s: %d", e.what(), e.err());
        throw std::runtime_error("Error creating OpenCL context.");
    }
    m_context = context;
    m_device = best_device;

    m_cl_args = getClArgs<net_t>();

    myprintf("Half precision compute support: ");
    if (m_device.getInfo<CL_DEVICE_EXTENSIONS>().find("cl_khr_fp16")
        != std::string::npos) {
        myprintf("Yes.\n");
        m_fp16_compute = true;
        m_cl_args += " -DFP16_SUPPORT";
    } else {
        myprintf("No.\n");
    }

    myprintf("Tensor Core support: ");
    {
        // if this is a nvidia GPU, test-compile a sample inline assembly code with
        // tensor wmma instructions. if not, don't bother trying
        std::string this_vendor = m_device.getInfo<CL_DEVICE_VENDOR>();
        if (boost::icontains(this_vendor, "nvidia")) {
            try {
                cl::Program(m_context, sourceCode_tensorcore_test).build(m_cl_args.c_str());
                m_tensorcore = true;
                myprintf("Yes.\n");
            } catch (...) {
                myprintf("No.\n");
            }
        } else {
            myprintf("No.\n");
        }
    }
}

template <typename net_t>
void OpenCL<net_t>::initialize(const int channels, size_t batch_size) {
    m_batch_size = batch_size;
    // Make program of the source code in the context
    try {
        m_program = cl::Program(m_context,
                                sourceCode_common
                                + sourceCode_config
                                + sourceCode_convolve1
                                + sourceCode_convolve3
                                + sourceCode_sgemm);
    } catch (const cl::Error &e) {
        myprintf("Error getting kernels: %s: %d", e.what(), e.err());
        throw std::runtime_error("Error getting OpenCL kernels.");
    }

    auto t = Tuner<net_t>(*this, m_context, m_device);
    if (m_tensorcore) {
        t.enable_tensorcore();
    }

    auto sgemm_tuners =
        t.load_sgemm_tuners(channels, batch_size * WINOGRAD_P, channels, WINOGRAD_TILE);

    // Some NVIDIA drivers are buggy and will fail to compile the rest of the
    // kernels after a tuning run.
    if (cfg_tune_only) {
        // Originally this was an exit() but this will make the tuner
        // only tune the first GPU.  Return instead.  Exit will be called
        // after all GPUs are created.
        return;
    }

    // Build program for these specific devices
    try {
        std::string args = m_cl_args;
        // Intel iGPUs need vector types for math for best performance
        if (m_device.getInfo<CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT>() > 1) {
            args += " -DWINOGRAD_SIMD";
        }

        args += sgemm_tuners;
        m_program.build(args.c_str());
    } catch (const cl::Error&) {
        myprintf("Error building kernels: %s\n",
                 m_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_device).c_str());
        throw std::runtime_error("Error building OpenCL kernels.");
    }

    OpenCLContext tdata;
    ensure_context_initialized(tdata);

    process_tuners(sgemm_tuners);

    m_wavefront_size =
        tdata.m_sgemm_kernel.getWorkGroupInfo<
            CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(m_device);
    myprintf("Wavefront/Warp size: %lu\n", m_wavefront_size);

    m_max_workgroup_size = m_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    m_max_workgroup_dims = m_device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();

    myprintf("Max workgroup size: %lu\n", m_max_workgroup_size);
    myprintf("Max workgroup dimensions: ");
    for (auto d : m_max_workgroup_dims) {
        myprintf("%lu ", d);
    }
    myprintf("\n");

    m_init_ok = true;
}

template <typename net_t>
bool OpenCL<net_t>::has_fp16_compute() {
    return m_fp16_compute;
}

template <typename net_t>
bool OpenCL<net_t>::has_tensor_cores() {
    return m_tensorcore;
}

template <typename net_t>
std::string OpenCL<net_t>::get_device_name() {
    std::stringstream ss;

    ss << "OpenCL: ";
    ss << m_device.getInfo<CL_DEVICE_VENDOR>() << " ";
    ss << m_device.getInfo<CL_DEVICE_NAME>() << " @ ";
    ss << m_device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "MHz";

    return ss.str();
}

template class OpenCL<float>;
template class OpenCL_Network<float>;

template class OpenCL<half_float::half>;
template class OpenCL_Network<half_float::half>;
