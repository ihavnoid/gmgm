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


#include <cstdio>

#include "half/half.hpp"
#include "Network.h"
#include "OpenCLScheduler.h"
#include "globals.h"

using gmgm::globals::myprintf;

static bool cfg_tune_only = false;

static size_t ceilMultiple(size_t a, size_t b) {
    if (a % b == 0) {
        return a;
    }

    auto ret = a + (b - a % b);
    return ret;
}

class from_float{
public:
    from_float(const std::vector<float> & f) : m_f(f) {}

    operator const std::vector<float>&() {
        return m_f;
    }

    operator std::vector<half_float::half>() {
        auto ret = std::vector<half_float::half>(m_f.size());
        std::copy(cbegin(m_f), cend(m_f), begin(ret));
        return ret;
    }
private:
    const std::vector<float>& m_f;
};

template <typename T>
static std::vector<T> zeropad_U(const std::vector<float>& U,
                                const int outputs, const int channels,
                                const int outputs_pad,
                                const int channels_pad) {
    // Fill with zeroes
    auto Upad =
        std::vector<T>(WINOGRAD_TILE * outputs_pad * channels_pad);

    for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++){
        for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
            for (auto c = 0; c < channels; c++) {
                for (auto o = 0; o < outputs; o++) {
                    Upad[xi * (WINOGRAD_ALPHA * outputs_pad * channels_pad)
                         + nu * (outputs_pad * channels_pad)
                         + c * outputs_pad +
                          o] =
                    U[xi * (WINOGRAD_ALPHA * outputs * channels)
                      + nu * (outputs * channels)
                      + c * outputs
                      + o];
                }
            }
        }
    }

    return Upad;
}

template <typename net_t>
OpenCLScheduler<net_t>::OpenCLScheduler() {
    auto silent{false};

    for(int gpu = 0; ; gpu++) {
        try {
            auto opencl = std::make_unique<OpenCL<net_t>>(gpu, silent);
            auto net = std::make_unique<OpenCL_Network<net_t>>(*opencl);
            m_opencl.push_back(std::move(opencl));
            m_networks.push_back(std::move(net));
    
            // Starting next GPU, let's not dump full list of GPUs.
            silent = true;
        } catch( std::runtime_error & ) {
            if(gpu == 0) {
                throw;
            } else {
                gmgm::globals::num_scheduler_threads = gmgm::globals::batch_size * gpu * 2;
                break;
            }
        }
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::initialize(const int channels) {
    // Launch the worker threads.  Minimum 1 worker per GPU, but use enough threads
    // so that we can at least concurrently schedule something to the GPU.
    auto num_worker_threads = gmgm::globals::num_scheduler_threads / gmgm::globals::batch_size / (m_opencl.size() + 1) + 1;
    myprintf("Scheduler threads : %lu\n", num_worker_threads);
    auto gnum = 0;
    for (auto & opencl : m_opencl) {
        opencl->initialize(channels, gmgm::globals::batch_size);

        for (auto i = unsigned{0}; i < num_worker_threads; i++) {
            auto t = std::thread(&OpenCLScheduler<net_t>::batch_worker, this, gnum);
            m_worker_threads.push_back(std::move(t));
        }
        gnum++;
    }

    // Exit immediately after tuning.  We should exit here because we skipped
    // initializing rest of the kernels due to some NVIDIA drivers crashing.
    if (cfg_tune_only) {
        exit(EXIT_SUCCESS);
    }
}

template <typename net_t>
OpenCLScheduler<net_t>::~OpenCLScheduler() {
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_running = false;
    }
    m_cv.notify_all();
    for (auto & x : m_worker_threads) {
        x.join();
    }
}

template<typename net_t>
bool OpenCLScheduler<net_t>::needs_autodetect() {
    for (auto& opencl : m_opencl) {
        // If any card has no native fp16 compute, we'll have to benchmark.
        if (!opencl->has_fp16_compute() && !opencl->has_tensor_cores()) {
            return true;
        }
    }
    return false;
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_input_convolution(
    unsigned int filter_size,
    unsigned int channels,
    unsigned int outputs,
    const std::vector<float>& weights,
    const std::vector<float>& means,
    const std::vector<float>& variances) {

    for (const auto& opencl_net : m_networks) {
        const auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();

        const auto mwg = tuners[0];
        const auto kwg = tuners[2];
        const auto vwm = tuners[3];

        const auto m_ceil = ceilMultiple(ceilMultiple(outputs, mwg), vwm);
        const auto k_ceil = ceilMultiple(ceilMultiple(channels, kwg), vwm);

        const auto Upad = zeropad_U<net_t>(weights,
                                           outputs, channels,
                                           m_ceil, k_ceil);
        opencl_net->push_input_convolution(
            filter_size, channels, outputs,
            Upad, from_float(means), from_float(variances)
        );
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_residual(unsigned int filter_size,
                                           unsigned int channels,
                                           unsigned int outputs,
                                           const std::vector<float>& weights_1,
                                           const std::vector<float>& means_1,
                                           const std::vector<float>& variances_1,
                                           const std::vector<float>& weights_2,
                                           const std::vector<float>& means_2,
                                           const std::vector<float>& variances_2,
                                           const std::vector<float>& se_1,
                                           const std::vector<float>& se_2
                                           ) {
    for (const auto& opencl_net : m_networks) {
        const auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();

        const auto mwg = tuners[0];
        const auto vwm = tuners[3];

        const auto m_ceil = ceilMultiple(ceilMultiple(outputs, mwg), vwm);
        const auto Upad1 = zeropad_U<net_t>(weights_1,
                                            outputs, outputs,
                                            m_ceil, m_ceil);
        const auto Upad2 = zeropad_U<net_t>(weights_2,
                                            outputs, outputs,
                                            m_ceil, m_ceil);
        opencl_net->push_residual(filter_size, channels, outputs,
                                  Upad1,
                                  from_float(means_1),
                                  from_float(variances_1),
                                  Upad2,
                                  from_float(means_2),
                                  from_float(variances_2),
                                  from_float(se_1),
                                  from_float(se_2));
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_convolve(unsigned int filter_size,
                                           unsigned int channels,
                                           unsigned int outputs,
                                           const std::vector<float>& conv_weights,
                                           const std::vector<float>& bn_pol_means,
                                           const std::vector<float>& bn_pol_stddevs,
                                           const std::vector<float>& fc_weights,
                                           const std::vector<float>& fc_biases) {
    for (const auto & opencl_net : m_networks) {
        opencl_net->push_convolve(filter_size, channels, outputs,
                                  from_float(conv_weights),
                                  from_float(bn_pol_means),
                                  from_float(bn_pol_stddevs),
                                  from_float(fc_weights),
                                  from_float(fc_biases)
        );
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_weights(
    unsigned int filter_size,
    unsigned int channels,
    unsigned int outputs,
    std::shared_ptr<const ForwardPipeWeights> weights) {

    auto weight_index = size_t{0};

    // Winograd filter transformation changes filter size to 4x4
    push_input_convolution(filter_size, channels, outputs,
                           weights->m_conv_weights[weight_index],
                           weights->m_batchnorm_means[weight_index],
                           weights->m_batchnorm_stddevs[weight_index]);
    weight_index++;

    // residual blocks : except the first entry,
    // the second ~ last entry is all on residual topwer
    for (auto i = size_t{0}; i < weights->m_conv_weights.size()/2; i++) {
        push_residual(filter_size, outputs, outputs,
                      weights->m_conv_weights[weight_index],
                      weights->m_batchnorm_means[weight_index],
                      weights->m_batchnorm_stddevs[weight_index],
                      weights->m_conv_weights[weight_index + 1],
                      weights->m_batchnorm_means[weight_index + 1],
                      weights->m_batchnorm_stddevs[weight_index + 1],
                      weights->m_squeeze_1[weight_index + 1],
                      weights->m_squeeze_2[weight_index + 1]
        );

        // first convolution don't have squeeze layers
        assert(weights->m_squeeze_1[weight_index].empty());
        assert(weights->m_squeeze_2[weight_index].empty());

        weight_index += 2;
    }

    // Output head convolutions
    push_convolve(1, outputs, Network::OUTPUTS_POLICY,
        weights->m_conv_pol_w,
        weights->m_bn_pol_w1,
        weights->m_bn_pol_w2,
        weights->m_ip_pol_w,
        weights->m_ip_pol_b
    );
    push_convolve(1, outputs, Network::OUTPUTS_VALUE,
        weights->m_conv_val_w,
        weights->m_bn_val_w1,
        weights->m_bn_val_w2,
        weights->m_ip_val_w,
        weights->m_ip_val_b
    );
}

template <typename net_t>
void OpenCLScheduler<net_t>::forward(const std::vector<float>& input,
                                     std::vector<float>& output_pol,
                                     std::vector<float>& output_val) {
    auto entry = std::make_shared<ForwardQueueEntry>(input, output_pol, output_val);
    std::unique_lock<std::mutex> lk(entry->mutex);
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_forward_queue.push_back(entry);
    }
    m_cv.notify_one();
    entry->cv.wait(lk);
}

#ifndef NDEBUG
struct batch_stats_t batch_stats;
#endif

template <typename net_t>
void OpenCLScheduler<net_t>::batch_worker(const size_t gnum) {
    constexpr auto in_size = Network::INPUT_CHANNELS * gmgm::BOARD_W * gmgm::BOARD_H;
    constexpr auto out_pol_size = Network::OUTPUTS_POLICY;
    constexpr auto out_val_size = Network::OUTPUTS_VALUE;

    OpenCLContext context;

    // batch scheduling heuristic.
    // Returns the batch picked up from the queue (m_forward_queue)
    // 1) Wait for m_waittime milliseconds for full batch
    // 2) if we don't have a full batch then just do a single eval
    //
    // The purpose of m_waittime is to prevent the system from deadlocking
    // because we were waiting for a job too long, while the job is never
    // going to come due to a control dependency (e.g., evals stuck on a
    // critical path).  To do so:
    //
    // 1) if we couldn't form a batch after waiting m_waittime ms, it means
    // that we hit the critical path and should do scalar evals.
    // Wait 1ms shorter next time.
    //
    // 2) if we picked up a single eval, but were getting additional evals
    // while that single eval was being processed, it means that we made
    // the wrong decision.  Wait 2ms longer next time.

    auto pickup_task = [this] () {
        std::list<std::shared_ptr<ForwardQueueEntry>> inputs;
        size_t count = 0;

        std::unique_lock<std::mutex> lk(m_mutex);
        while (true) {
            if (!m_running) return inputs;

            count = m_forward_queue.size();
            if (count >= static_cast<size_t>(gmgm::globals::batch_size)) {
                count = gmgm::globals::batch_size;
                break;
            }

            bool timeout = !m_cv.wait_for(
                lk,
                std::chrono::milliseconds(m_waittime),
                [this] () {
                    return !m_running || m_forward_queue.size() >= static_cast<size_t>(gmgm::globals::batch_size);
                }
            );

            if (!m_forward_queue.empty()) {
                if (timeout && m_single_eval_in_progress.exchange(true) == false) {
                    // Waited long enough but couldn't form a batch.
                    // Check if there is any other single eval in progress, and if not,
                    // do one from this thread.
                    if (m_waittime > 1) {
                        m_waittime--;
                    }
                    count = 1;
                    break;
                }
            }
        }
        // Move 'count' evals from shared queue to local list.
        auto end = begin(m_forward_queue);
        std::advance(end, count);
        std::move(begin(m_forward_queue), end, std::back_inserter(inputs));
        m_forward_queue.erase(begin(m_forward_queue), end);

        return inputs;
    };

    auto batch_input = std::vector<float>();
    auto batch_output_pol = std::vector<float>();
    auto batch_output_val = std::vector<float>();

    while (true) {
        auto inputs = pickup_task();
        auto count = inputs.size();

        if (!m_running) {
            return;
        }

#ifndef NDEBUG
        if (count == 1) {
            batch_stats.single_evals++;
        } else {
            batch_stats.batch_evals++;
        }
#endif

        // prepare input for forward() call
        batch_input.resize(in_size * count);
        batch_output_pol.resize(out_pol_size * count);
        batch_output_val.resize(out_val_size * count);

        auto index = size_t{0};
        for (auto & x : inputs) {
            std::unique_lock<std::mutex> lk(x->mutex);
            std::copy(begin(x->in), end(x->in), begin(batch_input) + in_size * index);
            index++;
        }

        // run the NN evaluation
        m_networks[gnum]->forward(
            batch_input, batch_output_pol, batch_output_val, context, count);

        // Get output and copy back
        index = 0;
        for (auto & x : inputs) {
            std::copy(begin(batch_output_pol) + out_pol_size * index,
                      begin(batch_output_pol) + out_pol_size * (index + 1),
                      begin(x->out_p));
            std::copy(begin(batch_output_val) + out_val_size * index,
                      begin(batch_output_val) + out_val_size * (index + 1),
                      begin(x->out_v));
            x->cv.notify_all();
            index++;
        }

        if (count == 1) {
            m_single_eval_in_progress = false;
        }
    }
}

template class OpenCLScheduler<float>;
template class OpenCLScheduler<half_float::half>;

