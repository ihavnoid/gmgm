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


#ifndef CPUPIPE_H_INCLUDED
#define CPUPIPE_H_INCLUDED

#include <vector>
#include <cassert>

#include "ForwardPipe.h"

class CPUPipe : public ForwardPipe {
public:
    virtual void initialize(const int channels);
    virtual void forward(const std::vector<float>& input,
                         std::vector<float>& output_pol,
                         std::vector<float>& output_val);

    virtual void push_weights(unsigned int filter_size,
                              unsigned int channels,
                              unsigned int outputs,
                              std::shared_ptr<const ForwardPipeWeights> weights);
private:
    void winograd_transform_in(const std::vector<float>& in,
                               std::vector<float>& V,
                               const int C);

    void winograd_sgemm(const std::vector<float>& U,
                        const std::vector<float>& V,
                        std::vector<float>& M,
                        const int C, const int K);

    void winograd_transform_out(const std::vector<float>& M,
                                std::vector<float>& Y,
                                const int K);

    void winograd_convolve3(const int outputs,
                            const std::vector<float>& input,
                            const std::vector<float>& U,
                            std::vector<float>& V,
                            std::vector<float>& M,
                            std::vector<float>& output);


    int m_input_channels;

    // Input + residual block tower
    std::shared_ptr<const ForwardPipeWeights> m_weights;

    std::vector<float> m_conv_pol_w;
    std::vector<float> m_conv_val_w;
    std::vector<float> m_conv_pol_b;
    std::vector<float> m_conv_val_b;
};
#endif
