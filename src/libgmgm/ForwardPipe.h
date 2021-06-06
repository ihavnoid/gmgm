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
*/



#ifndef FORWARDPIPE_H_INCLUDED
#define FORWARDPIPE_H_INCLUDED

#include <memory>
#include <vector>

class ForwardPipe {
public:
    class ForwardPipeWeights {
    public:
        // Input + residual block tower
        std::vector<std::vector<float>> m_conv_weights;
        std::vector<std::vector<float>> m_conv_biases;
        std::vector<std::vector<float>> m_batchnorm_means;
        std::vector<std::vector<float>> m_batchnorm_stddevs;
        
        // squeeze excitation layers (empty if we don't want SE)
        std::vector<std::vector<float>> m_squeeze_1;
        std::vector<std::vector<float>> m_squeeze_2;

        // Policy head
        std::vector<float> m_conv_pol_w;
        std::vector<float> m_conv_pol_b;
        std::vector<float> m_bn_pol_w1;
        std::vector<float> m_bn_pol_w2;
        std::vector<float> m_ip_pol_w;
        std::vector<float> m_ip_pol_b;

        std::vector<float> m_conv_val_w;
        std::vector<float> m_conv_val_b;
        std::vector<float> m_bn_val_w1;
        std::vector<float> m_bn_val_w2;
        std::vector<float> m_ip_val_w;
        std::vector<float> m_ip_val_b;
    };

    virtual ~ForwardPipe() = default;

    virtual void initialize(const int channels) = 0;
    virtual bool needs_autodetect() { return false; };
    virtual void forward(const std::vector<float>& input,
                         std::vector<float>& output_pol,
                         std::vector<float>& output_val) = 0;
    virtual void push_weights(unsigned int filter_size,
                              unsigned int channels,
                              unsigned int outputs,
                              std::shared_ptr<const ForwardPipeWeights> weights) = 0;
};

#endif
