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

#ifndef __GMGM_POSITION_EVAL_HH__
#define __GMGM_POSITION_EVAL_HH__

#include <unordered_map>
#include <functional>
#include <mutex>

#include "Board.h"
#include "Search.h"

namespace std {
    template<> struct hash<gmgm::Move>{
        std::size_t operator() (gmgm::Move const & s) const noexcept
        {
            return std::hash<int>{}(
                static_cast<int>(s.piece) * 1000000 + s.yx_from * 1000 + s.yx_to
            );
        }
    };
}

namespace gmgm {

class EvalResult {
public:
    /// possibility to take each move
    std::vector<std::pair<Move, float>> policy;

    /// likelihood of winning; -1 = opponent wins, 1 == player wins
    float value = 0.0f;
};

class PositionEval;

class PositionInputFeatures {
    friend class PositionEval;
public:
    std::vector<std::array<float, BOARD_W * BOARD_H>> features;
public:
    std::string tostring() const;
};

class PositionOutputFeatures {
    friend class PositionEval;
public:
    std::vector<std::array<float, BOARD_W * BOARD_H>> features;
    float value;
public:
    std::string tostring() const;
};

class PositionEval {
public:
    typedef std::pair<std::vector<float>,float> RawResult;
private:
    std::array<std::mutex,16> mutex;
    std::array<std::unordered_map<std::uint64_t, std::shared_ptr<EvalResult>>,16> primary_cache;
    std::array<std::unordered_map<std::uint64_t, std::shared_ptr<EvalResult>>,16> secondary_cache;
public:
    PositionEval();
    virtual ~PositionEval() {}
    PositionInputFeatures extract_input_features(const Board & b);
    PositionOutputFeatures extract_output_features(const Board & b, const std::vector<SearchResult> & result, Side final_winner, int final_movenum);
    PositionOutputFeatures extract_output_features(const Board & b, const Move & m, Side final_winner, int final_movenum);

    const std::shared_ptr<gmgm::EvalResult> evaluate(Board & b);
    virtual std::shared_ptr<gmgm::EvalResult> evaluate_raw(Board & b);
    virtual std::shared_ptr<RawResult> evaluate_raw(const std::vector<float> & v);

    int benchmark(Board & b, int ms);
};

}
#endif
