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

#include <cassert>
#include <cmath>

#include "PositionEval.h"

gmgm::PositionEval::PositionEval() {
    for(auto & p : primary_cache) {
        p.reserve(gmgm::globals::cache_size*2);
    }
}

gmgm::PositionOutputFeatures gmgm::PositionEval::extract_output_features(
    const gmgm::Board & b,
    const Move & m,
    gmgm::Side final_winner,
    int final_movenum)
{
    // this is used for evaluating using games without the evaluation results
    // (e.g., from some other engine or human play)
    // we only use visits and move on the original 'extract_output_features' method
    std::vector<gmgm::SearchResult> sr;
    sr.emplace_back(100, 0.5, 0.5, m);
    return extract_output_features(b, sr, final_winner, final_movenum);
}

gmgm::PositionOutputFeatures gmgm::PositionEval::extract_output_features(
    const gmgm::Board & b,
    const std::vector<SearchResult> & result,
    gmgm::Side final_winner,
    int final_movenum)
{
    PositionOutputFeatures ret;
    constexpr int feature_map_size = 16;
    
    ret.features.resize(feature_map_size);
    for(auto & b : ret.features) {
        std::fill(begin(b), end(b), 0.0f);
    }
    int total_cnt = 0;
    for(const auto & entry : result) {
        total_cnt += entry.visits;
    }
    for(const auto & entry : result) {
        int y2 = entry.move.yx_to / 10;
        int x2 = entry.move.yx_to % 10;
        auto piece = b.board[entry.move.yx_from];
        assert(piece != 0x20);
        
        float rate = 1.0f * entry.visits / total_cnt;
        ret.features.at(piece % 0x10)[y2 * BOARD_W + x2] = rate;
    }
    if(final_winner == b.to_move) {
        ret.value = 1.0f * std::exp(-final_movenum / 400.0);
    } else {
        ret.value = -1.0f * std::exp(-final_movenum / 400.0);
    }

    return ret;
}

gmgm::PositionInputFeatures gmgm::PositionEval::extract_input_features(const gmgm::Board & b)
{
    constexpr int feature_map_size = 66;
    PositionInputFeatures ret;
    ret.features.resize(feature_map_size);
    for(auto & x : ret.features) {
        for(auto & xx : x) {
            xx = 0.0f;
        }
    }

    bool han_to_move = b.to_move == Side::HAN;
    for(int yx = 0; yx < BOARD_W * BOARD_H; yx++) {
        auto p = b.board[(yx / BOARD_W) * 10 + (yx % BOARD_W)];
        if(han_to_move) {
            if(p < 0x10) {
                p += 0x10;
            } else if(p < 0x20) {
                p -= 0x10;
            }
        }
        if(p < 0x20) {
            ret.features[p][yx] = 1.0f;
        }
    }
    const auto & legal_moves = b.get_legal_moves();
    for(auto & m : legal_moves) {
        int y2 = m.yx_to / 10;
        int x2 = m.yx_to % 10;
        int p = b.board[m.yx_from];
        assert(p < 0x20);
        p = p % 16;
        ret.features[0x20 + p][y2 * BOARD_W + x2] = 1.0f;
    }

    const auto & legal_moves_opp = b.get_legal_moves_if_opponent();
    for(auto & m : legal_moves_opp) {
        int y2 = m.yx_to / 10;
        int x2 = m.yx_to % 10;
        int p = b.board[m.yx_from];
        assert(p < 0x20);
        p = p % 16;
        ret.features[0x30 + p][y2 * BOARD_W + x2] = 1.0f;
    }

    if (han_to_move) {
        std::fill(begin(ret.features[65]), end(ret.features[65]), 1.0f);
    } else {
        std::fill(begin(ret.features[64]), end(ret.features[64]), 1.0f);
    }
    return ret;
}

const std::shared_ptr<gmgm::EvalResult> gmgm::PositionEval::evaluate(Board & b) {
#if 0
    return evaluate_raw(b);
#else
    auto h = b.get_hash();
    auto pos = h%16;
    std::shared_ptr<gmgm::EvalResult> ret;
    bool found_result = false;

    {
        std::unique_lock<std::mutex> lk(mutex[pos]);
        auto iter1 = primary_cache[pos].find(h);
        auto iter2 = secondary_cache[pos].find(h);
        if(iter1 != primary_cache[pos].end()) {
            ret = iter1->second;
            assert(ret != nullptr);
            found_result = true;
        } else if(iter2 != secondary_cache[pos].end()) {
            primary_cache[pos][h] = std::move(iter2->second);
            secondary_cache[pos].erase(iter2);
            ret = primary_cache[pos][h];
            assert(ret != nullptr);
            found_result = true;
        }
    }

    if(!found_result) {
        ret = evaluate_raw(b);
        {
            std::unique_lock<std::mutex> lk(mutex[pos]);
            primary_cache[pos][h] = ret;
            if(primary_cache[pos].size() >= gmgm::globals::cache_size) {
                secondary_cache[pos].swap(primary_cache[pos]);
                primary_cache[pos].clear();
                primary_cache[pos].reserve(gmgm::globals::cache_size*2);
            }
        }
        assert(ret != nullptr);
    } else {
        auto lm = b.get_legal_moves();
        // validate if legal move matches
        for(auto i=size_t{0}; i<lm.size(); i++) {
            if(ret->policy[i].first != lm[i]) {
		// I don't know if we will ever touch this code, though.
		// Happens on hash collision - that is, two states with a same hash value
                std::cerr << "PositionEval collision" << std::endl;
                ret = evaluate_raw(b);
                std::unique_lock<std::mutex> lk(mutex[pos]);
                primary_cache[pos][h] = ret;
                return ret;
            }
        }
        assert(ret != nullptr);
    }
    assert(ret != nullptr);
    return ret;
#endif
}

std::shared_ptr<gmgm::EvalResult> gmgm::PositionEval::evaluate_raw(Board & b) {
    auto sptr = std::make_shared<gmgm::EvalResult>();
    auto & policy = sptr->policy;
    auto moves = b.get_legal_moves();
    policy.reserve(moves.size());
    float attack_delta = 0.0f;
    for(auto m : moves) {
        policy.emplace_back(m, 1.0f / moves.size());
        if(m.captured != 0x20 && m.yx_from != m.yx_to) {
            switch(m.captured % 16) {
                case 0: // Goong
                    attack_delta += 28.0f;
                    break;
                case 1: case 2: // Sa
                    attack_delta += 3.0f;
                    break;
                case 3: case 4: // Cha
                    attack_delta += 13.0f;
                    break;
                case 9: case 10: // Po
                    attack_delta += 7.0f;
                    break;
                case 5: case 6:
                    attack_delta += 3.0f;
                    break;
                case 7: case 8:
                    attack_delta += 5.0f;
                    break;
                case 11: case 12: case 13: case 14: case 15:
                    attack_delta += 2.0f;
                    break;
                default:
                    break;
            }
        }
    }
    auto move_opponent = b.get_legal_moves_if_opponent();
    for(auto m : move_opponent) {
        if(m.captured != 0x20 && m.yx_from != m.yx_to) {
            switch(m.captured % 16) {
                case 0: // Goong
                    attack_delta -= 28.0f;
                    break;
                case 1: case 2: // Sa
                    attack_delta -= 3.0f;
                    break;
                case 3: case 4: // Cha
                    attack_delta -= 13.0f;
                    break;
                case 9: case 10: // Po
                    attack_delta -= 7.0f;
                    break;
                case 5: case 6:
                    attack_delta -= 3.0f;
                    break;
                case 7: case 8:
                    attack_delta -= 5.0f;
                    break;
                case 11: case 12: case 13: case 14: case 15:
                    attack_delta -= 2.0f;
                    break;
                default:
                    break;
            }
        }
    }
    float value = 0.0f;

    // placeholder implementation
    if(b.winner() == b.to_move) value = 1.0f;
    else if(b.winner() != Side::NONE) value = -1.0f;
    else {
        value = (b.score_han() - b.score_cho()) / 14.4f;
        if(b.get_to_move() == Side::CHO) {
            value= -value;
        }
        float delta = 0.002f * moves.size() - 0.002f * move_opponent.size();
        value += delta;
        value += attack_delta / 70.0f;
        value = std::tanh(value);
    }

    sptr->value = value;

    return sptr;
}

std::string gmgm::PositionInputFeatures::tostring() const {
    std::ostringstream oss;
    for(const auto & x : features) {
        int v = 0;
        for(int i = 0; i < 90; i++) {
            if(x[i] > 0.5f) {
                v |= (1 << (i % 4) );
            }
            if(i % 4 == 3) {
                oss << std::hex << v;
                v = 0;
            }
        }
        oss << std::hex << v << std::endl;
    }

    return oss.str();
}

std::shared_ptr<gmgm::PositionEval::RawResult> gmgm::PositionEval::evaluate_raw(const std::vector<float> & ) {
    throw std::runtime_error("Not implemented");
    return nullptr;
}

std::string gmgm::PositionOutputFeatures::tostring() const {
    std::ostringstream oss;
    for(const auto & x : features) {
        for(const auto & xx : x) {
            oss << xx << " ";
        }
        oss << std::endl;
    }
    oss << value << std::endl;
    return oss.str();
}


int gmgm::PositionEval::benchmark(gmgm::Board & b, int runtime_ms) {
    std::atomic<bool> running{true};
    std::atomic<int> cnt{0};

    std::vector<std::thread> t;
    for (unsigned int i=0; i<globals::num_scheduler_threads; i++) {
        t.emplace_back([&running, &cnt, this, b](){
            auto b2 = b;
            while(running.load()) {
                evaluate_raw(b2);
                cnt++;
            }
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(runtime_ms));
    running = false;
    for(auto & x : t) {
        x.join();
    }

    return cnt.load();
}
