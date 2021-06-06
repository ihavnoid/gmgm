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

#include "SearchNode.h"
#include "PositionEval.h"
#include "Board.h"

std::string gmgm::SearchNode::print_best_path()
{
    std::string ret = "";
    SearchNode * max_child = nullptr;
    {
        if(acquire_expand()) {
            expand_cancel();
            return "";
        }

        expanded_rlock();
        int max_eval = 0;
        for(const auto &x : children) {
            if(x.get_child() != nullptr && x.get_child()->accum_visits > max_eval) {
                max_eval = x.get_child()->accum_visits;
                max_child = x.get_child();
                ret = x.move.string() + " ";
            }
        }
        expanded_runlock();
    }
    if(max_child != nullptr) {
        ret += max_child->print_best_path();
    }
    return ret;
}

void gmgm::SearchNode::create_children(std::shared_ptr<EvalResult> eval_result, Board & board)
{
    accum_visits++;
    accum_value = board.get_to_move() == Side::CHO ? (-eval_result->value) : eval_result->value;

    // net output is -1 ~ 1
    // we need 0 ~ 1 if we want to apply virtual loss
    accum_value = (accum_value + 1.0f) * 0.5f;

    // we have no short-term rewards.  Create some by putting score on bias
    float score_based_bias = board.score_han() - board.score_cho();
    accum_value = accum_value * (1.0f - gmgm::globals::score_based_bias_rate);
    accum_value = accum_value + gmgm::globals::score_based_bias_rate * 0.5f *
        (1.0f + std::tanh(score_based_bias / 14.4f));

    children.reserve(eval_result->policy.size());
    float total_policy = 0.0f;
    for(auto & x : eval_result->policy) {
        total_policy += x.second;
        total_policy += gmgm::globals::score_based_bias_rate / eval_result->policy.size();
    }
    for(auto & x : eval_result->policy) {
        float policy = x.second;
        if(policy < 0) policy = 0;
        policy = policy + gmgm::globals::score_based_bias_rate / eval_result->policy.size();
        policy = policy / total_policy;
        children.emplace_back(x.first, policy);
    }
}

float gmgm::SearchNode::expand(PositionEval & eval, Board & board)
{
    auto winner = board.winner();
    if(winner == Side::CHO) {
        add_value(0.0f);
        return 0.0f;
    } else if(winner == Side::HAN) {
        add_value(1.0f);
        return 1.0f;
    }

    
    std::shared_ptr<EvalResult> ev;
    
    if(!is_expanded()) {
        // pre-evaluate on the non-critical section
        ev = eval.evaluate(board);
    }

    if(acquire_expand()) {
        if(ev->policy.empty()) {
            ev = eval.evaluate(board);
        }

        assert(children.empty());
        vloss += VIRTUAL_LOSS;
        create_children(ev, board);
        float ret = accum_value;
        expand_done();
        vloss -= VIRTUAL_LOSS;
        return ret;
    } else {
        SearchCandidate * best = nullptr;
        float best_val = -9999.0f;
        vloss += VIRTUAL_LOSS;
        expanded_rlock();
        for(auto & candidate : children) {
            auto child = candidate.get_child();

            float _value = 
                (child != nullptr && child->accum_visits != 0)
                ? child->accum_value.load() : accum_value.load()
            ;
            int _vloss = 
                (child != nullptr && child->accum_visits != 0)
                ? child->vloss.load() : vloss.load()
            ;
            int _visits = 
                (child != nullptr && child->accum_visits != 0)
                ? child->accum_visits.load() : accum_visits.load()
            ;
            // For cho, 0 is winning and 1 is losing
            if(board.get_to_move() == Side::CHO) {
                _value = _visits - _value;
            }
            auto winrate = _value / (_visits + _vloss);

            const auto numerator = std::sqrt(double(accum_visits + vloss));
            const auto denom = 1.0 + (child != nullptr ? (child->accum_visits.load() + child->vloss.load()) : 0);
            const auto puct = candidate.policy * (numerator / denom);
            const auto value = winrate + 3.0f * puct;
            if(value > best_val) {
                best = &candidate;
                best_val = value;
            }
        }

        if(best == nullptr) {
            assert(false);
        }

        best->createChild();

        auto m = best->move;
        expanded_runlock();

        board.move(m);

        float ret = best->get_child()->expand(eval, board);
        add_value(ret);
        board.unmove();

        vloss -= VIRTUAL_LOSS;
        return ret;
    }
}
