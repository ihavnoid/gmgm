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

#ifndef __GMGM_SEARCH_NODE_HH__
#define __GMGM_SEARCH_NODE_HH__

#include <memory>
#include <set>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <cassert>

#include "Board.h"

namespace gmgm {

class SearchNode;
class PositionEval;
class Board;
class EvalResult;

class SearchCandidate {
private:
    std::atomic<SearchNode*> child;
public:
    Move move;
    float policy;
    SearchCandidate(Move m, float p) : child(nullptr), move(m), policy(p) {}
    SearchCandidate(const SearchCandidate & s) : child(nullptr), move(s.move), policy(s.policy){
        assert(false);
    }
    ~SearchCandidate();
    SearchNode * get_child() const {
        return child.load();
    }
    SearchNode * release_child() {
        SearchNode * ret = child.load();
        child.store(nullptr);
        return ret;
    }
    void createChild();
};

constexpr int VIRTUAL_LOSS = 3;
class SearchNode {
public:
    // value : 0 == cho wins, 1 == han wins
    std::atomic<float> accum_value{0.0f};
    std::atomic<int> accum_visits{0};
    std::atomic<int> vloss{0};
    std::vector<SearchCandidate> children;
    float expand(PositionEval & eval, Board & board);

    std::string print_best_path();
private:
    void create_children(std::shared_ptr<EvalResult> eval_result, Board & board);

    void add_value(float v) {
        accum_visits += 1;
        
        while(true) {
            float prev_v = accum_value.load();
            float new_v = prev_v + v;
            if (std::atomic_compare_exchange_weak
                (
                    &accum_value,
                    &prev_v,
                    new_v
                )
            ) { break; }
        }
    }

    // 0 : unexpanded
    // 1 : expanding
    // 3 : expanded
    // 2 : expanded, wlock
    // 4 ~ xxx : expanded, rlock
    std::atomic<int> state{0};
    bool acquire_expand() {
        int i = 0;
        while (true) {
            auto v = state.load();
            if(v == 0) {
                int expected = 0;
                if(state.compare_exchange_weak(expected, 1)) {
                    return true;
                }
            } else if(v >= 2) {
                return false;
            }

            i++;
            if (i % 1024 == 0) {
                std::this_thread::yield();
            }
        }
    }

    void expand_done() {
        state.store(3);
    }
    void expand_cancel() {
        state.store(0);
    }

    bool is_expanded() {
        return state.load() >= 2;
    }

    void expanded_rlock() {
        int i = 0;
        while (true) {
            auto x = state.load();
            if(x >= 3) {
                if(state.compare_exchange_weak(x, x+1)) {
                    return;
                }
            }
            i++;
            if (i % 1024 == 0) {
                std::this_thread::yield();
            }
        }
    }
    void expanded_wlock() {
        int i = 0;
        while (true) {
            auto x = state.load();
            if(x == 3) {
                if(state.compare_exchange_weak(x, 2)) {
                    return;
                }
            }
            i++;
            if (i % 1024 == 0) {
                std::this_thread::yield();
            }
        }
    }
    void expanded_runlock() {
        state--;
    }
    void expanded_wunlock() {
        state++;
    }
};

inline SearchCandidate::~SearchCandidate() {
    if(child.load() != nullptr) {
        delete child.load();
    }
}

inline void SearchCandidate::createChild() {
    SearchNode * n = new SearchNode();
    SearchNode * exp = nullptr;
    if(!child.compare_exchange_strong(exp, n)) {
        delete n;
    } 
}
}

#endif
