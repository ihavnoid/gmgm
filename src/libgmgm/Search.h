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

#ifndef __GMGM__SEARCH_HH__
#define __GMGM__SEARCH_HH__

#include <thread>
#include <atomic>
#include <future>

#include "SearchNode.h"

namespace gmgm {

class PositionEval;

class SearchResult {
public:
    int visits;
    float winrate;
    float policy;
    Move move;
    SearchResult(int v, float w, float p, Move m) : visits(v), winrate(w), policy(p), move(m) {}
};

class Search {
    class SearchTask {
    public:
        std::function<std::vector<SearchResult>()> searchfunc;
        std::promise<std::vector<SearchResult>> promise;
    };
    std::vector<std::thread> child_threads;

    std::condition_variable cv;
    std::mutex mutex;
    std::deque<SearchTask> taskqueue;
    std::atomic<bool> running{false};

    std::unique_ptr<SearchNode> rootcache;
    Board boardcache{StartingState::SMSM, StartingState::SMSM};
private:
    std::vector<SearchResult> analyze(SearchNode & root);
public:
    unsigned int num_threads = 1;
    unsigned int print_period = 0;
    Search();
    ~Search();
    std::vector<SearchResult> search(Board & b, PositionEval * eval, int visits, int ms);
    std::future<std::vector<gmgm::SearchResult>> search_async(Board & b, PositionEval * eval, int visits, int ms);
};

}

#endif
