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

#include "PositionEval.h"
#include "Search.h"

gmgm::Search::Search()
{
    running = true;
    for(unsigned int i=0; i<num_threads; i++) {
        child_threads.emplace_back([this]() {
            while(running) {
                SearchTask t;
                {
                    std::unique_lock<std::mutex> lk(mutex);
                    while(taskqueue.size() == 0) {
                        cv.wait(lk);
                        if(!running) {
                            return;
                        }
                    }
                    t = std::move(taskqueue.front());
                    taskqueue.pop_front();
                }
                t.promise.set_value(t.searchfunc());
            }
        });
    }
}

gmgm::Search::~Search() {
    running = false;
    cv.notify_all();
    for(auto & t : child_threads) {
        t.join();
    }
}

std::vector<gmgm::SearchResult> gmgm::Search::analyze(SearchNode & root)
{
    std::vector<SearchResult> ret;
    for(auto & x : root.children) {
        if(x.get_child() != nullptr) {
            ret.emplace_back(
                x.get_child()->accum_visits,
                x.get_child()->accum_value / x.get_child()->accum_visits,
                x.policy,
                x.move
            );
        } else {
            ret.emplace_back(
                0,
                0.0f,
                x.policy,
                x.move
            );
        }
    }
    return ret;
}

std::vector<gmgm::SearchResult> gmgm::Search::search(Board &b, PositionEval * eval, int visits, int ms)
{
    std::unique_ptr<SearchNode> root;

    // see if we can create root from rootcache.  This means we have to compare the cache board
    // and this board
    if(rootcache != nullptr) {
        if(boardcache.get_movenum() <= b.get_movenum()) {
            std::deque<Move> stack;
            while(boardcache.get_movenum() != b.get_movenum()) {
                auto m = b.unmove();
                stack.push_front(m);
            }

            if(boardcache.compare(b)) {
                root = std::move(rootcache); 
                for(auto & mv : stack) {
                    if(root == nullptr) {
                        break;
                    }
                    for(auto & x : root->children) {
                        if(x.move == mv) {
                            root.reset(x.release_child());
                            break;
                        }
                    }
                }
            }

            for(const auto & mv : stack) {
                b.move(mv);
            }
        }
    }

    if (root == nullptr) {
        root = std::make_unique<SearchNode>();
    }
    
    std::vector<std::thread> threads;
    std::atomic<size_t> runcount{(size_t)(root->accum_visits.load())};
#if 0
    if (root->accum_visits.load() > 0) {
        std::cout << "Tree reuse : " << root->accum_visits.load() << std::endl;
    }
#endif

    auto start = std::chrono::system_clock::now();
    auto next_print_time = start + std::chrono::milliseconds(2500);
    Board b2 = b;
    do {
        root->expand(*eval, b2);
        runcount++;

        auto now = std::chrono::system_clock::now();
        if(start + std::chrono::milliseconds(ms) < now) {
            break;
        }
        if (print_period > 0 && next_print_time < now) {
            auto winrate = root->accum_value.load() / root->accum_visits.load();
            std::cerr << winrate << " (" << root->accum_visits.load() << ") " 
                      << root->print_best_path() << std::endl;
            next_print_time = now + std::chrono::milliseconds(print_period);
        }

        // fork threads but not too many - too many will result in everybody spinning on root
        while(threads.size() < static_cast<size_t>(num_threads-1)
            && threads.size() < runcount.load()) {
            auto work_thread = [this, start, visits, ms, &runcount, &b, eval, &root] () {
                Board b2 = b;
                while(runcount.load() < static_cast<size_t>(visits)) {
                    root->expand(*eval, b2);
                    runcount++;
    
                    auto now = std::chrono::system_clock::now();
                    if(start + std::chrono::milliseconds(ms) < now) {
                        break;
                    }
                }
            };
            threads.emplace_back(work_thread);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    } while(runcount.load() < static_cast<size_t>(visits));
   
    for(auto & x : threads) {
        x.join();
    }
    
    boardcache = b;
    auto ret = analyze(*root);
    rootcache = std::move(root);
    
    return ret;
}

std::future<std::vector<gmgm::SearchResult>> gmgm::Search::search_async(Board &b, PositionEval * eval, int visits, int ms)
{
    SearchTask t;
    t.searchfunc = std::bind(&gmgm::Search::search, this, b, eval, visits, ms);

    auto f = t.promise.get_future();
    {
        std::lock_guard<std::mutex> lk(mutex);
        taskqueue.push_back(std::move(t));
    }
    cv.notify_one();
    return f;
}
