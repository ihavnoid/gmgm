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

#include <iostream>
#include <string>

#include "libgmgm/gmgm.h"

#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include <boost/format.hpp>


std::string net_filename = "";
std::unique_ptr<gmgm::PositionEval> position_eval;
gmgm::Board board("smsm", "smsm");
gmgm::Search search;

unsigned int search_num = 10000;
unsigned int search_time_ms = 10000;

#define CHECK_PARAM_4() \
{ \
    if(s1 == "") return false; \
    if(s2 == "") return false; \
    if(s3 == "") return false; \
    if(s4 == "") return false; \
}
#define CHECK_PARAM_3() \
{ \
    if(s1 == "") return false; \
    if(s2 == "") return false; \
    if(s3 == "") return false; \
    if(s4 != "") return false; \
}
#define CHECK_PARAM_2() \
{ \
    if(s1 == "") return false; \
    if(s2 == "") return false; \
    if(s3 != "") return false; \
    if(s4 != "") return false; \
}
#define CHECK_PARAM_1() \
{ \
    if(s1 == "") return false; \
    if(s2 != "") return false; \
    if(s3 != "") return false; \
    if(s4 != "") return false; \
}
#define CHECK_PARAM_0() \
{ \
    if(s1 != "") return false; \
    if(s2 != "") return false; \
    if(s3 != "") return false; \
    if(s4 != "") return false; \
}

class Command {
protected:
    std::string cmdname_;
    std::string usage_;
    std::string help_;
    std::function<bool(std::string,std::string,std::string,std::string)> f_;
public:
    Command(
        std::string cmdname, std::string usage, std::string help,
        std::function<bool(std::string,std::string,std::string,std::string)> f
    )
        : cmdname_(cmdname), usage_(usage), help_(help), f_(f) {}
    std::string cmdname() { return cmdname_; }
    std::string usage() { return usage_; }
    std::string help() { return help_; }
    std::string extended_help() { return help_; }
    bool do_command(std::string s1, std::string s2, std::string s3, std::string s4) { return f_(s1, s2, s3, s4); }
    void invalid_syntax() {
        std::cout << "command syntax : " << cmdname_ << " " << usage_ << std::endl;
    }
};

class Parameter {
private:
    std::string cmdname_;
    std::string help_;
public:
    Parameter(
        std::string cmdname, std::string help
    )
        : cmdname_(cmdname), help_(help) {}

    std::string cmdname() {
        return cmdname_;
    }
    std::string help() {
        return help_;
    }
    virtual std::string get() = 0;
    virtual bool set(std::string s) = 0;
};

class UIntSet : public Parameter {
private:
    unsigned int & ival_;
    std::function<void(void)> post_update_func_;
public:
    UIntSet(
        std::string cmdname, std::string help, unsigned int & ival,
        std::function<void(void)> post_update_func = [](){}
    ) : Parameter(cmdname, help), ival_(ival), post_update_func_(post_update_func) {}
    
    virtual std::string get() {
        return std::to_string(ival_);
    }

    virtual bool set(std::string s1) {
        try {
            ival_ = std::stoi(s1);
            post_update_func_();
            return true;
        } catch(...) {}
        return false;
    }
};


class BoolSet : public Parameter {
private:
    bool & bval_;
    std::function<void(void)> post_update_func_;
public:
    BoolSet(
        std::string cmdname, std::string help, bool & bval,
        std::function<void(void)> post_update_func = [](){}
    ) : Parameter(cmdname, help), bval_(bval), post_update_func_(post_update_func) {}

    virtual bool set(std::string s1) { 
        if(s1 == "true") {
            bval_ = true;
            post_update_func_();
            return true;
        } else if(s1 == "false") {
            bval_ = false;
            post_update_func_();
            return true;
        }
        return false;
    }
    virtual std::string get() {
        if(bval_) return "true";
        else return "false";
    }
};

static void help(std::string s);

static std::string load_net() {
    if(net_filename == "") {
        return "";
    }

    std::string ret = "";

    Network * network = new Network();
    try {
        network->initialize(net_filename);
    } catch(std::runtime_error &x) {
        delete network;
        network = nullptr;
        ret = x.what();
        net_filename = "";
    }

    if(network != nullptr) {
        position_eval.reset(network);
    }

    return ret;
}

static void think() {
    if(board.winner() != gmgm::Side::NONE) {
        std::cout << "Game already over. "
            << "Type new [cho_position] [han_position] for new game."
            << std::endl;
        return;
    }
    if(position_eval == nullptr) {
        std::cout << "No net loaded.  Type loadnet [net file] for new game."
            << std::endl;
        return;
    }

    std::cout << "Thinking..." << std::endl;
    std::vector<gmgm::SearchResult> candidate = search.search(board, position_eval.get(), search_num, search_time_ms);

    if(candidate.size() > 0) {
        std::sort(candidate.begin(), candidate.end(), 
            [](auto & x, auto & y) {
                return x.visits > y.visits;
            }
        );
        for(auto & x : candidate) {
            std::cout << boost::format("%-8s%8d %0.3f") % x.move.string() % x.visits % x.winrate << std::endl;
        }
        auto &max_candidate = candidate[0];
        std::cout << "move " << max_candidate.move.string() << std::endl;
        board.move(max_candidate.move);
    }
}

static void display() {
    board.print(std::cout);
    std::cout << std::endl;
    std::cout << "legal moves: " << std::endl;
    auto lm = board.get_legal_moves();
    int ptr = 0;
    for(auto m : lm) {
        std::cout << boost::format("%-8s") % m.string() << " ";
        ptr++;
        if(ptr == 8) {
            ptr = 0;
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
    if (position_eval == nullptr) {
        std::cout << "Neural net not loaded.  Please use loadnet [filename] to load neural network" << std::endl;
    }
}


auto create_param_list = []() {
    std::initializer_list<Parameter*> param_list = {
        new UIntSet("batch_size", "Neural net batch size.  Optimal size may differ from GPU to GPU", gmgm::globals::batch_size, 
            [](){
                if(position_eval != nullptr) {
                    std::cout << "Reloading net as we changed batch size..." << std::endl;
                    load_net();
                }
            }
        ),
        new UIntSet("cache_size", "Neural net evaluation cache size.  1 entry consumes roughly 4kB of system memory", gmgm::globals::cache_size),
        new UIntSet("num_threads", "Search evaluation parallelism.  Recommended size is at least 2x of batch_size", search.num_threads),
        new UIntSet("print_period", "Print period.  How often you print verbose messages while searching", search.print_period),
        new BoolSet("verbose_mode", "Verbose mode.  If true, will dump more diagnostic messages", gmgm::globals::verbose_mode),
        new UIntSet("search_num", "Amount of searches to do per move", search_num),
        new UIntSet("search_time_ms", "Maximum time to search, in milliseconds", search_time_ms)
    };

    std::vector<std::unique_ptr<Parameter>> p;
    for(auto x : param_list) {
        p.push_back(std::unique_ptr<Parameter>(x));
    }
    return p;
};
std::vector<std::unique_ptr<Parameter>> parameters = create_param_list();


std::vector<Command> command_list {
    Command("getparam", "[variable name]", "Get parameter value", [](auto s1, auto s2, auto s3, auto s4) {
            if(s1 == "") {
                for (auto & x : parameters) {
                    std::cout << x->cmdname() << " " << x->get() << std::endl;
                }
                return true;
            } else {
                CHECK_PARAM_1();
                for (auto & x : parameters) {
                    if(x->cmdname() == s1) {
                        std::cout << x->get() << std::endl;
                        return true;
                    }
                }
                std::cout << "Invalid variable " << s1 << std::endl;
                return true;
            }
        }
    ),
    Command("setparam", "[variable name] [value]", "Set parameter value", [](auto s1, auto s2, auto s3, auto s4) {
            CHECK_PARAM_2();
            for (auto & x : parameters) {
                if(x->cmdname() == s1) {
                    bool success = x->set(s2);
                    if (!success) {
                        std::cout << "Cannot set " << s2 << " on " << s1 << std::endl; 
                        return false;
                    }
                    return true;
                }
            }
            std::cout <<"Invalid variable " << s1 << std::endl;
            return true;
        }
    ),
    Command("exit", "", "Exit application", [](auto s1, auto s2, auto s3, auto s4) {
            CHECK_PARAM_0();
            std::cout << std::endl;
            std::cout << std::endl;
            std::exit(0);
            return true;
        }
    ),
    Command("new", "[starting_position_cho] [starting_position_han]",
        "Start new game.  starting_position is one of these:  smsm, smms, mssm or msms",
        [](auto s1, auto s2, auto s3, auto s4) {
            CHECK_PARAM_2();
            board = gmgm::Board(s1, s2);
            return true;
        }
    ),
    Command("display", "", "Show current board status",
        [](auto s1, auto s2, auto s3, auto s4) {
            CHECK_PARAM_0();
            display();
            return true;
        }
    ),
    Command("help", "", "Help message",
        [](auto s1, auto s2, auto s3, auto s4) {
            if(s1 != "") {
                CHECK_PARAM_1();
            }
            help(s1);
            return true;
        }
    ),
    Command("loadnet", "[neural_net_filename]", "Load neural net weight file",
        [](auto s1, auto s2, auto s3, auto s4) {
            CHECK_PARAM_1();

            std::cout << "Loading net " << s1 << "..." << std::endl;
            net_filename = s1;
            auto msg = load_net();
            
            if(msg != "") {
                std::cout <<"Failed loading net :" << msg << std::endl;
            }
            return true;
        }
    ),
    Command("think", "", "Let AI play",
        [](auto s1, auto s2, auto s3, auto s4) {
            CHECK_PARAM_0();
            think();
            return true;
        }
    ),
    Command("undo", "", "Undo move",
        [](auto s1, auto s2, auto s3, auto s4) {
            CHECK_PARAM_0();
            if(board.get_movenum() > 0) {
                board.unmove();
            }
            return true;
        }
    ),
    Command("move", "[move_number]", "Make a move.  move_number should be in a form of [source-destination]",
        [](auto s1, auto s2, auto s3, auto s4) {
            CHECK_PARAM_1();
            auto lm = board.get_legal_moves();
            for(auto & m : lm) {
                if (m.string() == s1) {
                    board.move(m);
                    return true;
                }
            }
            std::cout << "Invalid move" << std::endl;
            return true;
        }
    ),
    Command("play", "[move_number]", "Make a move, and let AI think. move_number should be in a form of [source-destination]",
        [](auto s1, auto s2, auto s3, auto s4) {
            CHECK_PARAM_1();
            auto lm = board.get_legal_moves();
            for(auto & m : lm) {
                if (m.string() == s1) {
                    board.move(m);
                    think();
                    return true;
                }
            }
            std::cout << "Invalid move" << std::endl;
            return true;
        }
    ),
    Command("flip", "", "Flip board - change side of Cho and Han",
        [](auto s1, auto s2, auto s3, auto s4) {
            CHECK_PARAM_0();
            gmgm::globals::flip_display = !gmgm::globals::flip_display;
            return true;
        }
    ),
};

void help(std::string s) {
    if (s == "") {
        std::cout << "These are the commands available." << std::endl;
        for (auto & x : command_list) {
            std::cout << (boost::format("%-11s %s") % x.cmdname() % x.usage())  << std::endl;

            boost::char_separator<char> sep("\n");
            auto ss = x.help();
            boost::tokenizer<boost::char_separator<char>> tok(ss, sep);
            for (auto xx: tok) {
                std::cout << "            " << xx << std::endl;
            }
        }
    } else {
        for (auto & x : command_list) {
            if(x.cmdname() == s) {
                std::cout << (boost::format("%-11s %s") % x.cmdname() % x.usage())  << std::endl;
    
                boost::char_separator<char> sep("\n");
                auto ss = x.extended_help();
                boost::tokenizer<boost::char_separator<char>> tok(ss, sep);
                for (auto xx: tok) {
                    std::cout << "            " << xx << std::endl;
                }
            }
        }
    }
}

static void process_command(std::string line) {
    boost::char_separator<char> sep(" ");
    boost::tokenizer<boost::char_separator<char>> tok(line, sep);
    std::vector<std::string> tokv;
    for(auto & x: tok) {
        tokv.push_back(x);
    }

    if(tokv.size() == 0) {
        return;
    }
    for(auto & x : command_list) {
        if(x.cmdname() == tokv[0]) {
            std::string s1 = "";
            std::string s2 = "";
            std::string s3 = "";
            std::string s4 = "";
            if(tokv.size() >= 2) { s1 = tokv[1]; }
            if(tokv.size() >= 3) { s2 = tokv[2]; }
            if(tokv.size() >= 4) { s3 = tokv[3]; }
            if(tokv.size() >= 5) { s4 = tokv[4]; }
            if(tokv.size() > 5) {
                x.invalid_syntax();
            } else {
                auto ret = x.do_command(s1, s2, s3, s4);
                if(!ret) {
                    x.invalid_syntax();
                }
            }
            return;
        }
    }
    std::cout << "Invalid command. Pleas type 'help' for commands" << std::endl;
}

static void console() {
    while (true) {
        display();
        std::cout << "> " << std::flush;
        std::string l;
        std::getline(std::cin, l);
        if(std::cin.eof() || std::cin.bad()) {
            return;
        }

        process_command(l);
    }
}

int main(int argc, const char ** argv) {
    namespace po = boost::program_options;
    
    gmgm::globals::cache_size = 20000;
    gmgm::globals::batch_size = 12;

    search.num_threads = 12;
    search.print_period = 2500;

    // Declare the supported options.
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
    ;
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    
    
    if (vm.count("help")) {
        std::cout << "gmgm: a modern, deep learning based Janggi AI" << std::endl << std::endl;
        std::cout << desc << "\n";
        return 0;
    }

    console();
    return 0;
}
