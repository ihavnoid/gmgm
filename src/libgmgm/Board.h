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

#ifndef __GMGM_BOARD_HH__
#define __GMGM_BOARD_HH__

#include <sstream>
#include <array>
#include <vector>
#include <deque>
#include <iostream>
#include <cassert>

#include <chrono>

#include "globals.h"

class lua_State;

namespace gmgm {

static auto constexpr BOARD_W = 9;
static auto constexpr BOARD_H = 10;

enum class PieceType : int8_t {
    GOONG = 0,
    SA = 1,
    CHA = 3,
    SANG = 5,
    MA = 7,
    PO = 9,
    JOL = 11,
    PIECE_COUNT = 16
};

enum class Side : int8_t {
    CHO,
    HAN = 16,
    NONE = 32
};

enum class StartingState : int8_t {
    SMSM = 0,
    SMMS = 1,
    MSSM = 2,
    MSMS = 3
};

class Move {
public:
    int8_t piece;
    int8_t yx_from;
    int8_t yx_to;
    int8_t captured;
    Move(int8_t piece_, int8_t yx_from_, int8_t yx_to_, int8_t captured_) :
        piece(piece_), yx_from(yx_from_), yx_to(yx_to_), captured(captured_) {}

    operator std::string() const {
        std::ostringstream oss;
        auto convert_to_printable = [](int v) {
            if( globals::flip_display) {
                auto y = v / 10;
                auto x = v % 10;
                y = 9 - y;
                v = y * 10 + x;
            }
            if (v >= 90) v -= 90;
            else v += 10;
            if (v % 10 == 9) v -= 9;
            else v++;
            return v;
        };
        oss << convert_to_printable(yx_from) << "-" << convert_to_printable(yx_to);
        return oss.str();
    }
    std::string string() const {
        return *this;
    }
    bool operator==(const Move & m) const {
        return (yx_from == m.yx_from) && (yx_to == m.yx_to) && piece == m.piece;
    }
    bool operator!=(const Move & m) const {
        return !(*this == m);
    }
    bool is_pass() const {
        // pass is the only move when 'from' equals 'to'
        return yx_from == yx_to;
    }
};

class PositionEval;

class Board {
    friend class PositionEval;
private:
    class stop_legal_move_search_exception {};
    float cached_score_han;
    float cached_score_cho;
    mutable Side to_move = Side::CHO;
    // value : PieceType + Side
    mutable std::array<std::int8_t, 100> board;
    // move - boardhash pair
    struct BoardHistory {
    public:
        Move move;
        std::uint64_t boardhash;
        std::uint64_t playhash;
        bool was_jang;
        BoardHistory(const Move & m, std::uint64_t bh, std::uint64_t ph, bool jang) :
            move(m), boardhash(bh), playhash(ph), was_jang(jang) {}
    };
    std::deque<BoardHistory> history;
    mutable std::uint64_t boardhash;
    std::uint64_t playhash;
    mutable std::vector<Move> legal_move_cache;
    mutable std::vector<Move> legal_move_opponent_cache;
    template <typename T> void __get_legal_moves(T callback) const;

    void move_piece_only(const Move & m) const;
    void unmove_piece_only(const Move & m) const;
    Side winner_piece_only() const;
    float __score(Side side) const;

    // return any losing sequence if player must lose, empty vector otherwise
    std::vector<Move> __must_lose(int max_depth, bool & timeout);
    std::vector<Move> __can_win(int max_depth, bool & timeout);
public:

    Board(std::string cho_state, std::string han_state);
    Board(StartingState cho_state, StartingState han_state);
    Board(const Board & b) = default;

    void __init(StartingState cho_state, StartingState han_state);
    
    void clear_cache() {
        legal_move_cache.clear();
        legal_move_opponent_cache.clear();
    }

    bool is_jang() const;
    bool can_win_immediately() const;
    bool compare(const Board & other) const;
    int get_piece_on(int yx) const;

    // alpha-beta search
    
    // return the winning sequence if player can win, empty vector otherwise
    std::vector<Move> can_win(int max_depth, int timeout_ms = 10000000);
    

    std::uint64_t get_hash() {
        return playhash;
    }

    const std::vector<Move>& get_legal_moves() const;

    // get legal moves if I was the opponent
    const std::vector<Move> & get_legal_moves_if_opponent() const;

    void print(std::ostream & os) const;

    Side winner() const;
    // float score_cho() const { assert(cached_score_cho == __score(Side::CHO)); return cached_score_cho; }
    // float score_han() const { assert(cached_score_han == __score(Side::HAN)); return cached_score_han; }
    float score_cho() const { return cached_score_cho; }
    float score_han() const { return cached_score_han; }

    int get_movenum() const { return history.size(); }
    void move(const Move & m);
    Move unmove();
    Side get_to_move() const { return to_move; }
    Side opponent(Side x) const {
        if (x == Side::CHO) return Side::HAN;
        if (x == Side::HAN) return Side::CHO;
        return Side::NONE;
    }
};

}

#endif // __GMGM_BOARD_HH__

// vim: set ts=4 sw=4 expandtab:
