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

#include <vector>
#include <cassert>
#include <iomanip>
#include <algorithm>
#include <thread>

#include "Board.h"

#include "globals.h"

#include "BoardHashConstants.h"

gmgm::Board::Board(std::string cho_state, std::string han_state) {
    auto convert = [](std::string x) {
        if(x == "msms") return StartingState::MSMS;
        if(x == "smms") return StartingState::SMMS;
        if(x == "mssm") return StartingState::MSSM;
        if(x == "smsm") return StartingState::SMSM;
        throw std::invalid_argument("Expecting starting state to be one of \"msms\", \"smsm\", \"mssm\", or \"smms\"");
    };
    __init(convert(cho_state), convert(han_state));
}

gmgm::Board::Board(StartingState cho_state, StartingState han_state) {
    __init(cho_state, han_state);
}

int gmgm::Board::get_piece_on(int yx) const {
    return board.at(yx);
}

void gmgm::Board::__init(StartingState cho_state, StartingState han_state) {
    std::array<std::array<std::uint8_t,9>,10> __board;
    // row 9 is cho, row 0 is han
    switch(cho_state) {
    case StartingState::SMSM:
        __board[9] = {
            0x03, 0x05, 0x07, 0x01, 0x20, 0x02, 0x06, 0x08, 0x04,
        };
        break;
    case StartingState::SMMS:
        __board[9] = {
            0x03, 0x05, 0x07, 0x01, 0x20, 0x02, 0x08, 0x06, 0x04,
        };
        break;
    case StartingState::MSSM:
        __board[9] = {
            0x03, 0x07, 0x05, 0x01, 0x20, 0x02, 0x06, 0x08, 0x04,
        };
        break;
    case StartingState::MSMS:
        __board[9] = {
            0x03, 0x07, 0x05, 0x01, 0x20, 0x02, 0x08, 0x06, 0x04,
        };
        break;
    }
    switch(han_state) {
    case StartingState::SMSM:
        __board[0] = {
            0x13, 0x15, 0x17, 0x11, 0x20, 0x12, 0x16, 0x18, 0x14,
        };
        break;
    case StartingState::SMMS:
        __board[0] = {
            0x13, 0x15, 0x17, 0x11, 0x20, 0x12, 0x18, 0x16, 0x14,
        };
        break;
    case StartingState::MSSM:
        __board[0] = {
            0x13, 0x17, 0x15, 0x11, 0x20, 0x12, 0x16, 0x18, 0x14,
        };
        break;
    case StartingState::MSMS:
        __board[0] = {
            0x13, 0x17, 0x15, 0x11, 0x20, 0x12, 0x18, 0x16, 0x14,
        };
        break;
    }
    __board[8] = { 0x20, 0x20, 0x20, 0x20, 0x00, 0x20, 0x20, 0x20, 0x20 };
    __board[1] = { 0x20, 0x20, 0x20, 0x20, 0x10, 0x20, 0x20, 0x20, 0x20 };

    __board[7] = { 0x20, 0x09, 0x20, 0x20, 0x20, 0x20, 0x20, 0x0a, 0x20 };
    __board[2] = { 0x20, 0x19, 0x20, 0x20, 0x20, 0x20, 0x20, 0x1a, 0x20 };

    __board[6] = { 0x0b, 0x20, 0x0c, 0x20, 0x0d, 0x20, 0x0e, 0x20, 0x0f };
    __board[3] = { 0x1b, 0x20, 0x1c, 0x20, 0x1d, 0x20, 0x1e, 0x20, 0x1f };

    __board[5] = { 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20 };
    __board[4] = { 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20 };

    boardhash = 0;
    for(int y=0; y<BOARD_H; y++) {
        for(int x = 0; x < BOARD_W; x++) {
            if(__board[y][x] < 0x20) {
                boardhash = boardhash ^ board_hash_constants[y * BOARD_W * 32 + x * 32 + __board[y][x]];
            }
        }
    }
    playhash = boardhash;

    for(int y=0; y<10; y++) {
        for(int x=0; x<9; x++) {
            board[y * 10 + x] = __board[y][x];
        }
    }
    cached_score_han = 73.5f;
    cached_score_cho = 72.0f;
}

void gmgm::Board::print(std::ostream & os) const {
    os << "To move : " << (to_move == Side::CHO ? "CHO" : "HAN") << std::endl;
    int highlight_x = -1;
    int highlight_y = -1;
    if(!history.empty()) {
        highlight_x = history.back().move.yx_to % 10;
        highlight_y = history.back().move.yx_to / 10;
    }
    os << "   1  2  3  4  5  6  7  8  9" << std::endl;
    for (int y = 0; y < BOARD_H; y++) {
        if (y == 9) {
            os << 0 << " ";
        } else {
            os << (y+1) << " ";
        }
        auto yy = globals::flip_display ? BOARD_H - y - 1 : y;
        for (int x = 0; x < BOARD_W; x++) {
            std::string tok = ".";
            switch(board[yy*10+x]) {
                case 0x00: tok="G"; break;
                case 0x10: tok="g"; break;
                case 0x01: case 0x02: tok="X"; break;
                case 0x11: case 0x12: tok="x"; break;
                case 0x03: case 0x04: tok="C"; break;
                case 0x13: case 0x14: tok="c"; break;
                case 0x05: case 0x06: tok="S"; break;
                case 0x15: case 0x16: tok="s"; break;
                case 0x07: case 0x08: tok="M"; break;
                case 0x17: case 0x18: tok="m"; break;
                case 0x09: case 0x0a: tok="P"; break;
                case 0x19: case 0x1a: tok="p"; break;
                case 0x0b: case 0x0c: case 0x0d: case 0x0e: case 0x0f: tok="J"; break;
                case 0x1b: case 0x1c: case 0x1d: case 0x1e: case 0x1f: tok="j"; break;
                default: tok="."; break;
            }
            if(board[yy*10+x] / 0x10 == 1) {
                os << "\x1b[1;31m";
            }
            if(board[yy*10+x] / 0x10 == 0) {
                os << "\x1b[1;32m";
            }
            if(highlight_x == x && highlight_y == yy) {
                os << "(" << tok << ")";
            } else {
                os << " " << tok << " ";
            }
            os << "\x1b[0m";
        }
        if (y == 9) {
            os << " " << 0;
        } else {
            os << " " << (y+1);
        }
        os << std::endl;
    }
    os << "   1  2  3  4  5  6  7  8  9" << std::endl;
}

template <typename T> inline void gmgm::Board::__get_legal_moves(T callback) const {
    auto is_empty = [this](int yx) {
        return (0x20 == board[yx]);
    };
    auto is_same_side = [this](int yx_from, int yx_to) {
        auto s1 = board[yx_from];
        auto s2 = board[yx_to];
        return (s1 >> 4) == (s2 >> 4);
    };

    auto append_sa_goong = [&] (int yx_from) {
        auto possible_legal_move = [&](int yx_to) {
            if (!is_same_side(yx_from, yx_to)) {
                callback(
                    yx_from,
                    yx_to,
                    board[yx_to]
                );
            }
        };
        switch(yx_from) {
            case 3: case 73:
                possible_legal_move(yx_from + 1); 
                possible_legal_move(yx_from + 10); 
                possible_legal_move(yx_from + 11); 
                break;
            case 4: case 74:
                possible_legal_move(yx_from - 1); 
                possible_legal_move(yx_from + 10); 
                possible_legal_move(yx_from + 1); 
                break;
            case 5: case 75:
                possible_legal_move(yx_from - 1); 
                possible_legal_move(yx_from + 10); 
                possible_legal_move(yx_from + 9); 
                break;
            case 13: case 83:
                possible_legal_move(yx_from - 10); 
                possible_legal_move(yx_from + 10); 
                possible_legal_move(yx_from + 1); 
                break;
            case 14: case 84: 
                possible_legal_move(yx_from - 11); 
                possible_legal_move(yx_from - 10); 
                possible_legal_move(yx_from - 9); 
                possible_legal_move(yx_from - 1); 
                possible_legal_move(yx_from + 1); 
                possible_legal_move(yx_from + 9); 
                possible_legal_move(yx_from + 10); 
                possible_legal_move(yx_from + 11); 
                break;
            case 15: case 85:
                possible_legal_move(yx_from - 1);
                possible_legal_move(yx_from - 10); 
                possible_legal_move(yx_from + 10); 
                break;
            case 23: case 93:
                possible_legal_move(yx_from + 1);
                possible_legal_move(yx_from - 10); 
                possible_legal_move(yx_from - 9); 
                break;
            case 24: case 94:
                possible_legal_move(yx_from - 1);
                possible_legal_move(yx_from + 1);
                possible_legal_move(yx_from - 10);
                break;
            case 25: case 95:
                possible_legal_move(yx_from - 11);
                possible_legal_move(yx_from - 10);
                possible_legal_move(yx_from - 1);
                break;
            default:
                assert(false);
        }
    };

    auto get_board_elem = [&](int yx) {
        return board[yx];
    };
    auto append_cha = [&] (int yx_from) {
        auto handle_as_cha = [&] (int yx_from, int yx_to) {
            auto empty = is_empty(yx_to);
            auto sameside = is_same_side(yx_from, yx_to);
            if (!sameside) {
                callback(
                    yx_from,
                    yx_to,
                    get_board_elem(yx_to)
                );
            }
            return !empty;
        };
        auto y = yx_from / 10;
        auto x = yx_from % 10;

        auto yo = y - 1;
        while(yo >= 0) {
            auto yx_to = yo * 10 + x;
            if(handle_as_cha(yx_from, yx_to)) break;
            yo--;
        }
        yo = y + 1;
        while(yo < BOARD_H) {
            auto yx_to = yo * 10 + x;
            if(handle_as_cha(yx_from, yx_to)) break;
            yo++;
        }
        auto xo = x - 1;
        while(xo >= 0) {
            auto yx_to = y * 10 + xo;
            if(handle_as_cha(yx_from, yx_to)) break;
            xo--;
        }
        xo = x + 1;
        while(xo < BOARD_W) {
            auto yx_to = y * 10 + xo;
            if(handle_as_cha(yx_from, yx_to)) break;
            xo++;
        }
        // diagnoal
        if(yx_from == 93 || yx_from == 84 || yx_from == 75
            || yx_from == 23 || yx_from == 14 || yx_from == 5) 
        {
            auto xo = x + 1;
            auto yo = y - 1;
            while(xo < 6) {
                auto yx_to = yo * 10 + xo;
                if(handle_as_cha(yx_from, yx_to)) break;
                xo++;
                yo--;
            }
            
            xo = x - 1;
            yo = y + 1;
            while(xo >= 3) {
                auto yx_to = yo * 10 + xo;
                if(handle_as_cha(yx_from, yx_to)) break;
                xo--;
                yo++;
            }
        }

        // diagnoal
        if(yx_from == 95 || yx_from == 84 || yx_from == 73
            || yx_from == 25 || yx_from == 14 || yx_from == 3) 
        {
            auto xo = x + 1;
            auto yo = y + 1;
            while(xo < 6) {
                auto yx_to = yo * 10 + xo;
                if(handle_as_cha(yx_from, yx_to)) break;
                xo++;
                yo++;
            }
            
            xo = x - 1;
            yo = y - 1;
            while(xo >= 3) {
                auto yx_to = yo * 10 + xo;
                if(handle_as_cha(yx_from, yx_to)) break;
                xo--;
                yo--;
            }
        }
    };

    auto append_po = [&] (int yx_from) {
        auto handle_as_po = [&] (int yx_from, int yx_to) {
            auto empty = is_empty(yx_to);
            auto sameside = is_same_side(yx_from, yx_to);
            auto piece = get_board_elem(yx_to);
            auto is_po = (piece == 0x9 || piece == 0xa || piece == 0x19 || piece == 0x1a);
            if (!is_po && !sameside) {
                callback(
                    yx_from,
                    yx_to,
                    get_board_elem(yx_to)
                );
            }
            return !empty;
        };
        auto y = yx_from / 10;
        auto x = yx_from % 10;

        bool found_jump = false;
        auto yo = y - 1;
        while(yo >= 0) {
            auto yx_to = yo * 10 + x;
            if(!found_jump) {
                if(!is_empty(yx_to)) {
                    if(board[yx_to] == 0x9 || board[yx_to] == 0xa || board[yx_to] == 0x19 || board[yx_to] == 0x1a) {
                        break;
                    } else {
                        found_jump = true;
                    }
                }
            } else {
                if(handle_as_po(yx_from, yx_to)) break;
            }
            yo--;
        }
        yo = y + 1;
        found_jump = false;
        while(yo < BOARD_H) {
            auto yx_to = yo * 10 + x;
            if(!found_jump) {
                if(!is_empty(yx_to)) {
                    if(board[yx_to] == 0x9 || board[yx_to] == 0xa || board[yx_to] == 0x19 || board[yx_to] == 0x1a) {
                        break;
                    } else {
                        found_jump = true;
                    }
                }
            } else {
                if(handle_as_po(yx_from, yx_to)) break;
            }
            yo++;
        }
        auto xo = x - 1;
        found_jump = false;
        while(xo >= 0) {
            auto yx_to = y * 10 + xo;
            if(!found_jump) {
                if(!is_empty(yx_to)) {
                    if(board[yx_to] == 0x9 || board[yx_to] == 0xa || board[yx_to] == 0x19 || board[yx_to] == 0x1a) {
                        break;
                    } else {
                        found_jump = true;
                    }
                }
            } else {
                if(handle_as_po(yx_from, yx_to)) break;
            }

            xo--;
        }
        xo = x + 1;
        found_jump = false;
        while(xo < BOARD_W) {
            auto yx_to = y * 10 + xo;
            if(!found_jump) {
                if(!is_empty(yx_to)) {
                    if(board[yx_to] == 0x9 || board[yx_to] == 0xa || board[yx_to] == 0x19 || board[yx_to] == 0x1a) {
                        break;
                    } else {
                        found_jump = true;
                    }
                }
            } else {
                if(handle_as_po(yx_from, yx_to)) break;
            }
            xo++;
        }
        // diagnoal
        if(yx_from == 93 || yx_from == 75
            || yx_from == 23 || yx_from == 5) 
        {
            auto xo = x + 1;
            auto yo = y - 1;
            found_jump = false;
            while(xo < 6) {
                auto yx_to = yo * 10 + xo;
                if(!found_jump) {
                    if(!is_empty(yx_to)) {
                        if(board[yx_to] == 0x9 || board[yx_to] == 0xa || board[yx_to] == 0x19 || board[yx_to] == 0x1a) {
                            break;
                        } else {
                            found_jump = true;
                        }
                    }
                } else {
                    if(handle_as_po(yx_from, yx_to)) break;
                }

                xo++;
                yo--;
            }
            
            xo = x - 1;
            yo = y + 1;
            found_jump = false;
            while(xo >= 3) {
                auto yx_to = yo * 10 + xo;
                if(!found_jump) {
                    if(!is_empty(yx_to)) {
                        if(board[yx_to] == 0x9 || board[yx_to] == 0xa || board[yx_to] == 0x19 || board[yx_to] == 0x1a) {
                            break;
                        } else {
                            found_jump = true;
                        }
                    }
                } else {
                    if(handle_as_po(yx_from, yx_to)) break;
                }
                xo--;
                yo++;
            }
        }

        // diagnoal
        if(yx_from == 95 || yx_from == 73
            || yx_from == 25 || yx_from == 3) 
        {
            auto xo = x + 1;
            auto yo = y + 1;
            found_jump = false;
            while(xo < 6) {
                auto yx_to = yo * 10 + xo;
                if(!found_jump) {
                    if(!is_empty(yx_to)) {
                        if(board[yx_to] == 0x9 || board[yx_to] == 0xa || board[yx_to] == 0x19 || board[yx_to] == 0x1a) {
                            break;
                        } else {
                            found_jump = true;
                        }
                    }
                } else {
                    if(handle_as_po(yx_from, yx_to)) break;
                }
                xo++;
                yo++;
            }
            
            xo = x - 1;
            yo = y - 1;
            found_jump = false;
            while(xo >= 3) {
                auto yx_to = yo * 10 + xo;
                if(!found_jump) {
                    if(!is_empty(yx_to)) {
                        if(board[yx_to] == 0x9 || board[yx_to] == 0xa || board[yx_to] == 0x19 || board[yx_to] == 0x1a) {
                            break;
                        } else {
                            found_jump = true;
                        }
                    }
                } else {
                    if(handle_as_po(yx_from, yx_to)) break;
                }

                xo--;
                yo--;
            }
        }
    };


    auto append_ma = [&](int yx_from) {
        int y = yx_from / 10; int x = yx_from % 10;
           
        auto check_x = [&](int direction) {
            int xo = x; int yo = y;
            auto in_range = [&xo, &yo]() {
                return yo >= 0 && yo < BOARD_H && xo >= 0 && xo < BOARD_W;
            };
            auto empty = [&]() {
                auto yx = yo * 10 + xo;
                return is_empty(yx);
            };
            auto sameside = [is_same_side, yx_from, &xo, &yo]() {
                return is_same_side(yx_from, yo * 10 + xo);
            };
 
            xo += direction; 
            if (!in_range() || !empty()) {
                return;
            }
            xo += direction; yo++;
            if (in_range() && !sameside()) {
                callback(yx_from, yo * 10 + xo, board[yo*10+xo]);
            }
            yo--; yo--;
            if (in_range() && !sameside()) {
                callback(yx_from, yo * 10 + xo, board[yo*10+xo]);
            }
        };

        auto check_y = [&](int direction) {
            int xo = x; int yo = y;
            auto in_range = [&xo, &yo]() {
                return yo >= 0 && yo < BOARD_H && xo >= 0 && xo < BOARD_W;
            };
            auto empty = [&]() {
                auto yx = yo * 10 + xo;
                return is_empty(yx);
            };
            auto sameside = [is_same_side, yx_from, &xo, &yo]() {
                return is_same_side(yx_from, yo * 10 + xo);
            };
 
            yo += direction; 
            if (!in_range() || !empty()) {
                return;
            }
            yo += direction; xo++;
            if (in_range() && !sameside()) {
                callback(yx_from, yo * 10 + xo, board[yo*10+xo]);
            }
            xo--; xo--;
            if (in_range() && !sameside()) {
                callback(yx_from, yo * 10 + xo, board[yo*10+xo]);
            }
        };
        check_x(1);
        check_x(-1);
        check_y(1);
        check_y(-1);
    };
    auto append_sang = [&](int yx_from) {
        int y = yx_from / 10; int x = yx_from % 10;
        auto check_x = [&](int direction) {
            int xo = x; int yo = y;
            auto in_range = [&xo, &yo]() {
                return yo >= 0 && yo < BOARD_H && xo >= 0 && xo < BOARD_W;
            };
            auto empty = [this, is_empty, &xo, &yo]() {
                auto yx = yo * 10 + xo;
                return is_empty(yx);
            };
            auto sameside = [this, is_same_side, yx_from, &xo, &yo]() {
                return is_same_side(yx_from, yo * 10 + xo);
            };
            
            xo += direction; 
            if (!in_range() || !empty()) {
                return;
            }
            xo += direction; yo++;
            if (in_range() && empty()) {
                xo += direction; yo++;
                if (in_range() && !sameside()) {
                    callback(yx_from, yo * 10 + xo, board[yo*10+xo]);
                }
                xo -= direction; yo--;
            }
            yo--; yo--;
            if (in_range() && empty()) {
                xo += direction; yo--;
                if (in_range() && !sameside()) {
                    callback(yx_from, yo * 10 + xo, board[yo*10+xo]);
                }
            }
        };

        auto check_y = [&](int direction) {
            int xo = x; int yo = y;
            auto in_range = [&xo, &yo]() {
                return yo >= 0 && yo < BOARD_H && xo >= 0 && xo < BOARD_W;
            };
            auto empty = [this, is_empty, &xo, &yo]() {
                auto yx = yo * 10 + xo;
                return is_empty(yx);
            };
            auto sameside = [this, is_same_side, yx_from, &xo, &yo]() {
                return is_same_side(yx_from, yo * 10 + xo);
            };
            
            yo += direction; 
            if (!in_range() || !empty()) {
                return;
            }
            yo += direction; xo++;
            if (in_range() && empty()) {
                yo += direction; xo++;
                if (in_range() && !sameside()) {
                    callback(yx_from, yo * 10 + xo, board[yo*10+xo]);
                }
                yo -= direction; xo--;
            }
            xo--; xo--;
            if (in_range() && empty()) {
                yo += direction; xo--;
                if (in_range() && !sameside()) {
                    callback(yx_from, yo * 10 + xo, board[yo*10+xo]);
                }
            }
        };
        check_x(1);
        check_x(-1);
        check_y(1);
        check_y(-1);
    };

    auto append_jol = [&](int yx_from) {
        int offsets[] = {(to_move == Side::CHO ? -1 : 1), 0, 0, 1, 0, -1};
        int xo, yo;
        int y = yx_from / 10;
        int x = yx_from % 10;
        auto in_range = [&xo, &yo]() {
            return yo >= 0 && yo < BOARD_H && xo >= 0 && xo < BOARD_W;
        };
        auto sameside = [this, is_same_side, yx_from, &xo, &yo]() {
            return is_same_side(yx_from, yo * 10 + xo);
        };
        for(int i=0; i<3; i++) {
            yo = y + offsets[2 * i];
            xo = x + offsets[2 * i + 1];
            if(in_range() && !sameside()) {
                callback(yx_from, yo * 10 + xo, board[yo*10+xo]);
            }
        }
        // diagonal cases
        if( yx_from == 14) {
            yo = y - 1;
            xo = x - 1;
            if(in_range() && !sameside()) {
                callback(yx_from, yo * 10 + xo, board[yo*10+xo]);
            }
            yo = y - 1;
            xo = x + 1;
            if(in_range() && !sameside()) {
                callback(yx_from, yo * 10 + xo, board[yo*10+xo]);
            }
        }
        if( yx_from == 84) {
            yo = y + 1;
            xo = x - 1;
            if(in_range() && !sameside()) {
                callback(yx_from, yo * 10 + xo, board[yo*10+xo]);
            }
            yo = y + 1;
            xo = x + 1;
            if(in_range() && !sameside()) {
                callback(yx_from, yo * 10 + xo, board[yo*10+xo]);
            }
        }
        if( yx_from == 23) {
            yo = y - 1;
            xo = x + 1;
            if(in_range() && !sameside()) {
                callback(yx_from, yo * 10 + xo, board[yo*10+xo]);
            }
        }
        if( yx_from == 25) {
            yo = y - 1;
            xo = x - 1;
            if(in_range() && !sameside()) {
                callback(yx_from, yo * 10 + xo, board[yo*10+xo]);
            }
        }
        if( yx_from == 73) {
            yo = y + 1;
            xo = x + 1;
            if(in_range() && !sameside()) {
                callback(yx_from, yo * 10 + xo, board[yo*10+xo]);
            }
        }
        if( yx_from == 75) {
            yo = y + 1;
            xo = x - 1;
            if(in_range() && !sameside()) {
                callback(yx_from, yo * 10 + xo, board[yo*10+xo]);
            }
        }
    };

    auto append_bikjang = [is_empty, callback, this](auto yx_from) {
        // can call bikjang only if captured at least one piece
        if(to_move == Side::CHO && score_han() >= 72.0f) {
            return;
        }
        if(to_move == Side::HAN && score_cho() >= 72.0f) {
            return;
        }
        auto y = yx_from / 10;
        auto x = yx_from % 10;
        if(y > 6) {
            while(--y >= 0) {
                if(is_empty(y * 10 + x)) {
                    continue;
                } else {
                    if(board[y*10+x] == 0x00 || board[y*10+x] == 0x10) {
                        callback(yx_from, y * 10 + x, board[y*10+x]);
                    } else {
                        break;
                    }
                }
            }
        } else {
            while(++y < BOARD_H) {
                if(is_empty(y * 10 + x)) {
                    continue;
                } else {
                    if(board[y*10+x] == 0x00 || board[y*10+x] == 0x10) {
                        callback(yx_from, y * 10 + x, board[y*10+x]);
                    } else {
                        break;
                    }
                }
            }
        }
    };
    int goongpos = -1;
    for (int y = 0; y<BOARD_H; y++) {
        for (int x = 0; x < BOARD_W; x++) {
            auto yx = y * 10 + x;
            auto side = static_cast<int>(to_move);
            auto move = board[yx];
            if (move / 16 == side / 16) {
                switch(move % 16) {
                    case 0: // Goong
                        goongpos = yx;
                        append_sa_goong(yx);
                        if(gmgm::globals::allow_bikjang) {
                            append_bikjang(yx);
                        }
                        break;
                    case 1: case 2: // Sa
                        append_sa_goong(yx);
                        break;
                    case 3: case 4: // Cha
                        append_cha(yx);
                        break;
                    case 9: case 10: // Po
                        append_po(yx);
                        break;
                    case 5: case 6:
                        append_sang(yx);
                        break;
                    case 7: case 8:
                        append_ma(yx);
                        break;
                    case 11: case 12: case 13: case 14: case 15:
                        append_jol(yx);
                        break;
                    default:
                        break;
                }
            }
        }
    }
    // pass : goong on same place
    callback(
        goongpos,
        goongpos,
        board[goongpos]
    );
}

gmgm::Side gmgm::Board::winner_piece_only() const {
    bool found_cho_goong = false;
    bool found_han_goong = false;

    // look for goong
    for (int y = 0; y<3; y++) {
        for (int x = 3; x < 6; x++) {
            auto move = board[y*10+x];
            if(move == 0x10) found_han_goong = true;
        }
    }

    for (int y = BOARD_H-3; y<BOARD_H; y++) {
        for (int x = 3; x < 6; x++) {
            auto move = board[y*10+x];
            if(move == 0x0) found_cho_goong = true;
        }
    }

    if(found_cho_goong && found_han_goong) return Side::NONE;
    else if(found_cho_goong) return Side::CHO;
    else if(found_han_goong) return Side::HAN;
    else return Side::NONE;
}

gmgm::Side gmgm::Board::winner() const {
    if(globals::jang_move_is_illegal) {
        // if nothing to move (happens when checkmate) current player loses
        if(get_legal_moves().size() == 0) {
            if(to_move == Side::HAN) return Side::CHO;
            if(to_move == Side::CHO) return Side::HAN;
        }
    }

    size_t repeat_cnt = 0;
    if (globals::board_based_repetitive_move) {
        int sz = history.size();
        int pt = sz-1;
    
        if( sz >= 8 && !history[sz-1].move.is_pass() ) {
            while(repeat_cnt < 3 && pt >= 0) {
                if(history[sz-1].boardhash == history[pt].boardhash) {
                    repeat_cnt++;
                    pt -= 4;
                } else {
                    break;
                }
            }
        }

        if(repeat_cnt >= 3) {
            // if we had to play a repeativie move to avoid jang,
            // let it happen
            if(history[sz-2].was_jang) {
                repeat_cnt = 0;
            }
        }

    } else {
        int sz = history.size();
        int pt = sz-5;
    
        // Ugh.  We need our custom == operator
        // Because KakaoJangi compares (piece,destination) rather
        // than (source,destination)
        // note that if we capture something successfully it doesn't count
        // as repeativie move (do we?)
        auto move_equals = [this](const Move & x, const Move &y) {
            return (x.piece == y.piece && x.yx_to == y.yx_to);
        };
        if(pt >= 0) {
            const auto & mv = history[sz-1].move;
            if(mv.is_pass()) {
                // pass doesn't count as repeatitive move
            } else if(mv.piece == 0 || mv.piece == 1 || mv.piece == 2 
                   || mv.piece == 16 || mv.piece == 17 || mv.piece == 18) {
                // goong and sa doesn't count as repeatitive move
            } else {
                while(repeat_cnt < 2 && pt >= 0) {
                    if(history[pt].move.is_pass()) {
                        pt -= 4;
                    } else if(mv.captured == 0x20 && history[pt].move.captured == 0x20 && move_equals(mv, history[pt].move)) {
                        repeat_cnt++;
                        pt -= 4;
                    } else {
                        break;
                    }
                }
            }
        }
        // jang doesn't count as repeat move
        if(sz >= 2 && history[sz-2].was_jang) {
            repeat_cnt = 0;
        }
    }

    auto is_passend = [this]() {
        if(history.size() < 2) {
            return false;
        }
        auto sz = history.size();
        if(sz < 2) {
            return false;
        }
        return history[sz-1].move.is_pass() && history[sz-2].move.is_pass();
    };

    // last move was a repetitive move; the current player wins
    if(repeat_cnt >= 2) {
        return to_move;
    }
    
    auto sc = score_cho();
    auto sh = score_han();
    if(sc < 10.0f || sh < 10.0f || history.size() >= 200) {
        // game too long; end game if nothing was captured last move
        // and there was no jang
        auto sz = history.size();
        if(sz > 0 && history[sz-1].move.captured == 0x20
            && !is_jang() && !history[sz-1].was_jang) {
            if(sc > sh) return Side::CHO;
            else return Side::HAN;
        }
    }

    // I know this is not part of the kakao rule,
    // but this is put in as a safeguard to prevent infinite games
    // later if we understand how it deals with infinite jangs then we can fix this.
    if(history.size() >= 240) {
        if(sc > sh) return Side::CHO;
        else return Side::HAN;
    }

    if(is_passend()) {
        if(sc > sh) return Side::CHO;
        else return Side::HAN;
    }

    return winner_piece_only();
}


float gmgm::Board::__score(Side s) const {
    float score = 0.0f;
    for (int y = 0; y<BOARD_H; y++) {
        for (int x = 0; x < BOARD_W; x++) {
            auto side = static_cast<int>(s);
            auto move = board[y*10+x];
            if (move / 16 == side / 16) {
                switch(move % 16) {
                    case 0: // Goong
                        break;
                    case 1: case 2: // Sa
                        score += 3.0f;
                        break;
                    case 3: case 4: // Cha
                        score += 13.0f;
                        break;
                    case 9: case 10: // Po
                        score += 7.0f;
                        break;
                    case 5: case 6:
                        score += 3.0f;
                        break;
                    case 7: case 8:
                        score += 5.0f;
                        break;
                    case 11: case 12: case 13: case 14: case 15:
                        score += 2.0f;
                        break;
                    default:
                        break;
                }
            }
        }
    }
    if(s == Side::HAN) score += 1.5f;
    return score;
}

std::vector<gmgm::Move> gmgm::Board::can_win(int depth, int timeout_ms) {
    bool timeout = false;
    bool done = false;
    auto timeout_thread = [timeout_ms, &timeout, &done]() {
        for(int i=0; !done && i<timeout_ms; i++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        timeout = true;
    };

    std::thread t(timeout_thread);
    std::vector<gmgm::Move> ret;
    for(int i=1; !timeout && i<=depth; i++) {
        auto x = __can_win(i, timeout);
        if(!x.empty()) {
            ret = x;
            break;
        }
    }
    done = true;

    t.join();

    return ret;
}

std::vector<gmgm::Move> gmgm::Board::__can_win(int depth, bool & timeout) {
    if(depth > 2) {
        if(timeout) {
            depth = 1;
        }
    }

    if(depth <= 0) {
        return {};
    }


    if(winner_piece_only() != Side::NONE) {
        // what's the point of searching if we have a winner?
        return {};
    }

    std::vector<gmgm::Move> retv;

    // iterate through the candidate moves and search for any sequence.
    // We don't need to find the shortest one, since we will be incrementally
    // increasing the depth.
    __get_legal_moves([&](int8_t yx_from_, int8_t yx_to_, int8_t captured_) {
        if(retv.size() > 0) return;
        auto elem_from = board[yx_from_];
        gmgm::Move m(elem_from, yx_from_, yx_to_, captured_);
        if(
            yx_from_ != yx_to_
            && (elem_from != 0x10 && elem_from != 0x00) // disallow bikjang
            && (captured_ == 0x10 || captured_ == 0x00)) 
        {
            // we immediately win
            retv.push_back(m);
        } else if(depth > 1) { // no point searching if depth is 1 (which will immediately drop to 0)
            move_piece_only(m);
            auto v = __must_lose(depth-1, timeout);
            if(v.size() > 0) {
                retv.clear();
                retv.reserve(100);
                retv.insert(end(retv), m);
                retv.insert(end(retv), begin(v), end(v));
            }
            unmove_piece_only(m);
        }
    });

    return retv;
}

std::vector<gmgm::Move> gmgm::Board::__must_lose(int depth, bool & timeout) {
    if(depth <= 0) {
        return {};
    }

    if(winner_piece_only() != Side::NONE) {
        // what's the point of searching if we have a winner?
        return {};
    }

    // we have to implement our version of is_jang because we are using the
    // 'piece only' versions of move/unmove.
    auto __is_jang = [this]() {
        bool ret = false;

        to_move = opponent(to_move);
        ret = can_win_immediately();
        to_move = opponent(to_move);

        return ret;
    };

    // we will be focused on the 'jang' moves.  non-jang moves will have shallower trees
    if(!__is_jang()) {
        depth--;
    }
    if(depth <= 0) {
        return {};
    }

    std::vector<gmgm::Move> vv;

    bool found_undecided = false;
    __get_legal_moves([&](int8_t yx_from_, int8_t yx_to_, int8_t captured_) {
        if(found_undecided) return;
        auto elem_from = board[yx_from_];

        Move m(elem_from, yx_from_, yx_to_, captured_);
        move_piece_only(m);
        auto v = __can_win(depth, timeout);

        if(v.size() > 0) {
            // find LONGEST sequence
            if(vv.size() <= v.size()) {
                vv.clear();
                vv.reserve(100);
                vv.insert(end(vv), m);
                vv.insert(end(vv), begin(v), end(v));
            }
        } else {
            vv.clear();
            found_undecided = true;
        }

        unmove_piece_only(m);
    });

    return vv;
}


void gmgm::Board::move_piece_only(const gmgm::Move & m) const {
    auto piece = board[m.yx_from];
    board[m.yx_from] = 0x20;
    board[m.yx_to] = piece;
    if(to_move == Side::CHO) {
        to_move = Side::HAN;
    } else {
        to_move = Side::CHO;
    }

    auto apply = [this](int y, int x, int piece) {
        boardhash = boardhash ^ board_hash_constants[y * BOARD_W * 32 + x * 32 + piece];
    };
    if(m.captured != 0x20)
        apply(m.yx_to/10, m.yx_to%10, m.captured);
    apply(m.yx_to/10, m.yx_to%10, piece);
    apply(m.yx_from/10, m.yx_from%10, piece);
    // this boardhash is here because nothing will change if we hit pass
    boardhash = boardhash ^ board_hash_constants[BOARD_HASH_SIZE-1];
}

void gmgm::Board::unmove_piece_only(const gmgm::Move & m) const {
    auto piece = board[m.yx_to];

    assert(piece < 0x20);

    board[m.yx_to] = m.captured;
    board[m.yx_from] = piece;
    if(to_move == Side::CHO) {
        to_move = Side::HAN;
    } else {
        to_move = Side::CHO;
    }

    auto apply = [this](int y, int x, int piece) {
        boardhash = boardhash ^ board_hash_constants[y * BOARD_W * 32 + x * 32 + piece];
    };
    apply(m.yx_from/10, m.yx_from%10, piece);
    apply(m.yx_to/10, m.yx_to%10, piece);
    if(m.captured != 0x20)
        apply(m.yx_to/10, m.yx_to%10, m.captured);

    // this boardhash is here because nothing will change if we hit pass
    boardhash = boardhash ^ board_hash_constants[BOARD_HASH_SIZE-1];
}

void gmgm::Board::move(const gmgm::Move & m) {
#if 0
    {
        auto moves = get_legal_moves();
        bool found_match = false;
        for(auto & x : moves) {
            if(x == m) {
                found_match = true; break;
            }
        }
        assert(found_match);
    }
#endif
    clear_cache();
    
    auto piece = board[m.yx_from];

    auto old_boardhash = boardhash;
    auto old_playhash = playhash;

    move_piece_only(m);

    auto apply = [this](int y, int x, int piece) {
        playhash = playhash ^ board_hash_constants[ (y * BOARD_W * 32 + x * 32 + piece + 37 * (1+history.size())) % BOARD_HASH_SIZE ];
        playhash = (playhash << 1) | (playhash >> 63);
    };
    if(m.captured != 0x20)
        apply(m.yx_to/10, m.yx_to%10, m.captured);
    apply(m.yx_to/10, m.yx_to%10, piece);
    apply(m.yx_from/10, m.yx_from%10, piece);

    // need to be called AFTER board changed
    history.emplace_back(m, old_boardhash, old_playhash, is_jang());

    switch(m.captured) {
        case 0: // Goong
            break;
        case 1: case 2: // Sa
            cached_score_cho -= 3.0f;
            break;
        case 3: case 4: // Cha
            cached_score_cho -= 13.0f;
            break;
        case 9: case 10: // Po
            cached_score_cho -= 7.0f;
            break;
        case 5: case 6:
            cached_score_cho -= 3.0f;
            break;
        case 7: case 8:
            cached_score_cho -= 5.0f;
            break;
        case 11: case 12: case 13: case 14: case 15:
            cached_score_cho -= 2.0f;
            break;
        case 16: // Goong
            break;
        case 17: case 18: // Sa
            cached_score_han -= 3.0f;
            break;
        case 19: case 20: // Cha
            cached_score_han -= 13.0f;
            break;
        case 25: case 26: // Po
            cached_score_han -= 7.0f;
            break;
        case 21: case 22:
            cached_score_han -= 3.0f;
            break;
        case 23: case 24:
            cached_score_han -= 5.0f;
            break;
        case 27: case 28: case 29: case 30: case 31:
            cached_score_han -= 2.0f;
            break;
        default:
            break;
    }
}

bool gmgm::Board::is_jang() const {
    auto mv = get_legal_moves_if_opponent();
    for(auto & m : mv) {
        auto piece = m.captured;
        if(m.yx_from != m.yx_to && (piece == 0x0 || piece == 0x10)) {
            return true;
        }
    }
    return false;
}

const std::vector<gmgm::Move> & gmgm::Board::get_legal_moves() const {
    if(legal_move_cache.empty()) {
        legal_move_cache.clear();
        if(globals::jang_move_is_illegal) {
            __get_legal_moves([&](int8_t yx_from_, int8_t yx_to_, int8_t captured_) {
                auto elem_from = board[yx_from_];
                Move m(elem_from, yx_from_, yx_to_, captured_);
                move_piece_only(m);
                if(!can_win_immediately()) {
                    legal_move_cache.emplace_back(elem_from, yx_from_, yx_to_, captured_);
                }
                unmove_piece_only(m);
            });
        } else {
            __get_legal_moves([&](int8_t yx_from_, int8_t yx_to_, int8_t captured_) {
                auto elem_from = board[yx_from_];
                Move m(elem_from, yx_from_, yx_to_, captured_);
                legal_move_cache.emplace_back(elem_from, yx_from_, yx_to_, captured_);
            });
        }
    }
    return legal_move_cache;
}
const std::vector<gmgm::Move>& gmgm::Board::get_legal_moves_if_opponent() const {
    if (!legal_move_opponent_cache.empty()) {
        return legal_move_opponent_cache;
    }

    legal_move_opponent_cache.clear();

    to_move = opponent(to_move);
    __get_legal_moves([&](int8_t yx_from_, int8_t yx_to_, int8_t captured_) {
        auto elem_from = board[yx_from_];
        legal_move_opponent_cache.emplace_back(elem_from, yx_from_, yx_to_, captured_);
    });
    to_move = opponent(to_move);

    return legal_move_opponent_cache;
}

gmgm::Move gmgm::Board::unmove() {
    if(history.empty()) {
        throw std::out_of_range("No move history");
    }
    clear_cache();
    const auto & m = history.back().move;
    const auto & prev_boardhash = history.back().boardhash;
    const auto & prev_playhash = history.back().playhash;

    auto piece = board[m.yx_to];

    unmove_piece_only(m);
    auto apply = [this](int y, int x, int piece) {
        playhash = (playhash >> 1) | (playhash << 63);
        playhash = playhash ^ board_hash_constants[ (y * BOARD_W * 32 + x * 32 + piece + 37 * history.size() ) % BOARD_HASH_SIZE];
    };
    apply(m.yx_from/10, m.yx_from%10, piece);
    apply(m.yx_to/10, m.yx_to%10, piece);
    if(m.captured != 0x20)
        apply(m.yx_to/10, m.yx_to%10, m.captured);

    assert(boardhash == prev_boardhash);
    assert(playhash == prev_playhash);

    auto ret = history.back().move;
    history.pop_back();

    switch(m.captured) {
        case 0: // Goong
            break;
        case 1: case 2: // Sa
            cached_score_cho += 3.0f;
            break;
        case 3: case 4: // Cha
            cached_score_cho += 13.0f;
            break;
        case 9: case 10: // Po
            cached_score_cho += 7.0f;
            break;
        case 5: case 6:
            cached_score_cho += 3.0f;
            break;
        case 7: case 8:
            cached_score_cho += 5.0f;
            break;
        case 11: case 12: case 13: case 14: case 15:
            cached_score_cho += 2.0f;
            break;
        case 16: // Goong
            break;
        case 17: case 18: // Sa
            cached_score_han += 3.0f;
            break;
        case 19: case 20: // Cha
            cached_score_han += 13.0f;
            break;
        case 25: case 26: // Po
            cached_score_han += 7.0f;
            break;
        case 21: case 22:
            cached_score_han += 3.0f;
            break;
        case 23: case 24:
            cached_score_han += 5.0f;
            break;
        case 27: case 28: case 29: case 30: case 31:
            cached_score_han += 2.0f;
            break;
        default:
            break;
    }
    return ret;
}

bool gmgm::Board::can_win_immediately() const {
    bool ret = false;
    __get_legal_moves([&](int8_t yx_from_, int8_t yx_to_, int8_t captured_) {
        auto elem_from = board[yx_from_];
        if(
            yx_from_ != yx_to_
            && (elem_from != 0x10 && elem_from != 0x00) // bikjang doesn't count
            && (captured_ == 0x10 || captured_ == 0x00)) 
        {
            ret = true;
        }
    });
    return ret;
}

bool gmgm::Board::compare(const gmgm::Board & other) const {
    for(auto y=0; y<BOARD_H; y++) {
        for(auto x = 0; x < BOARD_W; x++) {
            if(board[y*10+x] != other.board[y*10+x]) {
                assert(boardhash != other.boardhash);
                assert(playhash != other.playhash);
                return false;
            }
        }
    }
    if(to_move != other.to_move) {
        return false;
    }
    assert(boardhash == other.boardhash);
    if(history.size() != other.history.size()) {
        return false;
    }
    for(unsigned int i=0; i<history.size(); i++) {
        if(history[i].move != other.history[i].move) {
            return false;
        }
    }

    assert(playhash == other.playhash);
    return true;
}

