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

#include "libgmgm/gmgm.h"

const std::vector<gmgm::Move> get_legal_moves_wrapper(gmgm::Board & b)
{
    gmgm::globals::jang_move_is_illegal = true;
    b.clear_cache();

    auto lm = b.get_legal_moves();
    std::vector<gmgm::Move> ret;
    auto to_move = b.get_to_move();

    for(auto & x : lm) {
        b.move(x);
        if(b.winner() == gmgm::Side::NONE || b.winner() == to_move) {
            ret.push_back(x);
        }
        b.unmove();
    }

    gmgm::globals::jang_move_is_illegal = false;
    b.clear_cache();

    return ret;
}

