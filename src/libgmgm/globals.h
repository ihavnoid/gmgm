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

#ifndef __GMGM__GLOBALS_H__
#define __GMGM__GLOBALS_H__

#include <cstdlib>

namespace gmgm {
namespace globals {
extern unsigned int cache_size;
extern bool allow_bikjang;
extern bool flip_display;
extern unsigned int num_scheduler_threads;
extern unsigned int batch_size;
extern bool board_based_repetitive_move;
extern float score_based_bias_rate;
extern bool jang_move_is_illegal;
extern bool verbose_mode;

void myprintf(const char *fmt, ...);
}
}

#endif
