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

#include "globals.h"
#include <cstdio>
#include <cstdarg>

namespace gmgm {
namespace globals {

unsigned int cache_size = 20000;
bool allow_bikjang = false;
bool flip_display = false;
unsigned int num_scheduler_threads = 0;
unsigned int batch_size = 1;
bool board_based_repetitive_move = false;
bool jang_move_is_illegal = false;
float score_based_bias_rate = 0.0f;
bool verbose_mode = true;

void myprintf(const char *fmt, ...) {
    if (verbose_mode) {
        va_list args;
        va_start(args, fmt);
        std::vprintf(fmt, args);
        va_end(args);
    }
}

}
}


