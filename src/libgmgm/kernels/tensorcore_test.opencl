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
/*
    This source originated from Leela Zero (http://github.com/leela-zero/leela-zero
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors
*/

// This kernel simply tests if the host can compile a wmma insturction.
// Not intended to be run at all.

// Enables loading of this file using the C++ pre-processor's #include (C++11 standard raw string
// literal). Comment-out this line for syntax-highlighting when developing.
R"(

__kernel void tensorcore_test(__global int * ptr) {
    asm(
        ".reg .b32 a0, a1, a2, a3, a4, a5, a6, a7;\n"
        "wmma.load.a.sync.aligned.m16n16k16.shared.row.f16 {a0,a1,a2,a3,a4,a5,a6,a7}, [%0];\n" : : "l"(ptr)
    );
}

// End of the C++11 raw string literal
)"
