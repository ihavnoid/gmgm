# gmgm - A modern Janggi AI with a deep residual squeeze-excitation net (SENet) and MCTS

gmgm is a Janggi (Korean chess) AI which supports modern Janggi rules.  The author tried to faithfully reproduce
the rule from [Kakao Janggi](https://play.google.com/store/apps/details?id=com.monomob.jangki) which means:

* No draws
* Game ends when the game reaches 200 moves, both player passes, or any player goes less or equal to 10 points.
* No repetitive moves - I faithfully tried to reproduce Kakao's repeat move rules,
but there may be some holes due to the rule itself being undocumented.

# Weights

[There is a weight trained by myself](https://drive.google.com/file/d/11LLR74LHmRiKKEMZN-WPm8W6HZx1FnSM/view?usp=sharing)
This should be good enough to play against most casual players.

# Is there some way to try it on?

[I created a website that services gmgm on the cloud.](https://cbaduk.net/)

# Compiling

## Requirements (mostly copied from leela-zero, not that much default)

* GCC, Clang or MSVC, any C++14 compiler
* Boost 1.58.x or later, headers and program_options, filesystem and system libraries (libboost-dev, libboost-program-options-dev and libboost-filesystem-dev on Debian/Ubuntu)
* zlib library (zlib1g & zlib1g-dev on Debian/Ubuntu)
* Standard OpenCL C headers (opencl-headers on Debian/Ubuntu, or at
https://github.com/KhronosGroup/OpenCL-Headers/tree/master/CL)
* OpenCL ICD loader (ocl-icd-libopencl1 on Debian/Ubuntu, or reference implementation at https://github.com/KhronosGroup/OpenCL-ICD-Loader)
* An OpenCL capable device, preferably a very, very fast GPU, with recent
drivers is strongly recommended (OpenCL 1.1 support is enough).

## Example of compiling - Ubuntu & similar

    # Test for OpenCL support & compatibility
    sudo apt install clinfo && clinfo

    # Clone github repo
    git clone https://github.com/ihavnoid/gmgm
    cd gmgm
    git submodule update --init --recursive

    # Install build depedencies
    sudo apt install libboost-dev libboost-program-options-dev libboost-filesystem-dev opencl-headers ocl-icd-libopencl1 ocl-icd-opencl-dev zlib1g-dev

    # Use stand alone directory to keep source dir clean
    cd src
    make

    # Download net
    curl -O https://drive.google.com/file/d/11LLR74LHmRiKKEMZN-WPm8W6HZx1FnSM/view?usp=sharing

    # Run binary.  This will open up a gmgm shell
    ./gmgm_release

    # Load net
    loadnet sample_net.txt.gz

# Acknowledgements

A lot of other pieces of code (specifically the OpenCL code), makefiles and this markdown file itself
originated from [leela-zero](https://github.com/leela-zero/leela-zero/).  
The MCTS search and the game logic itself was written from scratch.

# License

The code is released under the GPLv3 or later, except for cl2.hpp, half.hpp and the eigen subdirs, which have specific licenses (compatible with GPLv3) mentioned in those files.

# Help needed

* Support for some standard Janggi protocol - unfortunately I have no idea what kind of standard protocol we should support.
