default:
	$(MAKE) CC=gcc CXX=g++ \
        TARGET=release \
		CXXFLAGS='$(CXXFLAGS) -Wall -Wextra -pipe -O3 -g -ffast-math -flto -march=native -std=c++14'  \
		LDFLAGS='$(LDFLAGS) -flto -g' \
		gmgm_release

debug:
	$(MAKE) CC=gcc CXX=g++ \
        TARGET=debug \
		CXXFLAGS='$(CXXFLAGS) -Wall -Wextra -pipe -Og -g -std=c++14' \
		LDFLAGS='$(LDFLAGS) -g' \
		gmgm_debug

TARGET ?= *
DYNAMIC_LIBS = -ltcmalloc -lboost_system -lboost_filesystem -lboost_program_options -lpthread -lz -lOpenCL
LIBS = libgmgm/libgmgm_$(TARGET).a

CXXFLAGS += -Wno-deprecated-copy
CPPFLAGS += -MD -MP

sources_cpp = gmgm.cpp util.cpp

objects = $(sources_cpp:.cpp=.$(TARGET).o)
deps = $(sources_cpp:%.cpp=%.$(TARGET).d)

-include $(deps)

%.$(TARGET).o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

gmgm_$(TARGET): $(objects) $(LIBS)
	$(CXX) $(LDFLAGS) -o $@ $^ $(DYNAMIC_LIBS)

clean:
	-$(RM) gmgm_$(TARGET) $(objects) $(deps)
	cd libgmgm && $(MAKE) clean

libgmgm/libgmgm_$(TARGET).a: FORCE
	cd libgmgm && $(MAKE)

FORCE: ;

.PHONY: clean default debug 
