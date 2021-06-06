default: libgmgm_$(TARGET).a

TARGET ?= *
INCPATH= -IEigen/
AR = ar

CPPFLAGS += -MD -MP

sources_cpp = Search.cpp globals.cpp \
    Board.cpp PositionEval.cpp SearchNode.cpp  \
    Network.cpp CPUPipe.cpp \
    Tuner.cpp OpenCL.cpp OpenCLScheduler.cpp

objects = $(sources_cpp:.cpp=.$(TARGET).o)
deps = $(sources_cpp:%.cpp=%.$(TARGET).d)

-include $(deps)


%.$(TARGET).o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCPATH) $(CPPFLAGS) -c -o $@ $<

libgmgm_$(TARGET).a: $(objects)
	$(AR) r $@ $^

clean:
	-$(RM) $(deps) $(objects) libgmgm_*.a

.PHONY: clean
