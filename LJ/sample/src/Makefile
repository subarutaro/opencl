CL_PATH=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk/System/Library/Frameworks/OpenCL.framework/Versions/A
CL_LIB=$(CL_PATH)/Libraries
CL_HED=$(CL_PATH)/Headers

CL_FLAGS=-framework OpenCL

#CXX_FLAGS = -std=c++11
CXX_FLAGS = -std=gnu++0x
CXX_FLAGS +=-g -O0

INC= -I../../src

SRC=$(shell ls *.cxx)
#SRC=clmanager.cxx
OBJ=$(SRC:.cxx=.o)
TARGET=md.out

.SUFFIXES: .cxx .hxx. .o
.cxx.o:	$(SRC) $(HED)
	g++ -L$(CL_LIB) -I$(CL_HED) $(CL_FLAGS) $(CXX_FLAGS) $(INC) $< -c -o $@

all:	$(TARGET)

$(TARGET):	$(OBJ)
	g++ -L$(CL_LIB) -I$(CL_HED) $(CL_FLAGS) $(CXX_FLAGS) $(INC) $^ -o $@
clean:
	rm *.o *.out
