
CL_PATH=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk/System/Library/Frameworks/OpenCL.framework/Versions/A
CL_LIB=$(CL_PATH)/Libraries
CL_HED=$(CL_PATH)/Headers

CL_FLAGS=-framework OpenCL

SRC=main.cxx

all:	$(SRC)
	g++ -L$(CL_LIB) -I$(CL_HED) $(CL_FLAGS) $(SRC)
