#ifndef HXX_CLMANAGER
#define HXX_CLMANAGER

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define PLATFORM_MAX 4
#define DEVICE_MAX 4
#define NDEVICE_MAX 10

#define MAX_BINARY_SIZE 100000

#define BUILD_SOURCE

#pragma OPENCL EXTENSION cl_intel_printf : enable

#include <iostream>
#include <fstream>
#include <sstream>

#include <vector>
#include <string>
#include <cstring>

#include <cassert>
using namespace std;

class clParams{
public:
  cl_uint ndevice;
  cl_uint device_index[NDEVICE_MAX];
  string src_path;
  string bin_path;
  string kernel_name;

  clParams(){};
  clParams(const clParams &_p){
    ndevice = _p.ndevice;
    for(int i=0;i<ndevice;i++)
      device_index[i] = _p.device_index[i];
    src_path = _p.src_path;
    bin_path = _p.bin_path;
    kernel_name = _p.kernel_name;
  }
  
  clParams(const string filename){
    ndevice = 0;

    ifstream is(filename.c_str());
    if(is.fail()){
      cerr << "error: open " << filename << " failed" << endl;
      exit(EXIT_FAILURE);
    }
    for(std::string line; std::getline(is, line);){
      std::istringstream ss(line);
      std::string tag;
      ss >> tag;
      if(ss.fail()) continue;
      if(!isalpha(tag[0])) continue; // check whether first string is alphabet or not
#define READ(name)                      \
      if(tag == string(#name)){		\
	ss >> name;			\
	continue;			\
      }
      READ(ndevice);
      if(tag == "device_index"){
	for(int i=0;i<ndevice;i++) ss >> device_index[i];
      }
      READ(src_path);
      READ(bin_path);
      READ(kernel_name);
    }
  };
  ~clParams(){};
};

class clManager{
private:
  clParams params;
public:
  cl_int           err;
  cl_platform_id   platforms[PLATFORM_MAX];
  cl_uint          platformCount;
  cl_device_id     devices[DEVICE_MAX];
  cl_uint          deviceCount;
  cl_context       context;
  cl_command_queue queue;
  cl_program       program;
  cl_kernel        kernel;
  cl_event         event;
  /*
  class Timer{
  private:
    cl_ulong timer_beg,timer_end;
    int      count;
    double   elapsed_time;
  public:
    Timer(){reset();};
    ~Timer(){};
    void reset(){elapsed_time = 0.0;count = 0;};
    cl_int measure(cl_event &event){
      cl_int err;
      err = clWaitForEvents(1,&event);
      err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(timer_beg), &timer_beg, NULL);
      err = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,   sizeof(timer_end), &timer_end, NULL);
      elapsed_time += (timer_end - timer_beg);
      count++;
    }
    void print(){
      cout << "# elapsed time:\t" << elapsed_time / 1e09 << " sec" << endl;
      cout << "# average time:\t" << elapsed_time / ((double)count * 1e09) << " sec" << endl;
    }
  };
  Timer timer;
  //*/
  clManager(const clParams&);
  ~clManager();
  void CreateContext();
  void CreateCommandQueue();
  void CreateProgramWithBinary();
  void CreateProgramWithSource();
  void CreateProgramWithSource(const char*);
  void BuildProgram();
  void CreateKernel();
  void EnqueueTask();
  void EnqueueNDRangeKernel(const size_t,const size_t);
  template<class T>
  void CreateBuffer(T*,cl_mem&,const cl_int);
  template<class T>
  void SetKernelArg(T&,const cl_int);

  void SaveBinaries();
    /*
  void StartTimer();
  void EndTimer();
    //*/
};
#endif
