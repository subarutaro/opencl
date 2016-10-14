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
  cl_uint     ndevice;
  cl_uint     device_index[NDEVICE_MAX];
  std::string src_path;
  std::string bin_path;
  std::string kernel_name;

  clParams(){};
  clParams(const clParams &_p){
    ndevice = _p.ndevice;
    for(int i=0;i<ndevice;i++)
      device_index[i] = _p.device_index[i];
    src_path = _p.src_path;
    bin_path = _p.bin_path;
    kernel_name = _p.kernel_name;
  }

  clParams(const std::string filename){
    ndevice = 0;

    ifstream is(filename.c_str());
    if(is.fail()){
      cerr << "error: open " << filename << " failed" << std::endl;
      exit(EXIT_FAILURE);
    }
    for(std::string line; std::getline(is, line);){
      std::istringstream ss(line);
      std::string tag;
      ss >> tag;
      if(ss.fail()) continue;
      if(!isalpha(tag[0])) continue; // check whether first std::string is alphabet or not
#define READ(name)                      \
      if(tag == std::string(#name)){		\
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
  cl_uint          platformCount;
  cl_uint          nplatform;
  cl_platform_id   *platforms;
  cl_device_id     devices[DEVICE_MAX];
  cl_uint          deviceCount;
  cl_context       context;
  cl_command_queue queue;
  cl_program       program;
  cl_kernel        kernel;
  cl_event         event;

  clManager(const clParams &_params){
    params.ndevice = _params.ndevice;
    for(int i=0;i<params.ndevice;i++)
      params.device_index[i] = _params.device_index[i];
    params.src_path = _params.src_path;
    params.bin_path = _params.bin_path;
    params.kernel_name = _params.kernel_name;

    std::cout << "# ndevice:\t " << params.ndevice << std::endl;
    std::cout << "# device_index:\t";
    for(int i=0;i<params.ndevice;i++)
      std::cout << " " << params.device_index[i];
    std::cout << std::endl;
    std::cout << "# src_path:\t " << params.src_path << std::endl;
    std::cout << "# bin_path:\t " << params.bin_path << std::endl;
    std::cout << "# kernel_name:\t " << params.kernel_name << std::endl;

    //get platform info
    clGetPlatformIDs(PLATFORM_MAX, NULL, &platformCount);
    if (platformCount == 0) {
      cerr << "# No platform found.\n";
      exit(EXIT_FAILURE);
    }
    if((platforms=(cl_platform_id*)malloc(sizeof(cl_platform_id)*platformCount))==NULL){
      cerr << "# malloc platform failed" << endl;
      exit(EXIT_FAILURE);
    }
    clGetPlatformIDs(platformCount, platforms, NULL);

    // print platforms
    for (int i = 0; i < platformCount; i++) {
      char vendor[100] = {0};
      char version[100] = {0};
      clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
      clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(version), version, NULL);
      std::cout << "# Platform id: " << platforms[i] << ", Vendor: " << vendor << ", Version: " << version << std::endl;
    }
    // get device info
    clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, DEVICE_MAX, devices, &deviceCount);
    if (deviceCount == 0) {
      cerr << "# No device found." << std::endl;
      exit(EXIT_FAILURE);
    }
    // print devices
    std::cout << "# " << deviceCount << " device(s) found.\n";
    for (int i = 0; i < deviceCount; i++) {
      char name[100] = {};
      char extensions[1024] = {};
      size_t len;
      size_t max_work_group_size;
      cl_uint max_compute_unit,max_work_dimensions;
      clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, &len);
      clGetDeviceInfo(devices[i], CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, &len);
      clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_dimensions), &max_work_dimensions, &len);
      size_t max_work_item_sizes[max_work_dimensions];
      clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_work_item_sizes), &max_work_item_sizes, &len);
      clGetDeviceInfo(devices[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, &len);
      clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_unit), &max_compute_unit, &len);
      std::cout << "# Device id: " << i << std::endl;
      std::cout << "# \tName:                  " << name << std::endl;
      std::cout << "# \tDeviceExtentions       " << extensions << std::endl;
      std::cout << "# \tMaxWorkItemDimensions: " << max_work_dimensions << std::endl;
      std::cout << "# \tMaxWorkItemSizes:     ";
      for(int i=0;i<max_work_dimensions;i++)
	std::cout << ' ' << max_work_item_sizes[i];
      std::cout << std::endl;
      std::cout << "# \tMaxWorkGroupSize:      " << max_work_group_size << std::endl;
      std::cout << "# \tMaxComputeUnits:       " << max_compute_unit << std::endl;
    }
    if(deviceCount < params.ndevice){
      cerr << "error: ndevice > deviceCount" << std::endl;
      exit(EXIT_FAILURE);
    }

    CreateContext();
    CreateCommandQueue();
#ifdef BUILD_SOURCE
    CreateProgramWithSource();
    BuildProgram();
#else
    CreateProgramWithBinary();
#endif
    CreateKernel();
#ifdef BUILD_SOURCE
    SaveBinaries();
#endif
  }

  ~clManager(){}
  void CreateContext(){
    context = clCreateContext(NULL, 1, &devices[0], NULL, NULL, &err);
    if(err!=CL_SUCCESS){
      fprintf(stderr,"error: CreateContext failed (err = %d)\n",err);
      exit(EXIT_FAILURE);
    }
  }

  void CreateCommandQueue(){
    queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
    if(err!=CL_SUCCESS){
      fprintf(stderr,"error: CreateCommandQueue failed (err = %d)\n",err);
      exit(EXIT_FAILURE);
    }
  }

  void CreateProgramWithBinary(){
    ifstream ifs(params.bin_path,ios::in | ios::binary);
    if(ifs.fail()){
      fprintf(stderr,"error: fopen %s failed in CreateProgramWithBinary\n",params.bin_path.c_str());
      exit(EXIT_FAILURE);
    }
    ifs.seekg(0,ifs.end);
    const size_t binary_size = ifs.tellg();
    ifs.seekg(0,ifs.beg);
    char* binary_buf = new char[binary_size];
    ifs.read(binary_buf,binary_size);

    cl_int status[deviceCount];
    cl_program program = clCreateProgramWithBinary(context, params.ndevice, devices, &binary_size, (const unsigned char **)&binary_buf, status, &err);
    if(err != CL_SUCCESS){
      cerr << "error: CreateProgramWithBinary failed (err= " << err << ")" << std::endl;
      exit(EXIT_FAILURE);
    }

    bool isSuccess = true;
    for(int i=0;i<params.ndevice;i++){
      if(status[i] != CL_SUCCESS){
	fprintf(stderr,"error: CreateProgram of device %d failed (error code: %d)",i,status[i]);
	isSuccess = false;
      }
    }
    if(!isSuccess) exit(EXIT_FAILURE);

    delete[] binary_buf;
  }

  void CreateProgramWithSource(){
    // create program
    const char* src_path = params.src_path.c_str();
    //ifstream ifs(src_path,ios::in | ios::binary);
    ifstream ifs(src_path,ios::in);
    if(ifs.fail()){
      fprintf(stderr,"error: fopen %s failed in CreateProgramWithSource\n",src_path);
      exit(EXIT_FAILURE);
    }
    const size_t beg = ifs.tellg();
    ifs.seekg(0,ifs.end);
    const size_t end = ifs.tellg();
    const size_t src_size = end - beg;
    char* src_buf = new char[src_size];
    ifs.seekg(0,ifs.beg);
    ifs.read(src_buf,src_size);

    program = clCreateProgramWithSource(context, 1, (const char **)&src_buf, &src_size, &err);
    if(err != CL_SUCCESS){;
      fprintf(stderr,"error: create program failed (error code: %d)",err);
      exit(EXIT_FAILURE);
    }
  }

  void CreateProgramWithSource(const char* source){
    // create program
    const char* src[] = {source};
    program = clCreateProgramWithSource(context, 1, src, NULL, &err);
    if(err != CL_SUCCESS){
      fprintf(stderr,"error: create program failed (error code: %d)",err);
      exit(EXIT_FAILURE);
    }
  }

  void BuildProgram(){
    // build program
    err = clBuildProgram(program, params.ndevice, devices, NULL, NULL, NULL);
    if(err != CL_SUCCESS){
      printf("BuildProgram failed (error code: %d)\n",err);
      // Determine the size of the log
      size_t log_size;
      clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      // Allocate memory for the log
      char *log = (char *) malloc(log_size);
      // Get the log
      clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
      // Print the log
      fprintf(stderr,"%s\n", log);
      free(log);
      exit(EXIT_FAILURE);
    }
  }

  void CreateKernel(){
    // build kernel
    kernel = clCreateKernel(program, params.kernel_name.c_str(), &err);
    if(err != CL_SUCCESS){
      fprintf(stderr,"error: CreateKernel (%s) failed (error code: %d)\n",params.kernel_name.c_str(),err);
      exit(EXIT_FAILURE);
    }
  }

  void EnqueueTask(){
    err = clEnqueueTask(queue, kernel, 0, NULL, &event);
    if(err != CL_SUCCESS){
      printf("error: EnqueTask failed (error code: %d)\n",err);
      exit(EXIT_FAILURE);
    }
  }

  void EnqueueNDRangeKernel(const size_t global,const size_t local){
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,0,NULL,&event);
    if(err != CL_SUCCESS){
      printf("error: EnqueNDRangeKernel failed (error code: %d)\n",err);
      exit(EXIT_FAILURE);
    }
  }

  template<class T>
  void CreateBuffer(T* hmem,cl_mem &dmem,const cl_int n){
    dmem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(T) * n, hmem, &err);
  }

  template<class T>
  void SetKernelArg(T &device_mem,const cl_int index){
    err = clSetKernelArg(kernel, index, sizeof(T), &device_mem);
    if(err != CL_SUCCESS){
      fprintf(stderr,"error: SetKernelArf failed (error code: %d)",err);
      exit(EXIT_FAILURE);
    }
  }

  void SaveBinaries(){
    cl_uint ndevice = 0;
    err = clGetProgramInfo(program, CL_PROGRAM_NUM_DEVICES, sizeof(ndevice), &ndevice, NULL);
    assert(err==CL_SUCCESS);

    size_t binary_size[ndevice];
    err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), binary_size, NULL);
    assert(err==CL_SUCCESS);

    vector<char*> bins(ndevice);
    for(size_t i=0;i<ndevice;i++){
      bins[i] = binary_size[i] == 0 ? NULL : new char[binary_size[i]];
    }
    err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(char*)*ndevice, bins.data(),NULL);
    assert(err==CL_SUCCESS);

    for(size_t i=0;i<ndevice;i++){
      if(bins[i]==NULL) continue;
      std::string filename = "device" + std::to_string((long long int)i) + ".bin";
      ofstream ofs(filename, ios::binary);
      if(ofs.is_open()){
	ofs.write(bins[i], binary_size[i]);
      }else{
	cerr << "error: failed to open: " << filename << std::endl;
      }
    }
  }

  void WaitForEvents(){
    err = clWaitForEvents(1,&event);
    if(err != CL_SUCCESS){
      std::cerr  << "error: clWaitForEvents returned invalid value (= " << err << ")" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
};
#endif

//#define TEST_CLMANAGER
#ifdef TEST_CLMANAGER
const char source[] = "__kernel void test(void)\n{\n}\n";

int main(void){
  const std::string input = "input.txt";
  clParams params(input);

  clManager clm(params);
  clm.CreateContext();
  clm.CreateCommandQueue();
  clm.CreateProgramWithSource();
  clm.BuildProgram();
  clm.CreateKernel();

  const size_t n = 1024;
  float *hmem;
  hmem = (float*)malloc(sizeof(float)*n);
  for(int i=0;i<n;i++) hmem[i] = i;
  // allocate device memory and copy data
  cl_mem dmem;
  clm.CreateBuffer(hmem,dmem,n);

  // set kernel argument
  clm.SetKernelArg(dmem,0);
  clm.SetKernelArg(n,1);

  //clm.EnqueueTask();
  clm.EnqueueNDRangeKernel(1,n,1);
  std::cout << "#kernel executed" << std::endl;
  return 0;
}
#endif
