#include <iostream>
#include <vector>

#include <cstring>

#include <opencl.h>
#define PLATFORM_MAX 4
#define DEVICE_MAX 4

#pragma OPENCL EXTENSION cl_intel_printf : enable

int main(int argc,char **argv){
  //get platform info
  cl_platform_id platforms[PLATFORM_MAX];
  cl_uint platformCount;
  clGetPlatformIDs(PLATFORM_MAX, platforms, &platformCount);
  if (platformCount == 0) {
    std::cerr << "No platform.\n";
    return EXIT_FAILURE;
  }
  // print platforms
  for (int i = 0; i < platformCount; i++) {
    char vendor[100] = {0};
    char version[100] = {0};
    clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
    clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(version), version, NULL);
    std::cout << "Platform id: " << platforms[i] << ", Vendor: " << vendor << ", Version: " << version << "\n";
  }

  // get device info
  cl_device_id devices[DEVICE_MAX];
  cl_uint deviceCount;
  clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, DEVICE_MAX, devices, &deviceCount);
  if (deviceCount == 0) {
    std::cerr << "No device.\n";
    return EXIT_FAILURE;
  }
  // print devices
  std::cout << deviceCount << " device(s) found.\n";
  for (int i = 0; i < deviceCount; i++) {
    char name[100] = {0};
    size_t len;
    clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(name), name, &len);
    std::cout << "Device id: " << i << ", Name: " << name << "\n";
  }

  cl_int err = CL_SUCCESS;
  // create context
  cl_context ctx = clCreateContext(NULL, 1, devices, NULL, NULL, &err);

#if 0
  // read compiled kernel program
  const char* bitcode_path = "OpenCL/kernel.cl.gpu_32.bc";
  size_t len = strlen(bitcode_path);
  cl_program program = clCreateProgramWithBinary(ctx, 1, devices, &len, (const unsigned char**)&bitcode_path, NULL, &err);
#else
  const char* source_path = "./kernel.cl";
  cl_program program = clCreateProgramWithSource(ctx, 1, &source_path, NULL, &err);
#endif
  // build program
  clBuildProgram(program, 1, devices, NULL, NULL, NULL);
  if(err != CL_SUCCESS) printf("program was not built (error number is %d)\n",err);
  // build kernel
  cl_kernel kernel = clCreateKernel(program, "test", &err);
  if(err != CL_SUCCESS) printf("kernel was not created (error number is %d)\n",err);
  //cl_kernel kernel = clCreateKernel(program, "calc_force", &err);

  // making data
  int n = 1000;
  std::vector<float> data(n);
  for(int i=0; i<n; i++) data[i] = 0.f;

  // allocate device memory and copy data
  cl_mem device_mem = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * n, data.data(), &err);

  // set kernel argument
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_mem);
  clSetKernelArg(kernel, 1, sizeof(int), &n);

  // making command queue
  cl_command_queue q = clCreateCommandQueue(ctx, devices[0], 0, &err);

  // execute kernel
  size_t local = 100;
  if(n%local != 0) std::cerr << "error: n is not divisible by local" << std::endl;
  size_t global = n;

  err = clEnqueueNDRangeKernel(q, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
  if(err != CL_SUCCESS) printf("kernel was not executed (error number is %d)\n",err);
  // read result
  clEnqueueReadBuffer(q, device_mem, CL_TRUE, 0, sizeof(float) * n, data.data(), 0, NULL, NULL);
  // print result
  for (int i = 0; i < n; i++) std::cout << data[i] << ", ";
  std::cout << std::endl;

  clReleaseCommandQueue(q);
  clReleaseMemObject(device_mem);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseContext(ctx);

  std::cout << "Done.\n";
  return EXIT_SUCCESS;
}

