#include "clmanager.hxx"

clManager::clManager(const clParams &_params){
  params.ndevice = _params.ndevice;
  for(int i=0;i<params.ndevice;i++)
    params.device_index[i] = _params.device_index[i];
  params.src_path = _params.src_path;
  params.bin_path = _params.bin_path;
  params.kernel_name = _params.kernel_name;

  cout << "# ndevice:\t " << params.ndevice << endl;
  cout << "# device_index:\t";
  for(int i=0;i<params.ndevice;i++)
    cout << " " << params.device_index[i];
  cout << endl;
  cout << "# src_path:\t " << params.src_path << endl;
  cout << "# bin_path:\t " << params.bin_path << endl;
  cout << "# kernel_name:\t " << params.kernel_name << endl;

  //get platform info
  clGetPlatformIDs(PLATFORM_MAX, platforms, &platformCount);
  if (platformCount == 0) {
    cerr << "# No platform found.\n";
    exit(EXIT_FAILURE);
  }
  // print platforms
  for (int i = 0; i < platformCount; i++) {
    char vendor[100] = {0};
    char version[100] = {0};
    clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(vendor), vendor, NULL);
    clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(version), version, NULL);
    cout << "# Platform id: " << platforms[i] << ", Vendor: " << vendor << ", Version: " << version << endl;
  }
  // get device info
  clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, DEVICE_MAX, devices, &deviceCount);
  if (deviceCount == 0) {
    cerr << "# No device found." << endl;
    exit(EXIT_FAILURE);
  }
  // print devices
  cout << "# " << deviceCount << " device(s) found.\n";
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
    cout << "# Device id: " << i << endl;
    cout << "# \tName:                  " << name << endl;
    cout << "# \tDeviceExtentions       " << extensions << endl;
    cout << "# \tMaxWorkItemDimensions: " << max_work_dimensions << endl;
    cout << "# \tMaxWorkItemSizes:     ";
    for(int i=0;i<max_work_dimensions;i++)
      cout << ' ' << max_work_item_sizes[i];
    cout << endl;
    cout << "# \tMaxWorkGroupSize:      " << max_work_group_size << endl;
    cout << "# \tMaxComputeUnits:       " << max_compute_unit << endl;
  }
  if(deviceCount < params.ndevice){
    cerr << "error: ndevice > deviceCount" << endl;
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

clManager::~clManager(){
  //timer.print();
  /*
  if(queue){
    err = clFlush(queue);
    err = clFinish(queue);
  }
  if(kernel){
    err = clReleaseKernel(kernel);
  }
  if(program) err = clReleaseProgram(program);
  if(context) err = clReleaseContext(context);
  //*/
}

void clManager::CreateContext(){
  context = clCreateContext(NULL, 1, &devices[0], NULL, NULL, &err);
  if(err!=CL_SUCCESS){
    fprintf(stderr,"error: CreateContext failed (err = %d)\n",err);
    exit(EXIT_FAILURE);
  }
}

void clManager::CreateCommandQueue(){
  queue = clCreateCommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &err);
  if(err!=CL_SUCCESS){
    fprintf(stderr,"error: CreateCommandQueue failed (err = %d)\n",err);
    exit(EXIT_FAILURE);
  }
}

void clManager::CreateProgramWithBinary(){
  ifstream ifs(params.bin_path,ios::in | ios::binary);
  if(ifs.fail()){
    fprintf(stderr,"error: fopen %s failed",params.bin_path.c_str());
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
    cerr << "error: CreateProgramWithBinary failed (err= " << err << ")" << endl;
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

void clManager::CreateProgramWithSource(){
  // create program
  const char* src_path = params.src_path.c_str();
  ifstream ifs(src_path);
  if(ifs.fail()){
    fprintf(stderr,"error: fopen %s failed",params.bin_path.c_str());
    exit(EXIT_FAILURE);
  }
  ifs.seekg(0,ifs.end);
  const size_t src_size = ifs.tellg();
  ifs.seekg(0,ifs.beg);
  char* src_buf = new char[src_size];
  ifs.read(src_buf,src_size);

  program = clCreateProgramWithSource(context, 1, (const char **)&src_path, &src_size, &err);
  if(err != CL_SUCCESS){;
    fprintf(stderr,"error: create program failed (error code: %d)",err);
    exit(EXIT_FAILURE);
  }
}

void clManager::CreateProgramWithSource(const char* source){
  // create program
  const char* src[] = {source};
  program = clCreateProgramWithSource(context, 1, src, NULL, &err);
  if(err != CL_SUCCESS){
    fprintf(stderr,"error: create program failed (error code: %d)",err);
    exit(EXIT_FAILURE);
  }
}

void clManager::BuildProgram(){
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
    exit(EXIT_FAILURE);
  }
}

void clManager::CreateKernel(){
  // build kernel
  kernel = clCreateKernel(program, params.kernel_name.c_str(), &err);
   if(err != CL_SUCCESS){
    fprintf(stderr,"error: CreateKernel (%s) failed (error code: %d)\n",params.kernel_name.c_str(),err);
    exit(EXIT_FAILURE);
  }
}

template<class T>
void clManager::SetKernelArg(T &device_mem,const cl_int index){
  err = clSetKernelArg(kernel, index, sizeof(T), &device_mem);
  if(err != CL_SUCCESS){
    fprintf(stderr,"error: SetKernelArf failed (error code: %d)",err);
    exit(EXIT_FAILURE);
  }
}

void clManager::EnqueueTask(){
  err = clEnqueueTask(queue, kernel, 0, NULL, &event);
  if(err != CL_SUCCESS){
    printf("error: EnqueTask failed (error code: %d)\n",err);
    exit(EXIT_FAILURE);
  }
}

void clManager::EnqueueNDRangeKernel(const size_t global,const size_t local){
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local,0,NULL,&event);
  if(err != CL_SUCCESS){
    printf("error: EnqueNDRangeKernel failed (error code: %d)\n",err);
    exit(EXIT_FAILURE);
  }
}

template<class T>
void clManager::CreateBuffer(T* hmem,cl_mem &dmem,const cl_int n){
 dmem = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(T) * n, hmem, &err);
}

void clManager::SaveBinaries(){
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
    string filename = "device" + to_string(i) + ".bin";
    ofstream ofs(filename, ios::binary);
    if(ofs.is_open()){
      ofs.write(bins[i], binary_size[i]);
    }else{
      cerr << "error: failed to open: " << filename << endl;
    }
  }

  //*/
}
/*
void clMD::StartTimer(){
}

void clMD::EndTimer(){
  timer.measure();
}
//*/

//#define TEST_CLMANAGER
#ifdef TEST_CLMANAGER
const char source[] = "__kernel void test(void)\n{\n}\n";

int main(void){
  const string input = "input.txt";
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
  cout << "#kernel executed" << endl;
  return 0;
}
#endif
