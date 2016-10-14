#ifndef HXX_CLMD
#define HXX_CLMD

#include "clmanager.hxx"
#include "md.hxx"

class clMD : public MD{
private:
  clManager *clmanager;
public:
  cl_mem r_dev,v_dev,f_dev;
  clMD(){
    clParams params("input.txt");
    clmanager = new clManager(params);
  }
  ~clMD(){}

  void Initialize(const int _nmol,const float _dens,const float _temp){
    MD::Initialize(_nmol,_dens,_temp);

    r_dev = clCreateBuffer(clmanager->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float3)*nmol, r, &clmanager->err);
    f_dev = clCreateBuffer(clmanager->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(cl_float3)*nmol, f, &clmanager->err);

    clSetKernelArg(clmanager->kernel,0,sizeof(cl_mem),&r_dev);
    clSetKernelArg(clmanager->kernel,1,sizeof(cl_mem),&f_dev);
    clSetKernelArg(clmanager->kernel,2,sizeof(float),&rcut);
    clSetKernelArg(clmanager->kernel,3,sizeof(float),&cs);
    clSetKernelArg(clmanager->kernel,4,sizeof(int),&nmol);

    clmanager->CreateCommandQueue();
  }

  void CalcForce(){
    clmanager->err = clEnqueueWriteBuffer(clmanager->queue,r_dev,CL_TRUE, 0,sizeof(cl_float3)*nmol,r, 0,NULL,NULL);

    clmanager->EnqueueNDRangeKernel(nmol,nmol/4);

    clmanager->err = clEnqueueReadBuffer(clmanager->queue,f_dev,CL_TRUE, 0,sizeof(cl_float3)*nmol,f, 0,NULL,NULL);
    //clmanager->WaitForEvents();
    //for(int i=0;i<nmol;i++) cout << i << ' ' << f[i] << ' ' << f[i+nmol] << ' ' << f[i+2*nmol] << endl;
  }
};
#endif
