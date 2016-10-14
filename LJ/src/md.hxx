#ifndef HXX_MD
#define HXX_MD

#include <iostream>
#include <fstream>
#include <string>

#include <cassert>
#include <cmath>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "timer.hxx"

#define PBC
//#define NEWTON3

class MD{
protected:
  cl_float3 *r;
  cl_float3 *v;
  cl_float3 *f;

  int  nmol;
  cl_float dens;
  cl_float temp;

  cl_float dt,dth;
  cl_float rcut;
  cl_float cs,csh;

public:
  cl_float  pot,kin,tot;
  cl_float4 vir;

  Timer timer;

  MD(){};

  ~MD(){
    if(r) free(r);
    if(v) free(v);
    if(f) free(f);
  }

  virtual void Initialize(const int,const cl_float,const cl_float);

  void SetCoorFCC(){
    int ndim=1;
    while(nmol > 4*ndim*ndim*ndim) ndim++;
    cl_float unit = cs / (cl_float)ndim;
    {
      int i = 0;
      for(int z=0;z<ndim;z++){
	for(int y=0;y<ndim;y++){
	  for(int x=0;x<ndim;x++){
	    r[i].x = unit * x;
	    r[i].y = unit * y;
	    r[i].z = unit * z;
	    i++;if(i==nmol) break;
	    r[i].x = unit * (x+0.5f);
	    r[i].y = unit * (y+0.5f);
	    r[i].z = unit * z;
	    i++;if(i==nmol) break;
	    r[i].x = unit * x;
	    r[i].y = unit * (y+0.5f);
	    r[i].z = unit * (z+0.5f);
	    i++;if(i==nmol) break;
	    r[i].x = unit * (x+0.5f);
	    r[i].y = unit * y;
	    r[i].z = unit * (z+0.5f);
	    i++;if(i==nmol) break;
	  }
	  if(i==nmol) break;
	}
	if(i==nmol) break;
      }
    }
    for(int i=0;i<nmol;i++){
      r[i].x -= 0.5f*cs;
      r[i].y -= 0.5f*cs;
      r[i].z -= 0.5f*cs;
    }
  }

  void SetVelRandom(){
    for(int i=0;i<nmol;i++){
      v[i].x = (cl_float)rand()/(cl_float)RAND_MAX;
      v[i].y = (cl_float)rand()/(cl_float)RAND_MAX;
      v[i].z = (cl_float)rand()/(cl_float)RAND_MAX;
    }
    KillMomentum();
    VelocityScaling();
  }

  void VelocityScaling(){
    CalcKineticEnergy();
    const cl_float scale = sqrt(1.5*nmol*temp/kin);
    for(int i=0;i<nmol;i++){
      v[i].x *= scale;
      v[i].y *= scale;
      v[i].z *= scale;
    }
    kin *= scale*scale;
    KillMomentum();
  }

  void KillMomentum(){
    cl_float3 mom = {0.f,0.f,0.f};
    for(int i=0;i<nmol;i++){
      mom.x += v[i].x;
      mom.y += v[i].y;
      mom.z += v[i].z;
    }
    mom.x /= (cl_float)nmol;
    mom.y /= (cl_float)nmol;
    mom.z /= (cl_float)nmol;
    for(int i=0;i<nmol;i++){
      v[i].x -= mom.x;
      v[i].y -= mom.y;
      v[i].z -= mom.z;
    }
  }

  void IntegrateCoor(){
    for(int i=0;i<nmol;i++){
      r[i].x += v[i].x * dt;
      r[i].y += v[i].y * dt;
      r[i].z += v[i].z * dt;
      if(r[i].x <= -csh) r[i].x += cs; if(r[i].x > csh) r[i].x -= cs;
      if(r[i].y <= -csh) r[i].y += cs; if(r[i].y > csh) r[i].y -= cs;
      if(r[i].z <= -csh) r[i].z += cs; if(r[i].z > csh) r[i].z -= cs;
    }
  }

  void IntegrateVel(){
    for(int i=0;i<nmol;i++){
      v[i].x += f[i].x * 0.5*dt;
      v[i].y += f[i].y * 0.5*dt;
      v[i].z += f[i].z * 0.5*dt;
    }
  }

  virtual void CalcForce();

  void CalcPotentialEnergy(){
    pot = 0.f;
    vir.x = vir.y = vir.z = vir.w = 0.f;
    const cl_float csh = 0.5f*cs;
    const cl_float rc2 = rcut*rcut;
    for(int i=0;i<nmol;i++){
      const cl_float3 ri = r[i];
      cl_float3 fi = {0.f,0.f,0.f};
      for(int j=0;j<nmol;j++){
	if(i==j) continue;
	cl_float dx = ri.x - r[j].x;
	cl_float dy = ri.y - r[j].y;
	cl_float dz = ri.z - r[j].z;
	if(dx <= -csh) dx += cs; if(dx > csh) dx -= cs;
	if(dy <= -csh) dy += cs; if(dy > csh) dy -= cs;
	if(dz <= -csh) dz += cs; if(dz > csh) dz -= cs;

	const cl_float r02 = dx*dx + dy*dy + dz*dz;
	if(r02>rc2) continue;
	const cl_float r02i = 1.f / r02;
	const cl_float r06i = r02i * r02i *r02i;
	pot += 2.f * r06i * (r06i - 1.f);
	const cl_float ftmp = 24.f * r06i * (r06i - 0.5f) * r02i;
	vir.x += ftmp * dx * dx;
	vir.y += ftmp * dy * dy;
	vir.z += ftmp * dz * dz;
	vir.w += ftmp * r02;
      }
    }
  }

  void CalcKineticEnergy(){
    kin = 0.0;
    for(int i=0;i<nmol;i++){
      kin += v[i].x*v[i].x + v[i].y*v[i].y + v[i].z*v[i].z;
    }
    kin *= 0.5f;
  }

  void CalcHamiltonian(){tot = pot + kin;}

  void OutputCDV(const std::string filename){
    std::ofstream ofs(filename.c_str());
    ofs << "'box_sx=" << -csh << ",box_ex="<< csh;
    ofs << ",box_sy=" << -csh << ",box_ey="<< csh;
    ofs << ",box_sz=" << -csh << ",box_ez="<< csh;
    ofs << std::endl;
    for(int i=0;i<nmol;i++){
      ofs << ' ' << i     << " " << "0";
      ofs << ' ' << r[i].x << ' ' << r[i].y << ' ' << r[i].z;
      ofs << ' ' << v[i].x << ' ' << v[i].y << ' ' << v[i].z;
      ofs << ' ' << f[i].x << ' ' << f[i].y << ' ' << f[i].z;
      ofs << std::endl;
    }
  }

  void DisplayEnergies(std::ostream &os){
    static int s = 0;
    os << s++ << ' ' << pot << ' ' << kin << ' ' << pot+kin << ' ' << (pot+kin - tot)/tot << std::endl;
  }

  void DisplayConditions(std::ostream &os){
    os << "# Number of molecules:\t" << nmol   << std::endl;
    os << "# Density:\t"        << dens        << std::endl;
    os << "# Temperature:\t"    << temp        << std::endl;
    os << "# Cell size:\t"      << cs          << std::endl;
    os << "# Cutoff radii:\t"   << rcut        << std::endl;
    os << "# Delta time:\t"     << dt << ' ' << dth << std::endl;
    os << std::endl;
  }
  template <class TResult>
  void CopyToResult(TResult&);
  void SetTemperature(cl_float _temp){temp = _temp;}
};

void MD::Initialize(const int _nmol,const cl_float _dens,const cl_float _temp){
  std::cout << "# ---construct MD---" << std::endl;
  nmol = _nmol;
  dt   = 0.000930f;
  dth  = 0.5f * dt;
  dens = _dens;
  temp = _temp;

  cs = powf((cl_float)nmol/dens,1./3.);
  csh = 0.5f * cs;

  if(csh > 5.f) rcut = 5.f;
  else          rcut = csh;

  assert((r = (cl_float3*)calloc(nmol,sizeof(cl_float3)))!=NULL);
  assert((v = (cl_float3*)calloc(nmol,sizeof(cl_float3)))!=NULL);
  assert((f = (cl_float3*)calloc(nmol,sizeof(cl_float3)))!=NULL);
}

void MD::CalcForce(){
  const cl_float csh = 0.5f*cs;
  const cl_float rc2 = rcut*rcut;
  for(int i=0;i<nmol;i++){
    const cl_float3 ri = r[i];
    cl_float3 fi = {0.f,0.f,0.f};
    for(int j=0;j<nmol;j++){
      if(i==j) continue;
      cl_float dx = ri.x - r[j].x;
      cl_float dy = ri.y - r[j].y;
      cl_float dz = ri.z - r[j].z;
      if(dx <= -csh) dx += cs; if(dx > csh) dx -= cs;
      if(dy <= -csh) dy += cs; if(dy > csh) dy -= cs;
      if(dz <= -csh) dz += cs; if(dz > csh) dz -= cs;

      const cl_float r02 = dx*dx + dy*dy + dz*dz;
      if(r02>rc2) continue;
      const cl_float r02i = 1.f / r02;
      const cl_float r06i = r02i * r02i *r02i;

      const cl_float ftmp = 48.f * r06i * (r06i - 0.5f) * r02i;
      fi.x += ftmp * dx;
      fi.y += ftmp * dy;
      fi.z += ftmp * dz;
    }
    f[i].x = fi.x;
    f[i].y = fi.y;
    f[i].z = fi.z;
  }
}

#endif
