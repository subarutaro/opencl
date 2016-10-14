__kernel
void molecular_dynamics(__global const float* r, __global float* f,
			const float rc,const float cs,const int nmol)
{
  const float rc2 = rc * rc;
  const float csh = 0.5f*cs;

  __global const float *rx = r;
  __global const float *ry = r + nmol;
  __global const float *rz = r + 2*nmol;

  __global float *fx = f;
  __global float *fy = f + nmol;
  __global float *fz = f + 2*nmol;

  const size_t lmem_size = get_local_size(0);
  __local float rlx[1024];
  __local float rly[1024];
  __local float rlz[1024];

  const size_t gid = get_global_id(0);
  const size_t gws = get_global_size(0);
  const size_t lid = get_local_id(0);
  const int i = gid;
  for(int i=gid;i<nmol;i+=gws){
    float3 r_i = {rx[i],ry[i],rz[i]};
    float3 f_i = {0.f,0.f,0.f};

    for(size_t l=0;l<(size_t)nmol;l+=lmem_size){
      barrier(CLK_LOCAL_MEM_FENCE);
      rlx[lid] = rx[l+lid];
      rly[lid] = ry[l+lid];
      rlz[lid] = rz[l+lid];
      barrier(CLK_LOCAL_MEM_FENCE);
      for(size_t j=0;j<lmem_size;j++){
	if(i==j+l) continue;
#if 1
	float x = r_i.x - rlx[j];
	float y = r_i.y - rly[j];
	float z = r_i.z - rlz[j];
#else
	float x = r_i.x - rx[j+l];
	float y = r_i.y - ry[j+l];
	float z = r_i.z - rz[j+l];
#endif
	if(x >= csh) x -= cs; if(x < -csh) x += cs;
	if(y >= csh) y -= cs; if(y < -csh) y += cs;
	if(z >= csh) z -= cs; if(z < -csh) z += cs;

	const float r2 = x*x + y*y + z*z;
	//if(r2 > rc2) continue;
	const float r2i = 1.f / r2;
	const float r6i = r2i*r2i*r2i;
	const float ftmp = r6i*(48.f*r6i - 24.f)*r2i;
	f_i.x += ftmp*x;
	f_i.y += ftmp*y;
	f_i.z += ftmp*z;
      }
    }
    fx[i] = f_i.x;
    fy[i] = f_i.y;
    fz[i] = f_i.z;
  }
}
