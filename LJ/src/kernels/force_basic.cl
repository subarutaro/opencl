__kernel void molecular_dynamics(__global const float3* r, __global float3* f,
			const float rc,const float cs,const int nmol)
{
  const float rc2 = rc * rc;
  const float csh = 0.5f*cs;

  const size_t gid = get_global_id(0);
  const size_t gws = get_global_size(0);

  for(size_t i=gid;i<nmol;i+=gws){
    float3 r_i = r[i];
    float3 f_i = {0.f,0.f,0.f};

    for(int j=0;j<nmol;j++){
      if(i==j) continue;
      float x = r_i.x - r[j].x;
      float y = r_i.y - r[j].y;
      float z = r_i.z - r[j].z;
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
    f[i] = f_i;
  }
}
