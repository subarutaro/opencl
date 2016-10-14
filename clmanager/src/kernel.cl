__kernel
void test(__global float* data, const int n)
{
    int index = get_global_id(0);
    if(index==0) printf("inside kernel: %d %d\n",get_global_size(0),get_local_size(0));
    if (index < n) {
      data[index] = get_local_id(0);
    }
}
