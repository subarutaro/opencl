#include <iostream>
#include <sys/time.h>

class Timer{
private:
  unsigned int count;
  double elapsed_time,max,min;
  struct timeval timer_start,timer_end;
public:
  Timer(){}
  ~Timer(){}
  void flush(){elapsed_time=max=0.0;min=1e32;count=0;}
  void start(){
    gettimeofday(&timer_start,NULL);
  }
  void end(){
    gettimeofday(&timer_end,NULL);
    double e = (timer_end.tv_sec - timer_start.tv_sec) + (timer_end.tv_usec - timer_start.tv_usec) * 0.000001;
    elapsed_time += e;
    if(max<e) max = e;
    if(min>e) min = e;
    count++;
  }
  void print(std::ostream &os){
    os << "# elapsed time[sec]:\t" << elapsed_time << std::endl;
    os << "# average time[sec]:\t" << elapsed_time / (double)count << std::endl;
    os << "# maximum time[sec]:\t" << max << std::endl;
    os << "# minimum time[sec]:\t" << min << std::endl;
  }
};

