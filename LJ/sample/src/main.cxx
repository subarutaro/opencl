#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

#include "clmd.hxx"

int main(int argc,char **argv){
  if(argc != 4){
    cerr << "usage: " << argv[0] << " nmol density temperature" << endl;
    exit(EXIT_FAILURE);
  }

  clMD md;
  //MD md;
  const int  nmol  = atoi(argv[1]);
  const float dens = atof(argv[2]);
  const float temp = atof(argv[3]);

  md.Initialize(nmol,dens,temp);
  md.DisplayConditions(cout);

  md.SetCoorFCC();
  md.SetVelRandom();
  md.CalcForce();

  md.CalcKineticEnergy();
  md.CalcPotentialEnergy();
  md.CalcHamiltonian();

  md.DisplayEnergies(cout);

  const int nstep = 1000000;
  const int nstep_vel_scale = 1000;
  const int output_interval = 100;
  md.timer.flush();
  for(int s=-nstep_vel_scale;s<nstep;s++){
    if(s>=0) md.timer.start();
    md.IntegrateVel();
    md.IntegrateCoor();

    md.CalcForce();

    md.IntegrateVel();
    if(s>=0) md.timer.end();
#if 1
    if(s>=0 && s%output_interval==0){
      /*
      ostringstream oss;
      oss << setfill('0') << setw(5) << s/output_interval;
      const string filename = oss.str() + ".cdv";
      md.OutputCDV(filename);
      //*/
      md.CalcKineticEnergy();
      md.CalcPotentialEnergy();
      if(s==0) md.CalcHamiltonian();
      md.DisplayEnergies(cout);
    }
#endif
    if(s<0) md.VelocityScaling();
  }

  md.timer.print(cout);
  //*/
  return EXIT_SUCCESS;
}

