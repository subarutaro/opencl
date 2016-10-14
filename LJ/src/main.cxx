#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>

#include "clmd.hxx"

int main(int argc,char **argv){
  //clMD md;
  MD md;
  if(argc != 4){
    cerr << "usage: " << argv[0] << " nmol density temperature" << endl;
    exit(EXIT_FAILURE);
  }
  const int  nmol = atoi(argv[1]);
  const DorF dens = atof(argv[2]);
  const DorF temp = atof(argv[3]);

  md.Initialize(nmol,dens,temp);
  md.DisplayConditions(cout);

  md.SetCoorFCC();
  md.SetVelRandom();
  md.CalcForce();

  const int nstep = 10000;
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
    if(s>=0 && s%output_interval==0){
      md.CalcPotentialEnergy();
      md.CalcKineticEnergy();
      if(s==0) md.CalcHamiltonian();
      md.DisplayEnergies(cout);
    }
    if(s<0) md.VelocityScaling();
  }
  md.timer.print(cout);

  return EXIT_SUCCESS;
}

