
#include "rann_state_zbl.h"
#include "pair_spin_rann.h"

#include <cmath>

using namespace LAMMPS_NS::RANN;

State_zbl::State_zbl(PairRANN *_pair) : State(_pair)
{
  n_body_type = 2;
  dr = 0;
  rc = 0;
  zi = 0;
  zj = 0;
  id = -1;
  style = "zbl";
  atomtypes = new int[n_body_type];
  empty = true;
  fullydefined = false;
  _pair->allscreen = false;
}

State_zbl::~State_zbl()
{
  delete [] atomtypes;
  delete [] table;
  delete [] dtable;
}

//called after state equnation is declared for i-j type, but before its parameters are read.
void State_zbl::init(int *i,int _id)
{
  empty = false;
  for (int j=0;j<n_body_type;j++) {atomtypes[j] = i[j];}
  id = _id;
}

void State_zbl::eos_function(double *ep,double **force,int ii,int nn,
                                double *xn,double *yn,double *zn,int *tn,int jnum,int* jl)
{
  int nelements = pair->nelements;
  int j;
  double rsq,f;
  int res = pair->res;
  double cutinv2 = 1/rc/rc;
  int jj;
  for (j=0;j<jnum;j++){
    if (atomtypes[1] != nelements && atomtypes[1] != tn[j])continue;
    rsq = xn[j]*xn[j]+yn[j]*yn[j]+zn[j]*zn[j];
    if (rsq > rc*rc)continue;
    //cubic interpolation from tables
    double r1 = (rsq*((double)res)*cutinv2);
    int m1 = (int)r1;
    if (m1>res || m1<1) {pair->errorf(FLERR,"invalid neighbor radius!");}
    if (table[m1]==0) {continue;}
    double *p = &table[m1-1];
    double *q = &dtable[m1-1];
    r1 = r1-trunc(r1);
    double de = q[1] + 0.5 * r1*(q[2] - q[0] + r1*(2.0*q[0] - 5.0*q[1] + 4.0*q[2] - q[3] + r1*(3.0*(q[1] - q[2]) + q[3] - q[0])));
    double e1 = p[1] + 0.5 * r1*(p[2] - p[0] + r1*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + r1*(3.0*(p[1] - p[2]) + p[3] - p[0])));
    ep[0] += e1;
    jj = jl[j];
    force[jj][0] += de*xn[j];
    force[jj][1] += de*yn[j];
    force[jj][2] += de*zn[j];
    force[ii][0] -= de*xn[j];
    force[ii][1] -= de*yn[j];
    force[ii][2] -= de*yn[j];
  }
  return;
}

bool State_zbl::parse_values(std::string constant,std::vector<std::string> line1) {
  int l;
  int nwords=line1.size();
  if (constant.compare("zi")==0) {
    zi = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("zj")==0) {
    zj = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("rc")==0) {
    rc = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("dr")==0) {
    dr = strtod(line1[0].c_str(),nullptr);
  }
  else pair->errorf(FLERR,"Undefined value for zbl equation of state");
  if (zi!=0 && rc!=0 && zj!=0 && dr!=0)return true;
  return false;
}

void State_zbl::generate_table()
{
  int buf = 5;
  int m;
  double r1,as,dfc,das,uberpoly,duberpoly;
  int res = pair->res;
  double a1,a2,a3,a4;
  double b1,b2,b3,b4;
  a1 = 0.18175;
  b1 = -3.19980;
  a2 = 0.50986;
  b2 = -0.94229;
  a3 = 0.28022;
  b3 = -0.40290;
  a4 = 0.02817;
  b4 = -0.20162;
  double a = 0.46850/(pow(zi,0.23)+pow(zj,0.23));
  table = new double[res + buf];
  dtable = new double[res + buf];
  r1 = rc*rc;
  double x = sqrt(r1)/a;
  double phi = a1*exp(b1)+a2*exp(b2)+a3*exp(b3)+a4*exp(b4);
  double dphi = a1*b1*exp(b1)+a2*b2*exp(b2)+a3*b3*exp(b3)+a4*b4*exp(b4);
  double ddphi = a1*b1*b1*exp(b1)+a2*b2*b2*exp(b2)+a3*b3*b3*exp(b3)+a4*b4*b4*exp(b4);
  dphi *= 1/a;
  ddphi *= 1/a/a;
  double Ecr = zi*zj*phi*14.3996454946;//e^2/4/pi/epsilon in eV*A
  double dEcr = Ecr*dphi/phi;
  double ddEcr = Ecr*ddphi/phi;
  double A = (-3*dEcr+dr*ddEcr)/dr/dr;
  double B = (2*dEcr-dr*ddEcr)/dr/dr/dr;
  double C = -Ecr+1/2*dr*dEcr-1/12*dr*dr*ddEcr;
  for (m = 0; m < (res + buf); m++) {
    r1 = rc * rc * (double) (m) / (double) (res);
    double x = sqrt(r1)/a;
    double phi = a1*exp(b1)+a2*exp(b2)+a3*exp(b3)+a4*exp(b4);
    double dphi = a1*b1*exp(b1)+a2*b2*exp(b2)+a3*b3*exp(b3)+a4*b4*exp(b4);
    double ddphi = a1*b1*b1*exp(b1)+a2*b2*b2*exp(b2)+a3*b3*b3*exp(b3)+a4*b4*b4*exp(b4);
    dphi *= 1/a;
    ddphi *= 1/a/a;
    double Ec = zi*zj*phi*14.3996454946;//e^2/4/pi/epsilon in eV*A
    double dEc = Ec*dphi/phi;
    double ddEC = Ec*ddphi/phi;
    double S = 0;
    if (sqrt(r1)>=rc){
      dtable[m]=0;
    }
    else if (sqrt(r1) <= (rc-dr)) {
      S = C;
      dtable[m] = dEc/sqrt(r1);
    }
    else{
      S = A/3*pow(sqrt(r1)-rc+dr,3)+B/4*pow(sqrt(r1)-rc+dr,4)+C;
      double dS = A*pow(sqrt(r1)-rc+dr,2)+B*pow(sqrt(r1)-rc+dr,3);
      dtable[m] = (dEc+dS)/sqrt(r1);
    }
    table[m] = Ec+S;
  }
}

void State_zbl::write_values(FILE *fid) {
  int i;
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:rc:\n",style,id);
  fprintf(fid,"%f\n",rc);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:dr:\n",style,id);
  fprintf(fid,"%f\n",dr);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:zi:\n",style,id);
  fprintf(fid,"%f\n",zi);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:zj:\n",style,id);
  fprintf(fid,"%f\n",zj);
}
