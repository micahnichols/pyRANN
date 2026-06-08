
#include "rann_state_rosescreened.h"
#include "pair_spin_rann.h"

#include <cmath>

using namespace LAMMPS_NS::RANN;

State_rosescreened::State_rosescreened(PairRANN *_pair) : State(_pair)
{
  n_body_type = 2;
  dr = 0;
  re = 0;
  rc = 0;
  alpha = 0;
  delta = 0;
  id = -1;
  style = "rosescreened";
  atomtypes = new int[n_body_type];
  empty = true;
  fullydefined = false;
  _pair->doscreen = true;
  screen = true;
}

State_rosescreened::~State_rosescreened()
{
  delete [] atomtypes;
  delete [] rosetable;
  delete [] rosedtable;
}

//called after state equnation is declared for i-j type, but before its parameters are read.
void State_rosescreened::init(int *i,int _id)
{
  empty = false;
  for (int j=0;j<n_body_type;j++) {atomtypes[j] = i[j];}
  id = _id;
}

void State_rosescreened::eos_function(double *ep,double **force,double *Sik, double *dSikx, 
                                double*dSiky, double *dSikz, double *dSijkx, double *dSijky, 
                                double *dSijkz, bool *Bij,int ii,int nn,double *xn,double *yn,
                                double *zn,int *tn,int jnum,int* jl)
{
  int nelements = pair->nelements;
  int j;
  double rsq,f;
  int res = pair->res;
  double cutinv2 = 1/rc/rc;
  int jj;
  for (j=0;j<jnum;j++){
    if (Bij[j]==false) {continue;}
    if (atomtypes[1] != nelements && atomtypes[1] != tn[j])continue;
    rsq = xn[j]*xn[j]+yn[j]*yn[j]+zn[j]*zn[j];
    if (rsq > rc*rc)continue;
    //cubic interpolation from tables
    double r1 = (rsq*((double)res)*cutinv2);
    int m1 = (int)r1;
    if (m1>res || m1<1) {pair->errorf(FLERR,"invalid neighbor radius!");}
    if (rosetable[m1]==0) {continue;}
    double *p = &rosetable[m1-1];
    double *q = &rosedtable[m1-1];
    r1 = r1-trunc(r1);
    double de = q[1] + 0.5 * r1*(q[2] - q[0] + r1*(2.0*q[0] - 5.0*q[1] + 4.0*q[2] - q[3] + r1*(3.0*(q[1] - q[2]) + q[3] - q[0])));
    double e1 = p[1] + 0.5 * r1*(p[2] - p[0] + r1*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + r1*(3.0*(p[1] - p[2]) + p[3] - p[0])));
    de *= Sik[j];
    e1 *= Sik[j];
    ep[0] += e1;
    jj = jl[j];
    force[jj][0] += de*xn[j]+e1*dSikx[j];
    force[jj][1] += de*yn[j]+e1*dSiky[j];
    force[jj][2] += de*zn[j]+e1*dSikz[j];
    force[ii][0] -= de*xn[j]+e1*dSikx[j];
    force[ii][1] -= de*yn[j]+e1*dSiky[j];
    force[ii][2] -= de*zn[j]+e1*dSikz[j];
    for (int k=0;k<jnum;k++) {
      if (Bij[k]==false){continue;}
      int kk = jl[k];
      force[kk][0] += e1*dSijkx[j*jnum+k];
      force[kk][1] += e1*dSijky[j*jnum+k];
      force[kk][2] += e1*dSijkz[j*jnum+k];
      force[ii][0] -= e1*dSijkx[j*jnum+k];
      force[ii][1] -= e1*dSijky[j*jnum+k];
      force[ii][2] -= e1*dSijkz[j*jnum+k];
    }
  }
  return;
}

bool State_rosescreened::parse_values(std::string constant,std::vector<std::string> line1) {
  int l;
  int nwords=line1.size();
  if (constant.compare("ec")==0) {
    ec = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("re")==0) {
    re = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("rc")==0) {
    rc = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("alpha")==0) {
    alpha = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("delta")==0) {
    delta = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("dr")==0) {
    dr = strtod(line1[0].c_str(),nullptr);
  }
  else pair->errorf(FLERR,"Undefined value for rosescreened equation of state");
 //allow undefined delta, (default = 0)
  if (re!=0 && rc!=0 && alpha!=0 && dr!=0 && ec!=0)return true;
  return false;
}

void State_rosescreened::generate_rose_table()
{
  int buf = 5;
  int m;
  double r1,as,dfc,das,uberpoly,duberpoly;
  int res = pair->res;
  rosetable = new double[res + buf];
  rosedtable = new double[res + buf];
  for (m = 0; m < (res + buf); m++) {
    r1 = rc * rc * (double) (m) / (double) (res);
    if (sqrt(r1)>=rc){
      rosetable[m]=0;
      rosedtable[m]=0;
    }
    else if (sqrt(r1) <= (rc-dr)) {
      as = alpha*(sqrt(r1)/re-1);
      das = alpha/re;
      uberpoly = -ec*(1+as+delta*alpha*as*as*as/(alpha+as));
      duberpoly = -ec*(1+3*delta*alpha*as*as/(alpha+as)-delta*alpha*as*as*as/(alpha+as)/(alpha+as));
      rosetable[m] = uberpoly*exp(-as);
      rosedtable[m] = duberpoly*exp(-as)*das-uberpoly*exp(-as)*das;
      rosedtable[m]*=1/sqrt(r1);
    }
    else{
      as = alpha*(sqrt(r1)/re-1);
      dfc = -8*pow(1-(rc-sqrt(r1))/dr,3)/dr/(1-pow(1-(rc-sqrt(r1))/dr,4));
      das = alpha/re;
      uberpoly = -ec*(1+as+delta*alpha*as*as*as/(alpha+as));
      duberpoly = -ec*(1+3*delta*alpha*as*as/(alpha+as)-delta*alpha*as*as*as/(alpha+as)/(alpha+as));
      rosetable[m] = uberpoly*exp(-as)*cutofffunction(sqrt(r1), rc, dr);
      rosedtable[m] = duberpoly*exp(-as)*cutofffunction(sqrt(r1),rc,dr)*das-
                      uberpoly*exp(-as)*cutofffunction(sqrt(r1),rc,dr)*das+
                      uberpoly*exp(-as)*cutofffunction(sqrt(r1),rc,dr)*dfc;
      rosedtable[m]*=1/sqrt(r1);
    }
  }
}

void State_rosescreened::write_values(FILE *fid) {
  int i;
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:re:\n",style,id);
  fprintf(fid,"%f\n",re);
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
  fprintf(fid,":%s_%d:alpha:\n",style,id);
  fprintf(fid,"%f\n",alpha);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:delta:\n",style,id);
  fprintf(fid,"%f\n",delta);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:ec:\n",style,id);
  fprintf(fid,"%f\n",ec);
}
