
#include "rann_state_eamscreenedsingle.h"
#include "pair_spin_rann.h"
#include <cmath>

using namespace LAMMPS_NS::RANN;

State_eamscreenedsingle::State_eamscreenedsingle(PairRANN *_pair) : State(_pair)
{
  n_body_type = 2;
  dr = 0;
  re = 0;
  rc = 0;
  alpha = 0;
  delta = 0;
  Z1 = 0;
  Z2 = 0;
  snn = 0;
  beta0 = 0;
  Asub = 0;
  C = 0;
  Cmin = 0;
  Cmax = 0;
  id = -1;
  style = "eamscreenedsingle";
  atomtypes = new int[n_body_type];
  empty = true;
  fullydefined = false;
  _pair->doscreen = true;
  screen = true;
}

State_eamscreenedsingle::~State_eamscreenedsingle()
{
  delete [] atomtypes;
  delete [] rosetable;
  delete [] rosedtable;
}

//called after state equnation is declared for i-j type, but before its parameters are read.
void State_eamscreenedsingle::init(int *i,int _id)
{
  empty = false;
  for (int j=0;j<n_body_type;j++) {atomtypes[j] = i[j];}
  id = _id;
}

void State_eamscreenedsingle::eos_function(double *ep,double **force,double *Sik, double *dSikx, 
                                double*dSiky, double *dSikz, double *dSijkx, double *dSijky, 
                                double *dSijkz, bool *Bij,int ii,int nn,double *xn,double *yn,
                                double *zn,int *tn,int jnum,int* jl)
{
  int nelements = pair->nelements;
  int j;
  double rsq,f;
  double rhobar = 0;
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
    double *emp = &embedtable[m1-1];
    double *emq = &embeddtable[m1-1];
    r1 = r1-trunc(r1);
    double de = q[1] + 0.5 * r1*(q[2] - q[0] + r1*(2.0*q[0] - 5.0*q[1] + 4.0*q[2] - q[3] + r1*(3.0*(q[1] - q[2]) + q[3] - q[0])));
    double e1 = p[1] + 0.5 * r1*(p[2] - p[0] + r1*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + r1*(3.0*(p[1] - p[2]) + p[3] - p[0])));
    double ee = emp[1] + 0.5 * r1*(emp[2] - emp[0] + r1*(2.0*emp[0] - 5.0*emp[1] + 4.0*emp[2] - emp[3] + r1*(3.0*(emp[1] - emp[2]) + emp[3] - emp[0])));
    double dee = emq[1] + 0.5 * r1*(emq[2] - emq[0] + r1*(2.0*emq[0] - 5.0*emq[1] + 4.0*emq[2] - emq[3] + r1*(3.0*(emq[1] - emq[2]) + emq[3] - emq[0])));
    de *= Sik[j];
    e1 *= Sik[j];
    ee *= Sik[j];
    dee *= Sik[j];
    ep[0] += e1;
    rhobar += ee;
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
  if (rhobar>0) {
    ep[0]+=Asub*ec*rhobar*log(rhobar);
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
      double *emp = &embedtable[m1-1];
      double *emq = &embeddtable[m1-1];
      r1 = r1-trunc(r1);
      double ee = emp[1] + 0.5 * r1*(emp[2] - emp[0] + r1*(2.0*emp[0] - 5.0*emp[1] + 4.0*emp[2] - emp[3] + r1*(3.0*(emp[1] - emp[2]) + emp[3] - emp[0])));
      double dee = emq[1] + 0.5 * r1*(emq[2] - emq[0] + r1*(2.0*emq[0] - 5.0*emq[1] + 4.0*emq[2] - emq[3] + r1*(3.0*(emq[1] - emq[2]) + emq[3] - emq[0])));
      ee *= Sik[j];
      dee *= Sik[j];
      jj = jl[j];
      force[jj][0] -= Asub*ec*(log(rhobar)+1) * (dee*xn[j] - ee*dSikx[j]);
      force[jj][1] -= Asub*ec*(log(rhobar)+1) * (dee*yn[j] - ee*dSiky[j]);
      force[jj][2] -= Asub*ec*(log(rhobar)+1) * (dee*zn[j] - ee*dSikz[j]);
      force[ii][0] += Asub*ec*(log(rhobar)+1) * (dee*xn[j] - ee*dSikx[j]);
      force[ii][1] += Asub*ec*(log(rhobar)+1) * (dee*yn[j] - ee*dSiky[j]);
      force[ii][2] += Asub*ec*(log(rhobar)+1) * (dee*zn[j] - ee*dSikz[j]);
      for (int k=0;k<jnum;k++) {
	if (Bij[k]==false){continue;}
	int kk = jl[k];
	force[kk][0] += Asub*ec*(log(rhobar)+1)*ee*dSijkx[j*jnum+k];
	force[kk][1] += Asub*ec*(log(rhobar)+1)*ee*dSijky[j*jnum+k];
	force[kk][2] += Asub*ec*(log(rhobar)+1)*ee*dSijkz[j*jnum+k];
	force[ii][0] -= Asub*ec*(log(rhobar)+1)*ee*dSijkx[j*jnum+k];
	force[ii][1] -= Asub*ec*(log(rhobar)+1)*ee*dSijky[j*jnum+k];
	force[ii][2] -= Asub*ec*(log(rhobar)+1)*ee*dSijkz[j*jnum+k];
      }
    }
  }
  return;
}

bool State_eamscreenedsingle::parse_values(std::string constant,std::vector<std::string> line1) {
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
  else if (constant.compare("Z1")==0) {
    Z1 = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("Z2")==0) {
    Z2 = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("snn")==0) {
    snn = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("beta0")==0) {
    beta0 = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("Asub")==0) {
    Asub = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("C")==0) {
    C = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("Cmax")==0) {
    Cmax = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("Cmin")==0) {
    Cmin = strtod(line1[0].c_str(),nullptr);
  }
  else pair->errorf(FLERR,"Undefined value for eamscreenedsingle equation of state");
 //allow undefined delta, (default = 0)
  if (re!=0 && rc!=0 && alpha!=0 && dr!=0 && ec!=0)return true;
  return false;
}

void State_eamscreenedsingle::generate_rose_table()
{
  int buf = 5;
  int m,n;
  double r1,as,bs,bs2,bsr,bsr2,dfc,das,dasr,dbs,dbsr,dbs2,dbsr2,uberpoly,duberpoly,uberpolyr,duberpolyr,S,asr,pre,rr;
  double embed,dembed,refden,embedr,dembedr;
  int res = pair->res;
  rosetable = new double[res + buf];
  rosedtable = new double[res + buf];
  embedtable = new double[res + buf];
  embeddtable = new double[res + buf];
  for (m = 0; m < (res + buf); m++) {
    r1 = rc * rc * (double) (m) / (double) (res);
    if (sqrt(r1)>=rc){
      rosetable[m]=0;
      rosedtable[m]=0;
      embedtable[m]=0;
      embeddtable[m]=0;
    }
    else {
      as = alpha*(sqrt(r1)/re-1);
      bs = beta0*(sqrt(r1)/re-1);
      bs2 = beta0*(snn*sqrt(r1)/re-1);
      das = alpha/re;
      dbs = beta0/re;
      dbs2 = beta0*snn/re;
      S = cutofffunction(Cmin, C, Cmax-Cmin);
      S = S*S*S*S;
      uberpoly = -(1+as+delta*alpha*as*as*as/(alpha+as));
      duberpoly = -(1+3*delta*alpha*as*as/(alpha+as)-delta*alpha*as*as*as/(alpha+as)/(alpha+as));
      refden = Z1+Z2*exp(-beta0*(snn-1))*S;
      embed = Asub*(Z1*exp(-bs)+Z2*exp(-bs2)*S)*log((Z1*exp(-bs)+Z2*exp(-bs2)*S)/refden);
      dembed = Asub*(Z1*exp(-bs)*dbs+Z2*exp(-bs2)*S*dbs2)*(1+log((Z1*exp(-bs)+Z2*exp(-bs2)*S)/refden));
      rosetable[m] = ec/Z1*(uberpoly*exp(-as)-embed/refden);
      rosedtable[m] = ec/Z1*(duberpoly*exp(-as)*das-uberpoly*exp(-as)*das+dembed/refden);
      embedtable[m] = exp(-bs)/refden;
      embeddtable[m] = exp(-bs)/refden*dbs;
      pre = 1;
      rr = sqrt(r1);
      dasr = das;
      dbsr = dbs;
      for (n = 1; n < 10; n++) {
	rr *= snn;
  if (rr>rc) break;
	asr = alpha*(rr/re-1);
	bsr = beta0*(rr/re-1);
	bsr2 = beta0*(snn*rr/re-1);
	dasr *= snn;
	dbsr *= snn;
	dbsr2 = dbsr*snn;
	pre *= -Z2*S/Z1;
	uberpolyr = -pre*(1+asr+delta*alpha*asr*asr*asr/(alpha+asr));
	duberpolyr = -pre*(1+3*delta*alpha*asr*asr/(alpha+asr)-delta*alpha*asr*asr*asr/(alpha+asr)/(alpha+asr));
	embedr = pre*Asub*(Z1*exp(-bsr)+Z2*exp(-bsr2)*S)*log((Z1*exp(-bsr)+Z2*exp(-bsr2)*S)/refden);
        dembedr = pre*Asub*(Z1*exp(-bsr)*dbsr+Z2*exp(-bsr2)*S*dbsr2)*(1+log((Z1*exp(-bsr)+Z2*exp(-bsr2)*S)/refden));	 
	rosetable[m] += ec/Z1*(uberpolyr*exp(-asr)-embedr/refden);
        rosedtable[m] += ec/Z1*(duberpolyr*exp(-asr)*dasr-uberpolyr*exp(-asr)*dasr+dembedr/refden);
      }
    }
    if ((sqrt(r1) > (rc-dr)) && (sqrt(r1)<rc)){
      dfc = -8*pow(1-(rc-sqrt(r1))/dr,3)/dr/(1-pow(1-(rc-sqrt(r1))/dr,4));
      rosetable[m] *= cutofffunction(sqrt(r1), rc, dr);
      rosedtable[m] = rosedtable[m]*cutofffunction(sqrt(r1),rc,dr)-
                      rosetable[m]*dfc;
      embedtable[m] *= cutofffunction(sqrt(r1),rc, dr);
      embeddtable[m] = embeddtable[m]*cutofffunction(sqrt(r1),rc,dr)-embedtable[m]*dfc;
    }
    rosedtable[m]*=1/sqrt(r1);
    embeddtable[m]*=1/sqrt(r1);
  }
  // FILE *file = fopen("eamtabletest2.csv","w");
  // for (m=0;m<(res+buf);m++){
  //   r1 = rc * rc * (double) (m) / (double) (res);
  //   double r = sqrt(r1);
  //   refden = Z1+Z2*exp(-beta0*(snn-1))*S;
  //   fprintf(file,"%f, %.10f, %.10f\n",r,rosetable[m],embedtable[m]*refden);
  // }
  // fclose(file);
}

void State_eamscreenedsingle::write_values(FILE *fid) {
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
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:Z1:\n",style,id);
  fprintf(fid,"%f\n",Z1);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:Z2:\n",style,id);
  fprintf(fid,"%f\n",Z2);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:snn:\n",style,id);
  fprintf(fid,"%f\n",snn);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:beta0:\n",style,id);
  fprintf(fid,"%f\n",beta0);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:Asub:\n",style,id);
  fprintf(fid,"%f\n",Asub);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:C:\n",style,id);
  fprintf(fid,"%f\n",C);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:Cmax:\n",style,id);
  fprintf(fid,"%f\n",Cmax);
  fprintf(fid,"stateequationconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:Cmin:\n",style,id);
  fprintf(fid,"%f\n",Cmin);

}
