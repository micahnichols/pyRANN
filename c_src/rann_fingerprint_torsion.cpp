// clang-format off
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */
/*  ----------------------------------------------------------------------
   Contributing authors: Christopher Barrett (MSU) barrett@me.msstate.edu
                              Doyl Dickel (MSU) doyl@me.msstate.edu
    ----------------------------------------------------------------------*/
/*
“The research described and the resulting data presented herein, unless
otherwise noted, was funded under PE 0602784A, Project T53 "Military
Engineering Applied Research", Task 002 under Contract No. W56HZV-17-C-0095,
managed by the U.S. Army Combat Capabilities Development Command (CCDC) and
the Engineer Research and Development Center (ERDC).  The work described in
this document was conducted at CAVS, MSU.  Permission was granted by ERDC
to publish this information. Any opinions, findings and conclusions or
recommendations expressed in this material are those of the author(s) and
do not necessarily reflect the views of the United States Army.​”

DISTRIBUTION A. Approved for public release; distribution unlimited. OPSEC#4918
 */

#include "rann_fingerprint_torsion.h"
#include "pair_spin_rann.h"

#include <cmath>

using namespace LAMMPS_NS::RANN;

Fingerprint_torsion::Fingerprint_torsion(PairRANN *_pair) : Fingerprint(_pair)
{
  n_body_type = 4;
  dr = 0;
  re = 0;
  rc = 0;
  rc_inner = 0;
  alpha_k = new double[1];
  alpha_k[0] = -1;
  kmax = 0;
  mlength = 0;
  id = -1;
  style = "torsion";
  atomtypes = new int[n_body_type];
  empty = true;
  _pair->allscreen = false;
}

Fingerprint_torsion::~Fingerprint_torsion() {
  delete[] alpha_k;
  delete[] atomtypes;
  delete[] expcuttable;
  delete[] dfctable;
  delete[] rinvsqrttable;
}

bool Fingerprint_torsion::parse_values(std::string constant,std::vector<std::string> line1) {
  int nwords,l;
  nwords=line1.size();
  if (constant.compare("re")==0) {
    re = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("rc")==0) {
    rc = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("rcinner")==0) {
    rc_inner = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("alphak")==0) {
    delete[] alpha_k;
    alpha_k = new double[nwords];
    for (l=0;l<nwords;l++) {
      alpha_k[l]=strtod(line1[l].c_str(),nullptr);
    }
  }
  else if (constant.compare("dr")==0) {
    dr = strtod(line1[0].c_str(),nullptr);
  }
  else if (constant.compare("k")==0) {
    kmax = strtol(line1[0].c_str(),nullptr,10);
  }
  else if (constant.compare("m")==0) {
    mlength = strtol(line1[0].c_str(),nullptr,10);
  }
  else pair->errorf(FLERR,"Undefined value for bond power");
  if (re!=0.0 && rc!=0.0 && rc_inner != 0.0 && alpha_k[0]!=-1 && dr!=0.0 && mlength!=0 && kmax!=0)return true;
  return false;
}

void Fingerprint_torsion::write_values(FILE *fid) {
  int i;
  fprintf(fid,"fingerprintconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:re:\n",style,id);
  fprintf(fid,"%f\n",re);
  fprintf(fid,"fingerprintconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:rc:\n",style,id);
  fprintf(fid,"%f\n",rc);
  fprintf(fid,"fingerprintconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:rcinner:\n",style,id);
  fprintf(fid,"%f\n",rc_inner);
  fprintf(fid,"fingerprintconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:alphak:\n",style,id);
  for (i=0;i<kmax;i++) {
    fprintf(fid,"%f ",alpha_k[i]);
  }
  fprintf(fid,"\n");
  fprintf(fid,"fingerprintconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:dr:\n",style,id);
  fprintf(fid,"%f\n",dr);
  fprintf(fid,"fingerprintconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:k:\n",style,id);
  fprintf(fid,"%d\n",kmax);
  fprintf(fid,"fingerprintconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:m:\n",style,id);
  fprintf(fid,"%d\n",mlength);
}

void Fingerprint_torsion::init(int *i,int _id) {
  for (int j=0;j<n_body_type;j++) {atomtypes[j] = i[j];}
  re = 0;
  rc = 0;
  mlength = 0;
  kmax = 0;
  delete[] alpha_k;
  alpha_k = new double[1];
  alpha_k[0]=-1;
  empty = false;
  id = _id;
}

//number of neurons defined by this fingerprint
int Fingerprint_torsion::get_length() {
  return mlength*kmax;
}

void Fingerprint_torsion::allocate() {
  generate_exp_cut_table();
  generate_rinvssqrttable();
}

//Generate table of complex functions for quick reference during compute. Used by do3bodyfeatureset_singleneighborloop and do3bodyfeatureset_doubleneighborloop.
void Fingerprint_torsion::generate_exp_cut_table() {
  int m,n;
  double r1;
  int buf = 5;
  int res = pair->res;
  double cutmax = pair->cutmax;
  expcuttable = new double[(res+buf)*(kmax)];
  dfctable = new double[res+buf];
  for (m=0;m<(res+buf);m++) {
    r1 = cutmax*cutmax*(double)(m)/(double)(res);
    for (n=0;n<(kmax);n++) {
      expcuttable[n+m*(kmax)] = exp(-alpha_k[n]/re*sqrt(r1))*cutofffunction(sqrt(r1),rc_inner,dr);
    }
    if (sqrt(r1)>=rc_inner || sqrt(r1) <= (rc_inner-dr)) {
      dfctable[m]=0;
    }
    else{
      dfctable[m]=-8*pow(1-(rc_inner-sqrt(r1))/dr,3)/dr/(1-pow(1-(rc_inner-sqrt(r1))/dr,4));
    }
  }
}

//Called by do3bodyfeatureset. Algorithm for low neighbor numbers and large series of bond angle powers
void Fingerprint_torsion::compute_fingerprint(double * features,double * dfeaturesx,double *dfeaturesy,double *dfeaturesz,int ii, int sid,double *xn,double *yn,double*zn,int *tn,int jnum,int * jl) {
  int i,jj,itype,jtype,ktype,ltype,kk,m,n,ll,k1;
  int *ilist;
  int jid,lnum;
  double delx,dely,delz,rsq;
  //atomtypes order: self, 2nd atom, 1st atom's neighbor, 2nd atom's neighbor
  int jtypes = atomtypes[1];
  int ktypes = atomtypes[2];
  int ltypes = atomtypes[3];
  int count=0;
  PairRANN::Simulation *sim = &pair->sims[sid];
  int *type = sim->type;
  int nelements = pair->nelements;
  int res = pair->res;
  double cutmax = pair->cutmax;
  double cutinv2 = 1/cutmax/cutmax;
  ilist = sim->ilist;
  i = ilist[ii];
  itype = pair->map[type[i]];
  int f = pair->fingerprintlength[itype];
  double expr[jnum][kmax];
  double y[jnum][3];
  double ri[jnum];
  double dfc[jnum];
  int kb = kmax;
  int mb = mlength;
  double c41[kmax];
  double c51[kmax];
  double c61[kmax];
  double c42[kmax];
  double c52[kmax];
  double c62[kmax];
  double c43[kmax];
  double c53[kmax];
  double c63[kmax];
  double c44[kmax];
  double c54[kmax];
  double c64[kmax];
  double ct[kmax];
  for (n = 0;n<kmax;n++) {
    ct[n] = alpha_k[n]/re;
  }
  //build tables for 2nd atom and 1st atom's neighbor
  for (kk = 0; kk < jnum; kk++) {
    ktype = tn[kk];
    if (ktypes != nelements && ktypes != ktype && jtypes != nelements && ktype != jtypes) {
      expr[kk][0]=0;
      continue;
    }
    delx = xn[kk];
    dely = yn[kk];
    delz = zn[kk];
    rsq = delx*delx + dely*dely + delz*delz;
    if (rsq>rc_inner*rc_inner) {
      expr[kk][0]=0;
      continue;
    }
    double r1 = (rsq*((double)res)*cutinv2);
    int m1 = (int)r1;
    if (!(m1>=1 && m1 <= res))pair->errorf(FLERR,"Neighbor list is invalid.");//usually results from nan somewhere.
    r1 = r1-trunc(r1);
    double *p0 = &expcuttable[(m1-1)*kmax];
    double *p1 = &expcuttable[m1*kmax];
    double *p2 = &expcuttable[(m1+1)*kmax];
    double *p3 = &expcuttable[(m1+2)*kmax];
    for (k1=0;k1<kmax;k1++) {
      expr[kk][k1] = p1[k1]+0.5*r1*(p2[k1]-p0[k1]+r1*(2.0*p0[k1]-5.0*p1[k1]+4.0*p2[k1]-p3[k1]+r1*(3.0*(p1[k1]-p2[k1])+p3[k1]-p0[k1])));
    }
    double* q = &dfctable[m1-1];
    double* r2 = &rinvsqrttable[m1-1];
    dfc[kk] = q[1] + 0.5 * r1*(q[2] - q[0] + r1*(2.0*q[0] - 5.0*q[1] + 4.0*q[2] - q[3] + r1*(3.0*(q[1] - q[2]) + q[3] - q[0])));
    ri[kk] = r2[1] + 0.5 * r1*(r2[2] - r2[0] + r1*(2.0*r2[0] - 5.0*r2[1] + 4.0*r2[2] - r2[3] + r1*(3.0*(r2[1] - r2[2]) + r2[3] - r2[0])));
    y[kk][0]=delx*ri[kk];
    y[kk][1]=dely*ri[kk];
    y[kk][2]=delz*ri[kk];
  }
  //loop over 2nd atoms
  for (jj = 0; jj < jnum; jj++) {
    if (expr[jj][0]==0)continue;
    jtype = tn[jj];
    if (jtypes != nelements && jtypes != jtype) {
      continue;
    }
    for (n = 0;n<kmax;n++) {
      c41[n]=(-ct[n]+dfc[jj])*y[jj][0];
      c51[n]=(-ct[n]+dfc[jj])*y[jj][1];
      c61[n]=(-ct[n]+dfc[jj])*y[jj][2];
    }
    jid = jl[jj];
    lnum = sim->numneigh[jid];
    double xn2[lnum];
    double yn2[lnum];
    double zn2[lnum];
    int tn2[lnum];
    int jl2[lnum];
    //build neighbor arrays for 2nd atom
    pair->cull_neighbor_list(xn2,yn2,zn2,tn2,&lnum,jl2,jid,sid,rc_inner);
    lnum-=1;
    double expr2[lnum][kmax];
    double y2[lnum][3];
    double ri2[lnum];
    double dfc2[lnum];
    int lmap[lnum];
    //build tables for 2nd atom's neighbor
    for (ll = 0; ll < lnum; ll++) {
      ltype = tn2[ll];
      if (ltypes != ltype && ltypes != nelements) {
        expr2[ll][0]=0;
        continue;
      }
      if (ilist[jl2[ll]]==i){
        expr2[ll][0]=0;
        continue;
      }
      delx = xn2[ll];
      dely = yn2[ll];
      delz = zn2[ll];
      rsq = delx*delx + dely*dely + delz*delz;
      if (rsq>rc_inner*rc_inner) {
          expr2[ll][0]=0;
          continue;
      }
      double r1 = (rsq*((double)res)*cutinv2);
      int m1 = (int)r1;
      if (!(m1>=1 && m1 <= res))pair->errorf(FLERR,"Neighbor list is invalid.");//usually results from nan somewhere.
      r1 = r1-trunc(r1);
      double *p0 = &expcuttable[(m1-1)*kmax];
      double *p1 = &expcuttable[m1*kmax];
      double *p2 = &expcuttable[(m1+1)*kmax];
      double *p3 = &expcuttable[(m1+2)*kmax];
      for (k1=0;k1<kmax;k1++) {
          expr2[ll][k1] = p1[k1]+0.5*r1*(p2[k1]-p0[k1]+r1*(2.0*p0[k1]-5.0*p1[k1]+4.0*p2[k1]-p3[k1]+r1*(3.0*(p1[k1]-p2[k1])+p3[k1]-p0[k1])));
      }
      double* q = &dfctable[m1-1];
      double* r2 = &rinvsqrttable[m1-1];
      dfc2[ll] = q[1] + 0.5 * r1*(q[2] - q[0] + r1*(2.0*q[0] - 5.0*q[1] + 4.0*q[2] - q[3] + r1*(3.0*(q[1] - q[2]) + q[3] - q[0])));
      ri2[ll] = r2[1] + 0.5 * r1*(r2[2] - r2[0] + r1*(2.0*r2[0] - 5.0*r2[1] + 4.0*r2[2] - r2[3] + r1*(3.0*(r2[1] - r2[2]) + r2[3] - r2[0])));
      y2[ll][0]=delx*ri2[ll];
      y2[ll][1]=dely*ri2[ll];
      y2[ll][2]=delz*ri2[ll];
      lmap[ll]=-1;
      for (int j2=0;j2<jnum;j2++){
        //if (ilist[jl2[ll]]==ilist[jl[j2]]){
        if (jl2[ll]==jl[j2]){
          lmap[ll]=j2;
        }
      }
      if (lmap[ll]==-1){
        for (int j2=0;j2<jnum;j2++){
          printf("%d %d %d %d %d %d %d\n",jl2[ll],jl[j2],i,jj,ll,jid,jl[jj]);
          //printf("%f %f %f %f %f %f %f %f %f %f %f %f\n",sim->x[jl2[ll]][0],sim->x[jl2[ll]][1],sim->x[jl2[ll]][2],sim->x[jl[j2]][0],sim->x[jl[j2]][1],sim->x[jl[j2]][2],sim->x[ii][0],sim->x[ii][1],sim->x[ii][2],sim->x[jid][0],sim->x[jid][1],sim->x[jid][2]);
        }
        printf("%.15f\n",expr[jj][0]);
        delx = xn[jj];
        dely = yn[jj];
        delz = zn[jj];
        rsq = delx*delx + dely*dely + delz*delz;
        printf("%.15f\n",sqrt(rsq));
        pair->errorf(FLERR,"couldn't map neighbor. rc_inner cannot be > rc/2.");
      }
    }
    //loop over 1st atom's neighbors
    for (kk=0;kk< jnum; kk++) {
      if (expr[kk][0]==0)continue;
      if (kk==jj)continue;
      ktype = tn[kk];
      if (ktypes != nelements && ktypes != ktype) {
        continue;
      }
      for (n = 0;n<kmax;n++) {
        double t = -ct[n]+dfc[kk];
        c42[n]=t*y[kk][0];
        c52[n]=t*y[kk][1];
        c62[n]=t*y[kk][2];
      }
      double ybg[3];
      ybg[0] = y[jj][1]*y[kk][2]-y[jj][2]*y[kk][1];
      ybg[1] = y[jj][2]*y[kk][0]-y[jj][0]*y[kk][2];
      ybg[2] = y[jj][0]*y[kk][1]-y[jj][1]*y[kk][0];
      //loop over 2nd atom's neighbors
      for (ll=0;ll<lnum;ll++) {
        int l2 = lmap[ll];
        if (ilist[jl2[ll]]==i || ilist[jl2[ll]]==ilist[jl[kk]])continue;
        if (expr2[ll][0]==0)continue;
        ltype = tn2[ll];
        if (ltypes != nelements && ltypes != ltype) {
          continue;
        }
        double ygd[3];
        double ybd[3];
        double dot;
        double ddb[3];
        double ddg[3];
        double ddd[3];
        ybd[0] = y[jj][1]*y2[ll][2]-y[jj][2]*y2[ll][1];
        ybd[1] = y[jj][2]*y2[ll][0]-y[jj][0]*y2[ll][2];
        ybd[2] = y[jj][0]*y2[ll][1]-y[jj][1]*y2[ll][0];
        dot = ybg[0]*ybd[0]+ybg[1]*ybd[1]+ybg[2]*ybd[2];
        ddb[0] = (ybg[1]*-y2[ll][2]+ybd[1]*-y[kk][2]+ybg[2]*y2[ll][1]+ybd[2]*y[kk][1]-2*dot*y[jj][0])*ri[jj];
        ddb[1] = (ybg[0]*y2[ll][2]+ybd[0]*y[kk][2]+ybg[2]*-y2[ll][0]+ybd[2]*-y[kk][0]-2*dot*y[jj][1])*ri[jj];
        ddb[2] = (ybg[0]*-y2[ll][1]+ybd[0]*-y[kk][1]+ybg[1]*y2[ll][0]+ybd[1]*y[kk][0]-2*dot*y[jj][2])*ri[jj];
        ddg[0] = (ybd[1]*y[jj][2]+ybd[2]*-y[jj][1]-dot*y[kk][0])*ri[kk];
        ddg[1] = (ybd[0]*-y[jj][2]+ybd[2]*y[jj][0]-dot*y[kk][1])*ri[kk];
        ddg[2] = (ybd[0]*y[jj][1]+ybd[1]*-y[jj][0]-dot*y[kk][2])*ri[kk];
        ddd[0] = (ybg[1]*y[jj][2]+ybg[2]*-y[jj][1]-dot*y2[ll][0])*ri2[ll];
        ddd[1] = (ybg[0]*-y[jj][2]+ybg[2]*y[jj][0]-dot*y2[ll][1])*ri2[ll];
        ddd[2] = (ybg[0]*y[jj][1]+ybg[1]*-y[jj][0]-dot*y2[ll][2])*ri2[ll];
        ddb[0] -= (ybg[1]*y[jj][2]+ybg[2]*-y[jj][1]-dot*y2[ll][0])*ri2[ll];
        ddb[1] -= (ybg[0]*-y[jj][2]+ybg[2]*y[jj][0]-dot*y2[ll][1])*ri2[ll];
        ddb[2] -= (ybg[0]*y[jj][1]+ybg[1]*-y[jj][0]-dot*y2[ll][2])*ri2[ll];
        double dot1;
        double dot3;
        for (n = 0;n<kmax;n++) {
          double t = -ct[n]+dfc2[ll];
          c43[n]=t*y2[ll][0];
          c53[n]=t*y2[ll][1];
          c63[n]=t*y2[ll][2];
          c44[n]=c41[n]-c43[n];
          c54[n]=c51[n]-c53[n];
          c64[n]=c61[n]-c63[n];
        }
        count = startingneuron;
        for (n=0;n<kb;n++) {
          //m=0
          dot1 = dot;
          dot3 = 1;
          double ex = expr[jj][n]*expr[kk][n]*expr2[ll][n];
          features[count]+=ex;
          dfeaturesx[jj*f+count]+=ex*c44[n];
          dfeaturesy[jj*f+count]+=ex*c54[n];
          dfeaturesz[jj*f+count]+=ex*c64[n];
          dfeaturesx[kk*f+count]+=ex*c42[n];
          dfeaturesy[kk*f+count]+=ex*c52[n];
          dfeaturesz[kk*f+count]+=ex*c62[n];
          dfeaturesx[l2*f+count]+=ex*c43[n];
          dfeaturesy[l2*f+count]+=ex*c53[n];
          dfeaturesz[l2*f+count]+=ex*c63[n];
          count++;
          for (m=1;m<mb;m++) {
            dfeaturesx[jj*f+count]+=ex*(m*ddb[0]*dot3+c44[n]*dot1);
            dfeaturesy[jj*f+count]+=ex*(m*ddb[1]*dot3+c54[n]*dot1);
            dfeaturesz[jj*f+count]+=ex*(m*ddb[2]*dot3+c64[n]*dot1);
            dfeaturesx[kk*f+count]+=ex*(m*ddg[0]*dot3+c42[n]*dot1);
            dfeaturesy[kk*f+count]+=ex*(m*ddg[1]*dot3+c52[n]*dot1);
            dfeaturesz[kk*f+count]+=ex*(m*ddg[2]*dot3+c62[n]*dot1);
            dfeaturesx[l2*f+count]+=ex*(m*ddd[0]*dot3+c43[n]*dot1);
            dfeaturesy[l2*f+count]+=ex*(m*ddd[1]*dot3+c53[n]*dot1);
            dfeaturesz[l2*f+count]+=ex*(m*ddd[2]*dot3+c63[n]*dot1);
            features[count]+=ex*dot1;
            count++;
            dot3 = dot1;
            dot1*=dot;
          }
        }
      }
    }
  }
  for (jj=0;jj<jnum;jj++) {
    count = startingneuron;
    for (n=0;n<kb;n++) {
      for (m=0;m<mb;m++) {
        dfeaturesx[jnum*f+count]-=dfeaturesx[jj*f+count];
        dfeaturesy[jnum*f+count]-=dfeaturesy[jj*f+count];
        dfeaturesz[jnum*f+count]-=dfeaturesz[jj*f+count];
        count++;
      }
    }
  }
}


