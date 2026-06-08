/* ----------------------------------------------------------------------
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
#include "omp.h"
#include "rann_net_exchangespin.h"
#include "rann_activation.h"
#include "pair_spin_rann.h"
#include "rann_fingerprint.h"

#include <cmath>

using namespace LAMMPS_NS::RANN;

Net_exchangespin::Net_exchangespin(PairRANN *_pair) : Net(_pair)
{
  empty = true;
  fullydefined = false;
  pair = _pair;
  n_body_type = 1;
  layers = 0;
  style = "exchangespin";
  atomtypes = new int[n_body_type];
  order = 1;
  fctable = nullptr;
  dfctable = nullptr;
  rc = 0;
  dr = 0;
  Eexchange=nullptr;
}

bool Net_exchangespin::parse_values(std::vector<std::string> line,std::vector<std::string> line1,FILE *fp,char *filename,int *linenum) {
  int nwords = line.size();
  std::string constant = line[nwords-1];
  if (constant.compare("layersize")==0) {
    parse_layer_size(line,line1,filename,*linenum);
  }
  else if (constant.compare("weight")==0) {
    parse_weight(line,line1,fp,filename,linenum);
  }
  else if (constant.compare("bias")==0) {
    parse_bias(line,line1,fp,filename,linenum);
  }
  else if (constant.compare("activation")==0) {
    parse_activation_functions(line,line1,filename,*linenum);
  }
  else if (constant.compare("fingerprintmap")==0){
    parse_fingerprint_map(line,line1,filename,*linenum);
  }
  else if (constant.compare("order")==0){
    order= pair->inumeric(filename,*linenum,line1[0]);
  }
  else if (constant.compare("dr")==0){
    dr= pair->numeric(filename,*linenum,line1[0]);
  }
  else if (constant.compare("rc")==0){
    rc= pair->numeric(filename,*linenum,line1[0]);
  }
  else pair->errorf(filename,*linenum,"Undefined value for net constants");
  
  return check_net_completeness();
}

bool Net_exchangespin::check_net_completeness(){
    if (dr==0.0)return false;
    if (rc==0.0)return false;
    return this->LAMMPS_NS::RANN::Net::check_net_completeness();
}

void Net_exchangespin::allocate(){
  int k;
  double r1;
  int buf = 5;
  int res = pair->res;
  double cutmax = pair->cutmax;
  fctable = new double[res+buf];
  dfctable = new double[res+buf];
  for (k=0;k<(res+buf);k++) {
    r1 = cutmax*cutmax*(double)(k)/(double)(res);
    if (sqrt(r1)>=rc) {
        dfctable[k]=0;
        fctable[k]=0;
    }
    else if (sqrt(r1) <= (rc-dr)){
        dfctable[k]=0;
        fctable[k]=1; 
    }
    else{
        double out = 1 - (rc - sqrt(r1)) / dr;
        out *= out;
        out *= out;
        out = 1 - out;
        out *= out;
        fctable[k]=out;
        dfctable[k]=-4*pow(1-(rc-sqrt(r1))/dr,3)/dr*(1-pow(1-(rc-sqrt(r1))/dr,4))/sqrt(r1);
    }
  }
}

void Net_exchangespin::jacobian_convolution(double *J,double *target,int *s,int sn,int natoms){
	//clock_t start = clock();
	double start_time = omp_get_wtime();
	#pragma omp parallel
	{
//	char str[MAXLINE];
	int nn,ii,n1,j;
	int count4 = 0;
	int aoff,soff;
	int n1dimi, n1sl, n1slM1, n4s, lM1, pIPk2, pLPk2;
	int sPcPiiX3, p1dlxyz, p2dlxyz, jPstartI, jjXfPk, iiX3, j1X3, p1dXw, p2dXw, p1ddXw, p2ddXw, i2n1W;
    double cutinv2 = 1/pair->cutmax/pair->cutmax;
    //printf("%d %d\n",start_index_sf,betalen);
	#pragma omp for schedule(guided)
	for (n1=0;n1<sn;n1++){
		nn = s[n1];
		n4s = pair->sims[nn].inum;
		double energy=0.0;
        for (ii=start_index_sf;ii<betalen+start_index_sf;ii++){
            J[n1*pair->betalen+ii]=0.0;
        }
		for (ii=0;ii<n4s;ii++){
			int itype,numneigh,jnum,**firstneigh,*jlist,i,j,k,j1,jj,starti,prevI,l,startL,prevL,k1,k2,k3;
			starti=0;
			itype = pair->sims[nn].type[ii];
            if (itype!=atomtypes[0])continue;
			double t2weight = pair->sims[nn].energy_weight/sqrt(pair->sims[nn].speciescount[itype]);
			int stype = pair->sims[nn].speciesmap[itype];
			int fex = pair->exchangelength[itype];
			//for (int k1 = 0;k1<pair->netperelement[itype];k1++){
            n1slM1 = sumlayers-1;
            iiX3 = ii*3;
            sPcPiiX3 = sn+count4+iiX3;
            // numneigh = pair->sims[nn].numneigh[ii];
            // jnum = numneigh;//extra value on the end of the array is the self term.
            jnum = pair->sims[nn].jnum[ii];
            int L = layers-1;
            double layer[sumlayers];
            double dlayer[sumlayers];
            double dlayerx[jnum*sumlayers];
            double dlayery[jnum*sumlayers];
            double dlayerz[jnum*sumlayers];

            int f = dimensions[0];
            n1dimi = dimensions[0];
            for (jj=0;jj<jnum;jj++){
                prevI = 0;
                double *si = pair->sims[nn].s[ii];
                int j1 = pair->sims[nn].jl[ii][jj];
                double *sj = pair->sims[nn].s[j1];
                double sp = si[0]*sj[0]+si[1]*sj[1]+si[2]*sj[2];
                sp = pow(sp,order);
                double xn = pair->sims[nn].xn[ii][jj];
                double yn = pair->sims[nn].yn[ii][jj];
                double zn = pair->sims[nn].zn[ii][jj];
                double rsq = xn*xn+yn*yn+zn*zn;
                if (rsq > rc*rc)continue;
                double r1 = (rsq*((double)pair->res)*cutinv2);
                int m1 = (int)r1;
                r1 = r1-trunc(r1);
                if (m1>pair->res || m1<1) {pair->errorf(FLERR,"invalid neighbor radius!");}
                if (fctable[m1]==0) {continue;}
            //cubic interpolation from tables
                double *q = &dfctable[m1-1];
                double *p = &fctable[m1-1];
                double dfc = q[1] + 0.5 * r1*(q[2] - q[0] + r1*(2.0*q[0] - 5.0*q[1] + 4.0*q[2] - q[3] + r1*(3.0*(q[1] - q[2]) + q[3] - q[0])));
                double fc =  p[1] + 0.5 * r1*(p[2] - p[0] + r1*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + r1*(3.0*(p[1] - p[2]) + p[3] - p[0])));
                for (k=0;k<n1dimi;k++){
                    //layer[k]=features[k];
                    dlayer[k] = 1.0;
                }
                int count1 = 0;
                for (k=0;k<map_length;k++){
                    RANN::Fingerprint *fingerprint = fingerprint_map[k];
                    int s = startingneuron[k];
                    //printf("%s %d\n",fingerprint->style,s);
                    if (fingerprint->exchange){
                        for (int l=0;l<fingerprint->get_length();l++){
                            layer[count1] = pair->sims[nn].exfeatures[ii][jj*fex+l+s];
                            if (pair->normalizeinput){
                                layer[count1]=(layer[count1]-normalshift[count1])/normalgain[count1];
                            }
                            count1++;
                        }
                    }
                    else {
                        for (int l=0;l<fingerprint->get_length();l++){
                            layer[count1] = pair->sims[nn].features[ii][l+s];
                            if (pair->normalizeinput){
                                layer[count1]=(layer[count1]-normalshift[count1])/normalgain[count1];
                            }
                            count1++;
                        }
                    }
                }
                for (k=dimensions[0];k<sumlayers;k++){layer[k]=0;}
                for (i=0;i<L;i++){
                    for (j=0;j<dimensions[i+1];j++){
                        starti = startI[i+1];
                        jPstartI = j+starti;
                        for (k=0;k<dimensions[i];k++){
                            layer[jPstartI] += Weight[i][j*dimensions[i]+k]*layer[k+prevI];
                        } 
                        layer[jPstartI] += Bias[i][j];
                    }
                    for (j=0;j<dimensions[i+1];j++){
                        starti = startI[i+1];
                        jPstartI = j+starti;
                        dlayer[jPstartI] = activations[i][j]->dactivation_function(layer[jPstartI]);
                        layer[jPstartI] =  activations[i][j]-> activation_function(layer[jPstartI]);
                        if (i==L-1){
                            energy += layer[jPstartI]*sp*fc;
                        }
                    }
                    prevI = starti;
                }
                prevI=0;
                int count2=start_index_sf;//skip frozen parameters
                int count3=start_index;//include frozen parameters
                //backpropagation
                for (i=0;i<L;i++){
                    int d1 = dimensions[i+1];
                    int d2 = dimensions[i];
                    double dXw[d1*sumlayers];
                    starti = startI[i+1];
                    prevI = startI[i];
                    for (k1=0;k1<sumlayers;k1++){
                        for (k2=0;k2<d1;k2++){
                            dXw[k1*d1+k2]=0.0;
                            if (k1==k2+starti){
                                dXw[k1*d1+k2]=1.0;
                            }
                        }
                    }
                    for (l=i+1;l<layers;l++){
                        int d3 = dimensions[l];
                        int d4 = dimensions[l-1];
                        int startL = startI[l];
                        int prevL = startI[l-1];
                        for (k1=0;k1<d3;k1++){
                            for (k2=0;k2<d4;k2++){
                                for (k3=0;k3<d1;k3++){
                                    //dlayer_l/dlayer_i
                                    dXw[(k1+startL)*d1+k3]+=dlayer[k2+prevL]*Weight[l-1][k1*d4+k2]*dXw[(k2+prevL)*d1+k3];
                                }
                            }
                        }
                    }
                    for (k1=0;k1<d1;k1++){
                        //weights
                        for (k2=0;k2<d2;k2++){
                            if (~freezebeta[count3]){
                                J[n1*pair->betalen+count2] += -dXw[(sumlayers-1)*d1+k1]*layer[k2+prevI]*pair->sims[nn].energy_weight*sp*fc;
                                count2++;
                            }
                            count3++;
                        }
                        //bias
                        if (~freezebeta[count3]){
                            J[n1*pair->betalen+count2] += -dXw[(sumlayers-1)*d1+k1]*pair->sims[nn].energy_weight*sp*fc;
                            count2++;
                        }
                        count3++;
                    }
                }
            }
		}
        target[n1] += energy*pair->sims[nn].energy_weight;
	}
	//regularizer
	if (pair->doregularizer & sn == pair->nsimr){
		int count2 = start_index_sf;
		int count3 = start_index;
		int snoff = sn;
		// #pragma omp for schedule(dynamic)
        for (int i=0;i<layers-1;i++){
            for (int k1=0;k1<dimensions[i+1];k1++){
                for (int k2=0;k2<dimensions[i];k2++){
                    if (~freezebeta[count3]){
                        J[(snoff+count2)*pair->betalen+count2] = pair->regularizer;
                        target[snoff+count2] = -pair->regularizer*Weight[i][k1*dimensions[i]+k2];
                        count2++;
                    }
                    count3++;
                }
                if (~freezebeta[count3]){
                    J[(sn+count2)*pair->betalen+count2] = pair->regularizer;
                    target[sn+count2] = -pair->regularizer*Bias[i][k1];
                    //force last bias to not count toward regularization
                    if (i+2==layers){
                        J[(snoff+count2)*pair->betalen+count2]=0;
                        target[(snoff+count2)*pair->betalen+count2]=0;
                    }
                    count2++;
                }
                count3++;
            }
        }
	}
    }

//	clock_t end = clock();
//	double time = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
	double time = (double) (omp_get_wtime() - start_time)*1000.0;
	// printf(" - compute_jacobian(): %f ms\n",time);

}

void Net_exchangespin::forward_pass(double *target,int *s,int sn){

	//clock_t start = clock();
	double start_time = omp_get_wtime();

	#pragma omp parallel
	{
	int nn,ii,n1,j;
	int countatoms = 0;
	int jPstartI, jjXfPk, n1slM1, p1dlxyz, p2dlxyz, sPcPiiX3, n4s, iiX3, j1X3;
	int count4 = 0;
    double cutinv2 = 1/pair->cutmax/pair->cutmax;
	#pragma omp for schedule(guided)
	for (n1=0;n1<sn;n1++){
		nn = s[n1];
		n4s = pair->sims[nn].inum;
		double energy=0.0;
		for (ii=0;ii<n4s;ii++){
			int itype,numneigh,jnum,**firstneigh,*jlist,i,j,k,j1,jj,starti,prevI;
			starti=0;
			itype = pair->sims[nn].type[ii];
			int fex = pair->exchangelength[itype];

            n1slM1 = sumlayers-1;
            // numneigh = pair->sims[nn].numneigh[ii];
            // jnum = numneigh;//extra value on the end of the array is the self term.
            jnum = pair->sims[nn].jnum[ii];
            int L = layers-1;
            double layer[sumlayers];
            for (jj=0;jj<jnum;jj++){
                prevI = 0;
                double *si = pair->sims[nn].s[ii];
                int j1 = pair->sims[nn].jl[ii][jj];
                double *sj = pair->sims[nn].s[j1];
                double sp = si[0]*sj[0]+si[1]*sj[1]+si[2]*sj[2];
                sp = pow(sp,order);
                double xn = pair->sims[nn].xn[ii][jj];
                double yn = pair->sims[nn].yn[ii][jj];
                double zn = pair->sims[nn].zn[ii][jj];
                double rsq = xn*xn+yn*yn+zn*zn;
                if (rsq > rc*rc)continue;
                double r1 = (rsq*((double)pair->res)*cutinv2);
                int m1 = (int)r1;
                r1 = r1-trunc(r1);
                if (m1>pair->res || m1<1) {pair->errorf(FLERR,"invalid neighbor radius!");}
                if (fctable[m1]==0) {continue;}
                //cubic interpolation from tables
                double *q = &dfctable[m1-1];
                double *p = &fctable[m1-1];
                double dfc = q[1] + 0.5 * r1*(q[2] - q[0] + r1*(2.0*q[0] - 5.0*q[1] + 4.0*q[2] - q[3] + r1*(3.0*(q[1] - q[2]) + q[3] - q[0])));
                double fc = p[1] + 0.5 * r1*(p[2] - p[0] + r1*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + r1*(3.0*(p[1] - p[2]) + p[3] - p[0])));
                int count1 = 0;
                for (k=0;k<map_length;k++){
                    RANN::Fingerprint *fingerprint = fingerprint_map[k];
                    int s = startingneuron[k];
                    //printf("%s %d\n",fingerprint->style,s);
                    if (fingerprint->exchange){
                        for (int l=0;l<fingerprint->get_length();l++){
                            layer[count1] = pair->sims[nn].exfeatures[ii][jj*fex+l+s];
                            if (pair->normalizeinput){
                                layer[count1]=(layer[count1]-normalshift[count1])/normalgain[count1];
                            }
                            count1++;
                        }
                    }
                    else {
                        for (int l=0;l<fingerprint->get_length();l++){
                            layer[count1] = pair->sims[nn].features[ii][l+s];
                            if (pair->normalizeinput){
                                layer[count1]=(layer[count1]-normalshift[count1])/normalgain[count1];
                            }
                            count1++;
                        }
                    }
                }
                for (k=dimensions[0];k<sumlayers;k++){layer[k]=0;}
                for (i=0;i<L;i++){
                    for (j=0;j<dimensions[i+1];j++){
                        starti = startI[i+1];
                        jPstartI = j+starti;
                        for (k=0;k<dimensions[i];k++){
                            layer[jPstartI] += Weight[i][j*dimensions[i]+k]*layer[k+prevI];
                        }
                        layer[jPstartI] += Bias[i][j];
                    }	
                    for (j=0;j<dimensions[i+1];j++){
                        starti = startI[i+1];
                        jPstartI = j+starti;
                        layer[jPstartI] =  activations[i][j]-> activation_function(layer[jPstartI]);
                        if (i==L-1){
                            energy += layer[jPstartI]*sp*fc;
                        }
                    }
                    prevI=starti;
                }
                prevI=0;
            }
			
		}
		//fill error vector
        target[n1] += energy*pair->sims[nn].energy_weight;
	}
	if (pair->doregularizer){
		int count2 = start_index_sf;
		int count3 = start_index;
		int snoff = sn;

		// #pragma omp for schedule(dynamic)

        for (int i=0;i<layers-1;i++){
            for (int k1=0;k1<dimensions[i+1];k1++){
                for (int k2=0;k2<dimensions[i];k2++){
                    if (~freezebeta[count3]){
                        target[snoff+count2] = -pair->regularizer*Weight[i][k1*dimensions[i]+k2];
                        count2++;
                    }
                    count3++;
                }
                if (~freezebeta[count3]){
                    target[sn+count2] = -pair->regularizer*Bias[i][k1];
                    //force last bias to not count toward regularization
                    if (i+2==layers){
                        target[(snoff+count2)*pair->betalen+count2]=0;
                    }
                    count2++;
                }
                count3++;
            }
        }

	}
	}
	double time = (double) (omp_get_wtime() - start_time)*1000.0;
}

void Net_exchangespin::get_per_atom_energy(double **energies,int *s,int sn){
	double start_time = omp_get_wtime();
    double cutinv2 = 1/pair->cutmax/pair->cutmax;
	#pragma omp parallel
	{
	int nn,ii,n1;
	int jPstartI, jjXfPk, n1slM1, p1dlxyz, p2dlxyz, sPcPiiX3, n4s, iiX3, j1X3;
	int count4 = 0;
    
	#pragma omp for schedule(guided)
	for (n1=0;n1<sn;n1++){
		nn = s[n1];
		n4s = pair->sims[nn].inum;
		double energy;
		energy = 0.0;
		for (ii=0;ii<n4s;ii++){
			int itype,numneigh,jnum,**firstneigh,*jlist,i,j,k,j1,jj,starti,prevI;
			starti=0;
			itype = pair->sims[nn].type[ii];
            n1slM1 = sumlayers-1;
            // numneigh = pair->sims[nn].numneigh[ii];
            jnum = pair->sims[nn].jnum[ii];//extra value on the end of the array is the self term.
            // firstneigh = pair->sims[nn].firstneigh;
            // jlist = firstneigh[ii];
            int L = layers-1;
            double layer[sumlayers];
            int f = dimensions[0];
            // double *features = pair->sims[nn].features[ii];
            int fex = pair->exchangelength[itype];
            prevI = 0;
            int startJ = 0;
            for (jj=startJ;jj<jnum;jj++){
                prevI = 0;
                double *si = pair->sims[nn].s[ii];
                int j1 = pair->sims[nn].jl[ii][jj];
                double *sj = pair->sims[nn].s[j1];
                double sp = si[0]*sj[0]+si[1]*sj[1]+si[2]*sj[2];
                sp = pow(sp,order);
                double xn = pair->sims[nn].xn[ii][jj];
                double yn = pair->sims[nn].yn[ii][jj];
                double zn = pair->sims[nn].zn[ii][jj];
                double rsq = xn*xn+yn*yn+zn*zn;
                if (rsq > rc*rc)continue;
                double r1 = (rsq*((double)pair->res)*cutinv2);
                int m1 = (int)r1;
                r1 = r1-trunc(r1);
                if (m1>pair->res || m1<1) {pair->errorf(FLERR,"invalid neighbor radius!");}
                if (fctable[m1]==0) {continue;}
                //cubic interpolation from tables
                double *q = &dfctable[m1-1];
                double *p = &fctable[m1-1];
                double dfc = q[1] + 0.5 * r1*(q[2] - q[0] + r1*(2.0*q[0] - 5.0*q[1] + 4.0*q[2] - q[3] + r1*(3.0*(q[1] - q[2]) + q[3] - q[0])));
                double fc = p[1] + 0.5 * r1*(p[2] - p[0] + r1*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + r1*(3.0*(p[1] - p[2]) + p[3] - p[0])));
                int count1 = 0;
                for (k=0;k<map_length;k++){
                    RANN::Fingerprint *fingerprint = fingerprint_map[k];
                    int s = startingneuron[k];
                    //printf("%s %d\n",fingerprint->style,s);
                    if (fingerprint->exchange){
                        for (int l=0;l<fingerprint->get_length();l++){
                            layer[count1] = pair->sims[nn].exfeatures[ii][jj*fex+l+s];
                            if (pair->normalizeinput){
                                layer[count1]=(layer[count1]-normalshift[count1])/normalgain[count1];
                            }
                            count1++;
                        }
                    }
                    else {
                        for (int l=0;l<fingerprint->get_length();l++){
                            layer[count1] = pair->sims[nn].features[ii][l+s];
                            if (pair->normalizeinput){
                                layer[count1]=(layer[count1]-normalshift[count1])/normalgain[count1];
                            }
                            count1++;
                        }
                    }
                }
                // if (nn==3 && ii==0)for (k=0;k<dimensions[0];k++)printf("get atom %d %d %f\n",k, jj, layer[k]);
                for (k=dimensions[0];k<sumlayers;k++){layer[k]=0;}
                for (i=0;i<L;i++){
                    // for (int i1 = 0;i1<bundles[i];i1++){
                    //     int s1 = bundleoutputsize[i][i1];
                    //     int s2 = bundleinputsize[i][i1];
                        for (j=0;j<dimensions[i+1];j++){
                            starti = startI[i+1];
                            // int j1 = bundleoutput[i][i1][j];
                            jPstartI = j+starti;
                            for (k=0;k<dimensions[i];k++){
                                // int k1 = bundleinput[i][i1][k];
                                layer[jPstartI] += Weight[i][j*dimensions[i]+k]*layer[k+prevI];
                            } 
                            layer[jPstartI] += Bias[i][j];
                        }
                    // }
                    for (j=0;j<dimensions[i+1];j++){
                        starti = startI[i+1];
                        jPstartI = j+starti;
                        layer[jPstartI] =  activations[i][j]-> activation_function(layer[jPstartI]);
                        if (i==L-1){
                            energy += layer[jPstartI]*sp*fc;
                            energies[n1][ii]+=layer[jPstartI]*sp*fc;
                        }
                    }
                    prevI = starti;
                }
            }
			
		}
	}
	}
	double time = (double) (omp_get_wtime() - start_time)*1000.0;
}

void Net_exchangespin::propagateforward(double *energy,double **force,int ii,int jnum,int itype,double *features, double *dfeaturesx,double *dfeaturesy, double *dfeaturesz,double *exfeatures, double *dexfeaturesx,double *dexfeaturesy, double *dexfeaturesz,int *jl,int nn,double *xn, double *yn, double *zn) {
	if (Eexchange!=nullptr){
    	delete [] Eexchange;
  	}
    Eexchange = new double [jnum];  
    int i,j,k,l,jj,kk;
	int L = layers-1;
	//energy output with forces from analytical derivatives
	double dsum1[maxlayer];
	int f = pair->fingerprintlength[itype];
	int fex = pair->exchangelength[itype];
	double sum[maxlayer];
	double layer[maxlayer];
	double dlayersumx[maxlayer];
	double dlayersumy[maxlayer];
	double dlayersumz[maxlayer];
	double dlayerx[maxlayer];
	double dlayery[maxlayer];
	double dlayerz[maxlayer];
    double dotnet[maxlayer][dimensions[0]];
    double dotnetsum[maxlayer][dimensions[0]];
    double cutinv2 = 1/pair->cutmax/pair->cutmax;
	for (jj=0;jj<jnum-1;jj++){
        int count1 = 0;
        for (k=0;k<map_length;k++){
            RANN::Fingerprint *fingerprint = fingerprint_map[k];
            int s = startingneuron[k];
            if (fingerprint->exchange){
                for (int l=0;l<fingerprint->get_length();l++){
                    layer[count1] = exfeatures[jj*fex+l+s];
                    count1++;
                }
            }
            else {
                for (int l=0;l<fingerprint->get_length();l++){
                    layer[count1] = features[l+s];
                    count1++;
                }
            }
        }
    	i = pair->sims[nn].ilist[ii];
        double *si = pair->sims[nn].s[i];
        double *sj = pair->sims[nn].s[jl[jj]];
        double sp = si[0]*sj[0]+si[1]*sj[1]+si[2]*sj[2];
        sp = pow(sp,order);
        double xn1 = xn[jj];
        double yn1 = yn[jj];
        double zn1 = zn[jj];
        double rsq = xn1*xn1+yn1*yn1+zn1*zn1;
        if (rsq > rc*rc)continue;
        double r1 = (rsq*((double)pair->res)*cutinv2);
        int m1 = (int)r1;
        r1 = r1-trunc(r1);
        if (m1>pair->res || m1<1) {pair->errorf(FLERR,"invalid neighbor radius!");}
        if (fctable[m1]==0) {continue;}
        //cubic interpolation from tables
        double *q = &dfctable[m1-1];
        double *p = &fctable[m1-1];
        double dfc = q[1] + 0.5 * r1*(q[2] - q[0] + r1*(2.0*q[0] - 5.0*q[1] + 4.0*q[2] - q[3] + r1*(3.0*(q[1] - q[2]) + q[3] - q[0])));
        double fc = p[1] + 0.5 * r1*(p[2] - p[0] + r1*(2.0*p[0] - 5.0*p[1] + 4.0*p[2] - p[3] + r1*(3.0*(p[1] - p[2]) + p[3] - p[0])));
        for (j=0;j<dimensions[1];j++){
            for (k=0;k<dimensions[0];k++){
                dotnetsum[j][k]=Weight[0][j*dimensions[0]+k]*fc*sp;
            }
        }
        for (i=0;i<layers-1;i++) {
            for (j=0;j<dimensions[i+1];j++){
                sum[j]=0;
                for (k=0;k<dimensions[i];k++){
                    double w1 = Weight[i][j*dimensions[i]+k];
                    sum[j] += w1*layer[k];
                    if (i>0){
                        for (l=0;l<dimensions[0];l++){
                            dotnetsum[j][l]+=w1*dotnet[k][l];
                        }
                    }
                }
                sum[j]+= Bias[i][j];
            }
            for (j=0;j<dimensions[i+1];j++) {
                dsum1[j] = activations[i][j]->dactivation_function(sum[j]);
                sum[j] = activations[i][j]->activation_function(sum[j]);
                for (l=0;l<dimensions[0];l++){
                    dotnetsum[j][l]*=dsum1[j];
                }
            }
            //update values for next iteration
            if (i<layers-2){
                for (j=0;j<dimensions[i+1];j++) {
                    layer[j]=sum[j];
                    for (l=0;l<dimensions[0];l++){
                        dotnet[j][l]=dotnetsum[j][l];
                    }
                }
                for (j=0;j<dimensions[i+2];j++){
                    for (l=0;l<dimensions[0];l++){
                        dotnetsum[j][l]=0.0;
                    }
                }
            }
        }
        energy[0] += sum[0]*sp*fc;
        Eexchange[jj]=sum[0]*fc;
        count1 = 0;
        int j2 = jl[jj];
        for (k=0;k<map_length;k++){
            RANN::Fingerprint *fingerprint = fingerprint_map[k];
            int s = startingneuron[k];
            if (fingerprint->exchange && fingerprint->screen){
                for (int l=0;l<fingerprint->get_length();l++){
                    for (kk=0;kk<jnum;kk++){
                        dlayerx[count1] = dexfeaturesx[jj*fex*jnum+kk*fex+l+s];
                        dlayery[count1] = dexfeaturesy[jj*fex*jnum+kk*fex+l+s];
                        dlayerz[count1] = dexfeaturesz[jj*fex*jnum+kk*fex+l+s];
                        int k2 = jl[kk];
                        force[k2][0]+=dotnetsum[0][count1]*dlayerx[count1];
                        force[k2][1]+=dotnetsum[0][count1]*dlayery[count1];
                        force[k2][2]+=dotnetsum[0][count1]*dlayerz[count1];
                    }
                    count1++;
                }
            }
            else if (fingerprint->exchange && !fingerprint->screen){
                for (int l=0;l<fingerprint->get_length();l++){
                    dlayerx[count1] = dexfeaturesx[jj*fex*jnum+jj*fex+l+s];
                    dlayery[count1] = dexfeaturesy[jj*fex*jnum+jj*fex+l+s];
                    dlayerz[count1] = dexfeaturesz[jj*fex*jnum+jj*fex+l+s];
                    force[j2][0]+=dotnetsum[0][count1]*dlayerx[count1];
                    force[j2][1]+=dotnetsum[0][count1]*dlayery[count1];
                    force[j2][2]+=dotnetsum[0][count1]*dlayerz[count1];
                    force[ii][0]-=dotnetsum[0][count1]*dlayerx[count1];
                    force[ii][1]-=dotnetsum[0][count1]*dlayery[count1];
                    force[ii][2]-=dotnetsum[0][count1]*dlayerz[count1];
                    count1++;
                }
            }
            else {
                for (int l=0;l<fingerprint->get_length();l++){
                    for (kk=0;kk<jnum;kk++){
                        dlayerx[count1] = dfeaturesx[kk*f+l+s];
                        dlayery[count1] = dfeaturesy[kk*f+l+s];
                        dlayerz[count1] = dfeaturesz[kk*f+l+s];
                        int k2 = jl[kk];
                        force[k2][0]+=dotnetsum[0][count1]*dlayerx[count1];
                        force[k2][1]+=dotnetsum[0][count1]*dlayery[count1];
                        force[k2][2]+=dotnetsum[0][count1]*dlayerz[count1];
                    }
                    count1++;
                }
            }
            force[j2][0]+=sp*xn1*dfc*sum[0];
            force[j2][1]+=sp*yn1*dfc*sum[0];
            force[j2][2]+=sp*zn1*dfc*sum[0];
            force[ii][0]-=sp*xn1*dfc*sum[0];
            force[ii][1]-=sp*yn1*dfc*sum[0];
            force[ii][2]-=sp*zn1*dfc*sum[0];
        }
    }
}

void Net_exchangespin::updatespin(double *energy,double **fm,double**hm,int ii,int jnum,int itype,int *jl,int nn){
    double newenergy[jnum];
    double *si = pair->sims[nn].s[ii];
    double spm,sph;
    double hbar = pair->hbar;
    int i2 = pair->sims->ilist[ii];
    for (int jj = 0;jj<jnum;jj++){
        int j2 = jl[jj];
        double *sj = pair->sims[nn].s[j2];
        double sp = si[0]*sj[0]+si[1]*sj[1]+si[2]*sj[2];
        sp = pow(sp,order);
        spm = pow(sp,order-1);
        sph = pow(sp,order-2);
        newenergy[jj] = Eexchange[jj]*sp;
        energy[0]+=newenergy[jj];
        for (int k=0;k<3;k++){
            fm[j2][k]+=Eexchange[jj]*spm*si[k]*order/hbar;
            fm[i2][k]+=Eexchange[jj]*spm*sj[k]*order/hbar;
        }
        hm[j2][0]+=Eexchange[jj]*sph*si[0]*si[0]*order*(order-1)/hbar;
        hm[j2][1]+=Eexchange[jj]*sph*si[0]*si[1]*order*(order-1)/hbar;
        hm[j2][2]+=Eexchange[jj]*sph*si[0]*si[2]*order*(order-1)/hbar;
        hm[j2][3]+=Eexchange[jj]*sph*si[1]*si[1]*order*(order-1)/hbar;
        hm[j2][4]+=Eexchange[jj]*sph*si[1]*si[2]*order*(order-1)/hbar;
        hm[j2][5]+=Eexchange[jj]*sph*si[2]*si[2]*order*(order-1)/hbar;
        hm[i2][0]+=Eexchange[jj]*sph*sj[0]*sj[0]*order*(order-1)/hbar;
        hm[i2][1]+=Eexchange[jj]*sph*sj[0]*sj[1]*order*(order-1)/hbar;
        hm[i2][2]+=Eexchange[jj]*sph*sj[0]*sj[2]*order*(order-1)/hbar;
        hm[i2][3]+=Eexchange[jj]*sph*sj[1]*sj[1]*order*(order-1)/hbar;
        hm[i2][4]+=Eexchange[jj]*sph*sj[1]*sj[2]*order*(order-1)/hbar;
        hm[i2][5]+=Eexchange[jj]*sph*sj[2]*sj[2]*order*(order-1)/hbar;
    }
}

void Net_exchangespin::propagateforwardspin(double *energy,double **force,double **fm,double **hm,int ii,int jnum,int itype,double *features, double *dfeaturesx,double *dfeaturesy, double *dfeaturesz,double *exfeatures, double *dexfeaturesx,double *dexfeaturesy, double *dexfeaturesz, double *sx, double *sy, double *sz, double *sxx, double *sxy, double *sxz, double *syy, double *syz, double *szz,int *jl,int nn) {
  int i,j,k,jj,j1,i1;
  double hbar = pair->hbar;
  if (Eexchange!=nullptr){
    delete [] Eexchange;
    
    //exchangenergy=Eexchange;
  }
  Eexchange = new double [jnum];
	int L = layers-1;
	//energy output with forces from analytical derivatives
	double dsum1[maxlayer];
	double ddsum1[maxlayer];
	double sum[maxlayer];
	double layer[maxlayer];
	double dlayersumx[jnum][maxlayer];
	double dlayersumy[jnum][maxlayer];
	double dlayersumz[jnum][maxlayer];
	double dlayerx[jnum][maxlayer];
	double dlayery[jnum][maxlayer];
	double dlayerz[jnum][maxlayer];
	double dsx[jnum][maxlayer];
	double dsy[jnum][maxlayer];
	double dsz[jnum][maxlayer];
	double dsxx[jnum][maxlayer];
	double dsxy[jnum][maxlayer];
	double dsxz[jnum][maxlayer];
	double dsyy[jnum][maxlayer];
	double dsyz[jnum][maxlayer];
	double dszz[jnum][maxlayer];
	double dssumx[jnum][maxlayer];
	double dssumy[jnum][maxlayer];
	double dssumz[jnum][maxlayer];
	double dssumxx[jnum][maxlayer];
	double dssumxy[jnum][maxlayer];
	double dssumxz[jnum][maxlayer];
	double dssumyy[jnum][maxlayer];
	double dssumyz[jnum][maxlayer];
	double dssumzz[jnum][maxlayer];
	int fex = pair->exchangelength[itype];
	int f = pair->fingerprintlength[itype];
	//energy output with forces from analytical derivatives
	for (jj=0;jj<jnum;jj++){
		int count1 = 0;
		for (k=0;k<map_length;k++){
			RANN::Fingerprint *fingerprint = fingerprint_map[k];
            int s = startingneuron[k];
			if (fingerprint->exchange){
				for (int l=0;l<fingerprint->get_length();l++){
					layer[count1] = exfeatures[jj*fex+l+s];
					dlayerx[jj][count1] = dexfeaturesx[jj*fex+count1];
					dlayery[jj][count1] = dexfeaturesy[jj*fex+count1];
					dlayerz[jj][count1] = dexfeaturesz[jj*fex+count1];
					dsx[jj][count1]=-sx[jj*fex+count1];
					dsy[jj][count1]=-sy[jj*fex+count1];
					dsz[jj][count1]=-sz[jj*fex+count1];
					dsxx[jj][count1]=-sxx[jj*fex+count1];
					dsxy[jj][count1]=-sxy[jj*fex+count1];
					dsxz[jj][count1]=-sxz[jj*fex+count1];
					dsyy[jj][count1]=-syy[jj*fex+count1];
					dsyz[jj][count1]=-syz[jj*fex+count1];
					dszz[jj][count1]=-szz[jj*fex+count1];
					count1++;
				}
			}
			else {
				for (int l=0;l<fingerprint->get_length();l++){
					layer[count1] = features[l+s];
					dlayerx[jj][count1] = dfeaturesx[jj*f+count1];
					dlayery[jj][count1] = dfeaturesy[jj*f+count1];
					dlayerz[jj][count1] = dfeaturesz[jj*f+count1];
					dsx[jj][count1]=-sx[jj*fex+count1];
					dsy[jj][count1]=-sy[jj*fex+count1];
					dsz[jj][count1]=-sz[jj*fex+count1];
					dsxx[jj][count1]=-sxx[jj*fex+count1];
					dsxy[jj][count1]=-sxy[jj*fex+count1];
					dsxz[jj][count1]=-sxz[jj*fex+count1];
					dsyy[jj][count1]=-syy[jj*fex+count1];
					dsyz[jj][count1]=-syz[jj*fex+count1];
					dszz[jj][count1]=-szz[jj*fex+count1];
					count1++;
					count1++;
				}
			}
		}
	}
	
    for (jj=0;jj<jnum;jj++){
        for (k=0;k<dimensions[0];k++) {
            layer[k]=features[k];
            for (jj=0;jj<jnum;jj++) {
            dlayerx[jj][k]=dfeaturesx[jj*f+k];
            dlayery[jj][k]=dfeaturesy[jj*f+k];
            dlayerz[jj][k]=dfeaturesz[jj*f+k];
            dsx[jj][k]=-sx[jj*f+k];
            dsy[jj][k]=-sy[jj*f+k];
            dsz[jj][k]=-sz[jj*f+k];
            dsxx[jj][k]=-sxx[jj*f+k];
            dsxy[jj][k]=-sxy[jj*f+k];
            dsxz[jj][k]=-sxz[jj*f+k];
            dsyy[jj][k]=-syy[jj*f+k];
            dsyz[jj][k]=-syz[jj*f+k];
            dszz[jj][k]=-szz[jj*f+k];
            }
        }
        for (i=0;i<layers-1;i++) {
            for (j=0;j<dimensions[i+1];j++) {
            sum[j]=0;
            }
            // for (i1=0;i1<bundles[i];i1++){
            // int s1=bundleoutputsize[i][i1];
            // int s2=bundleinputsize[i][i1];
            for (j=0;j<dimensions[i+1];j++){
                // int j1 = bundleoutput[i][i1][j];
                for (k=0;k<dimensions[i];k++){
                // int k1 = bundleinput[i][i1][k];
                sum[j1] += Weight[i][j*dimensions[i]+k]*layer[k];
                }
                sum[j1]+= Bias[i][j];
            }
            // }
            for (j=0;j<dimensions[i+1];j++) {
            ddsum1[j] = activations[i][j]->ddactivation_function(sum[j]);
            dsum1[j] = activations[i][j]->dactivation_function(sum[j]);
            sum[j] = activations[i][j]->activation_function(sum[j]);
            if (i==L-1) {
                energy[j] = sum[j];
                Eexchange[jj]=sum[j];
            }
            //force propagation
            for (jj=0;jj<jnum;jj++) {
                dlayersumx[jj][j]=0;
                dlayersumy[jj][j]=0;
                dlayersumz[jj][j]=0;
                dssumx[jj][j]=0;
                dssumy[jj][j]=0;
                dssumz[jj][j]=0;
                dssumxx[jj][j]=0;
                dssumxy[jj][j]=0;
                dssumxz[jj][j]=0;
                dssumyy[jj][j]=0;
                dssumyz[jj][j]=0;
                dssumzz[jj][j]=0;
            }
            }
            // for (i1=0;i1<bundles[i];i1++){
            // int s1 = bundleoutputsize[i][i1];
            // int s2 = bundleinputsize[i][i1];
            for (j=0;j<dimensions[i+1];j++){
                // int j1 = bundleoutput[i][i1][j];
                for (jj=0;jj<jnum;jj++){
                for (k=0;k<dimensions[i];k++){
                    // int k1= bundleinput[i][i1][k];
                    double w1 = Weight[i][j*dimensions[i]+k];
                    dlayersumx[jj][j1] += w1*dlayerx[jj][k];
                    dlayersumy[jj][j1] += w1*dlayery[jj][k];
                    dlayersumz[jj][j1] += w1*dlayerz[jj][k];
                    dssumx[jj][j1] += w1*dsx[jj][k];
                    dssumy[jj][j1] += w1*dsy[jj][k];
                    dssumz[jj][j1] += w1*dsz[jj][k];
                    dssumxx[jj][j1] += w1*dsxx[jj][k];
                    dssumxy[jj][j1] += w1*dsxy[jj][k];
                    dssumxz[jj][j1] += w1*dsxz[jj][k];
                    dssumyy[jj][j1] += w1*dsyy[jj][k];
                    dssumyz[jj][j1] += w1*dsyz[jj][k];
                    dssumzz[jj][j1] += w1*dszz[jj][k];
                }
                }
            }
            // }
            for (j=0;j<dimensions[i+1];j++){
            for (jj=0;jj<jnum;jj++){
                dlayersumx[jj][j]*= dsum1[j];
                dlayersumy[jj][j]*= dsum1[j];
                dlayersumz[jj][j]*= dsum1[j];
                dssumxx[jj][j] = dssumxx[jj][j]*dsum1[j]+dssumx[jj][j]*dssumx[jj][j]*ddsum1[j];
                dssumxy[jj][j] = dssumxy[jj][j]*dsum1[j]+dssumx[jj][j]*dssumy[jj][j]*ddsum1[j];
                dssumxz[jj][j] = dssumxz[jj][j]*dsum1[j]+dssumx[jj][j]*dssumz[jj][j]*ddsum1[j];
                dssumyy[jj][j] = dssumyy[jj][j]*dsum1[j]+dssumy[jj][j]*dssumy[jj][j]*ddsum1[j];
                dssumyz[jj][j] = dssumyz[jj][j]*dsum1[j]+dssumy[jj][j]*dssumz[jj][j]*ddsum1[j];
                dssumzz[jj][j] = dssumzz[jj][j]*dsum1[j]+dssumz[jj][j]*dssumz[jj][j]*ddsum1[j];
                dssumx[jj][j] *= dsum1[j];
                dssumy[jj][j] *= dsum1[j];
                dssumz[jj][j] *= dsum1[j];
            }
            }
            if (i==L-1) {
            for (j=0;j<dimensions[i+1];j++){
                for (jj=0;jj<jnum-1;jj++){
                int j2 = jl[jj];
                force[j2][0]+=dlayersumx[jj][j];
                force[j2][1]+=dlayersumy[jj][j];
                force[j2][2]+=dlayersumz[jj][j];
                fm[j2][0]+=dssumx[jj][j]/hbar;
                fm[j2][1]+=dssumy[jj][j]/hbar;
                fm[j2][2]+=dssumz[jj][j]/hbar;
                hm[j2][0]+=dssumxx[jj][j]/hbar;
                hm[j2][1]+=dssumxy[jj][j]/hbar;
                hm[j2][2]+=dssumxz[jj][j]/hbar;
                hm[j2][3]+=dssumyy[jj][j]/hbar;
                hm[j2][4]+=dssumyz[jj][j]/hbar;
                hm[j2][5]+=dssumzz[jj][j]/hbar;
                }
                int j2 = pair->sims->ilist[ii];
                jj = jnum-1;
                force[j2][0]+=dlayersumx[jj][j];
                force[j2][1]+=dlayersumy[jj][j];
                force[j2][2]+=dlayersumz[jj][j];
                fm[j2][0]+=dssumx[jj][j]/hbar;
                fm[j2][1]+=dssumy[jj][j]/hbar;
                fm[j2][2]+=dssumz[jj][j]/hbar;
                hm[j2][0]+=dssumxx[jj][j]/hbar;
                hm[j2][1]+=dssumxy[jj][j]/hbar;
                hm[j2][2]+=dssumxz[jj][j]/hbar;
                hm[j2][3]+=dssumyy[jj][j]/hbar;
                hm[j2][4]+=dssumyz[jj][j]/hbar;
                hm[j2][5]+=dssumzz[jj][j]/hbar;
            }
            }
            //update values for next iteration
            for (j=0;j<dimensions[i+1];j++) {
            layer[j]=sum[j];
            for (jj=0;jj<jnum;jj++) {
                dlayerx[jj][j] = dlayersumx[jj][j];
                dlayery[jj][j] = dlayersumy[jj][j];
                dlayerz[jj][j] = dlayersumz[jj][j];
                dsx[jj][j] = dssumx[jj][j];
                dsy[jj][j] = dssumy[jj][j];
                dsz[jj][j] = dssumz[jj][j];
                dsxx[jj][j] = dssumxx[jj][j];
                dsxy[jj][j] = dssumxy[jj][j];
                dsxz[jj][j] = dssumxz[jj][j];
                dsyy[jj][j] = dssumyy[jj][j];
                dsyz[jj][j] = dssumyz[jj][j];
                dszz[jj][j] = dssumzz[jj][j];
            }
            }
        }
    }
}

void Net_exchangespin::write_values(FILE *fid){
  int i;
  if (layers==0)return;
    fprintf(fid,"netconstants:");
    fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
    for (i=1;i<n_body_type;i++) {
        fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
    }
    fprintf(fid,":%s_%d:layersize:\n",style,id);
  for (int j=0;j<layers;j++){

    fprintf(fid,"%d\t",dimensions[j]);
  }
  fprintf(fid,"\n");

    fprintf(fid,"netconstants:");
    fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
    for (i=1;i<n_body_type;i++) {
        fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
    }
    fprintf(fid,":%s_%d:order:\n",style,id);
    fprintf(fid,"%d\t",order);
  fprintf(fid,"\n");
      fprintf(fid,"netconstants:");
    fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
    for (i=1;i<n_body_type;i++) {
        fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
    }
    fprintf(fid,":%s_%d:rc:\n",style,id);
    fprintf(fid,"%f\t",rc);
  fprintf(fid,"\n");
    fprintf(fid,"netconstants:");
    fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
    for (i=1;i<n_body_type;i++) {
        fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
    }
    fprintf(fid,":%s_%d:dr:\n",style,id);
    fprintf(fid,"%f\t",dr);
  fprintf(fid,"\n");
fprintf(fid,"netconstants:");
fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
}
fprintf(fid,":%s_%d:activation:\n",style,id);
  for (int j=1;j<layers;j++){
    // for (int k=0;k<dimensions[j];k++){
        fprintf(fid,"%s\t",activations[j-1][0]->style);
    // }
  }
  fprintf(fid,"\n");

  fprintf(fid,"netconstants:");
  fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
  for (i=1;i<n_body_type;i++) {
    fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
  }
  fprintf(fid,":%s_%d:fingerprintmap:\n",style,id);
  for (int k=0;k<map_length;k++){
        fprintf(fid,"%s_%d ",fingerprint_map[k]->style,fingerprint_map[k]->id);
  }
  fprintf(fid,"\n");

  for (int j=0;j<layers-1;j++){
    fprintf(fid,"netconstants:");
    fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
    for (i=1;i<n_body_type;i++) {
        fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
    }
    fprintf(fid,":%s_%d_%d:weight:\n",style,id,j);
    for (int k=0;k<dimensions[j+1];k++){
        for (int l=0;l<dimensions[j];l++){
            fprintf(fid,"%10.8f\t",Weight[j][k*dimensions[j]+l]);
        }
        fprintf(fid,"\n");
    }
  }

  for (int j=0;j<layers-1;j++){
    fprintf(fid,"netconstants:");
    fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
    for (i=1;i<n_body_type;i++) {
        fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
    }
    fprintf(fid,":%s_%d_%d:bias:\n",style,id,j);
    for (int k=0;k<dimensions[j+1];k++){
        fprintf(fid,"%10.8f\t",Bias[j][k]);
        fprintf(fid,"\n");
    }
  }
}

void Net_exchangespin::copy_network(Net *net_new){
    this->LAMMPS_NS::RANN::Net::copy_network(net_new);
    ((Net_exchangespin*) net_new)->order = order;
    ((Net_exchangespin*) net_new)->rc = rc;
    ((Net_exchangespin*) net_new)->dr = dr;
    ((Net_exchangespin*) net_new)->fctable = fctable;
    ((Net_exchangespin*) net_new)->dfctable = dfctable;
}
