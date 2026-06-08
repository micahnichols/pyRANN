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
#include "rann_net.h"
#include "rann_activation.h"
#include "pair_spin_rann.h"
#include "rann_fingerprint.h"

#include <cmath>

using namespace LAMMPS_NS::RANN;

Net::Net(PairRANN *_pair)
{
  empty = true;
  fullydefined = false;
  pair = _pair;
  n_body_type = 0;
  layers = 0;
  style = "empty";
  betalen = 0;
  map_length=0;
}

void Net::allocate(){

}

Net::~Net(){
    if (layers>0){
        for (int j=0;j<layers-1;j++){
            // delete [] bundleinputsize[j];
            // delete [] bundleoutputsize[j];
            for (int k=0;k<dimensions[j+1];k++){
                delete activations[j][k];
            }
            // for (int k=0;k<bundles[j];k++){
            //     delete [] bundleinput[j][k];
            //     delete [] bundleoutput[j][k];
            //     delete [] bundleW[j][k];
            //     delete [] bundleB[j][k];
            //     delete [] freezeW[j][k];
            //     delete [] freezeB[j][k];
            // }
            delete [] activations[j];
            // delete [] bundleinput[j];
            // delete [] bundleoutput[j];
            delete [] Weight[j];
            delete [] Bias[j];
            delete [] freezeW[j];
            delete [] freezeB[j];
        }
        delete [] activations;
        delete [] dimensions;
        delete [] startI;
        // delete [] bundleinput;
        // delete [] bundleoutput;
        // delete [] bundleinputsize;
        // delete [] bundleoutputsize;
        delete [] Weight;
        delete [] Bias;
        delete [] freezeW;
        delete [] freezeB;
        // delete [] bundles;
    }
}

void Net::updatespin(double *energy,double **fm,double**hm,int ii,int jnum,int itype,int *jl,int nn){

}

//called after fingerprint is declared for i-j type, but before its parameters are read.
void Net::init(int *i,int _id)
{
  empty = false;
  for (int j=0;j<n_body_type;j++) {atomtypes[j] = i[j];}
  id = _id;
  int nelementsp = pair->nelementsp;
  weightdefined = new bool[nelementsp];
  biasdefined = new bool [nelementsp];
  dimensiondefined = new bool[nelementsp];
//   bundle_inputdefined = new bool*[nelementsp];
//   bundle_outputdefined = new bool*[nelementsp];
  activations = new RANN::Activation**[nelementsp];
}

void Net::flatten_beta(double *beta){
	int itype,i,k1,k2,count2;
	count2 = 0;
    for (i=0;i<layers-1;i++){
        // for (int i1=0;i1<bundles[i];i1++){
            // if (identitybundle[i][i1])continue;
            for (k1=0;k1<dimensions[i+1];k1++){
                for (k2=0;k2<dimensions[i];k2++){
                    if (freezeW[i][k1*dimensions[i]+k2])continue;
                    beta[count2]=Weight[i][k1*dimensions[i]+k2];
                    count2++;
                }
                if (freezeB[i][k1])continue;
                beta[count2]=Bias[i][k1];
                count2++;
            }
        // }
    }
}

void Net::unflatten_beta(double *beta){
	int itype,i,k1,k2,count2;
	count2 = 0;
    for (i=0;i<layers-1;i++){
        // for (int i1=0;i1<bundles[i];i1++){
            // if (identitybundle[i][i1])continue;
            for (k1=0;k1<dimensions[i+1];k1++){
                for (k2=0;k2<dimensions[i];k2++){
                    if (freezeW[i][k1*dimensions[i]+k2])continue;
                    Weight[i][k1*dimensions[i]+k2]=beta[count2];
                    count2++;
                }
                if (freezeB[i][k1])continue;
                Bias[i][k1]=beta[count2];
                count2++;
            }
        // }
    }
}

void Net::normalize_data(){
    int i,n,ii,j,itype,k,count1;
	int natoms;
	//initialize
		//if (net[i].layers==0)continue;
    normalgain = new double [dimensions[0]];
    normalshift = new double [dimensions[0]];
    for (j=0;j<dimensions[0];j++){
        normalgain[j]=0;
        normalshift[j]=0;
    }
    natoms = 0;
	//get mean value of each 1st layer neuron input
	for (n=0;n<pair->nsims;n++){
		for (ii=0;ii<pair->sims[n].inum;ii++){
			itype = pair->sims[n].type[ii];
            if (itype!=atomtypes[0])continue;
			natoms++;
            count1=0;
            for (k=0;k<map_length;k++){
                RANN::Fingerprint *fingerprint = fingerprint_map[k];
                int s = startingneuron[k];
                //printf("%s %d\n",fingerprint->style,s);
                if (fingerprint->exchange){
                    int fex = pair->exchangelength[itype];
                    int jnum = pair->sims[n].jnum[ii];
                    for (int l=0;l<fingerprint->get_length();l++){
                        for (int jj=0;jj<jnum;jj++){
                            normalshift[count1] += pair->sims[n].exfeatures[ii][jj*fex+l+s]/jnum;
                        }
                        count1++;
                    }
                }
                else {
                    for (int l=0;l<fingerprint->get_length();l++){
                        normalshift[count1] += pair->sims[n].features[ii][l+s];
                        count1++;
                    }
                }
            }
		}
	}
    if (natoms==0)return;
    for (j=0;j<dimensions[0];j++){
        normalshift[j]/=natoms;
    }
	//get standard deviation
	for (n=0;n<pair->nsims;n++){
		for (ii=0;ii<pair->sims[n].inum;ii++){
			itype = pair->sims[n].type[ii];
            if (itype!=atomtypes[0])continue;
            count1=0;
            for (k=0;k<map_length;k++){
                RANN::Fingerprint *fingerprint = fingerprint_map[k];
                int s = startingneuron[k];
                //printf("%s %d\n",fingerprint->style,s);
                if (fingerprint->exchange){
                    int fex = pair->exchangelength[itype];
                    int jnum = pair->sims[n].jnum[ii];
                    for (int l=0;l<fingerprint->get_length();l++){
                        for (int jj=0;jj<jnum;jj++){
                            normalgain[count1]+=(pair->sims[n].exfeatures[ii][jj*fex+l+s]-normalshift[count1])*(pair->sims[n].exfeatures[ii][jj*fex+l+s]-normalshift[count1])/jnum;
                        }
                        count1++;
                    }
                }
                else {
                    for (int l=0;l<fingerprint->get_length();l++){
                        normalgain[count1]+=(pair->sims[n].features[ii][l+s]-normalshift[count1])*(pair->sims[n].features[ii][l+s]-normalshift[count1]);
                        count1++;
                    }
                }
            }
		}
	}
    for (j=0;j<dimensions[0];j++){
        normalgain[j]=sqrt(normalgain[j]/natoms);
    }
	//shift input to mean=0, std = 1
	// for (n=0;n<nsims;n++){
	// 	for (ii=0;ii<sims[n].inum;ii++){
	// 		itype = sims[n].type[ii];
            
	// 		for (j=0;j<fingerprintlength[itype];j++){
	// 			if (normalgain[itype][j]>0){
	// 				sims[n].features[ii][j] -= normalshift[itype][j];
	// 				sims[n].features[ii][j] /= normalgain[itype][j];
	// 			}
	// 		}
	// 		itype = nelements;
	// 		for (j=0;j<fingerprintlength[itype];j++){
	// 			if (normalgain[itype][j]>0){
	// 				sims[n].features[ii][j] -= normalshift[itype][j];
	// 				sims[n].features[ii][j] /= normalgain[itype][j];
	// 			}
	// 		}
	// 	}
	// }
}

void Net::unnormalize_net(Net *net_out){
	int i,j,k;
	double temp;
	copy_network(net_out);
    i=0;
    if (layers>0){
        // for (int i1=0;i1<bundles[0];i1++){
            for (j=0;j<dimensions[1];j++){
                temp = 0.0;
                for (k=0;k<dimensions[0];k++){
                    if (normalgain[k]>0){
                        if (weightdefined[0])net_out[i].Weight[0][j*dimensions[0]+k]/=normalgain[k];
                        temp+=net_out[i].Weight[0][j*dimensions[0]+k]*normalshift[k];
                    }
                }
                if (biasdefined[0])net_out[i].Bias[0][j]-=temp;
            }
        // }
    }
}

void Net::normalize_net(Net *net_out){
	int i,j,k;
	double temp;
    i=0;
	copy_network(net_out);
    if (layers>0){
        // for (int i1=0;i1<bundles[0];i1++){
            for (j=0;j<dimensions[1];j++){
                temp = 0.0;
                for (k=0;k<dimensions[0];k++){
                    if (normalgain[k]>0){
                        temp+=net_out[i].Weight[0][j*dimensions[0]+k]*normalshift[k];
                        if (weightdefined[0])net_out[i].Weight[0][j*dimensions[0]+k]*=normalgain[k];
                    }
                }
                if (biasdefined[0])net_out[i].Bias[0][j]+=temp;
            }
        // }
    }
}

void Net::copy_network(Net *net_new){
	int i,j,k;
    i=0;
    net_new->layers = layers;
    if (layers>0){
        net_new[i].style = new char[SHORTLINE];
        strncpy(net_new[i].style,style,strlen(style)+1);
        net_new[i].empty=empty;
        net_new[i].fullydefined=fullydefined;
        net_new[i].start_index=start_index;
        net_new[i].start_index_sf=start_index_sf;
        net_new[i].end_index=end_index;
        net_new[i].length=length;
        net_new[i].id=id;    //based on ordering of fingerprints listed for i-j in potential file
        net_new[i].pair=pair;
        net_new[i].n_body_type=n_body_type; 
        net_new[i].atomtypes = new int[n_body_type];
        for (j=0;j<n_body_type;j++)net_new[i].atomtypes[j]=atomtypes[j];
        net_new[i].maxlayer = maxlayer;
        net_new[i].sumlayers=sumlayers;
        net_new[i].dimensions = new int [net_new[i].layers];
        net_new[i].startI = new int [net_new[i].layers];
        net_new[i].Weight = new double*[net_new[i].layers-1];
        net_new[i].Bias = new double*[net_new[i].layers-1];
        net_new[i].freezeW = new bool*[net_new[i].layers-1];
        net_new[i].freezeB = new bool*[net_new[i].layers-1];
        net_new[i].weightdefined = weightdefined;
        net_new[i].biasdefined = biasdefined;
        net_new[i].dimensiondefined = dimensiondefined;
        net_new->activations=new Activation** [layers-1];
        net_new->map_length = map_length;
        net_new->startingneuron = new int[map_length];
        net_new->exchange = exchange;
        net_new->fingerprint_map = new Fingerprint*[map_length];
        net_new->neuronmap = neuronmap;

        for (j=0;j<map_length;j++){
            net_new->fingerprint_map[j] = fingerprint_map[j];
            net_new->startingneuron[j] = startingneuron[j];
        }
        net_new->betalen = betalen;
        net_new->betalen_f = betalen_f;
        net_new->freezebeta = new bool[betalen];
        for (j=0;j<betalen;j++){
            net_new->freezebeta[j]=freezebeta[j];
        }
        if (pair->normalizeinput){
            net_new->normalgain = new double[dimensions[0]];
            net_new->normalshift = new double[dimensions[0]];
            for (j=0;j<dimensions[0];j++){
                net_new->normalgain[j]=normalgain[j];
                net_new->normalshift[j]=normalshift[j];
            }
        }
        for (j=0;j<layers;j++){
            net_new[i].dimensions[j]=dimensions[j];
            net_new[i].startI[j]=startI[j];
            if (j>0){
                net_new->activations[j-1] = new Activation*[dimensions[j]];
                for (int k=0;k<dimensions[j];k++){
                    net_new->activations[j-1][k] = pair->create_activation(activations[j-1][k]->style);
                }
            }
            if (j==layers-1)break;
            net_new[i].dimensions[j+1] = dimensions[j+1];
            net_new[i].Weight[j] = new double[net_new[i].dimensions[j]*net_new[i].dimensions[j+1]];
            net_new[i].Bias[j] = new double[net_new[i].dimensions[j+1]];
            net_new[i].freezeW[j] = new bool[net_new[i].dimensions[j]*net_new[i].dimensions[j+1]];
            net_new[i].freezeB[j] = new bool[net_new[i].dimensions[j+1]];
            for (int k=0;k<net_new[i].dimensions[j]*net_new[i].dimensions[j+1];k++){
                net_new[i].Weight[j][k] = Weight[j][k];
                net_new[i].freezeW[j][k] = freezeW[j][k];
            }
            for (int k=0;k<net_new[i].dimensions[j+1];k++){
                net_new[i].Bias[j][k]=Bias[j][k];
                net_new[i].freezeB[j][k]=freezeB[j][k];
            }
        }
    }
}

// void Net::create_identity_wb(int rows,int columns,int itype,int layer,int ){
//   Weight[layer] = new double [rows*columns];
//   Bias[layer] = new double [rows];
//   freezeW[layer] = new bool [rows*columns];
//   freezeB[layer] = new bool [rows];
//   double r;
//   for (int i=0;i<rows;i++){
//     for (int j=0;j<columns;j++){
//       Weight[layer][j*rows+i] = 0.0;
//       freezeW[layer][j*rows+i] = 1;
//       if (i==j){
//         Weight[layer][j*rows+i] = 1.0;
//       }
//     }
//     Bias[layer][i] = 0.0;
//     freezeB[layer][i] = 0;
//   }
//   weightdefined[layer]=true;
//   biasdefined[layer]=true;
// }

void Net::create_random_weights(int rows,int columns,int itype,int layer,int ){
	Weight[layer] = new double [rows*columns];
	freezeW[layer] = new bool [rows*columns];
	double r;
	for (int i=0;i<rows;i++){
		for (int j=0;j<columns;j++){
			r = (double)rand()/RAND_MAX*2-1;//flat distribution from -1 to 1
			Weight[layer][i*columns+j] = r;
			freezeW[layer][i*columns+j] = 0;
		}
	}
	weightdefined[layer]=true;
}

void Net::create_random_biases(int rows,int itype, int layer,int ){
	Bias[layer] = new double [rows];
	freezeB[layer] = new bool [rows];
	double r;
	for (int i=0;i<rows;i++){
		r = (double) rand()/RAND_MAX*2-1;
		Bias[layer][i] = r;
		freezeB[layer][i] = 0;
	}
	biasdefined[layer]=true;
}

bool Net::parse_values(std::vector<std::string> line,std::vector<std::string> line1,FILE *fp,char *filename,int *linenum) {
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
  else pair->errorf(filename,*linenum,"Undefined value for net constants");
  
  if (check_net_completeness())return true;
  return false;
}

//Check for completeness of dimensions and activations. Check for correctness comes later. Weights, biases, and bundles are optional and can be automatically generated later if not defined.
bool Net::check_net_completeness(){
    int j;
    if (layers==0){
        printf("WARNING: Found empty net. Proceeding. I hope you know what you're doing!\n");
        return true;
    }//approve empy nets
    startI = new int [layers];
    for (j=0;j<layers;j++){
        if (dimensions[j]<=0)return false;//incomplete network definition
    }
    for (j=0;j<layers-1;j++){
        for (int k=0;k<dimensions[j+1];k++) {
            if (activations[j][k]->empty)return false;
        }
    }
    int neuronlength=0;
    for (j=0;j<map_length;j++){
        neuronlength+=fingerprint_map[j]->get_length();
    }
    if (neuronlength<dimensions[0])return false;
    else if (neuronlength>dimensions[0])pair->errorf(FLERR,"fingerprints mapped to net have too many neurons for input layersize");
    return true;
}

bool Net::check_net_correctness(){
    int i,j,l,layer,bundle,rows,columns,c,r;
    if (layers==0){
        return true;
    }//no definitions for this starting element, not considered an error.
    maxlayer=0;
    sumlayers=0;
    for (j=0;j<layers;j++){
        startI[j] = sumlayers;
        sumlayers+=dimensions[j];
        if (dimensions[j]<=0)pair->errorf(FLERR,"missing layersize");//incomplete network definition
        if (dimensions[j]>maxlayer)maxlayer = dimensions[j];
    }    
    if (maxlayer>pair->fnmax) {pair->fnmax = maxlayer;}
    if (dimensions[layers-1]!=1)pair->errorf(FLERR,"output layer must have single neuron");//output layer must have single neuron (the energy)
    if (dimensions[0]>pair->fmax)pair->fmax=dimensions[0];
    for (j=0;j<layers-1;j++){
    for (int k=0;k<dimensions[j+1];k++) {
        if (activations[j][k]->empty)pair->errorf(FLERR,"undefined activations");//undefined activations
    }
    // for (int k=0;k<bundles[j];k++){
    //     if (!bundle_inputdefined[j][k]){
    //         bundleinputsize[j][k] = dimensions[j];
    //         bundleinput[j][k] = new int[bundleinputsize[j][k]];
    //         for (int l=0;l<bundleinputsize[j][k];l++)bundleinput[j][k][l]=l;
    //     }
    //     if (!bundle_outputdefined[j][k]){
    //         bundleoutputsize[j][k] = dimensions[j+1];
    //         bundleoutput[j][k] = new int[bundleoutputsize[j][k]];
    //         for (int l=0;l<bundleoutputsize[j][k];l++)bundleoutput[j][k][l]=l;
    //     }
    //     if (identitybundle[j][k]){
    //         create_identity_wb(bundleinputsize[j][k],bundleoutputsize[j][k],i,j,k);
    //     if (bundleinputsize[j][k]!=bundleoutputsize[j][k]) pair->errorf(FLERR,"input and output of identity bundles must be equal size");
    //     }
        if (!weightdefined[j] && !pair->is_lammps)create_random_weights(dimensions[j],dimensions[j+1],i,j,0);//undefined weights
        else if (!weightdefined[j]) pair->errorf(FLERR,"undefined weight matrix!");
        if (!biasdefined[j] && !pair->is_lammps)create_random_biases(dimensions[j+1],i,j,0);//undefined biases
        else if (!biasdefined[j]) pair->errorf(FLERR,"undefined bias vector!");
    }
    // for (int i1=0;i1<bundles[j];i1++){
    //     if (identitybundle[j][i1])continue;
    // }
    // for (int k=0;k<dimensions[j];k++){
    //     bool foundoutput=false;
    //     for (int l=0;l<bundles[j];l++){
    //         if (foundoutput)continue;
    //         for (int m=0;m<bundleinputsize[j][l];m++){
    //             if (bundleinput[j][l][m]==k){
    //             foundoutput = true;
    //             continue;
    //             }
    //         }
    //     }
    //     if (!foundoutput){
    //     pair->errorf(FLERR,"found neuron with no output\n");}
    // }
    // for (int k=0;k<dimensions[j+1];k++){
    //     bool foundinput=false;
    //     for (int l=0;l<bundles[j];l++){
    //         if (foundinput)continue;
    //         for (int m=0;m<bundleoutputsize[j][l];m++){
    //             if (bundleoutput[j][l][m]==k){
    //             foundinput = true;
    //             continue;
    //             }
    //         }
    //     }
    //     if (!foundinput){pair->errorf(FLERR,"found neuron with no input\n");}
    // }
//   }
    startingneuron = new int[map_length];
    int inputlength = 0;
    for (int k=0;k<map_length;k++){
        startingneuron[k]=fingerprint_map[k]->startingneuron;
        inputlength+=fingerprint_map[k]->get_length();
        if (fingerprint_map[k]->exchange){exchange=true;}
    }
    // printf("%d %d\n",dimensions[0],startingneuron[map_length]);
    if (dimensions[0]!=inputlength)pair->errorf(FLERR,"fingerprint length does not match input layersize");
    betalen=0;
    for (layer=0;layer<layers-1;layer++){
        // for (bundle=0;bundle<bundles[layer];bundle++){
            // if (identitybundle[layer][bundle]){continue;}
            rows = dimensions[layer+1];
            columns = dimensions[layer];
            betalen += rows*columns+rows;
        // }
    }
    freezebeta = new bool[betalen];
    start_index = pair->betalen_f;
    start_index_sf = pair->betalen;
    int count = 0;
    for (layer=0;layer<layers-1;layer++){
        // for (bundle=0;bundle<bundles[layer];bundle++){
        //     if (identitybundle[layer][bundle]){continue;}
            rows = dimensions[layer+1];
            columns = dimensions[layer];
            for (r=0;r<rows;r++){
                for (c=0;c<columns;c++){
                    if (freezeW[layer][r*columns+c]){
                        freezebeta[count] = 1;
                        betalen_f++;
                        count++;
                    }
                    else {
                        freezebeta[count] = 0;
                        count++;
                    }
                }
                if (freezeB[layer][r]){
                    freezebeta[count] = 1;
                    betalen_f++;
                    count++;
                }
                else {
                    freezebeta[count] = 0;
                    count++;
                }
            }
        // }
    }
    allocate();
  return true;
}

 
void Net::parse_layer_size(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
    int i;
    int nwords = line1.size();
    layers = nwords;
    weightdefined = new bool [layers];
    biasdefined = new bool [layers];
    dimensiondefined = new bool [layers];
    // bundle_inputdefined = new bool *[layers];
    // bundle_outputdefined = new bool *[layers];
    dimensions = new int [layers];
    // bundles = new int[layers];
    // identitybundle = new bool *[layers];
    // bundleinputsize = new int *[layers];
    // bundleoutputsize = new int *[layers];
    // bundleinput = new int **[layers];
    // bundleoutput = new int **[layers];
    Weight = new double *[layers];
    Bias = new double *[layers];
    freezeW = new bool *[layers];
    freezeB = new bool *[layers];
    for (int j=0;j<layers;j++){
        dimensions[j]=0;
        // bundles[j] = 1;
        // bundleinputsize[j] = new int [bundles[j]];
        // bundleoutputsize[j] = new int [bundles[j]];
        // bundleinput[j] = new int *[bundles[j]];
        // bundleoutput[j] = new int *[bundles[j]];
        // Weight[j] = new double *[bundles[j]];
        // Bias[j] = new double *[bundles[j]];
        // freezeW[j] = new bool *[bundles[j]];
        // freezeB[j] = new bool *[bundles[j]];
        // identitybundle[j] = new bool[bundles[j]];
        // weightdefined[j] = new bool[bundles[j]];
        // biasdefined[j] = new bool[bundles[j]];
        dimensiondefined[j]=false;
        // bundle_inputdefined[j]=new bool [bundles[j]];
        // bundle_outputdefined[j]=new bool [bundles[j]];
        // for (int k=0;k<bundles[j];k++){
            weightdefined[j]=false;
            biasdefined[j]=false;
            // bundle_inputdefined[j][k]=false;
            // bundle_outputdefined[j][k]=false;
            // identitybundle[j][k]=false;
        // }
    }
    activations=new Activation** [layers-1];
    for (int j=0;j<layers;j++){
        dimensions[j]= pair->inumeric(filename,linenum,line1[j]);
        dimensiondefined[j]=true;
        if (j>0){
            activations[j-1] = new Activation*[dimensions[j]];
            for (int k=0;k<dimensions[j];k++){
                activations[j-1][k] = new Activation(pair);
            }
        }
    }
    return;
}

// void Net::parse_bundles(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum){
//       if (layers==0)pair->errorf(filename,linenum-1,"networklayers for each atom type must be defined before the corresponding layer sizes.");
//       int nwords = line.size();
//       int nwords1 = line1.size();
//       if (nwords1 != layers)pair->errorf(filename,linenum-1,"invalid number of layers in bundle definition");
//       for (int j=0;j<layers;j++){
//         bundles[j]= utils::inumeric(filename,linenum,line1[0],true,pair->lmp);
//         delete [] bundleinputsize[j];
//         delete [] bundleoutputsize[j];
//         delete [] bundleinput[j];
//         delete [] bundleoutput[j];
//         delete [] bundleW[j];
//         delete [] bundleB[j];
//         delete [] freezeW[j];
//         delete [] freezeB[j];
//         delete [] identitybundle[j];
//         delete [] weightdefined[j];
//         delete [] biasdefined[j];
//         delete [] bundle_inputdefined[j];
//         delete [] bundle_outputdefined[j];
//         bundleinputsize[j] = new int [bundles[j]];
//         bundleoutputsize[j] = new int [bundles[j]];
//         bundleinput[j] = new int *[bundles[j]];
//         bundleoutput[j] = new int *[bundles[j]];
//         bundleW[j] = new double *[bundles[j]];
//         bundleB[j] = new double *[bundles[j]];
//         freezeW[j] = new bool *[bundles[j]];
//         freezeB[j] = new bool *[bundles[j]];
//         identitybundle[j] = new bool[bundles[j]];
//         weightdefined[j] = new bool[bundles[j]];
//         biasdefined[j] = new bool[bundles[j]];
//         bundle_inputdefined[j] = new bool[bundles[j]];
//         bundle_outputdefined[j] = new bool[bundles[j]];
//         for (int k=0;k<bundles[j];k++){
//             weightdefined[j][k]=false;
//             biasdefined[j][k]=false;
//             bundle_inputdefined[j][k]=false;
//             bundle_outputdefined[j][k]=false;
//             identitybundle[j][k]=false;
//         }
//       }
//       return;
// }

// void Net::parse_bundle_id(std::vector<std::string> line,std::vector<std::string> line1,FILE *fp,char *filename,int *linenum){
//     int nwords = line.size();
//     int nwords1 = line1.size();
//     char *ptr;
//     char **words1;
//     int longline = 4096;
//     char linetemp [longline];
//     for (int j=0;j<layers;j++){
//         if (nwords1 != bundles[j])pair->errorf(filename,*linenum-1,"invalid number of bundles in bundle id definition");
//         for (int b=0;b<bundles[j];b++){
//             int id =utils::inumeric(filename,*linenum,line1[j],true,pair->lmp);
//             if (id!= 1 && id != 0)pair->errorf(filename,*linenum-1,"bundle id must be 1 or 0\n");
//             identitybundle[j][b]=id;
// //       if (net[i].layers==0)errorf(filename,linenum-1,"networklayers for each atom type must be defined before the corresponding layer sizes.");
// //       int j = utils::inumeric(filename,linenum,line[2],true,lmp);
// //       if (j>=net[i].layers || j<0){errorf(filename,linenum-1,"invalid layer in bundle inputs definition");};
// //       int b = utils::inumeric(filename,linenum,line[3],true,lmp);
// //       if (b>net[i].bundles[j] || b<0){errorf(filename,linenum-1,"invalid bundle in bundle inputs");}
// //       int id =utils::inumeric(filename,linenum,line1[0],true,lmp);
// //       if (id!= 1 && id != 0)errorf(filename,linenum-1,"bundle id must be 1 or 0\n");
// //       net[i].identitybundle[j][b]=id;
// //       return;
//         }
//         ptr = fgets(linetemp,longline,fp);
//         (*linenum)++;
//         line1 = pair->tokenmaker(linetemp,": ,\t_\n");
//         if (ptr==nullptr)pair->errorf(filename,*linenum-1,"unexpected end of potential file!");
//         nwords1 = line1.size();
//     }
//     return;
// }

// void Net::parse_bundle_input(std::vector<std::string> line,std::vector<std::string> line1,FILE * fid,char *filename,int *linenum){
//     int nwords;
//     if (layers==0)pair->errorf(filename,*linenum-1,"networklayers for each atom type must be defined before the corresponding layer sizes.");
//     int j = utils::inumeric(filename,*linenum,line1[0],true,pair->lmp);
//     if (j>=layers || j<0){pair->errorf(filename,*linenum-1,"invalid layer in bundle inputs definition");};
//     int b = utils::inumeric(filename,*linenum,line[3],true,pair->lmp);
//     if (b>bundles[j] || b<0){pair->errorf(filename,*linenum-1,"invalid bundle in bundle inputs");}
//     nwords = line1.size();
//     bundleinputsize[j][b] = nwords;
//     bundleinput[j][b] = new int [nwords];
//     for (int k=0;k<nwords;k++){
//       bundleinput[j][b][k]=utils::inumeric(filename,*linenum,line1[k],true,pair->lmp);
//       if (bundleinput[j][b][k]>dimensions[j] || bundleinput[j][b][k]<0){pair->errorf(filename,*linenum-1,"bundle input neurons exceed layer size");}
//     }
//     bundle_inputdefined[j][b]=true;
//     return;
// }

// void Net::parse_bundle_output(std::vector<std::string> line,std::vector<std::string> line1,FILE *fid,char *filename,int *linenum){
//   int i,nwords;
//     if (layers==0)pair->errorf(filename,*linenum-1,"networklayers for each atom type must be defined before the corresponding layer sizes.");
//     int j = utils::inumeric(filename,*linenum,line[2],true,pair->lmp);
//     if (j>=layers || j<0){pair->errorf(filename,*linenum-1,"invalid layer in bundle inputs definition");};
//     int b = utils::inumeric(filename,*linenum,line[3],true,pair->lmp);
//     if (b>bundles[j] || b<0){pair->errorf(filename,*linenum-1,"invalid bundle in bundle inputs");}
//     nwords = line1.size();
//     bundleoutputsize[j][b] = nwords;
//     bundleoutput[j][b] = new int [nwords];
//     for (int k=0;k<nwords;k++){
//         bundleoutput[j][b][k]=utils::inumeric(filename,*linenum,line1[k],true,pair->lmp);
//         if (bundleoutput[j][b][k]>dimensions[j+1] || bundleoutput[j][b][k]<0)pair->errorf(filename,*linenum-1,"bundle output neurons exceed layer size");
//     }
//     bundle_outputdefined[j][b]=true;
//     return;
// }

void Net::parse_weight(std::vector<std::string> line,std::vector<std::string> line1,FILE* fp,char *filename,int *linenum){
  int i,j,k,l,ins,ops,nwords;
  char *ptr;
  char **words1;
  int longline = 4096;
  char linetemp [longline];
  nwords = line.size();
  if (layers==0)pair->errorf(filename,*linenum-1,"networklayers must be defined before weights.");
  i=pair->inumeric(filename,*linenum,line[4]);
  if (i>=layers || i<0)pair->errorf(filename,*linenum-1,"invalid weight layer");
  if (dimensiondefined[i]==false || dimensiondefined[i+1]==false) pair->errorf(FLERR,"network layer sizes must be defined before corresponding weight");
  ins = dimensions[i];
  ops = dimensions[i+1];
  Weight[i] = new double [ins*ops];
  freezeW[i] = new bool [ins*ops];
  for (j=0;j<ins*ops;j++){freezeW[i][j]=0;}
  weightdefined[i] = true;
  nwords = line1.size();
  if (nwords != ins)pair->errorf(filename,*linenum-1,"invalid weights per line");
  for (k=0;k<ins;k++){
    Weight[i][k] = pair->numeric(filename,*linenum,line1[k]);
  }
  for (j=1;j<ops;j++){
    ptr = fgets(linetemp,longline,fp);
    (*linenum)++;
    line1 = pair->tokenmaker(linetemp,": ,\t_\n");
    if (ptr==nullptr)pair->errorf(filename,*linenum-1,"unexpected end of potential file!");
    nwords = line1.size();
    if (nwords != ins)pair->errorf(filename,*linenum-1,"invalid weights per line");
    for (k=0;k<ins;k++) {
        Weight[i][j*ins+k] = pair->numeric(filename,*linenum,line1[k]);
    }
  }
  return;
}

void Net::parse_freezeW(std::vector<std::string> line,std::vector<std::string> line1,FILE* fp,char *filename,int *linenum){
  int i,j,k,l,ins,ops,nwords;
  char *ptr;
  char **words1;
  int longline = 4096;
  char linetemp [longline];
  nwords = line.size();
//   if (nwords == 3){b=0;}
//   else if (nwords>3){b = utils::inumeric(filename,*linenum,line[3],true,pair->lmp);}
      if (layers==0)pair->errorf(filename,*linenum-1,"networklayers must be defined before freeze weights.");
      i=pair->inumeric(filename,*linenum,line[4]);
      if (!weightdefined[i]) pair-> errorf(filename,*linenum-1,"weights must be defined before freezeW");
      if (i>=layers || i<0)pair->errorf(filename,*linenum-1,"invalid freeze weight layer");
      if (dimensiondefined[i]==false || dimensiondefined[i+1]==false) pair->errorf(FLERR,"network layer sizes must be defined before corresponding freeze weight");
    //   if (bundle_inputdefined[i][b]==false && b!=0) pair->errorf(filename,*linenum-1,"bundle inputs must be defined before freeze weights");
    //   if (bundle_outputdefined[i][b]==false && b!=0) pair->errorf(filename,*linenum-1,"bundle outputs must be defined before freeze weights");
    //   if (identitybundle[i][b]) pair->errorf(filename,*linenum-1,"do not define freeze weights for an identity bundle");
    //   if (bundle_inputdefined[i][b]==false){ins = dimensions[i];}
    //   else {ins = bundleinputsize[i][b];}
    //   if (bundle_outputdefined[i][b]==false){ops = dimensions[i+1];}
    //   else {ops = bundleoutputsize[i][b];}
      ins = dimensions[i];
      ops = dimensions[i+1];
	  nwords = line1.size();
      if (nwords != ins)pair->errorf(filename,*linenum-1,"invalid freeze weights per line");
      for (k=0;k<ins;k++){
        freezeW[i][k] = pair->numeric(filename,*linenum,line1[k]);
      }
      for (j=1;j<ops;j++){
        ptr = fgets(linetemp,longline,fp);
        (*linenum)++;
        line1 = pair->tokenmaker(linetemp,": ,\t_\n");
        if (ptr==nullptr)pair->errorf(filename,*linenum-1,"unexpected end of potential file!");
        nwords = line1.size();
        if (nwords != ins)pair->errorf(filename,*linenum-1,"invalid freeze weights per line");
        for (k=0;k<ins;k++) {
          freezeW[i][j*ins+k] = pair->numeric(filename,*linenum,line1[k]);
        }
      }
      return;
    }

void Net::parse_bias(std::vector<std::string> line,std::vector<std::string> line1,FILE* fp,char *filename,int *linenum){
  int i,j,l,ops;
  char *ptr;
  int longline = 1024;
  int nwords = line.size();
  char linetemp [longline];
    // if (nwords == 3){b=0;}
    // else if (nwords>3){b = utils::inumeric(filename,*linenum,line[3],true,pair->lmp);}
      if (layers==0)pair->errorf(filename,*linenum-1,"networklayers must be defined before biases.");
      i=pair->inumeric(filename,*linenum,line[4]);
      if (i>=layers || i<0)pair->errorf(filename,*linenum-1,"invalid bias layer");
      if (dimensiondefined[i]==false) pair->errorf(filename,*linenum-1,"network layer sizes must be defined before corresponding bias");
    //   if (bundle_outputdefined[i][b]==false && b!=0) pair->errorf(filename,*linenum-1,"bundle outputs must be defined before bias");
    //   if (identitybundle[i][b]) pair->errorf(filename,*linenum-1,"cannot define bias for an identity bundle");
    //   if (bundle_outputdefined[i][b]==false){ops=dimensions[i+1];}
    //   else {ops = bundleoutputsize[i][b];}
    ops = dimensions[i+1];
      Bias[i] = new double [ops];
      freezeB[i] = new bool[ops];
      for (j=0;j<ops;j++){freezeB[i][j]=0;}
      biasdefined[i] = true;
      Bias[i][0] = pair->numeric(filename,*linenum,line1[0]);
      for (j=1;j<ops;j++){
		    ptr=fgets(linetemp,longline,fp);
        if (ptr==nullptr)pair->errorf(filename,*linenum,"unexpected end of potential file!");
        (*linenum)++;
        line1 = pair->tokenmaker(linetemp,": ,\t_\n");
        Bias[i][j] = pair->numeric(filename,*linenum,line1[0]);
      }
      return;
    }

void Net::parse_freezeB(std::vector<std::string> line,std::vector<std::string> line1,FILE* fp,char *filename,int *linenum){
  int i,j,l,ops;
  char *ptr;
  int longline = 1024;
  int nwords = line.size();
  char linetemp [longline];
    // if (nwords == 3){b=0;}
    // else if (nwords>3){b = utils::inumeric(filename,*linenum,line[3],true,pair->lmp);}
      if (layers==0)pair->errorf(filename,*linenum-1,"networklayers must be defined before freeze biases.");
      i=pair->inumeric(filename,*linenum,line[4]);
      if (!biasdefined[i])pair->errorf(filename,*linenum-1,"bias must be defined before freezeB");
      if (i>=layers || i<0)pair->errorf(filename,*linenum-1,"invalid bias layer");
      if (dimensiondefined[i]==false) pair->errorf(filename,*linenum-1,"network layer sizes must be defined before corresponding freeze bias");
    //   if (bundle_outputdefined[i][b]==false && b!=0) pair->errorf(filename,*linenum-1,"bundle outputs must be defined before freeze bias");
    //   if (identitybundle[i][b]) pair->errorf(filename,*linenum-1,"do not define bias for an identity bundle");
    //   if (bundle_outputdefined[i][b]==false){ops=dimensions[i+1];}
    //   else {ops = bundleoutputsize[i][b];}
    ops = dimensions[i+1];
      biasdefined[i] = true;
      freezeB[i][0] = pair->numeric(filename,*linenum,line1[0]);
      for (j=1;j<ops;j++){
		    ptr=fgets(linetemp,longline,fp);
        if (ptr==nullptr)pair->errorf(filename,*linenum,"unexpected end of potential file!");
        (*linenum)++;
        line1 = pair->tokenmaker(linetemp,": ,\t_\n");
        freezeB[i][j] = pair->numeric(filename,*linenum,line1[0]);
      }
      return;
    }

void Net::parse_activation_functions(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum){
    int i,l;
    int nwords = line.size();
    int nwords1 = line1.size();
    if (nwords1 != layers-1)pair->errorf(filename,linenum-1,"invalid number of layers in activation definition");
    for (i=0;i<layers-1;i++){
        if (dimensiondefined[i+1]==false) pair->errorf(filename,linenum-1,"network layer sizes must be defined before corresponding activation");
        for (int j = 0;j<dimensions[i+1];j++){
            delete activations[i][j];
            activations[i][j]=pair->create_activation(line1[i].c_str());
        }
    }
    return;
    }

void Net::parse_fingerprint_map(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum){
    int i,j,l;
    int nwords = line.size();
    int nwords1 = line1.size();
    int type = atomtypes[0];
    int count = 0;
    int neuroncount =0;
    map_length = nwords1>>1;
    neuronmap = new int[map_length];
    fingerprint_map = new RANN::Fingerprint*[map_length];
    for (i=0;i<nwords1-1;i++){
        bool found=false;
        for (j=0;j<pair->fingerprintperelement[type];j++){
            RANN::Fingerprint *fingerprint = pair->fingerprints[type][j];
            if (line1[i].compare(fingerprint->style)==0 & fingerprint->id==pair->inumeric(filename,linenum,line1[i+1])){
                neuroncount+=fingerprint->get_length();
                fingerprint_map[count] = fingerprint;
                i+=1;
                found = true;
                neuronmap[count] = neuroncount;
                count++;
                break;
            }
        }
        if (!found){pair->errorf(filename,linenum-1,"Could not find fingerprint for network in list of defined fingerprints");}
    }
    
    return;
}

void Net::jacobian_convolution(double *J,double *target,int *s,int sn,int natoms){
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
    int targettype = pair->targettype;
	#pragma omp for schedule(guided)
	for (n1=0;n1<sn;n1++){
		nn = s[n1];
		n4s = pair->sims[nn].inum;
		double energy;
		double force[n4s*3];
		energy = 0.0;
		soff = pair->sims[nn].speciesoffset;
        for (ii=start_index_sf;ii<betalen+start_index_sf;ii++){
            J[n1*pair->betalen+ii]=0.0;
        }
		for (ii=0;ii<n4s;ii++){
			int itype,numneigh,jnum,**firstneigh,*jlist,i,j,k,j1,jj,starti,prevI,l,startL,prevL,k1,k2,k3;
			starti=0;
			itype = pair->sims[nn].type[ii];
            if (itype!=atomtypes[0])continue;
			int fex = pair->exchangelength[itype];
			//for (int k1 = 0;k1<pair->netperelement[itype];k1++){
            n1slM1 = sumlayers-1;
            iiX3 = ii*3;
            sPcPiiX3 = sn+count4+iiX3;
            numneigh = pair->sims[nn].numneigh[ii];
            jnum = numneigh+1;//extra value on the end of the array is the self term.
            firstneigh = pair->sims[nn].firstneigh;
            jlist = firstneigh[ii];
            int L = layers-1;
            double layer[sumlayers];
            double dlayer[sumlayers];
            double dlayerx[jnum*sumlayers];
            double dlayery[jnum*sumlayers];
            double dlayerz[jnum*sumlayers];

            int f = dimensions[0];
            prevI = 0;
            n1dimi = dimensions[0];
            int startJ = 0;

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
                        energy += layer[jPstartI];
                    }
                }
                prevI = starti;
            }
            prevI=0;
            int count2=start_index_sf;
            int count3=start_index;

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
                            for (k3=0;k3<dimensions[i+1];k3++){
                                dXw[(k1+startL)*d1+k3]+=dlayer[k2+prevL]*Weight[l-1][k1*dimensions[l-1]+k2]*dXw[(k2+prevL)*d1+k3]*dlayer[startI[layers-1]];
                            }
                        }
                    }
                }
                for (k1=0;k1<dimensions[i+1];k1++){
                    //weights
                    for (k2=0;k2<dimensions[i];k2++){
                        if (~freezebeta[count3]){
                            J[n1*pair->betalen+count2] += -dXw[(sumlayers-1)*d1+k1]*layer[k2+prevI]*pair->sims[nn].energy_weight;
                            count2++;
                        }
                        count3++;
                    }
                    //bias
                    if (~freezebeta[count3]){
                        J[n1*pair->betalen+count2] += -dXw[(sumlayers-1)*d1+k1]*pair->sims[nn].energy_weight;
                        count2++;
                    }
                    count3++;
                }
            }
		}
        target[n1] += energy*pair->sims[nn].energy_weight;
	}
	//regularizer
	if (pair->doregularizer){
		int count2 = start_index_sf;
		int count3 = start_index;
		int snoff = sn;
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

void Net::force_fit(double *J,double *target,int *s,int sn,int natoms){
	//clock_t start = clock();
	double start_time = omp_get_wtime();
    #pragma omp parallel
	{
    int n1,nn,inum,ii;
	#pragma omp for schedule(guided)
	for (n1=0;n1<sn;n1++){
		nn = s[n1];
		inum = pair->sims[nn].inum;
		double energy;
		double force[inum*3];
        double stress[3][3]={{0,0,0},{0,0,0},{0,0,0}};
        double volumei = 1/(pair->sims[nn].box[0][0]*pair->sims[nn].box[1][1]*pair->sims[nn].box[2][2]);
        energy = 0.0;
        for (ii=0;ii<inum*3;ii++){
            force[ii]=0.0;
        }
		for (ii=0;ii<inum;ii++){
			int itype,numneigh,jnum,**firstneigh,*jlist,i,j,k,j1,jj,starti,prevI,l,startL,prevL,k1,k2,k3;
			starti=0;
			itype = pair->sims[nn].type[ii];
            if (itype!=atomtypes[0])continue;
            numneigh = pair->sims[nn].numneigh[ii];
            jnum = numneigh+1;//extra value on the end of the array is the self term.
            firstneigh = pair->sims[nn].firstneigh;
            jlist = firstneigh[ii];
            int L = layers-1;
            double layer[sumlayers];
            double dlayer[sumlayers];
            double ddlayer[sumlayers];
            int f = dimensions[0];
            double dlayerx[jnum*f];
            double dlayery[jnum*f];
            double dlayerz[jnum*f];
            int f1 = pair->fingerprintlength[itype];
            prevI = 0;
            int startJ = 0;
            for (k=0;k<f;k++){
                dlayer[k] = 1.0;
                ddlayer[k] = 0.0;
            }
            //initial layer derivatives:
            int count1 = 0;
            for (k=0;k<map_length;k++){
                RANN::Fingerprint *fingerprint = fingerprint_map[k];
                int s = startingneuron[k];
                for (int l=0;l<fingerprint->get_length();l++){
                    layer[count1] = pair->sims[nn].features[ii][l+s];
                    if (pair->normalizeinput){
                        layer[count1]=(layer[count1]-normalshift[count1])/normalgain[count1];
                    }
                    for (jj=0;jj<jnum;jj++){
                        dlayerx[jj*f+count1]=pair->sims[nn].dfx[ii][jj*f1+l+s];
                        dlayery[jj*f+count1]=pair->sims[nn].dfy[ii][jj*f1+l+s];
                        dlayerz[jj*f+count1]=pair->sims[nn].dfz[ii][jj*f1+l+s];
                        if (pair->normalizeinput){
                            dlayerx[jj*f+count1]/=normalgain[count1];
                            dlayery[jj*f+count1]/=normalgain[count1];
                            dlayerz[jj*f+count1]/=normalgain[count1];
                        }
                    }
                    count1++;
                }
            }
            for (k=f;k<sumlayers;k++){layer[k]=0;}
            //initial propagation forward:
            for (i=0;i<L;i++){
                for (j=0;j<dimensions[i+1];j++){
                    int ji = startI[i+1]+j;
                    for (k=0;k<dimensions[i];k++){
                        layer[ji] += Weight[i][j*dimensions[i]+k]*layer[k+prevI];
                    } 
                    layer[ji] += Bias[i][j];
                }
                for (j=0;j<dimensions[i+1];j++){
                    int ji = startI[i+1]+j;
                    ddlayer[ji] = activations[i][j]->ddactivation_function(layer[ji]);
                    dlayer[ji]  = activations[i][j]-> dactivation_function(layer[ji]);
                    layer[ji]   = activations[i][j]->  activation_function(layer[ji]);
                    if (i==L-1){
                        energy += layer[ji];
                    }
                }
                prevI = startI[i+1];
            }
            prevI=0;
            int count2=start_index_sf;
            int count3=start_index;
            int il,il1,ik,ik1,m,im,im1;
            //chain rule derivative terms:
            double C[layers][maxlayer][maxlayer];
            for (l=1;l<layers;l++){
                for (il=0;il<dimensions[l];il++){
                    for (il1=0;il1<dimensions[l-1];il1++){
                        C[l][il][il1] = dlayer[il+startI[l]]*Weight[l-1][il*dimensions[l-1]+il1];
                    }
                }
            }
            double D[layers][layers][maxlayer][maxlayer];
            double E[layers][layers][maxlayer][maxlayer];
            for (k=1;k<layers;k++){
                for (ik=0;ik<dimensions[k];ik++){
                    for (ik1=0;ik1<dimensions[k-1];ik1++){
                        D[k][k][ik][ik1]=C[k][ik][ik1];
                    }
                }
                // if (n1==0 && ii==0 && k==1){
                //     printf("D%d\n",1);
                //     for (int im=0;im<dimensions[1];im++){
                //         printf("%f %f\n",dlayer[im+startI[1]],layer[im+startI[1]]);
                //         for (int i0=0;i0<dimensions[0];i0++){
                //             printf(" %f ",D[1][1][im][i0]);
                //         }
                //         printf("\n");
                //     }
                //     printf("\n");
                // }
                for (m=k+1;m<layers;m++){
                    for (im=0;im<dimensions[m];im++){
                        for (ik1=0;ik1<dimensions[k-1];ik1++){
                            D[m][k][im][ik1]=0;
                            for (im1=0;im1<dimensions[m-1];im1++){
                                D[m][k][im][ik1]+=C[m][im][im1]*D[m-1][k][im1][ik1];
                            }
                        }
                    }
                    // if (n1==0 && ii==0 && k==1){
                    //     printf("D%d\n",m);
                    //     for (int im=0;im<dimensions[m];im++){
                    //         printf("%f %f\n",dlayer[im+startI[m]],layer[im+startI[m]]);
                    //         for (int i0=0;i0<dimensions[0];i0++){
                    //             printf(" %f ",D[m][1][im][i0]);
                    //         }
                    //         printf("\n");
                    //     }
                    //     printf("\n");
                    // }
                }
                // for (ik=0;ik<dimensions[k];ik++){
                //     for (int ik2=0;ik2<dimensions[k-1];ik2++){
                //         D[k-1][k][ik2][ik]=0;
                //     }
                //     D[k-1][k][ik][ik]=1;
                // }
                for (m=k+1;m<layers;m++){
                    for (int im=0;im<dimensions[m];im++){
                        for (int jk1=0;jk1<dimensions[k-1];jk1++){
                            E[m][k][im][jk1]=0.0;
                            for (int lm1=0;lm1<dimensions[m-1];lm1++){
                                E[m][k][im][jk1]+=Weight[m-1][im*dimensions[m-1]+lm1]*D[m-1][k][lm1][jk1];
                            }
                        }
                    }
                }
            }

            //get dE/dW, dE/dB
            double dW[layers][maxlayer][maxlayer];
            double dB[layers][maxlayer];
            for (l=1;l<layers;l++){
                for (il=0;il<dimensions[l];il++){
                    if (layers-1>=l+1){
                        dB[l][il]=D[layers-1][l+1][0][il]*dlayer[startI[l]+il];
                        for (il1=0;il1<dimensions[l-1];il1++){
                            dW[l][il][il1]=dB[l][il]*layer[startI[l-1]+il1];
                        }
                    }
                    else {
                        dB[l][il]=dlayer[startI[l]+il];
                        for (il1=0;il1<dimensions[l-1];il1++){
                            dW[l][il][il1]=dB[l][il]*layer[startI[l-1]+il1];
                        }
                    }
                }
            }
            // if (ii==0&&n1==0){
            // for (int i0=0;i0<f;i0++){
            //     printf("%f ",D[layers-1][1][0][i0]);
            // }
            // printf("\n");
            // }
            //get forces and stress:
            int n = layers-1;
            for (jj=0;jj<jnum;jj++){
                int j1 = jlist[jj];
                int j2 = pair->sims[nn].id[j1];
                for (ik=0;ik<f;ik++){
                    force[j2*3  ]+=D[n][1][0][ik]*dlayerx[jj*f+ik];
                    force[j2*3+1]+=D[n][1][0][ik]*dlayery[jj*f+ik];
                    force[j2*3+2]+=D[n][1][0][ik]*dlayerz[jj*f+ik];
                    //note that stress uses j1 to get ghost atom position, not j2 which is local
                    stress[0][0]+=D[n][1][0][ik]*dlayerx[jj*f+ik]*pair->sims[nn].x[j1][0];
                    stress[1][1]+=D[n][1][0][ik]*dlayery[jj*f+ik]*pair->sims[nn].x[j1][1];
                    stress[2][2]+=D[n][1][0][ik]*dlayerz[jj*f+ik]*pair->sims[nn].x[j1][2];
                    stress[0][1]+=D[n][1][0][ik]*dlayerx[jj*f+ik]*pair->sims[nn].x[j1][1];
                    stress[0][2]+=D[n][1][0][ik]*dlayerx[jj*f+ik]*pair->sims[nn].x[j1][2];
                    stress[1][2]+=D[n][1][0][ik]*dlayery[jj*f+ik]*pair->sims[nn].x[j1][2];
                    //for verification only:
                    stress[1][0]+=D[n][1][0][ik]*dlayery[jj*f+ik]*pair->sims[nn].x[j1][0];
                    stress[2][0]+=D[n][1][0][ik]*dlayerz[jj*f+ik]*pair->sims[nn].x[j1][0];
                    stress[2][1]+=D[n][1][0][ik]*dlayerz[jj*f+ik]*pair->sims[nn].x[j1][1];
                }
            }
            //get d2E/dlayer0/dW to be multiplied by dlayer0/dx later
            double dfdW[layers][maxlayer][maxlayer][f];
            double dfdB[layers][maxlayer][f];
            for (l=0;l<layers;l++){
                for (int il = 0;il<dimensions[l];il++){
                    for (int i0=0;i0<dimensions[0];i0++){
                        dfdB[l][il][i0]=0.0;
                        for (int il1=0;il1<dimensions[l-1];il1++){
                            dfdW[l][il][il1][i0]=0.0;
                        }
                    }
                }
            }
            
            for (l=1;l<layers;l++){
                for (m=l+1;m<layers;m++){
                    for (int jl=0;jl<dimensions[l];jl++){
                        for (int i0=0;i0<dimensions[0];i0++){
                            double ugly=0.0;
                            double d1,d2,d3;
                            for (im=0;im<dimensions[m];im++){
                                if (n>=m+1){
                                    d1=D[n][m+1][0][im];
                                }
                                else d1=1;
                                if (m>=l+2){
                                    d2=E[m][l+1][im][jl];
                                }
                                else d2=Weight[m-1][im*dimensions[m-1]+jl];
                                if (m>=2){
                                    d3=E[m][1][im][i0];
                                }
                                else d3=Weight[m-1][im*dimensions[m-1]+i0];
                                ugly+=d1*ddlayer[im+startI[m]]*d2*dlayer[jl+startI[l]]*d3;
                            }
                            dfdB[l][jl][i0]+=ugly;
                            for (int jl1=0;jl1<dimensions[l-1];jl1++){
                                dfdW[l][jl][jl1][i0]+=ugly*layer[jl1+startI[l-1]];
                            }
                        }
                    }
                }
                for (int jl=0;jl<dimensions[l];jl++){
                    for (int i0=0;i0<dimensions[0];i0++){
                        double d1,d3;
                        for (int jl1=0;jl1<dimensions[l-1];jl1++){
                            if (l-1>=1){
                                d3=D[l-1][1][jl1][i0];
                            }
                            else d3=(i0==jl1);
                            if (n>=l+1){
                                d1=D[n][l+1][0][jl];
                            }
                            else d1=1;
                            dfdW[l][jl][jl1][i0]+=d1*dlayer[startI[l]+jl]*d3;
                            dfdB[l][jl][i0]+=d1*Weight[l-1][jl*dimensions[l-1]+jl1]*ddlayer[jl+startI[l]]*d3;
                            if (l-1>=1){
                                d3=E[l][1][jl][i0];
                            }
                            else d3=Weight[l-1][jl*dimensions[l-1]+i0];
                            dfdW[l][jl][jl1][i0]+=d1*ddlayer[jl+startI[l]]*layer[jl1+startI[l-1]]*d3;
                        }
                    }
                }
            }

            //get d2E/dW/dx, d2E/dB/dx
            int row = pair->sims[nn].startrow;
            if (sn==1){
                row = 0;
            }
            int column=start_index_sf;
            int column_all=start_index;
            int stressrow=row+1;
            if (pair->doforces)stressrow += inum*3;
            for (jj=0;jj<jnum;jj++){
                int j1 = pair->sims[nn].firstneigh[ii][jj];
                int j2 = pair->sims[nn].id[j1];
                int fr =j2*3+row;
                for (int i0=0;i0<f;i0++){
                    int column=start_index_sf;
                    int column_all=start_index;
                    for (l=1;l<layers;l++){
                        for (il=0;il<dimensions[l];il++){
                            for (il1=0;il1<dimensions[l-1];il1++){
                                if (~freezebeta[column_all]){
                                    if (pair->doforces){
                                        J[fr*pair->betalen+column]-=dfdW[l][il][il1][i0]*dlayerx[jj*f+i0]*pair->sims[nn].force_weight;
                                        fr++;
                                        J[fr*pair->betalen+column]-=dfdW[l][il][il1][i0]*dlayery[jj*f+i0]*pair->sims[nn].force_weight;
                                        fr++;
                                        J[fr*pair->betalen+column]-=dfdW[l][il][il1][i0]*dlayerz[jj*f+i0]*pair->sims[nn].force_weight;
                                        column++;
                                        fr=fr-2;
                                    }
                                    if (pair->dostresses){
                                        int sr=stressrow;
                                        J[sr*pair->betalen+column]-=dfdW[l][il][il1][i0]*dlayerx[jj*f+i0]*pair->sims[nn].x[j1][0]*pair->sims[nn].force_weight*volumei;
                                        sr++;
                                        J[sr*pair->betalen+column]-=dfdW[l][il][il1][i0]*dlayerx[jj*f+i0]*pair->sims[nn].x[j1][1]*pair->sims[nn].force_weight*volumei;
                                        sr++;
                                        J[sr*pair->betalen+column]-=dfdW[l][il][il1][i0]*dlayerx[jj*f+i0]*pair->sims[nn].x[j1][2]*pair->sims[nn].force_weight*volumei;
                                        sr++;
                                        J[sr*pair->betalen+column]-=dfdW[l][il][il1][i0]*dlayery[jj*f+i0]*pair->sims[nn].x[j1][1]*pair->sims[nn].force_weight*volumei;
                                        sr++;
                                        J[sr*pair->betalen+column]-=dfdW[l][il][il1][i0]*dlayery[jj*f+i0]*pair->sims[nn].x[j1][2]*pair->sims[nn].force_weight*volumei;
                                        sr++;
                                        J[sr*pair->betalen+column]-=dfdW[l][il][il1][i0]*dlayerz[jj*f+i0]*pair->sims[nn].x[j1][2]*pair->sims[nn].force_weight*volumei;
                                    }
                                }
                                column_all++;
                            }
                            if (~freezebeta[column_all]){
                                if (pair->doforces){
                                    J[fr*pair->betalen+column]-=dfdB[l][il][i0]*dlayerx[jj*f+i0]*pair->sims[nn].force_weight;
                                    fr++;
                                    J[fr*pair->betalen+column]-=dfdB[l][il][i0]*dlayery[jj*f+i0]*pair->sims[nn].force_weight;
                                    fr++;
                                    J[fr*pair->betalen+column]-=dfdB[l][il][i0]*dlayerz[jj*f+i0]*pair->sims[nn].force_weight;
                                    column++;
                                    fr=fr-2;
                                }
                                if (pair->dostresses){
                                    int sr=stressrow;
                                    J[sr*pair->betalen+column]-=dfdB[l][il][i0]*dlayerx[jj*f+i0]*pair->sims[nn].x[j1][0]*pair->sims[nn].force_weight*volumei;
                                    sr++;
                                    J[sr*pair->betalen+column]-=dfdB[l][il][i0]*dlayerx[jj*f+i0]*pair->sims[nn].x[j1][1]*pair->sims[nn].force_weight*volumei;
                                    sr++;
                                    J[sr*pair->betalen+column]-=dfdB[l][il][i0]*dlayerx[jj*f+i0]*pair->sims[nn].x[j1][2]*pair->sims[nn].force_weight*volumei;
                                    sr++;
                                    J[sr*pair->betalen+column]-=dfdB[l][il][i0]*dlayery[jj*f+i0]*pair->sims[nn].x[j1][1]*pair->sims[nn].force_weight*volumei;
                                    sr++;
                                    J[sr*pair->betalen+column]-=dfdB[l][il][i0]*dlayery[jj*f+i0]*pair->sims[nn].x[j1][2]*pair->sims[nn].force_weight*volumei;
                                    sr++;
                                    J[sr*pair->betalen+column]-=dfdB[l][il][i0]*dlayerz[jj*f+i0]*pair->sims[nn].x[j1][2]*pair->sims[nn].force_weight*volumei;
                                }
                            }
                            column_all++;
                        }
                    }
                }
            }
            //append dE/dW, dE/dB to jacobian
            int r=row;
            if (pair->doforces)r+=inum*3;
            for (l=1;l<layers;l++){
                for (il=0;il<dimensions[l];il++){
                    for (il1=0;il1<dimensions[l-1];il1++){
                        if (~freezebeta[column_all]){
                            J[r*pair->betalen+column]-=dW[l][il][il1]*pair->sims[nn].energy_weight;
                            column++;
                        }
                        column_all++;
                    }
                    if (~freezebeta[column_all]){
                        J[r*pair->betalen+column]-=dB[l][il]*pair->sims[nn].energy_weight;
                        column++;
                    }
                    column_all++;
                }
            }
		}
        int row = pair->sims[nn].startrow;
        if (pair->doforces){
            for (ii=0;ii<inum*3;ii++){
                target[row+ii] += force[ii]*pair->sims[nn].force_weight;
                // if (ii<3&&n1==0)printf("force[ii] %f %f\n",force[ii],target[row+ii]);
            }
            row+=inum*3;
        }
        target[row] += energy*pair->sims[nn].energy_weight;
        if (pair->dostresses){
            target[row+1] += stress[0][0]*pair->sims[nn].force_weight*volumei;
            target[row+2] += stress[0][1]*pair->sims[nn].force_weight*volumei;
            target[row+3] += stress[0][2]*pair->sims[nn].force_weight*volumei;
            target[row+4] += stress[1][1]*pair->sims[nn].force_weight*volumei;
            target[row+5] += stress[1][2]*pair->sims[nn].force_weight*volumei;
            target[row+6] += stress[2][2]*pair->sims[nn].force_weight*volumei;
            // printf("%d %f %f %f %f %f %f %f %f %f\n",nn,stress[0][0],stress[1][1],stress[2][2],stress[0][1],stress[0][2],stress[1][2],stress[1][0],stress[2][0],stress[2][1]);
        }
	}
	//regularizer
	if (pair->doregularizer){
		int count2 = start_index_sf;
		int count3 = start_index;
        int snoff;
        if (pair->dostresses)snoff = sn*7;
        else snoff = sn;
        if (pair->doforces){
            if (sn==1){
                snoff += pair->sims[s[0]].inum;
            }
            else {
		        snoff += pair->natomsr*3;
            }
        }
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
                    J[(snoff+count2)*pair->betalen+count2] = pair->regularizer;
                    target[snoff+count2] = -pair->regularizer*Bias[i][k1];
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

void Net::forward_pass(double *target,int *s,int sn){

	//clock_t start = clock();
	double start_time = omp_get_wtime();

	#pragma omp parallel
	{
	int nn,ii,n1,j;
	int countatoms = 0;
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
            if (itype!=atomtypes[0])continue;
			int fex = pair->exchangelength[itype];

            n1slM1 = sumlayers-1;
            numneigh = pair->sims[nn].numneigh[ii];
            jnum = numneigh+1;//extra value on the end of the array is the self term.
            firstneigh = pair->sims[nn].firstneigh;
            jlist = firstneigh[ii];
            int L = layers-1;
            double layer[sumlayers];
            // double dlayer[sumlayers];
            // double dlayerx[jnum*sumlayers];
            // double dlayery[jnum*sumlayers];
            // double dlayerz[jnum*sumlayers];
            int f = dimensions[0];
            double *features = pair->sims[nn].features[ii];
            // double *dfeaturesx;
            // double *dfeaturesy;
            // double *dfeaturesz;
            // if (pair->doforces){
            //     dfeaturesx = pair->sims[nn].dfx[ii];
            //     dfeaturesy = pair->sims[nn].dfy[ii];
            //     dfeaturesz = pair->sims[nn].dfz[ii];
            // }
            prevI = 0;
            // int startJ = 0;
            // for (k=0;k<dimensions[0];k++){
            //     dlayer[k] = 1.0;
            // }
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
                // for (int i1=0;i1<bundles[i];i1++){
                //     int s1 = bundleoutputsize[i][i1];
                //     int s2 = bundleinputsize[i][i1];
                    for (j=0;j<dimensions[i+1];j++){
                        starti = startI[i+1];
                        // j1 = bundleoutput[i][i1][j];
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
                    // dlayer[jPstartI] = activations[i][j]->dactivation_function(layer[jPstartI]);
                    layer[jPstartI] =  activations[i][j]-> activation_function(layer[jPstartI]);
                    if (i==L-1){
                        energy += layer[jPstartI];
                    }
                }
                prevI=starti;
            }
            prevI=0;
			
		}
		//fill error vector
        target[n1] += energy*pair->sims[nn].energy_weight;
	}
	if (pair->doregularizer & sn == pair->nsimr){
		int count2 = start_index_sf;
		int count3 = start_index;
		int snoff = sn;

		// #pragma omp for schedule(dynamic)

        for (int i=0;i<layers-1;i++){
            // for (int i1=0;i1<bundles[i];i1++){
            //     if (identitybundle[i][i1])continue;
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
            // }
        }

	}
	}
	double time = (double) (omp_get_wtime() - start_time)*1000.0;
}

void Net::forward_pass_force(double *target,int *s,int sn){
	//clock_t start = clock();
	double start_time = omp_get_wtime();
    int f = dimensions[0];
    #pragma omp parallel
	{
    int n1,nn,inum,ii;
	#pragma omp for schedule(guided)
	for (n1=0;n1<sn;n1++){
		nn = s[n1];
		inum = pair->sims[nn].inum;
		double energy;
		double force[inum*3];
        double stress[3][3]={{0,0,0},{0,0,0},{0,0,0}};
        double volumei = 1/(pair->sims[nn].box[0][0]*pair->sims[nn].box[1][1]*pair->sims[nn].box[2][2]);  
        energy = 0.0;
        for (ii=0;ii<inum*3;ii++){
            force[ii]=0.0;
        }
		for (ii=0;ii<inum;ii++){
			int itype,numneigh,jnum,**firstneigh,*jlist,i,j,k,j1,jj,starti,prevI,l,startL,prevL,k1,k2,k3;
			starti=0;
			itype = pair->sims[nn].type[ii];
            if (itype!=atomtypes[0])continue;
            numneigh = pair->sims[nn].numneigh[ii];
            jnum = numneigh+1;//extra value on the end of the array is the self term.
            firstneigh = pair->sims[nn].firstneigh;
            jlist = firstneigh[ii];
            int L = layers-1;
            double layer[sumlayers];
            double dlayer[sumlayers];
            double dlayerx[jnum*f];
            double dlayery[jnum*f];
            double dlayerz[jnum*f];
            int f = dimensions[0];
            int f1 = pair->fingerprintlength[itype];
            prevI = 0;
            int startJ = 0;
            for (k=0;k<f;k++){
                dlayer[k] = 1.0;
            }
            //initial layer derivatives:
            int count1 = 0;
            for (k=0;k<map_length;k++){
                RANN::Fingerprint *fingerprint = fingerprint_map[k];
                int s = startingneuron[k];
                for (int l=0;l<fingerprint->get_length();l++){
                    layer[count1] = pair->sims[nn].features[ii][l+s];
                    if (pair->normalizeinput){
                        layer[count1]=(layer[count1]-normalshift[count1])/normalgain[count1];
                    }
                    for (jj=0;jj<jnum;jj++){
                        dlayerx[jj*f+count1]=pair->sims[nn].dfx[ii][jj*f1+l+s];
                        dlayery[jj*f+count1]=pair->sims[nn].dfy[ii][jj*f1+l+s];
                        dlayerz[jj*f+count1]=pair->sims[nn].dfz[ii][jj*f1+l+s];
                        if (pair->normalizeinput){
                            dlayerx[jj*f+count1]/=normalgain[count1];
                            dlayery[jj*f+count1]/=normalgain[count1];
                            dlayerz[jj*f+count1]/=normalgain[count1];
                        }
                    }
                    count1++;
                }
            }
            for (k=f;k<sumlayers;k++){layer[k]=0;}
            //initial propagation forward:
            for (i=0;i<L;i++){
                for (j=0;j<dimensions[i+1];j++){
                    int ji = startI[i+1]+j;
                    for (k=0;k<dimensions[i];k++){
                        layer[ji] += Weight[i][j*dimensions[i]+k]*layer[k+prevI];
                    } 
                    layer[ji] += Bias[i][j];
                }
                for (j=0;j<dimensions[i+1];j++){
                    int ji = startI[i+1]+j;
                    dlayer[ji] = activations[i][j]->dactivation_function(layer[ji]);
                    layer[ji] =  activations[i][j]-> activation_function(layer[ji]);
                    if (i==L-1){
                        energy += layer[ji];
                    }
                }
                prevI = startI[i+1];
            }
            prevI=0;
            int count2=start_index_sf;
            int count3=start_index;
            int il,il1,ik,ik1,m,im,im1;
            //chain rule derivative terms:
            double C[layers][maxlayer][maxlayer];
            for (l=1;l<layers;l++){
                for (il=0;il<dimensions[l];il++){
                    for (il1=0;il1<dimensions[l-1];il1++){
                        C[l][il][il1] = dlayer[il+startI[l]]*Weight[l-1][il*dimensions[l-1]+il1];
                    }
                }
            }
            double D[layers][layers][maxlayer][maxlayer];
            for (k=1;k<layers;k++){
                for (ik=0;ik<dimensions[k];ik++){
                    for (ik1=0;ik1<dimensions[k-1];ik1++){
                        D[k][k][ik][ik1]=C[k][ik][ik1];
                    }
                }
                for (m=k+1;m<layers;m++){
                    for (im=0;im<dimensions[m];im++){
                        for (ik1=0;ik1<dimensions[k-1];ik1++){
                            D[m][k][im][ik1]=0;
                            for (im1=0;im1<dimensions[m-1];im1++){
                                D[m][k][im][ik1]+=C[m][im][im1]*D[m-1][k][im1][ik1];
                            }
                        }
                    }
                }
            }
            //get forces and stress:
            int n = layers-1;
            for (jj=0;jj<jnum;jj++){
                int j1 = jlist[jj];
                int j2 = pair->sims[nn].id[j1];
                for (ik=0;ik<f;ik++){
                    force[j2*3  ]+=D[n][1][0][ik]*dlayerx[jj*f+ik];
                    force[j2*3+1]+=D[n][1][0][ik]*dlayery[jj*f+ik];
                    force[j2*3+2]+=D[n][1][0][ik]*dlayerz[jj*f+ik];
                    if (pair->dostresses){
                        //note that stress uses j1 to get ghost atom position, not j2 which is local
                        stress[0][0]+=D[n][1][0][ik]*dlayerx[jj*f+ik]*pair->sims[nn].x[j1][0];
                        stress[1][1]+=D[n][1][0][ik]*dlayery[jj*f+ik]*pair->sims[nn].x[j1][1];
                        stress[2][2]+=D[n][1][0][ik]*dlayerz[jj*f+ik]*pair->sims[nn].x[j1][2];
                        stress[0][1]+=D[n][1][0][ik]*dlayerx[jj*f+ik]*pair->sims[nn].x[j1][1];
                        stress[0][2]+=D[n][1][0][ik]*dlayerx[jj*f+ik]*pair->sims[nn].x[j1][2];
                        stress[1][2]+=D[n][1][0][ik]*dlayery[jj*f+ik]*pair->sims[nn].x[j1][2];
                        //for verification only:
                        // stress[1][0]+=D[n][1][0][ik]*dlayery[jj*f+ik]*pair->sims[nn].x[j1][0];
                        // stress[2][0]+=D[n][1][0][ik]*dlayerz[jj*f+ik]*pair->sims[nn].x[j1][0];
                        // stress[2][1]+=D[n][1][0][ik]*dlayerz[jj*f+ik]*pair->sims[nn].x[j1][1];
                    }
                }
            }
		}
        int row = pair->sims[nn].startrow;
        if (pair->doforces){
            for (ii=0;ii<inum*3;ii++){
                target[row+ii] += force[ii]*pair->sims[nn].force_weight;
            }
            row+=inum*3;
        }
        target[row] += energy*pair->sims[nn].energy_weight;
        if (pair->dostresses){
            target[row+1] += stress[0][0]*pair->sims[nn].force_weight*volumei;
            target[row+2] += stress[0][1]*pair->sims[nn].force_weight*volumei;
            target[row+3] += stress[0][2]*pair->sims[nn].force_weight*volumei;
            target[row+4] += stress[1][1]*pair->sims[nn].force_weight*volumei;
            target[row+5] += stress[1][2]*pair->sims[nn].force_weight*volumei;
            target[row+6] += stress[2][2]*pair->sims[nn].force_weight*volumei;
        }
	}
    }

//	clock_t end = clock();
//	double time = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
	double time = (double) (omp_get_wtime() - start_time)*1000.0;
	// printf(" - compute_jacobian(): %f ms\n",time);

}


void Net::get_per_atom_energy(double **energies,int *s,int sn){
	double start_time = omp_get_wtime();
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
            numneigh = pair->sims[nn].numneigh[ii];
            jnum = numneigh+1;//extra value on the end of the array is the self term.
            firstneigh = pair->sims[nn].firstneigh;
            jlist = firstneigh[ii];
            int L = layers-1;
            double layer[sumlayers];
            int f = dimensions[0];
            double *features = pair->sims[nn].features[ii];
            int fex = pair->exchangelength[itype];
            prevI = 0;
            int startJ = 0;
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
                        energy += layer[jPstartI];
                        energies[n1][ii]+=layer[jPstartI];
                    }
                }
                prevI = starti;
            }
		}
	}
	}
	double time = (double) (omp_get_wtime() - start_time)*1000.0;
}

void Net::propagateforward(double *energy,double **force,int ii,int jnum,int itype,double *features, double *dfeaturesx,double *dfeaturesy, double *dfeaturesz,double *exfeatures, double *dexfeaturesx,double *dexfeaturesy, double *dexfeaturesz,int *jl,int nn,double *xn, double *yn, double *zn) {
  int i,j,k,jj,l;
	int L = layers-1;
	//energy output with forces from analytical derivatives
	double dsum1[maxlayer];
	int f = pair->fingerprintlength[itype];
	int fex = pair->exchangelength[itype];
	double sum[maxlayer];
	double layer[maxlayer];
	double dlayerx[maxlayer];
	double dlayery[maxlayer];
	double dlayerz[maxlayer];
    double dotnet[maxlayer][dimensions[0]];
    double dotnetsum[maxlayer][dimensions[0]];
    int count1 = 0;
    for (k=0;k<map_length;k++){
        RANN::Fingerprint *fingerprint = fingerprint_map[k];
        int s = startingneuron[k];
        for (int l=0;l<fingerprint->get_length();l++){
            layer[count1] = features[l+s];
            count1++;
        }
    }
    for (j=0;j<dimensions[1];j++){
        for (k=0;k<dimensions[0];k++){
            dotnetsum[j][k]=Weight[0][j*dimensions[0]+k];
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
        // if (ii==0)printf("P%d\n",i+1);
        for (j=0;j<dimensions[i+1];j++) {
            dsum1[j] = activations[i][j]->dactivation_function(sum[j]);
            sum[j] = activations[i][j]->activation_function(sum[j]);
            // if (ii==0)printf("%f %f\n",dsum1[j],sum[j]);
            for (l=0;l<dimensions[0];l++){
                dotnetsum[j][l]*=dsum1[j];
                // if (ii==0)printf(" %f ",dotnetsum[j][l]);
            }
            // if (ii==0)printf("\n");
        }
        // if (ii==0)printf("\n");
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
    energy[0] += sum[0];
    count1 = 0;
    // if (ii==0){
    //     for (int i0=0;i0<dimensions[0];i0++){
    //         printf("%f ",dotnetsum[0][i0]);
    //     }
    //     printf("\n");
    // }
    // if (ii==0){
    //     for (int i0=0;i0<dimensions[0];i0++){
    //         printf("P %f\n",dotnetsum[0][i0]);
    //     }
    // }
    for (k=0;k<map_length;k++){
        RANN::Fingerprint *fingerprint = fingerprint_map[k];
        int s = startingneuron[k];
        for (int l=0;l<fingerprint->get_length();l++){
            for (jj=0;jj<jnum;jj++){
                dlayerx[count1] = dfeaturesx[jj*f+l+s];
                dlayery[count1] = dfeaturesy[jj*f+l+s];
                dlayerz[count1] = dfeaturesz[jj*f+l+s];
                int j2 = jl[jj];
                force[j2][0]+=dotnetsum[0][count1]*dlayerx[count1];
                force[j2][1]+=dotnetsum[0][count1]*dlayery[count1];
                force[j2][2]+=dotnetsum[0][count1]*dlayerz[count1];
            }
            count1++;
        }
    }
}

// void Net::propagateforward(double *energy,double **force,int ii,int jnum,int itype,double *features, double *dfeaturesx,double *dfeaturesy, double *dfeaturesz,double *exfeatures, double *dexfeaturesx,double *dexfeaturesy, double *dexfeaturesz,int *jl,int nn,double *xn, double *yn, double *zn) {
//   int i,j,k,jj;
// 	int L = layers-1;
// 	//energy output with forces from analytical derivatives
// 	double dsum1[maxlayer];
// 	int f = pair->fingerprintlength[itype];
// 	int fex = pair->exchangelength[itype];
// 	double sum[maxlayer];
// 	double layer[maxlayer];
// 	double dlayersumx[jnum][maxlayer];
// 	double dlayersumy[jnum][maxlayer];
// 	double dlayersumz[jnum][maxlayer];
// 	double dlayerx[jnum][maxlayer];
// 	double dlayery[jnum][maxlayer];
// 	double dlayerz[jnum][maxlayer];
// 	for (jj=0;jj<jnum;jj++){
// 		int count1 = 0;
// 		for (k=0;k<map_length;k++){
// 			RANN::Fingerprint *fingerprint = fingerprint_map[k];
//             int s = startingneuron[k];
// 			if (fingerprint->exchange){
// 				for (int l=0;l<fingerprint->get_length();l++){
// 					layer[count1] = exfeatures[jj*fex+l+s];
// 					dlayerx[jj][count1] = dexfeaturesx[jj*fex+count1];
// 					dlayery[jj][count1] = dexfeaturesy[jj*fex+count1];
// 					dlayerz[jj][count1] = dexfeaturesz[jj*fex+count1];
// 					count1++;
// 				}
// 			}
// 			else {
// 				for (int l=0;l<fingerprint->get_length();l++){
// 					layer[count1] = features[l+s];
// 					dlayerx[jj][count1] = dfeaturesx[jj*f+count1];
// 					dlayery[jj][count1] = dfeaturesy[jj*f+count1];
// 					dlayerz[jj][count1] = dfeaturesz[jj*f+count1];
// 					count1++;
// 				}
// 			}
// 		}
// 	}
//     for (i=0;i<layers-1;i++) {
//         for (j=0;j<dimensions[i+1];j++){
//             sum[j]=0;
//         }
//         for (j=0;j<dimensions[i+1];j++){
//             for (k=0;k<dimensions[i];k++){
//                 sum[j] += Weight[i][j*dimensions[i]+k]*layer[k];
//             }
//             sum[j]+= Bias[i][j];
//         }
//         for (j=0;j<dimensions[i+1];j++) {
//         dsum1[j] = activations[i][j]->dactivation_function(sum[j]);
//         sum[j] = activations[i][j]->activation_function(sum[j]);
//         if (i==L-1) {
//             energy[j] += sum[j];
//         }
//         //force propagation
//         for (jj=0;jj<jnum;jj++) {
//             dlayersumx[jj][j]=0;
//             dlayersumy[jj][j]=0;
//             dlayersumz[jj][j]=0;
//         }
//         }
//         for (j=0;j<dimensions[i+1];j++){
//             for (jj=0;jj<jnum;jj++){
//                 for (k=0;k<dimensions[i];k++){
//                     double w1 = Weight[i][j*dimensions[i]+k];
//                     dlayersumx[jj][j] += w1*dlayerx[jj][k];
//                     dlayersumy[jj][j] += w1*dlayery[jj][k];
//                     dlayersumz[jj][j] += w1*dlayerz[jj][k];
//                 }
//             }
//         }
//         for (j=0;j<dimensions[i+1];j++){
//             for (jj=0;jj<jnum;jj++){
//                 dlayersumx[jj][j]*= dsum1[j];
//                 dlayersumy[jj][j]*= dsum1[j];
//                 dlayersumz[jj][j]*= dsum1[j];
//             }
//         }
//         if (i==L-1) {
//             for (j=0;j<dimensions[i+1];j++){
//                 for (jj=0;jj<jnum-1;jj++){
//                     int j2 = jl[jj];
//                     force[j2][0]+=dlayersumx[jj][j];
//                     force[j2][1]+=dlayersumy[jj][j];
//                     force[j2][2]+=dlayersumz[jj][j];
//                 }
//                 int j2 = pair->sims[nn].ilist[ii];
//                 jj = jnum-1;
//                 force[j2][0]+=dlayersumx[jj][j];
//                 force[j2][1]+=dlayersumy[jj][j];
//                 force[j2][2]+=dlayersumz[jj][j];
//             }
//         }
//         //update values for next iteration
//         for (j=0;j<dimensions[i+1];j++) {
//             layer[j]=sum[j];
//             for (jj=0;jj<jnum;jj++) {
//                 dlayerx[jj][j] = dlayersumx[jj][j];
//                 dlayery[jj][j] = dlayersumy[jj][j];
//                 dlayerz[jj][j] = dlayersumz[jj][j];
//             }
//         }
//     }
// }

void Net::propagateforwardspin(double *energy,double **force,double **fm,double **hm,int ii,int jnum,int itype,double *features, double *dfeaturesx,double *dfeaturesy, double *dfeaturesz,double *exfeatures, double *dexfeaturesx,double *dexfeaturesy, double *dexfeaturesz, double *sx, double *sy, double *sz, double *sxx, double *sxy, double *sxz, double *syy, double *syz, double *szz,int *jl,int nn) {
  int i,j,k,jj,j1,i1;
  double hbar = pair->hbar;
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
					layer[count1] = features[jj*fex+l+s];
					dlayerx[jj][count1] = dfeaturesx[jj*fex+count1];
					dlayery[jj][count1] = dfeaturesy[jj*fex+count1];
					dlayerz[jj][count1] = dfeaturesz[jj*fex+count1];
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
                // int k1=bundleinput[i][i1][k];
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

void Net::write_values(FILE *fid){
  int i;
  if (layers==0)return;
//   fprintf(fid,"netconstants:");
//   fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
//   for (i=1;i<n_body_type;i++) {
//     fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
//   }
//   fprintf(fid,":%s_%d:layers:\n",style,id);
//   fprintf(fid,"%d\n",layers);
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

//   for (int j=0;j<layers-1;j++){
//     fprintf(fid,"netconstants:");
//     fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
//     for (i=1;i<n_body_type;i++) {
//         fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
//     }
//     fprintf(fid,":%s_%d_%d:bundles:\n",style,id,j);
//     fprintf(fid,"%d ",bundles[j]);
//     fprintf(fid,"\n");
//   }

//   for (int j=0;j<layers-1;j++){
//     for (int k=0;k<bundles[j];k++){
//         fprintf(fid,"netconstants:");
//         fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
//         for (i=1;i<n_body_type;i++) {
//             fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
//         }
//         fprintf(fid,":%s_%d_%d_%d:bundleinput:\n",style,id,j,k);
//         for (int l=0;l<bundleinputsize[j][k];l++){
//         fprintf(fid,"%d ",bundleinput[j][k][l]);
//         }
//         fprintf(fid,"\n");
//     }
//   }

//   for (int j=0;j<layers-1;j++){
//     for (int k=0;k<bundles[j];k++){
//         fprintf(fid,"netconstants:");
//         fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
//         for (i=1;i<n_body_type;i++) {
//             fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
//         }
//         fprintf(fid,":%s_%d_%d_%d:bundleoutput:\n",style,id,j,k);
//         for (int l=0;l<bundleoutputsize[j][k];l++){
//         fprintf(fid,"%d ",bundleoutput[j][k][l]);
//         }
//         fprintf(fid,"\n");
//     }
//   }

//   for (int j=0;j<layers-1;j++){
//     for (int k=0;k<bundles[j];k++){
//         fprintf(fid,"netconstants:");
//         fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
//         for (i=1;i<n_body_type;i++) {
//             fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
//         }
//         if (identitybundle[j][k])continue;
//         fprintf(fid,":%s_%d_%d_%d:weight:\n",style,id,j,k);
//         for (int l=0;l<bundleoutputsize[j][k];l++){
//             for (int m=0;m<bundleinputsize[j][k];m++){
//                 fprintf(fid,"%.15e\t",bundleW[j][k][l*bundleinputsize[j][k]+m]);
//             }
//             fprintf(fid,"\n");
//         }
//     }
//   }

//   for (int j=0;j<layers-1;j++){
//     for (int k=0;k<bundles[j];k++){
//         fprintf(fid,"netconstants:");
//         fprintf(fid,"%s",pair->elementsp[atomtypes[0]]);
//         for (i=1;i<n_body_type;i++) {
//             fprintf(fid,"_%s",pair->elementsp[atomtypes[i]]);
//         }
//         if (identitybundle[j][k])continue;
//         fprintf(fid,":%s_%d_%d_%d:bias:\n",style,id,j,k);
//         for (int l=0;l<bundleoutputsize[j][k];l++){
//             fprintf(fid,"%.15e\t",bundleB[j][k][l]);
//             fprintf(fid,"\n");
//         }
//     }
//   }
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
