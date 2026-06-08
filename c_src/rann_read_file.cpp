#include "pair_spin_rann.h"
#include "rann_activation.h"
#include "rann_fingerprint.h"
#include "rann_stateequation.h"
#include "rann_net.h"
using namespace LAMMPS_NS;


void PairRANN::read_file(char *filename)
{
  FILE *fp;
  int eof = 0;
  std::string line,line1;
  const int longline = 4096;
  int linenum=0;
  char linetemp[longline];
  std::string strtemp;
  char *ptr;
  std::vector<std::string> linev,line1v;
  fp = utils::open_potential(filename,lmp,nullptr);
  if (fp == nullptr) {errorf(FLERR,"Cannot open RANN potential file");}
  ptr=fgets(linetemp,longline,fp);
  linenum++;
  strtemp=utils::trim_comment(linetemp);
  while (strtemp.empty()) {
          ptr=fgets(linetemp,longline,fp);
          strtemp=utils::trim_comment(linetemp);
          linenum++;
  }
  line=strtemp;
  while (eof == 0) {
    ptr=fgets(linetemp,longline,fp);
    linenum++;
    if (ptr == nullptr) {
      fclose(fp);
      if (check_potential()) {
        errorf(FLERR,"Invalid syntax in potential file, values are inconsistent or missing");
      }
      else {
        eof = 1;
        break;
      }
    }
    strtemp=utils::trim_comment(linetemp);
    while (strtemp.empty()) {
        ptr=fgets(linetemp,longline,fp);
        strtemp=utils::trim_comment(linetemp);
        linenum++;
    }
    line1=linetemp;
    linev = tokenmaker(line,": ,\t_\n");
    line1v = tokenmaker(line1,": ,\t_\n");
    if (linev[0]=="atomtypes") read_atom_types(line1v,filename,linenum);
    else if (linev[0]=="mass") read_mass(linev,line1v,filename,linenum);
    else if (linev[0]=="fingerprintsperelement") read_fpe(linev,line1v,filename,linenum);
    else if (linev[0]=="fingerprints") read_fingerprints(linev,line1v,filename,linenum);
    else if (linev[0]=="fingerprintconstants") read_fingerprint_constants(linev,line1v,filename,linenum);
    else if (linev[0]=="netsperelement") read_npe(linev,line1v,filename,linenum);
    else if (linev[0]=="nets") read_nets(linev,line1v,filename,linenum);
    else if (linev[0]=="netconstants") read_net_constants(linev,line1v,fp,filename,&linenum);
    else if (linev[0]=="screening") read_screening(linev,line1v,filename,linenum);
    else if (linev[0]=="stateequationsperelement") read_eospe(linev,line1v,filename,linenum);
    else if (linev[0]=="stateequations") read_eos(linev,line1v,filename,linenum);
    else if (linev[0]=="stateequationconstants") read_eos_constants(linev,line1v,filename,linenum);
    else if (linev[0]=="calibrationparameters") {
        if (~is_lammps)read_parameters(linev,line1v,fp,filename,&linenum,linetemp);
    }
    //deprecated
    //else if (linev[0]=="networklayers") read_network_layers(linev,line1v,filename,linenum);
    //else if (linev[0]=="layersize") read_layer_size(linev,line1v,filename,linenum);
    //else if (linev[0]=="weight") read_weight(linev,line1v,fp,filename,&linenum);
    //else if (linev[0]=="bias") read_bias(linev,line1v,fp,filename,&linenum);
    //else if (linev[0]=="activationfunctions") read_activation_functions(linev,line1v,filename,linenum);
    //else if (linev[0]=="bundles")read_bundles(linev,line1v,filename,linenum);
    //else if (linev[0]=="bundleinput")read_bundle_input(linev,line1v,filename,linenum);
    //else if (linev[0]=="bundleoutput")read_bundle_output(linev,line1v,filename,linenum);
    //else if (linev[0]=="bundleid")read_bundle_id(linev,line1v,filename,linenum);

    else errorf(filename,linenum-1,"Could not understand file syntax: unknown keyword");
    ptr=fgets(linetemp,longline,fp);
    linenum++;
    strtemp=utils::trim_comment(linetemp);
    while (strtemp.empty()) {
        ptr=fgets(linetemp,longline,fp);
        strtemp=utils::trim_comment(linetemp);
        linenum++;
    }
    if (ptr == nullptr) {
      if (check_potential()) {
        errorf(FLERR,"Invalid syntax in potential file, values are inconsistent or missing");
      }
      else {
        eof = 1;
        break;
      }
    }
    line=linetemp;
  }
}

void PairRANN::read_atom_types(std::vector<std::string> line,char *filename,int linenum) {
  int nwords = line.size();
  if (nwords < 1) errorf(filename,linenum,"Incorrect syntax for atom types");
  nelements = nwords;
  line.emplace_back("all");
  allocate(line);
}

void PairRANN::read_mass(const std::vector<std::string> &line1, const std::vector<std::string> &line2, const char *filename,int linenum) {
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before mass in potential file.");
  for (int i=0;i<nelements;i++) {
    if (line1[1].compare(elements[i])==0) {
      mass[i]=utils::numeric(filename,linenum,line2[0],true,lmp);
      return;
    }
  }
  errorf(filename,linenum-1,"mass element not found in atom types.");
}

void PairRANN::read_fpe(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int i;
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before fingerprints per element in potential file.");
  for (i=0;i<nelementsp;i++) {
    if (line[1].compare(elementsp[i])==0) {
      fingerprintperelement[i] = utils::inumeric(filename,linenum,line1[0],true,lmp);
      fingerprints[i] = new RANN::Fingerprint *[fingerprintperelement[i]];
      for (int j=0;j<fingerprintperelement[i];j++) {
        fingerprints[i][j]=new RANN::Fingerprint(this);
      }
      return;
    }
  }
  errorf(filename,linenum-1,"fingerprint-per-element element not found in atom types");
}

void PairRANN::read_fingerprints(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int nwords1,nwords,i,j,k,i1,*atomtypes;
  bool found;
  nwords1 = line1.size();
  nwords = line.size();
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before fingerprints in potential file.");
  atomtypes = new int[nwords-1];
  for (i=1;i<nwords;i++) {
    found = false;
    for (j=0;j<nelementsp;j++) {
      if (line[i].compare(elementsp[j])==0) {
        atomtypes[i-1]=j;
        found = true;
        break;
      }
    }
    if (!found) {errorf(filename,linenum-1,"fingerprint element not found in atom types");}
  }
  i = atomtypes[0];
  k = 0;
  if (fingerprintperelement[i]==-1) {errorf(filename,linenum-1,"fingerprint per element must be defined before fingerprints");}
  while (k<nwords1) {
    i1 = fingerprintcount[i];
    if (i1>=fingerprintperelement[i]) {errorf(filename,linenum,"more fingerprints defined than fingerprint per element");}
    delete fingerprints[i][i1];
    fingerprints[i][i1] = create_fingerprint(line1[k].c_str());
    if (fingerprints[i][i1]->n_body_type!=nwords-1) {errorf(filename,linenum,"invalid fingerprint for element combination");}
    k++;
    fingerprints[i][i1]->init(atomtypes,utils::inumeric(filename,linenum,line1[k++],true,lmp));
    fingerprintcount[i]++;
  }
  delete[] atomtypes;
}

void PairRANN::read_fingerprint_constants(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int i,j,k,i1,*atomtypes;
  bool found;
  int nwords = line.size();
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before fingerprints in potential file.");
  int n_body_type = nwords-4;
  atomtypes = new int[n_body_type];
  for (i=1;i<=n_body_type;i++) {
    found = false;
    for (j=0;j<nelementsp;j++) {
      if (line[i].compare(elementsp[j])==0) {
        atomtypes[i-1]=j;
        found = true;
        break;
      }
    }
    if (!found) {errorf(filename,linenum-1,"fingerprint element not found in atom types");}
  }
  i = atomtypes[0];
  found = false;
  for (k=0;k<fingerprintperelement[i];k++) {
    if (fingerprints[i][k]->empty) {continue;}
    if (n_body_type!=fingerprints[i][k]->n_body_type) {continue;}
    for (j=0;j<n_body_type;j++) {
      if (fingerprints[i][k]->atomtypes[j]!=atomtypes[j]) {break;}
      if (j==n_body_type-1) {
        if (line[nwords-3].compare(fingerprints[i][k]->style)==0 && utils::inumeric(filename,linenum,line[nwords-2],true,lmp)==fingerprints[i][k]->id) {
          found=true;
          i1 = k;
          break;
        }
      }
    }
    if (found) {break;}
  }
  if (!found) {errorf(filename,linenum-1,"cannot define constants for unknown fingerprint");}
  fingerprints[i][i1]->fullydefined=fingerprints[i][i1]->parse_values(line[nwords-1],line1);
  delete[] atomtypes;
}

void PairRANN::read_npe(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int i;
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before nets per element in potential file.");
  for (i=0;i<nelementsp;i++) {
    if (line[1].compare(elementsp[i])==0) {
      netperelement[i] = utils::inumeric(filename,linenum,line1[0],true,lmp);
      nets[i] = new RANN::Net *[netperelement[i]];
      for (int j=0;j<netperelement[i];j++) {
        nets[i][j]=new RANN::Net(this);
      }
      return;
    }
  }
  errorf(filename,linenum-1,"net-per-element element not found in atom types");
}

void PairRANN::read_nets(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int nwords1,nwords,i,j,k,i1,*atomtypes;
  bool found;
  nwords1 = line1.size();
  nwords = line.size();
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before nets in potential file.");
  atomtypes = new int[nwords-1];
  for (i=1;i<nwords;i++) {
    found = false;
    for (j=0;j<nelementsp;j++) {
      if (line[i].compare(elementsp[j])==0) {
        atomtypes[i-1]=j;
        found = true;
        break;
      }
    }
    if (!found) {errorf(filename,linenum-1,"net element not found in atom types");}
  }
  i = atomtypes[0];
  k = 0;
  if (netperelement[i]==-1) {errorf(filename,linenum-1,"net per element must be defined before fingerprints");}
  while (k<nwords1) {
    i1 = netcount[i];
    if (i1>=netperelement[i]) {errorf(filename,linenum,"more nets defined than fingerprint per element");}
    delete nets[i][i1];
    nets[i][i1] = create_net(line1[k].c_str());
    if (nets[i][i1]->n_body_type!=nwords-1) {
      printf("n_body_type:%d, nwords-1:%d,style:%s\n",nets[i][i1]->n_body_type,nwords-1,nets[i][i1]->style);
      errorf(filename,linenum,"invalid net for element combination");}
    k++;
    nets[i][i1]->init(atomtypes,utils::inumeric(filename,linenum,line1[k++],true,lmp));
    netcount[i]++;
  }
  delete[] atomtypes;
}

void PairRANN::read_net_constants(std::vector<std::string> line,std::vector<std::string> line1,FILE *fp,char *filename,int *linenum) {
  int i,j,k,i1,*atomtypes;
  bool found;
  int nwords = line.size();
  if (nelements == -1)errorf(filename,*linenum-1,"atom types must be defined before networks in potential file.");
  int n_body_type = nwords-4;
  n_body_type = 1;//fix later if needed
  atomtypes = new int[n_body_type];
  for (i=1;i<=n_body_type;i++) {
    found = false;
    for (j=0;j<nelementsp;j++) {
      if (line[i].compare(elementsp[j])==0) {
        atomtypes[i-1]=j;
        found = true;
        break;
      }
    }
    if (!found) {errorf(filename,*linenum-1,"network element not found in atom types");}
  }
  i = atomtypes[0];
  found = false;
  for (k=0;k<netperelement[i];k++) {
    if (nets[i][k]->empty) {continue;}
    if (n_body_type!=nets[i][k]->n_body_type) {continue;}
    for (j=0;j<n_body_type;j++) {
      if (nets[i][k]->atomtypes[j]!=atomtypes[j]) {break;}
      if (j==n_body_type-1) {
        if (line[2].compare(nets[i][k]->style)==0 && utils::inumeric(filename,*linenum,line[3],true,lmp)==nets[i][k]->id) {
          found=true;
          i1 = k;
          break;
        }
      }
    }
    if (found) {break;}
  }
  if (!found) {errorf(filename,*linenum-1,"cannot define constants for unknown network");}
  nets[i][i1]->fullydefined=nets[i][i1]->parse_values(line,line1,fp,filename,linenum);
  delete[] atomtypes;
}

// void PairRANN::read_network_layers(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
//   int i,j;
//   if (nelements == -1)errorf(FLERR,"atom types must be defined before network layers in potential file.");
//   for (i=0;i<nelements;i++){
//     if (line[1].compare(elements[i])==0){
//       net[i].layers = utils::inumeric(filename,linenum,line1[0],true,lmp);
//       if (net[i].layers < 1)errorf(filename,linenum,"invalid number of network layers");
//       weightdefined[i] = new bool *[net[i].layers];
//       biasdefined[i] = new bool *[net[i].layers];
//       dimensiondefined[i] = new bool [net[i].layers];
//       bundle_inputdefined[i] = new bool *[net[i].layers];
//       bundle_outputdefined[i] = new bool *[net[i].layers];
//       net[i].dimensions = new int [net[i].layers];
//       net[i].bundles = new int[net[i].layers];
//       net[i].identitybundle = new bool *[net[i].layers];
//       net[i].bundleinputsize = new int *[net[i].layers];
//       net[i].bundleoutputsize = new int *[net[i].layers];
//       net[i].bundleinput = new int **[net[i].layers];
//       net[i].bundleoutput = new int **[net[i].layers];
//       net[i].bundleW = new double **[net[i].layers];
//       net[i].bundleB = new double **[net[i].layers];
//       net[i].freezeW = new bool **[net[i].layers];
//       net[i].freezeB = new bool **[net[i].layers];
//       for (j=0;j<net[i].layers;j++){
//         net[i].dimensions[j]=0;
//         net[i].bundles[j] = 1;
//         net[i].bundleinputsize[j] = new int [net[i].bundles[j]];
//         net[i].bundleoutputsize[j] = new int [net[i].bundles[j]];
//         net[i].bundleinput[j] = new int *[net[i].bundles[j]];
//         net[i].bundleoutput[j] = new int *[net[i].bundles[j]];
//         net[i].bundleW[j] = new double *[net[i].bundles[j]];
//         net[i].bundleB[j] = new double *[net[i].bundles[j]];
//         net[i].freezeW[j] = new bool *[net[i].bundles[j]];
//         net[i].freezeB[j] = new bool *[net[i].bundles[j]];
//         net[i].identitybundle[j] = new bool[net[i].bundles[j]];
//         weightdefined[i][j] = new bool[net[i].bundles[j]];
//         biasdefined[i][j] = new bool[net[i].bundles[j]];
//         dimensiondefined[i][j]=false;
//         bundle_inputdefined[i][j]=new bool [net[i].bundles[j]];
//         bundle_outputdefined[i][j]=new bool [net[i].bundles[j]];
//         for (int k=0;k<net[i].bundles[j];k++){
//           weightdefined[i][j][k]=false;
//           biasdefined[i][j][k]=false;
//           bundle_inputdefined[i][j][k]=false;
//           bundle_outputdefined[i][j][k]=false;
//           net[i].identitybundle[j][k]=false;
//         }
//       }
//       activation[i]=new RANN::Activation** [net[i].layers-1];
// 	  for (int j=0;j<net[i].layers-1;j++) {
//         activation[i][j]= new RANN::Activation*;
//       }
//       return;
//     }
//   }
//   errorf(filename,linenum-1,"network layers element not found in atom types");
// }

// void PairRANN::read_layer_size(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
//   int i;
//   for (i=0;i<nelements;i++) {
//     if (line[1].compare(elements[i])==0) {
//       if (net[i].layers==0)errorf(filename,linenum-1,"networklayers for each atom type must be defined before the corresponding layer sizes.");
//       int j = utils::inumeric(filename,linenum,line[2],true,lmp);
//       if (j>=net[i].layers || j<0) {errorf(filename,linenum,"invalid layer in layer size definition");};
//       net[i].dimensions[j]= utils::inumeric(filename,linenum,line1[0],true,lmp);
//       dimensiondefined[i][j]=true;
//       if (j>0){
//         activation[i][j-1] = new RANN::Activation*[net[i].dimensions[j]];
//         for (int k=0;k<net[i].dimensions[j];k++){
//           activation[i][j-1][k] = new RANN::Activation(this);
//         }
//       }
//       return;
//     }
//   }
//   errorf(filename,linenum-1,"layer size element not found in atom types");
// }

// void PairRANN::read_exnetwork_layers(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
//   int i,j,exnum;
//   if (nelements == -1)errorf(FLERR,"atom types must be defined before network layers in potential file.");
//   for (i=0;i<nelements;i++){
//     if (line[1].compare(elements[i])==0){
//       exnum = utils::inumeric(filename,linenum,line[2],true,lmp);
//       exnet[i][exnum].layers = utils::inumeric(filename,linenum,line1[0],true,lmp);
//       if (exnet[i][exnum].layers < 1)errorf(filename,linenum,"invalid number of exnetwork layers");
//       weightdefinedex[i][exnum] = new bool *[exnet[i][exnum].layers];
//       biasdefinedex[i][exnum] = new bool *[exnet[i][exnum].layers];
//       dimensiondefinedex[i][exnum] = new bool [exnet[i][exnum].layers];
//       bundle_inputdefinedex[i][exnum] = new bool *[exnet[i][exnum].layers];
//       bundle_outputdefinedex[i][exnum] = new bool *[exnet[i][exnum].layers];
//       exnet[i][exnum].dimensions = new int [exnet[i][exnum].layers];
//       exnet[i][exnum].bundles = new int[exnet[i][exnum].layers];
//       exnet[i][exnum].identitybundle = new bool *[exnet[i][exnum].layers];
//       exnet[i][exnum].bundleinputsize = new int *[exnet[i][exnum].layers];
//       exnet[i][exnum].bundleoutputsize = new int *[exnet[i][exnum].layers];
//       exnet[i][exnum].bundleinput = new int **[exnet[i][exnum].layers];
//       exnet[i][exnum].bundleoutput = new int **[exnet[i][exnum].layers];
//       exnet[i][exnum].bundleW = new double **[exnet[i][exnum].layers];
//       exnet[i][exnum].bundleB = new double **[exnet[i][exnum].layers];
//       exnet[i][exnum].freezeW = new bool **[exnet[i][exnum].layers];
//       exnet[i][exnum].freezeB = new bool **[exnet[i][exnum].layers];
//       for (j=0;j<exnet[i][exnum].layers;j++){
//         exnet[i][exnum].dimensions[j]=0;
//         exnet[i][exnum].bundles[j] = 1;
//         exnet[i][exnum].bundleinputsize[j] = new int [exnet[i][exnum].bundles[j]];
//         exnet[i][exnum].bundleoutputsize[j] = new int [exnet[i][exnum].bundles[j]];
//         exnet[i][exnum].bundleinput[j] = new int *[exnet[i][exnum].bundles[j]];
//         exnet[i][exnum].bundleoutput[j] = new int *[exnet[i][exnum].bundles[j]];
//         exnet[i][exnum].bundleW[j] = new double *[exnet[i][exnum].bundles[j]];
//         exnet[i][exnum].bundleB[j] = new double *[exnet[i][exnum].bundles[j]];
//         exnet[i][exnum].freezeW[j] = new bool *[exnet[i][exnum].bundles[j]];
//         exnet[i][exnum].freezeB[j] = new bool *[exnet[i][exnum].bundles[j]];
//         exnet[i][exnum].identitybundle[j] = new bool[exnet[i][exnum].bundles[j]];
//         weightdefinedex[i][exnum][j] = new bool[exnet[i][exnum].bundles[j]];
//         biasdefinedex[i][exnum][j] = new bool[exnet[i][exnum].bundles[j]];
//         dimensiondefinedex[i][exnum][j]=false;
//         bundle_inputdefinedex[i][exnum][j]=new bool [exnet[i][exnum].bundles[j]];
//         bundle_outputdefinedex[i][exnum][j]=new bool [exnet[i][exnum].bundles[j]];
//         for (int k=0;k<exnet[i][exnum].bundles[j];k++){
//           weightdefinedex[i][exnum][j][k]=false;
//           biasdefinedex[i][exnum][j][k]=false;
//           bundle_inputdefinedex[i][exnum][j][k]=false;
//           bundle_outputdefinedex[i][exnum][j][k]=false;
//           exnet[i][exnum].identitybundle[j][k]=false;
//         }
//       }
//       exactivation[i][exnum]=new RANN::Activation** [exnet[i][exnum].layers-1];
// 	  for (int j=0;j<exnet[i][exnum].layers-1;j++) {
// 	    exactivation[i][exnum][j]= new RANN::Activation*;
//       }
//       return;
//     }
//   }
//   errorf(filename,linenum-1,"exnetwork layers element not found in atom types");
// }

// void PairRANN::read_exlayer_size(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
//   int i,exnum;
//   for (i=0;i<nelements;i++) {
//     if (line[1].compare(elements[i])==0) {
//       int exnum = utils::inumeric(filename,linenum,line[2],true,lmp);      
//       if (exnet[i][exnum].layers==0)errorf(filename,linenum-1,"exnetworklayers for each atom type must be defined before the corresponding layer sizes.");
//       int j = utils::inumeric(filename,linenum,line[3],true,lmp);
//       if (j>=exnet[i][exnum].layers || j<0) {errorf(filename,linenum,"invalid layer in exlayer size definition");};
//       exnet[i][exnum].dimensions[j]= utils::inumeric(filename,linenum,line1[0],true,lmp);
//       dimensiondefinedex[i][exnum][j]=true;
//       if (j>0){
//         exactivation[i][exnum][j-1] = new RANN::Activation*[exnet[i][exnum].dimensions[j]];
//         for (int k=0;k<exnet[i][exnum].dimensions[j];k++){
//           exactivation[i][exnum][j-1][k] = new RANN::Activation(this);
//         }
//       }
//       return;
//     }
//   }
//   errorf(filename,linenum-1,"layer size element not found in atom types");
// }

// void PairRANN::read_bundles(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum){
//   int i;
//   for (i=0;i<nelements;i++){
//     if (line[1].compare(elements[i])==0){
//       if (net[i].layers==0)errorf(filename,linenum-1,"networklayers for each atom type must be defined before the corresponding layer sizes.");
//       int j = utils::inumeric(filename,linenum,line[2],true,lmp);
//       if (j>=net[i].layers || j<0){errorf(filename,linenum-1,"invalid layer in bundles definition");};
//       net[i].bundles[j]= utils::inumeric(filename,linenum,line1[0],true,lmp);
//       delete [] net[i].bundleinputsize[j];
//       delete [] net[i].bundleoutputsize[j];
//       delete [] net[i].bundleinput[j];
//       delete [] net[i].bundleoutput[j];
//       delete [] net[i].bundleW[j];
//       delete [] net[i].bundleB[j];
//       delete [] net[i].freezeW[j];
//       delete [] net[i].freezeB[j];
//       delete [] net[i].identitybundle[j];
//       delete [] weightdefined[i][j];
//       delete [] biasdefined[i][j];
//       delete [] bundle_inputdefined[i][j];
//       delete [] bundle_outputdefined[i][j];
//       net[i].bundleinputsize[j] = new int [net[i].bundles[j]];
//       net[i].bundleoutputsize[j] = new int [net[i].bundles[j]];
//       net[i].bundleinput[j] = new int *[net[i].bundles[j]];
//       net[i].bundleoutput[j] = new int *[net[i].bundles[j]];
//       net[i].bundleW[j] = new double *[net[i].bundles[j]];
//       net[i].bundleB[j] = new double *[net[i].bundles[j]];
//       net[i].freezeW[j] = new bool *[net[i].bundles[j]];
//       net[i].freezeB[j] = new bool *[net[i].bundles[j]];
//       net[i].identitybundle[j] = new bool[net[i].bundles[j]];
//       weightdefined[i][j] = new bool[net[i].bundles[j]];
//       biasdefined[i][j] = new bool[net[i].bundles[j]];
//       bundle_inputdefined[i][j] = new bool[net[i].bundles[j]];
//       bundle_outputdefined[i][j] = new bool[net[i].bundles[j]];
//       for (int k=0;k<net[i].bundles[j];k++){
//         weightdefined[i][j][k]=false;
//         biasdefined[i][j][k]=false;
//         bundle_inputdefined[i][j][k]=false;
//         bundle_outputdefined[i][j][k]=false;
//         net[i].identitybundle[j][k]=false;
//       }
//       return;
//     }
//   }
//   errorf(FLERR,"bundle element not found in atom types");
// }

// void PairRANN::read_bundle_id(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum){
//   int i;
//   for (i=0;i<nelements;i++){
//     if (line[1].compare(elements[i])==0){
//       if (net[i].layers==0)errorf(filename,linenum-1,"networklayers for each atom type must be defined before the corresponding layer sizes.");
//       int j = utils::inumeric(filename,linenum,line[2],true,lmp);
//       if (j>=net[i].layers || j<0){errorf(filename,linenum-1,"invalid layer in bundle inputs definition");};
//       int b = utils::inumeric(filename,linenum,line[3],true,lmp);
//       if (b>net[i].bundles[j] || b<0){errorf(filename,linenum-1,"invalid bundle in bundle inputs");}
//       int id =utils::inumeric(filename,linenum,line1[0],true,lmp);
//       if (id!= 1 && id != 0)errorf(filename,linenum-1,"bundle id must be 1 or 0\n");
//       net[i].identitybundle[j][b]=id;
//       return;
//     }
//   }
//   errorf(filename,linenum-1,"bundle id element not found in atom types");
// }

// void PairRANN::read_bundle_input(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum){
//   int i,nwords;
//   for (i=0;i<nelements;i++){
//     if (line[1].compare(elements[i])==0){
//       if (net[i].layers==0)errorf(filename,linenum-1,"networklayers for each atom type must be defined before the corresponding layer sizes.");
//       int j = utils::inumeric(filename,linenum,line[2],true,lmp);
//       if (j>=net[i].layers || j<0){errorf(filename,linenum-1,"invalid layer in bundle inputs definition");};
//       int b = utils::inumeric(filename,linenum,line[3],true,lmp);
//       if (b>net[i].bundles[j] || b<0){errorf(filename,linenum-1,"invalid bundle in bundle inputs");}
//       nwords = line1.size();
//       net[i].bundleinputsize[j][b] = nwords;
//       net[i].bundleinput[j][b] = new int [nwords];
//       for (int k=0;k<nwords;k++){
//         net[i].bundleinput[j][b][k]=utils::inumeric(filename,linenum,line1[k],true,lmp);
//         if (net[i].bundleinput[j][b][k]>net[i].dimensions[j] || net[i].bundleinput[j][b][k]<0){errorf(filename,linenum-1,"bundle input neurons exceed layer size");}
//       }
//       bundle_inputdefined[i][j][b]=true;
//       return;
//     }
//   }
//   errorf(filename,linenum-1,"bundle input element not found in atom types");
// }

// void PairRANN::read_bundle_output(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum){
//   int i,nwords;
//   for (i=0;i<nelements;i++){
//     if (line[1].compare(elements[i])==0){
//       if (net[i].layers==0)errorf(filename,linenum-1,"networklayers for each atom type must be defined before the corresponding layer sizes.");
//       int j = utils::inumeric(filename,linenum,line[2],true,lmp);
//       if (j>=net[i].layers || j<0){errorf(filename,linenum-1,"invalid layer in bundle inputs definition");};
//       int b = utils::inumeric(filename,linenum,line[3],true,lmp);
//       if (b>net[i].bundles[j] || b<0){errorf(filename,linenum-1,"invalid bundle in bundle inputs");}
//       nwords = line1.size();
//       net[i].bundleoutputsize[j][b] = nwords;
//       net[i].bundleoutput[j][b] = new int [nwords];
//       for (int k=0;k<nwords;k++){
//         net[i].bundleoutput[j][b][k]=utils::inumeric(filename,linenum,line1[k],true,lmp);
//         if (net[i].bundleoutput[j][b][k]>net[i].dimensions[j+1] || net[i].bundleoutput[j][b][k]<0)errorf(filename,linenum-1,"bundle output neurons exceed layer size");
//       }
//       bundle_outputdefined[i][j][b]=true;
//       return;
//     }
//   }
//   errorf(filename,linenum-1,"bundle output element not found in atom types");
// }

// void PairRANN::read_weight(std::vector<std::string> line,std::vector<std::string> line1,FILE* fp,char *filename,int *linenum){
//   int i,j,k,l,b,ins,ops,nwords;
//   char *ptr;
//   char **words1;
//   int longline = 4096;
//   char linetemp [longline];
//   nwords = line.size();
//   if (nwords == 3){b=0;}
//   else if (nwords>3){b = utils::inumeric(filename,*linenum,line[3],true,lmp);}
//   for (l=0;l<nelements;l++){
//     if (line[1].compare(elements[l])==0){
//       if (net[l].layers==0)errorf(filename,*linenum-1,"networklayers must be defined before weights.");
//       i=utils::inumeric(filename,*linenum,line[2],true,lmp);
//       if (i>=net[l].layers || i<0)errorf(filename,*linenum-1,"invalid weight layer");
//       if (dimensiondefined[l][i]==false || dimensiondefined[l][i+1]==false) errorf(FLERR,"network layer sizes must be defined before corresponding weight");
//       if (bundle_inputdefined[l][i][b]==false && b!=0) errorf(filename,*linenum-1,"bundle inputs must be defined before weights");
//       if (bundle_outputdefined[l][i][b]==false && b!=0) errorf(filename,*linenum-1,"bundle outputs must be defined before weights");
//       if (net[l].identitybundle[i][b]) errorf(filename,*linenum-1,"cannot define weights for an identity bundle");
//       if (bundle_inputdefined[l][i][b]==false){ins = net[l].dimensions[i];}
//       else {ins = net[l].bundleinputsize[i][b];}
//       if (bundle_outputdefined[l][i][b]==false){ops = net[l].dimensions[i+1];}
//       else {ops = net[l].bundleoutputsize[i][b];}
//       net[l].bundleW[i][b] = new double [ins*ops];
//       net[l].freezeW[i][b] = new bool [ins*ops];
//       for (j=0;j<ins*ops;j++){net[l].freezeW[i][b][j]=0;}
//       weightdefined[l][i][b] = true;
// 	    nwords = line1.size();
//       if (nwords != ins)errorf(filename,*linenum-1,"invalid weights per line");
//       for (k=0;k<ins;k++){
//         net[l].bundleW[i][b][k] = utils::numeric(filename,*linenum,line1[k],true,lmp);
//       }
//       for (j=1;j<ops;j++){
//         ptr = fgets(linetemp,longline,fp);
//         (*linenum)++;
//         line1 = tokenmaker(linetemp,": ,\t_\n");
//         if (ptr==nullptr)errorf(filename,*linenum-1,"unexpected end of potential file!");
//         nwords = line1.size();
//         if (nwords != ins)errorf(filename,*linenum-1,"invalid weights per line");
//         for (k=0;k<ins;k++) {
//           net[l].bundleW[i][b][j*ins+k] = utils::numeric(filename,*linenum,line1[k],true,lmp);
//         }
//       }
//       return;
//     }
//   }
//   errorf(FLERR,"weight element not found in atom types");
// }

// void PairRANN::read_bias(std::vector<std::string> line,std::vector<std::string> line1,FILE* fp,char *filename,int *linenum){
//   int i,j,l,b,ops;
//   char *ptr;
//   int longline = 1024;
//   int nwords = line.size();
//   char linetemp [longline];
//     if (nwords == 3){b=0;}
//     else if (nwords>3){b = utils::inumeric(filename,*linenum,line[3],true,lmp);}
//   for (l=0;l<nelements;l++){
//     if (line[1].compare(elements[l])==0){
//       if (net[l].layers==0)errorf(filename,*linenum-1,"networklayers must be defined before biases.");
//       i=utils::inumeric(filename,*linenum,line[2],true,lmp);
//       if (i>=net[l].layers || i<0)errorf(filename,*linenum-1,"invalid bias layer");
//       if (dimensiondefined[l][i]==false) errorf(filename,*linenum-1,"network layer sizes must be defined before corresponding bias");
//       if (bundle_outputdefined[l][i][b]==false && b!=0) errorf(filename,*linenum-1,"bundle outputs must be defined before bias");
//       if (net[l].identitybundle[i][b]) errorf(filename,*linenum-1,"cannot define bias for an identity bundle");
//       if (bundle_outputdefined[l][i][b]==false){ops=net[l].dimensions[i+1];}
//       else {ops = net[l].bundleoutputsize[i][b];}
//       net[l].bundleB[i][b] = new double [ops];
//       net[l].freezeB[i][b] = new bool[ops];
//       for (j=0;j<ops;j++){net[l].freezeB[i][b][j]=0;}
//       biasdefined[l][i][b] = true;
//       net[l].bundleB[i][b][0] = utils::numeric(filename,*linenum,line1[0],true,lmp);
//       for (j=1;j<ops;j++){
// 		    ptr=fgets(linetemp,longline,fp);
//         if (ptr==nullptr)errorf(filename,*linenum,"unexpected end of potential file!");
//         (*linenum)++;
//         line1 = tokenmaker(linetemp,": ,\t_\n");
//         net[l].bundleB[i][b][j] = utils::numeric(filename,*linenum,line1[0],true,lmp);
//       }
//       return;
//     }
//   }
//   errorf(filename,*linenum-1,"bias element not found in atom types");
// }

// void PairRANN::read_activation_functions(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum){
//   int i,l;
//   int nwords = line.size();
//   for (l=0;l<nelements;l++){
//     if (line[1].compare(elements[l])==0){
//       i = utils::inumeric(filename,linenum,line[2],true,lmp);
//       if (i>=net[l].layers || i<0)errorf(filename,linenum-1,"invalid activation layer");
//       if (dimensiondefined[l][i+1]==false) errorf(filename,linenum-1,"network layer sizes must be defined before corresponding activation");
//       if (nwords==3){
//         for (int j = 0;j<net[l].dimensions[i+1];j++){
//           delete activation[l][i][j];
//           activation[l][i][j]=create_activation(line1[0].c_str());
//         }
// 	    }
// 	    else if (nwords>3){
// 		    int j = utils::inumeric(filename,linenum,line[3],true,lmp);
// 		    delete activation[l][i][j];
// 		    activation[l][i][j]=create_activation(line1[0].c_str());
// 	    }
//       return;
//     }
//   }
//   errorf(filename,linenum-1,"activation function element not found in atom types");
// }

// void PairRANN::read_exweight(std::vector<std::string> line,std::vector<std::string> line1,FILE* fp,char *filename,int *linenum){
//   int i,j,k,l,b,ins,ops,nwords,exnum;
//   char *ptr;
//   char **words1;
//   int longline = 4096;
//   char linetemp [longline];
//   nwords = line.size();
//   if (nwords == 4){b=0;}
//   else if (nwords>4){b = utils::inumeric(filename,*linenum,line[4],true,lmp);}
//   for (l=0;l<nelements;l++){
//     exnum=utils::inumeric(filename,*linenum,line[2],true,lmp);
//     if (line[1].compare(elements[l])==0){
//       if (exnet[l][exnum].layers==0)errorf(filename,*linenum-1,"exnetworklayers must be defined before exweights.");
//       i=utils::inumeric(filename,*linenum,line[3],true,lmp);
//       if (i>=exnet[l][exnum].layers || i<0)errorf(filename,*linenum-1,"invalid exweight layer");
//       if (dimensiondefinedex[l][exnum][i]==false || dimensiondefinedex[l][exnum][i+1]==false) errorf(FLERR,"exnetwork layer sizes must be defined before corresponding exweight");
//       if (bundle_inputdefinedex[l][exnum][i][b]==false && b!=0) errorf(filename,*linenum-1,"exbundle inputs must be defined before exweights");
//       if (bundle_outputdefinedex[l][exnum][i][b]==false && b!=0) errorf(filename,*linenum-1,"exbundle outputs must be defined before exweights");
//       if (exnet[l][exnum].identitybundle[i][b]) errorf(filename,*linenum-1,"cannot define exweights for an identity bundle");
//       if (bundle_inputdefinedex[l][exnum][i][b]==false){ins = exnet[l][exnum].dimensions[i];}
//       else {ins = exnet[l][exnum].bundleinputsize[i][b];}
//       if (bundle_outputdefinedex[l][exnum][i][b]==false){ops = exnet[l][exnum].dimensions[i+1];}
//       else {ops = exnet[l][exnum].bundleoutputsize[i][b];}
//       exnet[l][exnum].bundleW[i][b] = new double [ins*ops];
//       exnet[l][exnum].freezeW[i][b] = new bool [ins*ops];
//       for (j=0;j<ins*ops;j++){exnet[l][exnum].freezeW[i][b][j]=0;}
//       weightdefinedex[l][exnum][i][b] = true;
// 	    nwords = line1.size();
//       if (nwords != ins)errorf(filename,*linenum-1,"invalid exweights per line");
//       for (k=0;k<ins;k++){
//         exnet[l][exnum].bundleW[i][b][k] = utils::numeric(filename,*linenum,line1[k],true,lmp);
//       }
//       for (j=1;j<ops;j++){
//         ptr = fgets(linetemp,longline,fp);
//         (*linenum)++;
//         line1 = tokenmaker(linetemp,": ,\t_\n");
//         if (ptr==nullptr)errorf(filename,*linenum-1,"unexpected end of potential file!");
//         nwords = line1.size();
//         if (nwords != ins)errorf(filename,*linenum-1,"invalid exweights per line");
//         for (k=0;k<ins;k++) {
//           exnet[l][exnum].bundleW[i][b][j*ins+k] = utils::numeric(filename,*linenum,line1[k],true,lmp);
//         }
//       }
//       return;
//     }
//   }
//   errorf(FLERR,"exweight element not found in atom types");
// }

// void PairRANN::read_exbias(std::vector<std::string> line,std::vector<std::string> line1,FILE* fp,char *filename,int *linenum){
//   int i,j,l,b,ops,exnum;
//   char *ptr;
//   int longline = 1024;
//   int nwords = line.size();
//   char linetemp [longline];
//     if (nwords == 4){b=0;}
//     else if (nwords>4){b = utils::inumeric(filename,*linenum,line[4],true,lmp);}
//   for (l=0;l<nelements;l++){
//     exnum=utils::inumeric(filename,*linenum,line[2],true,lmp);
//     if (line[1].compare(elements[l])==0){
//       if (exnet[l][exnum].layers==0)errorf(filename,*linenum-1,"exnetworklayers must be defined before biases.");
//       i=utils::inumeric(filename,*linenum,line[3],true,lmp);
//       if (i>=exnet[l][exnum].layers || i<0)errorf(filename,*linenum-1,"invalid exbias layer");
//       if (dimensiondefinedex[l][exnum][i]==false) errorf(filename,*linenum-1,"exnetwork layer sizes must be defined before corresponding exbias");
//       if (bundle_outputdefinedex[l][exnum][i][b]==false && b!=0) errorf(filename,*linenum-1,"bundle outputs must be defined before exbias");
//       if (exnet[l][exnum].identitybundle[i][b]) errorf(filename,*linenum-1,"cannot define exbias for an identity bundle");
//       if (bundle_outputdefinedex[l][exnum][i][b]==false){ops=exnet[l][exnum].dimensions[i+1];}
//       else {ops = exnet[l][exnum].bundleoutputsize[i][b];}
//       exnet[l][exnum].bundleB[i][b] = new double [ops];
//       exnet[l][exnum].freezeB[i][b] = new bool[ops];
//       for (j=0;j<ops;j++){exnet[l][exnum].freezeB[i][b][j]=0;}
//       biasdefinedex[l][exnum][i][b] = true;
//       exnet[l][exnum].bundleB[i][b][0] = utils::numeric(filename,*linenum,line1[0],true,lmp);
//       for (j=1;j<ops;j++){
// 		    ptr=fgets(linetemp,longline,fp);
//         if (ptr==nullptr)errorf(filename,*linenum,"unexpected end of potential file!");
//         (*linenum)++;
//         line1 = tokenmaker(linetemp,": ,\t_\n");
//         exnet[l][exnum].bundleB[i][b][j] = utils::numeric(filename,*linenum,line1[0],true,lmp);
//       }
//       return;
//     }
//   }
//   errorf(filename,*linenum-1,"exbias element not found in atom types");
// }

// void PairRANN::read_exactivation_functions(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum){
//   int i,l,exnum;
//   int nwords = line.size();
//   for (l=0;l<nelements;l++){
//     exnum=utils::inumeric(filename,linenum,line[2],true,lmp);
//     if (line[1].compare(elements[l])==0){
//       i = utils::inumeric(filename,linenum,line[3],true,lmp);
//       if (i>=exnet[l][exnum].layers || i<0)errorf(filename,linenum-1,"invalid exactivation layer");
//       if (dimensiondefinedex[l][exnum][i+1]==false) errorf(filename,linenum-1,"exnetwork layer sizes must be defined before corresponding exactivation");
//       if (nwords==4){
//         for (int j = 0;j<exnet[l][exnum].dimensions[i+1];j++){
//           delete exactivation[l][exnum][i][j];
//           exactivation[l][exnum][i][j]=create_activation(line1[0].c_str());
//         }
// 	    }
// 	    else if (nwords>4){
// 		    int j = utils::inumeric(filename,linenum,line[4],true,lmp);
// 		    delete exactivation[l][exnum][i][j];
// 		    exactivation[l][exnum][i][j]=create_activation(line1[0].c_str());
// 	    }
//       return;
//     }
//   }
//   errorf(filename,linenum-1,"activation function element not found in atom types");
// }

void PairRANN::read_eospe(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int i;
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before equations of state per element in potential file.");
  for (i=0;i<nelementsp;i++) {
    if (line[1].compare(elementsp[i])==0) {
      stateequationperelement[i] = utils::inumeric(filename,linenum,line1[0],true,lmp);
      state[i] = new RANN::State *[stateequationperelement[i]];
      for (int j=0;j<stateequationperelement[i];j++) {
        state[i][j]=new RANN::State(this);
      }
      return;
    }
  }
  errorf(filename,linenum-1,"state equations-per-element element not found in atom types");
}

void PairRANN::read_eos(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int nwords1,nwords,i,j,k,i1,*atomtypes;
  bool found;
  nwords1 = line1.size();
  nwords = line.size();
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before state equations in potential file.");
  atomtypes = new int[nwords-1];
  for (i=1;i<nwords;i++) {
    found = false;
    for (j=0;j<nelementsp;j++) {
      if (line[i].compare(elementsp[j])==0) {
        atomtypes[i-1]=j;
        found = true;
        break;
      }
    }
    if (!found) {errorf(filename,linenum-1,"state equation element not found in atom types");}
  }
  i = atomtypes[0];
  k = 0;
  if (stateequationperelement[i]==-1) {errorf(filename,linenum-1,"eos per element must be defined before fingerprints");}
  while (k<nwords1) {
    i1 = stateequationcount[i];
    if (i1>=stateequationperelement[i]) {errorf(filename,linenum,"more state equations defined than eos per element");}
    delete state[i][i1];
    state[i][i1] = create_state(line1[k].c_str());
    if (state[i][i1]->n_body_type!=nwords-1) {errorf(filename,linenum,"invalid state equation for element combination");}
    k++;
    state[i][i1]->init(atomtypes,utils::inumeric(filename,linenum,line1[k++],true,lmp));
    stateequationcount[i]++;
  }
  delete[] atomtypes;
}


void PairRANN::read_eos_constants(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int i,j,k,i1,*atomtypes;
  bool found;
  int nwords = line.size();
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before state equations in potential file.");
  int n_body_type = nwords-4;
  atomtypes = new int[n_body_type];
  for (i=1;i<=n_body_type;i++) {
    found = false;
    for (j=0;j<nelementsp;j++) {
      if (line[i].compare(elementsp[j])==0) {
        atomtypes[i-1]=j;
        found = true;
        break;
      }
    }
    if (!found) {errorf(filename,linenum-1,"equation of state element not found in atom types");}
  }
  i = atomtypes[0];
  found = false;
  for (k=0;k<stateequationperelement[i];k++) {
    if (state[i][k]->empty) {continue;}
    if (n_body_type!=state[i][k]->n_body_type) {continue;}
    for (j=0;j<n_body_type;j++) {
      if (state[i][k]->atomtypes[j]!=atomtypes[j]) {break;}
      if (j==n_body_type-1) {
        if (line[nwords-3].compare(state[i][k]->style)==0 && utils::inumeric(filename,linenum,line[nwords-2],true,lmp)==state[i][k]->id) {
          found=true;
          i1 = k;
          break;
        }
      }
    }
    if (found) {break;}
  }
  if (!found) {errorf(filename,linenum-1,"cannot define constants for unknown equation of state");}
  state[i][i1]->fullydefined=state[i][i1]->parse_values(line[nwords-1],line1);
  delete[] atomtypes;
}

void PairRANN::read_screening(std::vector<std::string> line,std::vector<std::string> line1,char *filename,int linenum) {
  int i,j,k,*atomtypes;
  bool found;
  int nwords = line.size();
  if (nelements == -1)errorf(filename,linenum-1,"atom types must be defined before fingerprints in potential file.");
  if (nwords!=5)errorf(filename,linenum-1,"invalid screening command");
  int n_body_type = 3;
  atomtypes = new int[n_body_type];
  for (i=1;i<=n_body_type;i++) {
    found = false;
    for (j=0;j<nelementsp;j++) {
      if (line[i].compare(elementsp[j])==0) {
        atomtypes[i-1]=j;
        found = true;
        break;
      }
    }
    if (!found) {errorf(filename,linenum-1,"fingerprint element not found in atom types");}
  }
  i = atomtypes[0];
  j = atomtypes[1];
  k = atomtypes[2];
  int index = i*nelements*nelements+j*nelements+k;
  if (line[4].compare("Cmin")==0)  {
    screening_min[index] = utils::numeric(filename,linenum,line1[0],true,lmp);
  }
  else if (line[4].compare("Cmax")==0) {
    screening_max[index] = utils::numeric(filename,linenum,line1[0],true,lmp);
  }
  else errorf(filename,linenum-1,"unrecognized screening keyword");
  delete[] atomtypes;
}

//Called after finishing reading the potential file to make sure it is complete. True is bad.
//also allocates maxlayer and fingerprintlength and random weights and biases if ones were not provided.
bool PairRANN::check_potential(){
  int i,j,k,l,exnum,itype;
  if (nelements==-1){errorf(FLERR,"elements not defined!");}
  betalen = 0;
  betalen_f = 0;
  for (i=0;i<=nelements;i++){
    if (i<nelements){
      if (mass[i]<0)errorf(FLERR,"mass not defined");//uninitialized mass
    }
    fingerprintlength[i]=0;
    exchangelength[i]=0;
    for (j=0;j<fingerprintperelement[i];j++){
      for (int k=0;k<j;k++){
        if (fingerprints[i][j]->id==fingerprints[i][k]->id && fingerprints[i][j]->style==fingerprints[i][k]->style)errorf(FLERR,"duplicate fingerprint ids");
      }
      if (fingerprints[i][j]->fullydefined==false)errorf(FLERR,"undefined fingerprint parameters");
      if (fingerprints[i][j]->rc>cutmax){cutmax = fingerprints[i][j]->rc;}
      if (fingerprints[i][j]->spin==true){dospin=true;}
      if (fingerprints[i][j]->exchange==true){doexchange=true;}
      fingerprints[i][j]->startingneuron = fingerprintlength[i];
      if (fingerprints[i][j]->exchange==false){fingerprintlength[i]+=fingerprints[i][j]->get_length();}
      if (fingerprints[i][j]->exchange==true){
        fingerprints[i][j]->startingneuron=exchangelength[i];
        exchangelength[i]+=fingerprints[i][j]->get_length();
        }
    }
    for (int i1=0;i1<netperelement[i];i1++){
      if (!nets[i][i1]->fullydefined)errorf(FLERR,"undefined network parameters");
      bool check = nets[i][i1]->check_net_correctness();
      betalen += nets[i][i1]->betalen;
      betalen_f += nets[i][i1]->betalen_f;
      //printf("betalen:%d %d\n",betalen,nets[i][i1]->betalen);
    }
    for (j=0;j<stateequationperelement[i];j++){
      if (state[i][j]->fullydefined==false)errorf(FLERR,"undefined state equation parameters");
      if (state[i][j]->rc>cutmax){cutmax = state[i][j]->rc;}
    }
    if (netperelement[i]>0)energyexchange[i]=new double*[netperelement[i]];
    if (fingerprintlength[i]>fmax)fmax=fingerprintlength[i];
    if (exchangelength[i]>fexmax)fexmax=exchangelength[i];
  }

  return false;//everything looks good
}



