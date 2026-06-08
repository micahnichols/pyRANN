#include "omp.h"
#include "pair_spin_rann.h"
#include "rann_activation.h"
#include "rann_fingerprint.h"
#include "rann_stateequation.h"
#include "rann_net.h"
#include <pybind11/pybind11.h>
#include <sys/resource.h>
#include <iostream>
namespace py = pybind11;

using namespace LAMMPS_NS;

PairRANN::PairRANN(char *potential_file){
	cutmax = 0.0;
	nelementsp = -1;
	nelements = -1;
	fingerprintlength = NULL;
	netlength = NULL;
	mass = NULL;
	betalen = 0;
	doregularizer = false;
	normalizeinput = true;
	fingerprints = NULL;
	nets = NULL;
	max_epochs = 1e7;
	regularizer = 0.0;
	res = 10000;
	fingerprintcount = 0;
	netcount = 0;
	stateequationcount = 0;
	elementsp = NULL;
	elements = NULL;
	state = NULL;
	tolerance = 1e-6;
	sims = NULL;
	doscreen = false;
	allscreen = true;
	dospin = false;
	map = NULL;//check this
	natoms = 0;
	nsims = 0;
	doforces = false;
	dostresses = false;
	doexchange = false;
	fingerprintperelement = NULL;
	stateequationperelement = NULL;
	validation = 0.0;
	potential_output_freq = 100;
	algorithm = new char [SHORTLINE];
	potential_input_file = new char [strlen(potential_file)+1];
	dump_directory = new char [SHORTLINE];
	log_file = new char [SHORTLINE];
	potential_output_file = new char [SHORTLINE];
	strncpy(this->potential_input_file,potential_file,strlen(potential_file)+1);
	char temp1[] = ".\0";
	char temp2[] = "calibration.log\0";
	char temp3[] = "potential_output.nn\0";
	strncpy(dump_directory,temp1,strlen(temp1)+1);
	strncpy(log_file,temp2,strlen(temp2)+1);
	strncpy(potential_output_file,temp3,strlen(temp3)+1);
	strncpy(algorithm,"LMch",strlen("LMch")+1);
	overwritepotentials = false;
	debug_level1_freq = 10;
	debug_level2_freq = 0;
	debug_level3_freq = 0;
	debug_level4_freq = 0;
	debug_level5_freq = 0;
	debug_level5_spin_freq = 0;
	debug_level6_freq = 0;
  debug_level7_freq = 0;
	adaptive_regularizer = false;
	seed = time(0);
	lambda_initial = 1000;
	lambda_increase = 10;
	lambda_reduce = 0.3;
	targettype = 1;
	inum_weight = 1.0;
	freeenergy = 0;
	hbar = 4.135667403e-3/6.283185307179586476925286766559;
}

PairRANN::~PairRANN(){
	//clear memory
	delete [] algorithm;
	delete [] potential_input_file;
	delete [] dump_directory;
	delete [] log_file;
	delete [] potential_output_file;
	delete [] r;
	delete [] v;
	delete [] Xset;
	delete [] mass;
	for (int i=0;i<nsims;i++){
		for (int j=0;j<sims[i].inum;j++){
//			delete [] sims[i].x[j];
			if (doforces)delete [] sims[i].f[j];
			if (sims[i].spins)delete [] sims[i].s[j];
			delete [] sims[i].firstneigh[j];
			delete [] sims[i].features[j];
			if (doforces)delete [] sims[i].dfx[j];
			if (doforces)delete [] sims[i].dfy[j];
			if (doforces)delete [] sims[i].dfz[j];
		}
//		delete [] sims[i].x;
		if (doforces)delete [] sims[i].f;
		if (sims[i].spins)delete [] sims[i].s;
		if (doforces)delete [] sims[i].dfx;
		if (doforces)delete [] sims[i].dfy;
		if (doforces)delete [] sims[i].dfz;
		if (targettype>1)delete [] sims[i].total_ea;
		delete [] sims[i].firstneigh;
		delete [] sims[i].id;
		delete [] sims[i].features;
		delete [] sims[i].ilist;
		delete [] sims[i].numneigh;
		delete [] sims[i].type;
	}
	delete [] sims;
	for (int i=0;i<nelements;i++){delete [] elements[i];}
	delete [] elements;
	for (int i=0;i<nelementsp;i++){delete [] elementsp[i];}
	delete [] elementsp;
	
	delete [] map;
	for (int i=0;i<nelementsp;i++){
		if (fingerprintperelement[i]>0){
			for (int j=0;j<fingerprintperelement[i];j++){
				delete fingerprints[i][j];
			}
			delete [] fingerprints[i];
		}
		if (stateequationperelement[i]>0){
			for (int j=0;j<stateequationperelement[i];j++){
				delete state[i][j];
			}
			delete [] state[i];
		}
	}
	delete [] fingerprints;
	delete [] nets;
	delete [] state;
	delete [] fingerprintcount;
	delete [] fingerprintperelement;
	delete [] fingerprintlength;
	delete [] stateequationcount;
	delete [] stateequationperelement;
	delete [] freezebeta;
}

void PairRANN::setup(){

	int nthreads=1;
	#pragma omp parallel
	nthreads=omp_get_num_threads();

	std::cout << std::endl;
	std::cout << "# Number Threads     : " << nthreads << std::endl;

	double start_time = omp_get_wtime();

	read_file(potential_input_file);
	check_parameters();
	for (int i=0;i<nelementsp;i++){
		for (int j=0;j<fingerprintperelement[i];j++){
			  fingerprints[i][j]->allocate();
		}
		for (int j=0;j<stateequationperelement[i];j++){
			state[i][j]->allocate();
		}
	}
	read_dump_files();
	create_neighbor_lists();
	compute_fingerprints();
	if (normalizeinput){
		normalize_data();
	}
	separate_validation();

	double end_time = omp_get_wtime();
	double time = (end_time-start_time);
	printf("finished setup(): %f seconds\n",time);
}

void PairRANN::run(){
  struct rlimit rl;
  // Get current limits
  if (getrlimit(RLIMIT_STACK, &rl) == 0) {
    // Set soft limit to unlimited
    rl.rlim_cur = RLIM_INFINITY;
    // Apply changes
    if (setrlimit(RLIMIT_STACK, &rl) != 0) {
      std::cerr << "Failed to set unlimited stack" << std::endl;
    }
  }
    else {
      std::cout << "Set unlimited stack\n" << std:: endl;
    }
	if (strcmp(algorithm,"LMqr")==0){
		//DEPRECATED. Do not use.
		//slow but robust.
		//levenburg_marquardt_qr();
		errorf(FLERR,"QR algorithm is discontinued.");
	}
	else if (strcmp(algorithm,"LMch")==0){
		//faster. crashes if Jacobian has any columns of zeros.
		//usually will find exactly the same step for each iteration as qr.
		levenburg_marquardt_ch();
	}
	else if (strcmp(algorithm,"CG")==0){
		conjugate_gradient();
	}
	else if (strcmp(algorithm,"LMsearch")==0){
		levenburg_marquardt_linesearch();
	}
	else if (strcmp(algorithm,"bfgs")==0){
		bfgs();
	}
	else {
		errorf("unrecognized algorithm");
	}
}


void PairRANN::finish(){

//	write_potential_file(true);
}

void PairRANN::read_parameters(std::vector<std::string> line,std::vector<std::string> line1,FILE* fp,char *filename,int *linenum,char *linetemp){
	if (line[1]=="algorithm"){
		if (line[1].size()>SHORTLINE){
			delete [] algorithm;
			algorithm = new char[line[1].size()+1];
		}
		strncpy(algorithm,line1[0].c_str(),line1[0].size()+1);
	}
	else if (line[1]=="dumpdirectory"){
		if (line1[0].size()>SHORTLINE){
			delete [] dump_directory;
			dump_directory = new char[line1[0].size()+1];
		}
		strncpy(dump_directory,line1[0].c_str(),line1[0].size()+1);
	}
	else if (line[1]=="doforces"){
		int temp = strtol(line1[0].c_str(),NULL,10);
		doforces = (temp>0);
	}
	else if (line[1]=="dostresses"){
		int temp = strtol(line1[0].c_str(),NULL,10);
		dostresses = (temp>0);
	}
	else if (line[1]=="normalizeinput"){
		int temp = strtol(line1[0].c_str(),NULL,10);
		normalizeinput = (temp>0);
	}
	else if (line[1]=="tolerance"){
		tolerance = strtod(line1[0].c_str(),NULL);
	}
	else if (line[1]=="regularizer"){
		regularizer = strtod(line1[0].c_str(),NULL);
		if (regularizer!=0.0){doregularizer = true;}
	}
	else if (line[1]=="logfile"){
		delete [] log_file;
		log_file = new char[strlen(linetemp)];
		strncpy(log_file,linetemp,strlen(linetemp));
		log_file[strlen(linetemp)-1] = '\0';
	}
	else if (line[1]=="potentialoutputfreq"){
		potential_output_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="potentialoutputfile"){
		delete [] potential_output_file;
		potential_output_file = new char[strlen(linetemp)];
		strncpy(potential_output_file,linetemp,strlen(linetemp));
		potential_output_file[strlen(linetemp)-1] = '\0';
	}
	else if (line[1]=="maxepochs"){
		max_epochs = strtol(line1[0].c_str(),NULL,10);
	}
	// else if (line[1]=="dimsreserved"){
	// 	int i;
	// 	for (i=0;i<nelements;i++){
	// 		if (strcmp(words[2],elements[i])==0){
	// 			if (net[i].layers==0)errorf("networklayers for each atom type must be defined before the corresponding layer sizes.");
	// 			int j = strtol(words[3],NULL,10);
	// 			net[i].dimensionsr[j]= strtol(line1,NULL,10);
	// 			return;
	// 		}
	// 	}
	// 	errorf("dimsreserved element not found in atom types");
	// }
	// else if (line[1]=="freezeW"){
	// 	int i,j,k,b,l,ins,ops;
	// 	char **words1,*ptr;
	// 	char linetemp [MAXLINE];
	// 	int nwords = line.size();
	// 	if (nwords == 4){b=0;}
	// 	else if (nwords>4){b = strtol(line[4].c_str(),NULL,10);}
	// 	for (l=0;l<nelements;l++){
	// 		if (line[2]==elements[l]){
	// 			if (net[l].layers==0)errorf("networklayers must be defined before weights.");
	// 			i=strtol(line[3].c_str(),NULL,10);
	// 			if (i>=net[l].layers || i<0)errorf("invalid weight layer");
	// 			if (dimensiondefined[l][i]==false || dimensiondefined[l][i+1]==false) errorf("network layer sizes must be defined before corresponding weight");
	// 			if (bundle_inputdefined[l][i][b]==false && b!=0) errorf("bundle inputs must be defined before weights");
	// 			if (bundle_outputdefined[l][i][b]==false && b!=0) errorf("bundle outputs must be defined before weights");
	// 			if (net[l].identitybundle[i][b]) errorf("cannot define weights for an identity bundle");
	// 			if (bundle_inputdefined[l][i][b]==false){ins = net[l].dimensions[i];}
	// 			else {ins = net[l].bundleinputsize[i][b];}
	// 			if (bundle_outputdefined[l][i][b]==false){ops = net[l].dimensions[i+1];}
	// 			else {ops = net[l].bundleoutputsize[i][b];}
	// 			net[l].freezeW[i][b] = new bool [ins*ops];
	// 			nwords = line1.size();
	// 			if (nwords != ins)errorf("invalid weights per line");
	// 			for (k=0;k<ins;k++){
	// 				net[l].freezeW[i][b][k] = strtol(line1[k].c_str(),NULL,10);
	// 			}
	// 			for (j=1;j<ops;j++){
	// 				ptr = fgets(linetemp,MAXLINE,fp);
	// 				(linenum)++;
	// 				line1 = tokenmaker(linetemp,": ,\t_\n");
	// 				if (ptr==NULL)errorf("unexpected end of potential file!");
	// 				nwords = line1.size();
	// 				if (nwords != ins)errorf("invalid weights per line");
	// 				for (k=0;k<ins;k++){
	// 					net[l].freezeW[i][b][j*ins+k] = strtol(line1[k].c_str(),NULL,10);
	// 				}
	// 			}
	// 			delete [] words1;
	// 			return;
	// 		}
	// 	}
	// 	errorf("weight element not found in atom types");
	// }
	// else if (line[1]=="freezeB"){
	// 	int i,j,l,b,ops;
	// 	char *ptr;
	// 	int nwords = line.size();
	// 	char linetemp[MAXLINE];
	// 	if (nwords == 4){b=0;}
	// 	else if (nwords>4){b = strtol(line[4].c_str(),NULL,10);}
	// 	for (l=0;l<nelements;l++){
	// 		if (line[2]==elements[l]){
	// 			if (net[l].layers==0)errorf("networklayers must be defined before biases.");
	// 			i=strtol(line[3].c_str(),NULL,10);
	// 			if (i>=net[l].layers || i<0)errorf("invalid bias layer");
	// 			if (dimensiondefined[l][i]==false) errorf("network layer sizes must be defined before corresponding bias");
	// 			if (bundle_outputdefined[l][i][b]==false && b!=0) errorf("bundle outputs must be defined before bias");
	// 			if (net[l].identitybundle[i][b]) errorf("cannot define bias for an identity bundle");
	// 			if (bundle_outputdefined[l][i][b]==false){ops=net[l].dimensions[i+1];}
	// 			else {ops = net[l].bundleoutputsize[i][b];}
	// 			net[l].freezeB[i][b] = new bool [ops];
	// 			net[l].freezeB[i][b][0] = strtol(line1[0].c_str(),NULL,10);
	// 			for (j=1;j<ops;j++){
	// 				ptr = fgets(linetemp,MAXLINE,fp);
	// 				if (ptr==NULL)errorf("unexpected end of potential file!");
	// 				line1 = tokenmaker(linetemp," ,\t:_\n");
	// 				net[l].freezeB[i][b][j] = strtol(line1[0].c_str(),NULL,10);
	// 			}
	// 			return;
	// 		}
	// 	}
	// 	errorf("bias element not found in atom types");
	// }
	else if (line[1]=="validation"){
		validation = strtod(line1[0].c_str(),NULL);
	}
	else if (line[1]=="overwritepotentials") {
		int temp = strtol(line1[0].c_str(),NULL,10);
		overwritepotentials = (temp>0);
	}
	else if (line[1]=="debug1freq") {
		debug_level1_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug2freq") {
		debug_level2_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug3freq") {
		debug_level3_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug4freq") {
		debug_level4_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug5freq") {
		debug_level5_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug5spinfreq") {
		debug_level5_spin_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug6freq") {
		debug_level6_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="debug7freq") {
		debug_level7_freq = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="adaptiveregularizer") {
		double temp = strtod(line1[0].c_str(),NULL);
		adaptive_regularizer = (temp>0);
		if (adaptive_regularizer>0)doregularizer = true;
	}
	else if (line[1]=="lambdainitial"){
		lambda_initial = strtod(line1[0].c_str(),NULL);
	}
	else if (line[1]=="lambdaincrease"){
		lambda_increase = strtod(line1[0].c_str(),NULL);
	}
	else if (line[1]=="lambdareduce"){
		lambda_reduce = strtod(line1[0].c_str(),NULL);
	}
	else if (line[1]=="seed"){
		seed = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="targettype"){
		targettype = strtol(line1[0].c_str(),NULL,10);
	}
	else if (line[1]=="inumweight"){
		inum_weight = strtod(line1[0].c_str(),NULL);
	}
	else {
		char str[MAXLINE];
		sprintf(str,"unrecognized keyword in parameter file: %s\n",line[1]);
		errorf(filename,*linenum,str);
	}
}

void PairRANN::allocate(const std::vector<std::string> &elementwords)
{
	int i,n;
	cutmax = 0;
	nelementsp=nelements+1;
	//initialize arrays
	elements = new char *[nelements];
	elementsp = new char *[nelementsp];//elements + 'all'
	map = new int[nelementsp];
	mass = new double[nelements];
	nets = new RANN::Net **[nelementsp];
	screening_min = new double [nelements*nelements*nelements];
	screening_max = new double [nelements*nelements*nelements];
	for (i=0;i<nelements;i++){
		for (int j =0;j<nelements;j++){
			for (int k=0;k<nelements;k++){
				screening_min[i*nelements*nelements+j*nelements+k] = 0.8;//default values. Custom values may be read from potential file later.
				screening_max[i*nelements*nelements+j*nelements+k] = 2.8;//default values. Custom values may be read from potential file later.
			}
		}
	}

	fingerprints = new RANN::Fingerprint**[nelementsp];
	state = new RANN::State**[nelementsp];
	fingerprintlength = new int[nelementsp];
	exchangelength = new int[nelementsp];
	fingerprintperelement = new int [nelementsp];
	netperelement = new int [nelementsp];
	fingerprintcount = new int[nelementsp];
	netcount = new int[nelementsp];
	stateequationperelement = new int [nelementsp];
	stateequationcount = new int [nelementsp];
	energyexchange = new double **[nelementsp];
	for (i=0;i<=nelements;i++){
		n = elementwords[i].size();
		fingerprintlength[i]=0;
		fingerprintperelement[i] = -1;
		netperelement[i] = 0;
		fingerprintcount[i] = 0;
		netcount[i] = 0;
		stateequationperelement[i] = 0;
		stateequationcount[i] = 0;
		map[i] = i;
		if (i<nelements){
			mass[i]=-1.0;
			elements[i]= utils::strdup(elementwords[i]);
		}
		elementsp[i]= utils::strdup(elementwords[i]);
	}
}


void PairRANN::update_stack_size(){
	//TO DO: fix. Still getting stack overflow from underestimating memory needs.
	//get very rough guess of memory usage
	int jlen = nsims;
	if (doregularizer){
		jlen+=betalen-1;
	}
	if (doforces){
		jlen+=natoms*3;
	}
	//neighborlist memory use:
	memguess = 0;
	for (int i=0;i<nelementsp;i++){
		for (int j=0;j<netperelement[i];j++){
		memguess+=8*nets[i][j]->dimensions[0]*20*3;
	}}
	memguess+=8*20*12;
	memguess+=8*20*20*3;
	//separate validation memory use:
	memguess+=nsims*8*2;
	//levenburg marquardt ch memory use:
	memguess+=8*jlen*betalen*2;
	memguess+=8*betalen*betalen;
	memguess+=8*jlen*4;
	memguess+=8*betalen*4;
	//chsolve memory use:
	memguess+=8*betalen*betalen;
	//generous buffer:
	memguess *= 16;
	const rlim_t kStackSize = memguess;
	struct rlimit rl;
	int result;
	result = getrlimit(RLIMIT_STACK, &rl);
	if (result == 0)
	{
		if (rl.rlim_cur < kStackSize)
		{
			rl.rlim_cur += kStackSize;
			result = setrlimit(RLIMIT_STACK, &rl);
			if (result != 0)
			{
				fprintf(stderr, "setrlimit returned result = %d\n", result);
			}
		}
	}
}

bool PairRANN::check_parameters(){
	int itype,layer,bundle,rows,columns,r,c,count;
	if (strcmp(algorithm,"LMqr")!=0 && strcmp(algorithm,"LMch")!=0 && strcmp(algorithm,"CG")!=0 && strcmp(algorithm,"LMsearch")!=0 && strcmp(algorithm,"bfgs")!=0)errorf(FLERR,"Unrecognized algorithm. Must be CG, LMch or LMqr\n");//add others later maybe
	if (tolerance==0.0)errorf(FLERR,"tolerance not correctly initialized\n");
	if (tolerance<0.0 || max_epochs < 0 || regularizer < 0.0 || potential_output_freq < 0)errorf(FLERR,"detected parameter with negative value which must be positive.\n");
	if (targettype>3 || targettype<1)errorf(FLERR,"targettype must be 1, 2, or 3.");
	srand(seed);

	return false;//everything looks good
}

//part of setup. Do not optimize:
void PairRANN::read_dump_files(){
	DIR *folder;
//	char str[MAXLINE];
	struct dirent *entry;
	int file = 0;
	char line[MAXLINE],*ptr;
	char **words;
	int nwords,nwords1,sets;
	folder = opendir(dump_directory);

	if(folder == NULL)
	{
		errorf("unable to open dump directory");
	}
	std::cout<<"reading dump files\n";
	int nsims = 0;
	int nsets = 0;
	//count files
	while( (entry=readdir(folder)) )
	{
		if (strstr(entry->d_name,"dump")==NULL){continue;}
		FILE *fid = fopen(entry->d_name,"r");
		if (!fid){continue;}
		nsets++;
		fclose(fid);
	}
	closedir(folder);
	folder = opendir(dump_directory);
	this->nsets = nsets;
	Xset=new int[nsets];
	dumpfilenames = new char*[nsets];
	int count=0;
	//count snapshots per file
	while( (entry=readdir(folder)) )
	{
		if (strstr(entry->d_name,"dump")==NULL){continue;}
		FILE *fid = fopen(entry->d_name,"r");
		if (!fid){continue;}
		dumpfilenames[count] = new char[strlen(entry->d_name)+10];
		strcpy(dumpfilenames[count],entry->d_name);
		ptr = fgets(line,MAXLINE,fid);//ITEM: TIMESTEP
		ptr = fgets(line,MAXLINE,fid);
		nwords = 0;
		words = new char* [strlen(line)];
		words[nwords++] = strtok(line," ,\t\n");
		while ((words[nwords++] = strtok(NULL," ,\t\n"))) continue;
		nwords--;
		if (nwords!=5 && nwords != 11 && nwords != 6 && nwords != 12){errorf(entry->d_name,2,"dumpfile must contain 2nd line with timestep, energy, energy_weight, force_weight, snapshots\n");}
		sets = strtol(words[4],NULL,10);
		delete [] words;
		nsims+=sets;
		Xset[count++]=sets;
		fclose(fid);
	}
	closedir(folder);
	folder = opendir(dump_directory);
	sims = new Simulation[nsims];
	this->nsims = nsims;
	sims[0].startI=0;
	//read dump files
	while((entry=readdir(folder))){
		if (strstr(entry->d_name,"dump")==NULL){continue;}
		FILE *fid = fopen(entry->d_name,"r");
		printf("\t%s\n",entry->d_name);
		if (!fid){continue;}
		ptr = fgets(line,MAXLINE,fid);//ITEM: TIMESTEP
		while (ptr!=NULL){
			if (strstr(line,"ITEM: TIMESTEP")==NULL)errorf("invalid dump file line 1");
			ptr = fgets(line,MAXLINE,fid);//timestep
			nwords = 0;
			char *words1[strlen(line)];
			words1[nwords++] = strtok(line," ,\t");
			while ((words1[nwords++] = strtok(NULL," ,\t\n"))) continue;
			nwords--;
			if (nwords!=5 && nwords != 11 && nwords != 6 && nwords != 12)errorf("error: dump file line 2 must contain 5 entries: timestep, energy, energy_weight, force_weight, snapshots");
			int timestep = strtol(words1[0],NULL,10);
			sims[file].filename = new char [strlen(entry->d_name)+10];
			sims[file].timestep = timestep;
			strcpy(sims[file].filename,entry->d_name);
			sims[file].energy = strtod(words1[1],NULL);
			sims[file].energy_weight = strtod(words1[2],NULL);
			sims[file].force_weight = strtod(words1[3],NULL);
			sims[file].spinvec[0] = 0;
			sims[file].spinvec[1] = 0;
			sims[file].spinvec[2] = 0;
			sims[file].spinaxis[0] = 0;
			sims[file].spinaxis[1] = 0;
			sims[file].spinaxis[2] = 0;
			sims[file].temp = 0;
			if (nwords==6){
				freeenergy = 1;
				sims[file].temp = strtod(words1[5],NULL);
			}
			if (nwords==11){
				sims[file].spinspirals = true;
				double spinvec[3],spinaxis[3];
				spinvec[0] = strtod(words1[5],NULL);
				spinvec[1] = strtod(words1[6],NULL);
				spinvec[2] = strtod(words1[7],NULL);
				spinaxis[0] = strtod(words1[8],NULL);
				spinaxis[1] = strtod(words1[9],NULL);
				spinaxis[2] = strtod(words1[10],NULL);
				double norm = spinaxis[0]*spinaxis[0]+spinaxis[1]*spinaxis[1]+spinaxis[2]*spinaxis[2];
				if (norm<1e-14){errorf("spinaxis cannot be zero\n");}
				spinaxis[0]=spinaxis[0]/sqrt(norm);
				spinaxis[1]=spinaxis[1]/sqrt(norm);
				spinaxis[2]=spinaxis[2]/sqrt(norm);
				sims[file].spinvec[0] = spinvec[0];
				sims[file].spinvec[1] = spinvec[1];
				sims[file].spinvec[2] = spinvec[2];
				sims[file].spinaxis[0] = spinaxis[0];
				sims[file].spinaxis[1] = spinaxis[1];
				sims[file].spinaxis[2] = spinaxis[2];
			}
			if (nwords==12){
				freeenergy = 1;
				sims[file].temp = strtod(words1[11],NULL);
			}
			ptr = fgets(line,MAXLINE,fid);//ITEM: NUMBER OF ATOMS
			if (strstr(line,"ITEM: NUMBER OF ATOMS")==NULL)errorf("invalid dump file line 3");
			ptr = fgets(line,MAXLINE,fid);//natoms
			int natoms = strtol(line,NULL,10);
			//printf("%d %d %f\n",file,timestep,sims[file].energy/natoms);
			if (file>0){sims[file].startI=sims[file-1].startI+natoms*3;}
			this->natoms+=natoms;
			if (targettype==1)sims[file].energy_weight /=pow(natoms,inum_weight);
			sims[file].force_weight =sims[file].force_weight;
			ptr = fgets(line,MAXLINE,fid);//ITEM: BOX BOUNDS xy xz yz pp pp pp
			if (strstr(line,"ITEM: BOX BOUNDS")==NULL)errorf("invalid dump file line 5");
			double box[3][3];
			double origin[3];
			bool cols[12];
			bool dostress=true;
			for (int i= 0;i<11;i++){
				cols[i]=false;
			}
			box[0][1] = box[0][2] = box[1][2] = 0.0;
			for (int i = 0;i<3;i++){
				ptr = fgets(line,MAXLINE,fid);//box line
				char *words[7];
				nwords = 0;
				words[nwords++] = strtok(line," ,\t\n");
				while ((words[nwords++] = strtok(NULL," ,\t\n"))) continue;
				nwords--;
				if (nwords != 6 && nwords!=3 && nwords!=2){errorf("invalid dump box definition");}
				origin[i] = strtod(words[0],NULL);
				box[i][i] = strtod(words[1],NULL);
				if (nwords==3 || nwords==6){
					if (i==0){
						box[0][1]=strtod(words[2],NULL);
						if (box[0][1]>0){box[0][0]-=box[0][1];}
						else origin[0] -= box[0][1];
					}
					else if (i==1){
						box[0][2]=strtod(words[2],NULL);
						if (box[0][2]>0){box[0][0]-=box[0][2];}
						else origin[0] -= box[0][2];
					}
					else{
						box[1][2]=strtod(words[2],NULL);
						if (box[1][2]>0)box[1][1]-=box[1][2];
						else origin[1] -=box[1][2];
					}
				}
				if (nwords==6){
					for (int j=0;j<3;j++){
						sims[file].stress[i][j]=strtod(words[j+3],NULL);
					}
				}
				if (nwords!=6)dostress=false;
			}
			sims[file].stresses=dostress;
			for (int i=0;i<3;i++)box[i][i]-=origin[i];
			box[1][0]=box[2][0]=box[2][1]=0.0;
			ptr = fgets(line,MAXLINE,fid);//ITEM: ATOMS id type x y z c_energy fx fy fz sx sy sz
			nwords = 0;
			char *words[count_words(line)+1];
			words[nwords++] = strtok(line," ,\t\n");
			while ((words[nwords++] = strtok(NULL," ,\t\n"))) continue;
			nwords--;
			int colid = -1;
			int columnmap[11];
			for (int i=0;i<nwords-2;i++){columnmap[i]=-1;}
			for (int i=2;i<nwords;i++){
				if (strcmp(words[i],"type")==0){colid = 0;}
				else if (strcmp(words[i],"x")==0){colid=1;}
				else if (strcmp(words[i],"y")==0){colid=2;}
				else if (strcmp(words[i],"z")==0){colid=3;}
				else if (strcmp(words[i],"fx")==0){colid=4;}
				else if (strcmp(words[i],"fy")==0){colid=5;}
				else if (strcmp(words[i],"fz")==0){colid=6;}
				else if (strcmp(words[i],"sx")==0){colid=7;}
				else if (strcmp(words[i],"sy")==0){colid=8;}
				else if (strcmp(words[i],"sz")==0){colid=9;}
				else if (strcmp(words[i],"c_eng")==0){colid=10;}
				else {continue;}
				cols[colid] = true;
				if (colid!=-1){columnmap[colid]=i-2;}
			}
			for (int i=0;i<4;i++){
				if (!cols[i]){errorf("dump file must include type, x, y, and z data columns (other recognized keywords are fx, fy, fz, sx, sy, sz)");}
			}
			bool doforce = false;
			bool dospin = false;
			sims[file].inum = natoms;
			sims[file].ilist = new int [natoms];
			sims[file].type = new int [natoms];
			sims[file].x= new double *[natoms];
			for (int i=0;i<3;i++){
				for (int j=0;j<3;j++)sims[file].box[i][j]=box[i][j];
				sims[file].origin[i]=origin[i];
			}
			for (int i=0;i<natoms;i++){
				sims[file].x[i]=new double [3];
			}
			//if forces are given in dump file
			if (cols[4] && cols[5] && cols[6]){
				doforce = true;
				sims[file].forces=doforce;
			}
			//if force calibration is on
			if (doforces || doforce){
				sims[file].f = new double *[natoms];
				for (int i=0;i<natoms;i++){
					sims[file].f[i] = new double [3];
				}
			}
			//if per-atom energies will be used
			if (targettype>1)sims[file].total_ea = new double [natoms];
			//if spin vectors are provided
			if (cols[7] && cols[8] && cols[9]){
				dospin = true;
				sims[file].s = new double *[natoms];
				for (int i=0;i<natoms;i++){
					sims[file].s[i] = new double [3];
				}
			}
			else if (this->dospin){
				errorf(FLERR,"spin vectors must be defined for all input simulations when magnetic fingerprints are used");
			}
			if (!cols[10] && targettype>1){
				errorf(FLERR,"per-atom energies must be specified using keyword \"c_eng\" in all input dump files when per-atom or per-species training is enabled");
			}
			if ((dostress || dostresses) && (!doforce || !dostress)){
				errorf(FLERR,"stress cannot be computed unless forces and stresses are given in all dump files");
			}
			for (int i=0;i<natoms;i++){
				ptr = fgets(line,MAXLINE,fid);
				char *words2[count_words(line)+1];
				nwords1 = 0;
				words2[nwords1++] = strtok(line," ,\t");
				while ((words2[nwords1++] = strtok(NULL," ,\t"))) continue;
				nwords1--;
				if (nwords1!=nwords-2){errorf("incorrect number of data columns in dump file.");}
				sims[file].ilist[i]=i;//ignore any id mapping in the dump file, just id them based on line number.
				sims[file].type[i]=strtol(words2[columnmap[0]],NULL,10)-1;//lammps type counting starts at 1 instead of 0
				sims[file].x[i][0]=strtod(words2[columnmap[1]],NULL);
				sims[file].x[i][1]=strtod(words2[columnmap[2]],NULL);
				sims[file].x[i][2]=strtod(words2[columnmap[3]],NULL);
				//sims[file].energy[i]=strtod(words[columnmap[4]],NULL);
				if (doforce){
					sims[file].f[i][0]=strtod(words2[columnmap[4]],NULL);
					sims[file].f[i][1]=strtod(words2[columnmap[5]],NULL);
					sims[file].f[i][2]=strtod(words2[columnmap[6]],NULL);
				}
				//if force calibration is on, but forces are not given in file, assume they are zero.
				else if (doforces){
					sims[file].f[i][0]=0.0;
					sims[file].f[i][1]=0.0;
					sims[file].f[i][2]=0.0;
				}
				if (dospin){
					sims[file].s[i][0]=strtod(words2[columnmap[7]],NULL);
					sims[file].s[i][1]=strtod(words2[columnmap[8]],NULL);
					sims[file].s[i][2]=strtod(words2[columnmap[9]],NULL);
					double sm = sims[file].s[i][0]*sims[file].s[i][0]+sims[file].s[i][1]*sims[file].s[i][1]+sims[file].s[i][2]*sims[file].s[i][2];
					sims[file].s[i][0]/=sqrt(sm);
					sims[file].s[i][1]/=sqrt(sm);
					sims[file].s[i][2]/=sqrt(sm);
				}
				sims[file].spins = dospin;
				if (targettype>1){
					sims[file].total_ea[i] = strtod(words2[columnmap[10]],NULL);
				}
			}
			ptr = fgets(line,MAXLINE,fid);//ITEM: TIMESTEP
			file++;
			if (file>nsims){errorf("Too many dump files found. Nsims is incorrect.\n");}
		}
		fclose(fid);
	}
	closedir(folder);
	sprintf(line,"imported %d atoms, %d simulations\n",natoms,nsims);
	std::cout<<line;
}

int PairRANN::count_unique_species(int *s,int nsims){
	int nn,n1,j,count1,count2,count3;
	count1=0;
	count3=0;
	for (n1=0;n1<nsims;n1++){
		nn = s[n1];
		sims[nn].speciescount = new int[nelements];
		sims[nn].speciesoffset = count1;
		sims[nn].atomoffset = count3;
		sims[nn].speciesmap = new int[nelements];
		bool tp[nelements];
		for (j=0;j<nelements;j++){
			tp[j]=false;
			sims[nn].speciescount[j] = 0;
		}
		for (j=0;j<sims[nn].inum;j++){
			tp[sims[nn].type[j]]=true;
			sims[nn].speciescount[sims[nn].type[j]]++;
		}
		count2 = 0;
		for (j=0;j<nelements;j++){
			if (tp[j]==true) {
				sims[nn].speciesmap[j]=count2;
				count1++;
				count2++;
			}
		}
		count3 += sims[nn].inum;
		sims[nn].uniquespecies=count2;	
	}
	return count1;
}

//part of setup. Do not optimize:
void PairRANN::create_neighbor_lists(){
	//brute force search technique rather than tree search because we only do it once and most simulations are small.
	//I did optimize for low memory footprint by only adding ghost neighbors
	//within cutoff distance of the box
	int i,ix,iy,iz,j,k;
//	char str[MAXLINE];
	double buffer = 0.01;//over-generous compensation for roundoff error
	std::cout<<"building neighbor lists\n";
	for (i=0;i<nsims;i++){
		double box[3][3];
		for (ix=0;ix<3;ix++){
			for (iy=0;iy<3;iy++)box[ix][iy]=sims[i].box[ix][iy];
		}
		double *origin = sims[i].origin;
		int natoms = sims[i].inum;
		int xb = floor(cutmax/box[0][0]+1);
		int yb = floor(cutmax/box[1][1]+1);
		int zb = floor(cutmax/box[2][2]+1);
		int buffsize = natoms*(xb*2+1)*(yb*2+1)*(zb*2+1);
		double x[buffsize][3];
		int type[buffsize];
		int id[buffsize];
		double spins[buffsize][3];
		int count = 0;
		//force all atoms to be inside the box:
		double xtemp[3];
		double xp[3];
		double boxt[9];
		for (j=0;j<3;j++){
			for (k=0;k<3;k++){
				boxt[j*3+k]=box[j][k];
			}
		}
		for (j=0;j<natoms;j++){
			for (k=0;k<3;k++){
				xp[k] = sims[i].x[j][k]-origin[k];
			}
			qrsolve(boxt,3,3,xp,xtemp);//convert coordinates from Cartesian to box basis (uses qrsolve for matrix inversion)
			for (k=0;k<3;k++){
				xtemp[k]-=floor(xtemp[k]);//if atom is outside box find periodic replica in box
			}
			for (k=0;k<3;k++){
				sims[i].x[j][k] = 0.0;
				for (int l=0;l<3;l++){
					sims[i].x[j][k]+=box[k][l]*xtemp[l];//convert back to Cartesian
				}
				sims[i].x[j][k]+=origin[k];
			}
		}

		//calculate box face normal directions and plane intersections
		double xpx,xpy,xpz,ypx,ypy,ypz,zpx,zpy,zpz;
		zpx = 0;zpy=0;zpz =1;
		double ym,xm;
		ym = sqrt(box[1][2]*box[1][2]+box[2][2]*box[2][2]);
		xm = sqrt(box[1][1]*box[2][2]*box[1][1]*box[2][2]+box[0][1]*box[0][1]*box[2][2]*box[2][2]+(box[0][1]*box[1][2]-box[0][2]*box[1][1])*(box[0][1]*box[1][2]-box[0][2]*box[1][1]));
		//unit vectors normal to box faces:
		ypx = 0;
		ypy = box[2][2]/ym;
		ypz = -box[1][2]/ym;
		xpx = box[1][1]*box[2][2]/xm;
		xpy = -box[0][1]*box[2][2]/xm;
		xpz = (box[0][1]*box[1][2]-box[0][2]*box[1][1])/xm;
		double fxn,fxp,fyn,fyp,fzn,fzp;
		//minimum distances from origin to planes aligned with box faces:
		fxn = origin[0]*xpx+origin[1]*xpy+origin[2]*xpz;
		fyn = origin[0]*ypx+origin[1]*ypy+origin[2]*ypz;
		fzn = origin[0]*zpx+origin[1]*zpy+origin[2]*zpz;
		fxp = (origin[0]+box[0][0])*xpx+(origin[1]+box[1][0])*xpy+(origin[2]+box[2][0])*xpz;
		fyp = (origin[0]+box[0][1])*ypx+(origin[1]+box[1][1])*ypy+(origin[2]+box[2][1])*ypz;
		fzp = (origin[0]+box[0][2])*zpx+(origin[1]+box[1][2])*zpy+(origin[2]+box[2][2])*zpz;
		//fill buffered atom list
		double px,py,pz;
		double xe,ye,ze;
		double theta,sx,sy,sz;
		for (j=0;j<natoms;j++){
			x[count][0] = sims[i].x[j][0];
			x[count][1] = sims[i].x[j][1];
			x[count][2] = sims[i].x[j][2];
			type[count] = sims[i].type[j];
			if (sims[i].spins){
				spins[count][0] = sims[i].s[j][0];
				spins[count][1] = sims[i].s[j][1];
				spins[count][2] = sims[i].s[j][2];
			}
			id[count] = j;
			count++;
		}

		//add ghost atoms outside periodic boundaries:
		for (ix=-xb;ix<=xb;ix++){
			for (iy=-yb;iy<=yb;iy++){
				for (iz=-zb;iz<=zb;iz++){
					if (ix==0 && iy == 0 && iz == 0)continue;
					for (j=0;j<natoms;j++){
						xe = ix*box[0][0]+iy*box[0][1]+iz*box[0][2]+sims[i].x[j][0];
						ye = iy*box[1][1]+iz*box[1][2]+sims[i].x[j][1];
						ze = iz*box[2][2]+sims[i].x[j][2];
						px = xe*xpx+ye*xpy+ze*xpz;
						py = xe*ypx+ye*ypy+ze*ypz;
						pz = xe*zpx+ye*zpy+ze*zpz;
						//include atoms if their distance from the box face is less than cutmax
						if (px>cutmax+fxp+buffer || px<fxn-cutmax-buffer){continue;}
						if (py>cutmax+fyp+buffer || py<fyn-cutmax-buffer){continue;}
						if (pz>cutmax+fzp+buffer || pz<fzn-cutmax-buffer){continue;}
						x[count][0] = xe;
						x[count][1] = ye;
						x[count][2] = ze;
						type[count] = sims[i].type[j];
						id[count] = j;
						if (sims[i].spinspirals && sims[i].spins){
							// sims[i].s[j][0]=1;
							// sims[i].s[j][2]=0;
							// spins[j][0]=1;
							// spins[j][2]=0;
							double twopi = 3.1415926535897932384626433*2;
							theta = twopi*(sims[i].spinvec[0]*ix + sims[i].spinvec[1]*iy + sims[i].spinvec[2]*iz);
							//theta = sims[i].spinvec[0]*(ix*box[0][0]+iy*box[0][1]+iz*box[0][2]) + sims[i].spinvec[1]*(iy*box[1][1]+iz*box[1][2]) + sims[i].spinvec[2]*iz*box[2][2];
							double sxi = sims[i].s[j][0];
							double syi = sims[i].s[j][1];
							double szi = sims[i].s[j][2];
							double ax = sims[i].spinaxis[0];
							double ay = sims[i].spinaxis[1];
							double az = sims[i].spinaxis[2];
							ax = 0;//REMOVE
							az = 1;//REMOVE
							sx = sxi*(ax*ax*(1-cos(theta))+cos(theta))+syi*(ax*ay*(1-cos(theta))-az*sin(theta))+szi*(ax*az*(1-cos(theta))+ay*sin(theta));
							sy = sxi*(ax*ay*(1-cos(theta))+az*sin(theta))+syi*(ay*ay*(1-cos(theta))+cos(theta))+szi*(-ax*sin(theta)+ay*az*(1-cos(theta)));
							sz = sxi*(ax*az*(1-cos(theta))-ay*sin(theta))+syi*(ax*sin(theta)+ay*az*(1-cos(theta)))+szi*(az*az*(1-cos(theta))+cos(theta));
							//sx = ax*(ax*sxi+ay*syi+az*szi)+(ay*szi-az*syi)*sin(theta)+(-ay*(ax*syi-ay*sxi)+az*(-ax*szi+az*sxi))*cos(theta);
							//sy = ay*(ax*sxi+ay*syi+az*szi)+(-ax*szi+az*sxi)*sin(theta)+(ax*(ax*syi-ay*sxi)-az*(ay*szi-az*syi))*cos(theta);
							//sz = az*(ax*sxi+ay*syi+az*szi)+(ax*syi-ay*sxi)*sin(theta)+(-ax*(-ax*szi+az*sxi)-ay*(ay*szi-az*syi))*cos(theta);
							spins[count][0]=sx;
							spins[count][1]=sy;
							spins[count][2]=sz;
						}
						else if (sims[i].spins) {
							spins[count][0]=sims[i].s[j][0];
							spins[count][1]=sims[i].s[j][1];
							spins[count][2]=sims[i].s[j][2];
						}
						count++;
						if (count>buffsize){errorf("neighbor overflow!\n");}
					}
				}
			}
		}

		//update stored lists
		buffsize = count;
		for (j=0;j<natoms;j++){
			delete [] sims[i].x[j];
		}
		delete [] sims[i].x;
		delete [] sims[i].type;
		delete [] sims[i].ilist;
		if (sims[i].spins){
			for (j=0;j<natoms;j++){
				delete [] sims[i].s[j];
			}
			delete [] sims[i].s;
			sims[i].s = new double *[buffsize];
		}
		sims[i].type = new int [buffsize];
		sims[i].x = new double *[buffsize];
		sims[i].id = new int [buffsize];
		sims[i].ilist = new int [buffsize];

		for (j=0;j<buffsize;j++){
			sims[i].x[j] = new double [3];
			for (k=0;k<3;k++){
				sims[i].x[j][k] = x[j][k];
			}
			sims[i].type[j] = type[j];
			sims[i].id[j] = id[j];
			sims[i].ilist[j] = j;
			if (sims[i].spins){
				sims[i].s[j] = new double [3];
				for (k=0;k<3;k++){
					sims[i].s[j][k]=spins[j][k];
				}
			}
		}
		sims[i].inum = natoms;
		sims[i].gnum = buffsize-natoms;
		sims[i].numneigh = new int[natoms];
		sims[i].firstneigh = new int*[natoms];
		//do double count, slow, but enables getting the exact size of the neighbor list before filling it.
		for (j=0;j<natoms;j++){
			sims[i].numneigh[j]=0;
			for (k=0;k<buffsize;k++){
				if (k==j)continue;
				double xtmp = sims[i].x[j][0]-sims[i].x[k][0];
				double ytmp = sims[i].x[j][1]-sims[i].x[k][1];
				double ztmp = sims[i].x[j][2]-sims[i].x[k][2];
				double r2 = xtmp*xtmp+ytmp*ytmp+ztmp*ztmp;
				if (r2<cutmax*cutmax){
					sims[i].numneigh[j]++;
				}
			}
			sims[i].firstneigh[j] = new int[sims[i].numneigh[j]+1];
			count = 0;
			for (k=0;k<buffsize;k++){
				if (k==j)continue;
				double xtmp = sims[i].x[j][0]-sims[i].x[k][0];
				double ytmp = sims[i].x[j][1]-sims[i].x[k][1];
				double ztmp = sims[i].x[j][2]-sims[i].x[k][2];
				double r2 = xtmp*xtmp+ytmp*ytmp+ztmp*ztmp;
				if (r2<cutmax*cutmax){
					sims[i].firstneigh[j][count] = k;
					count++;
				}
			}
			sims[i].firstneigh[j][sims[i].numneigh[j]]=j;//This will be useful...
		}
	}
}

//part of setup. Do not optimize:
//TO DO: fix stack size problem
void PairRANN::compute_fingerprints(){
	std::cout<<"computing fingerprints\n";
	int nn,j,ii,f,i,itype,jnum,fex;
	for (nn=0;nn<nsims;nn++){
		sims[nn].features = new double *[sims[nn].inum];
		sims[nn].exfeatures = new double *[sims[nn].inum];
		sims[nn].state_e = 0;
		sims[nn].jl = new int *[sims[nn].inum];
		sims[nn].jnum = new int[sims[nn].inum];
		sims[nn].xn = new double*[sims[nn].inum];
		sims[nn].yn = new double*[sims[nn].inum];
		sims[nn].zn = new double*[sims[nn].inum];
		sims[nn].tn = new int*[sims[nn].inum];
		if (doforces || dostresses){
			sims[nn].dfx = new double *[sims[nn].inum];
			sims[nn].dfy = new double *[sims[nn].inum];
			sims[nn].dfz = new double *[sims[nn].inum];
			if (doexchange){
				sims[nn].dexfx = new double *[sims[nn].inum];
				sims[nn].dexfy = new double *[sims[nn].inum];
				sims[nn].dexfz = new double *[sims[nn].inum];
			}
			if (dospin){
				sims[nn].dsx = new double *[sims[nn].inum];
				sims[nn].dsy = new double *[sims[nn].inum];
				sims[nn].dsz = new double *[sims[nn].inum];
			}
		}
		sims[nn].force = new double*[sims[nn].inum+sims[nn].gnum];
		sims[nn].fm = new double*[sims[nn].inum+sims[nn].gnum];
		  for (j=0;j<sims[nn].inum+sims[nn].gnum;j++){
			  sims[nn].force[j]=new double[3];
			  sims[nn].fm[j]=new double[3];
			  sims[nn].force[j][0]=0;
			  sims[nn].force[j][1]=0;
			  sims[nn].force[j][2]=0;
			  sims[nn].fm[j][0]=0;
			  sims[nn].fm[j][1]=0;
			  sims[nn].fm[j][2]=0;
		  }
			for (ii=0;ii<sims[nn].inum;ii++){
				i = sims[nn].ilist[ii];
			  	itype = map[sims[nn].type[i]];
				//if (net[itype].layers==0){errorf(FLERR,"atom type found without corresponding network defined");}
			    f = fingerprintlength[itype];
				fex = exchangelength[itype];
				jnum = sims[nn].numneigh[i]+1;
				sims[nn].features[ii] = new double [f];

		  }
		}
		#pragma omp parallel
		{
		int i,ii,itype,f,jnum,len,j,nn;
		double **force,**fm;
		#pragma omp for schedule(guided)
		for (nn=0;nn<nsims;nn++){
		  clock_t start = clock();
		
		  double start_time = omp_get_wtime();
		  force = sims[nn].force;
		  fm = sims[nn].fm;
		  if (debug_level2_freq>0){
			  sims[nn].state_ea = new double [sims[nn].inum];
		  }
		  for (ii=0;ii<sims[nn].inum;ii++){
			  i = sims[nn].ilist[ii];
			  itype = map[sims[nn].type[i]];
			  f = fingerprintlength[itype];
			  fex = exchangelength[itype];
			  jnum = sims[nn].numneigh[i];
			  double xn[jnum];
			  double yn[jnum];
			  double zn[jnum];
			  int tn[jnum];
			  int jl[jnum+1];
			  cull_neighbor_list(xn,yn,zn,tn,&jnum,jl,i,nn,cutmax);
			  double features [f];
			  double exfeatures[fex*jnum];
			  double dfeaturesx[f*jnum];
			  double dfeaturesy[f*jnum];
			  double dfeaturesz[f*jnum];
			  double dexfeaturesx[fex*jnum*jnum];
			  double dexfeaturesy[fex*jnum*jnum];
			  double dexfeaturesz[fex*jnum*jnum];
			  for (j=0;j<f;j++){
				  features[j]=0;
			  }
   	          for (j=0;j<fex*jnum;j++) {
				exfeatures[j]=0;
			  }
			  for (j=0;j<f*jnum;j++){
				  dfeaturesx[j]=dfeaturesy[j]=dfeaturesz[j]=0;
			  }
			  for (j=0;j<fex*jnum*jnum;j++) {
        		dexfeaturesx[j]=dexfeaturesy[j]=dexfeaturesz[j]=0;
      		  }
			  //screening is calculated once for all atoms if any fingerprint uses it.
			  double Sik[jnum];
			  double dSikx[jnum];
			  double dSiky[jnum];
			  double dSikz[jnum];
			  //TO D0: stack overflow often happens here from stack limit too low.
			  double dSijkx[jnum*jnum];
			  double dSijky[jnum*jnum];
			  double dSijkz[jnum*jnum];
			  //TO D0: stack overflow often happens here from stack limit too low.
			  bool Bij[jnum];
			  double sx[jnum*f];
			  double sy[jnum*f];
			  double sz[jnum*f];
			  double sxx[jnum*f];
			  double sxy[jnum*f];
			  double sxz[jnum*f];
			  double syy[jnum*f];
			  double syz[jnum*f];
			  double szz[jnum*f];
			  for (j=0;j<f*jnum;j++){
				  sx[j]=sy[j]=sz[j]=0;
				  sxx[j]=sxy[j]=sxz[j]=syy[j]=syz[j]=szz[j]=0;
			  }
			  if (doscreen){
					screen(Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1);//jnum is neighlist + self term, hence jnum-1 in function inputs
			  }
			  if (allscreen){
				  screen_neighbor_list(xn,yn,zn,tn,&jnum,jl,i,nn,Bij,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz);
			  }

				sims[nn].xn[ii] = new double[jnum];
				sims[nn].yn[ii] = new double[jnum];
				sims[nn].zn[ii] = new double[jnum];
				sims[nn].tn[ii] = new int[jnum];
				if (doforces || dostresses){
				  sims[nn].dfx[ii] = new double[f*jnum];
				  sims[nn].dfy[ii] = new double[f*jnum];
				  sims[nn].dfz[ii] = new double[f*jnum];
				  if (dospin){
					  sims[nn].dsx[ii] = new double[f*jnum];
					  sims[nn].dsy[ii] = new double[f*jnum];
					  sims[nn].dsz[ii] = new double[f*jnum];
				  }
			   }
			   if (doexchange){
				sims[nn].exfeatures[ii] = new double[fex*jnum];
				if (doforces || dostresses){
					sims[nn].dexfx[ii] = new double[fex*jnum];
					sims[nn].dexfy[ii] = new double[fex*jnum];
					sims[nn].dexfz[ii] = new double[fex*jnum];
				}
			   }

			  //do fingerprints for atom type
			  len = fingerprintperelement[itype];
			  for (j=0;j<len;j++) {
				if (fingerprints[itype][j]->exchange==false){
					if      (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,ii,nn,xn,yn,zn,tn,jnum-1,jl);		
					else if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
			        // else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,sxx,sxy,sxz,syy,syz,szz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
			        // else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,sxx,sxy,sxz,syy,syz,szz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				}
			  }
			  itype = nelements;
			  //do fingerprints for type "all"
			  len = fingerprintperelement[itype];
			  for (j=0;j<len;j++) {
				if (fingerprints[itype][j]->exchange==false){	
					if      (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
					else if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
					//        else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,sxx,sxy,sxz,syy,syz,szz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
					//        else if (fingerprints[itype][j]->spin==true  && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(features,dfeaturesx,dfeaturesy,dfeaturesz,sx,sy,sz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				}
				// else if (fingerprints[itype][j]->exchange==true){
				// 	if      (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(exfeatures,dexfeaturesx,dexfeaturesy,dexfeaturesz,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				// 	else if (fingerprints[itype][j]->spin==false && fingerprints[itype][j]->screen==true) fingerprints[itype][j]->compute_fingerprint(exfeatures,dexfeaturesx,dexfeaturesy,dexfeaturesz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jnum-1,jl);
				// }
     		  }
			  //copy features from stack to heap
			  for (j=0;j<f;j++){
				  sims[nn].features[ii][j] = features[j];
			  }
			  itype = map[sims[nn].type[i]];
			  len = fingerprintperelement[itype];
			  if (doexchange){
				//printf("%d %d %d\n",fex,jnum,len);
				for (int jj=0;jj<jnum-1;jj++){
					for (j=0;j<len;j++){
						if (fingerprints[itype][j]->exchange==true){	
							if      (fingerprints[itype][j]->screen==false)fingerprints[itype][j]->compute_fingerprint(exfeatures,dexfeaturesx,dexfeaturesy,dexfeaturesz,ii,nn,xn,yn,zn,tn,jj,jnum-1,jl);
							else if (fingerprints[itype][j]->screen==true)fingerprints[itype][j]->compute_fingerprint(exfeatures,dexfeaturesx,dexfeaturesy,dexfeaturesz,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,ii,nn,xn,yn,zn,tn,jj,jnum-1,jl);
						}
					}
					for (int k=0;k<fex;k++){
						sims[nn].exfeatures[ii][jj*fex+k] = exfeatures[jj*fex+k];
						//printf("%f\n",exfeatures[k]);
					}
				}
				for (int k=0;k<fex;k++){
					sims[nn].exfeatures[ii][(jnum-1)*fex+k] = exfeatures[(jnum-1)*fex+k];
				}
				sims[nn].jnum[ii] = jnum-1;
				sims[nn].jl[ii]=new int[jnum-1];
				for (int jj=0;jj<jnum-1;jj++){
					sims[nn].jl[ii][jj]=jl[jj];
					sims[nn].xn[ii][jj]=xn[jj];
					sims[nn].yn[ii][jj]=yn[jj];
					sims[nn].zn[ii][jj]=zn[jj];
					sims[nn].tn[ii][jj]=tn[jj];
				}
			  }
			  if (doforces || dostresses){
				  for (j=0;j<f*jnum;j++){
					  sims[nn].dfx[ii][j]=dfeaturesx[j];
					  sims[nn].dfy[ii][j]=dfeaturesy[j];
					  sims[nn].dfz[ii][j]=dfeaturesz[j];
					  if (doexchange){
						sims[nn].dexfx[ii][j]=dexfeaturesx[j];
						sims[nn].dexfy[ii][j]=dexfeaturesy[j];
						sims[nn].dexfz[ii][j]=dexfeaturesz[j];
					  }
				  }
				  if (dospin){
					  for (j=0;j<f*jnum;j++){
						  sims[nn].dsx[ii][j] = sx[j];
						  sims[nn].dsy[ii][j] = sy[j];
						  sims[nn].dsz[ii][j] = sz[j];
					  }
				  }
			  }
			  double e=0.0;
			  itype = map[sims[nn].type[i]];
			  len = stateequationperelement[itype];
			  for (j=0;j<len;j++){
				       if (state[itype][j]->screen==false && state[itype][j]->spin==false){state[itype][j]->eos_function(&e,force,i,nn,xn,yn,zn,tn,jnum-1,jl);}
				  else if (state[itype][j]->screen==true  && state[itype][j]->spin==false){state[itype][j]->eos_function(&e,force,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
 			      else if (state[itype][j]->screen==false && state[itype][j]->spin==true ){state[itype][j]->eos_function(&e,force,fm,i,nn,xn,yn,zn,tn,jnum-1,jl);}
				  else if (state[itype][j]->screen==true  && state[itype][j]->spin==true ){state[itype][j]->eos_function(&e,force,fm,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
			  }
			  itype = nelements;
			  len = stateequationperelement[itype];
			  for (j=0;j<len;j++){
				       if (state[itype][j]->screen==false && state[itype][j]->spin==false){state[itype][j]->eos_function(&e,force,i,nn,xn,yn,zn,tn,jnum-1,jl);}
				  else if (state[itype][j]->screen==true  && state[itype][j]->spin==false){state[itype][j]->eos_function(&e,force,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
 			      else if (state[itype][j]->screen==false && state[itype][j]->spin==true ){state[itype][j]->eos_function(&e,force,fm,i,nn,xn,yn,zn,tn,jnum-1,jl);}
				  else if (state[itype][j]->screen==true  && state[itype][j]->spin==true ){state[itype][j]->eos_function(&e,force,fm,Sik,dSikx,dSiky,dSikz,dSijkx,dSijky,dSijkz,Bij,i,nn,xn,yn,zn,tn,jnum-1,jl);}
			  }
			  sims[nn].energy-=e;
			  sims[nn].state_e+=e;
			  if (debug_level2_freq>0){sims[nn].state_ea[ii]=e;}
			  if (targettype>1){sims[nn].total_ea[ii]-=e;}
		  }
		  clock_t end = clock();
		  sims[nn].time = (double)(end-start)/ CLOCKS_PER_SEC;
	}
	}
}

void PairRANN::normalize_data(){
	for (int i=0;i<nelementsp;i++){
		for (int k=0;k<netperelement[i];k++){
			if (nets[i][k]->layers>0)nets[i][k]->normalize_data();
		}
	}
	int i,j;
	int count = 0;
	RANN::Net ***net_new = new RANN::Net **[nelementsp];
	for (i=0;i<nelementsp;i++){
		net_new[i] = new RANN::Net *[netperelement[i]];
		for (j=0;j<netperelement[i];j++){
			net_new[i][j] = create_net(nets[i][j]->style);
			nets[i][j]->normalize_net(net_new[i][j]);
		}
	}
	copy_network(&nets,net_new);
	// RANN::Net *net_new = new RANN::Net;
	// normalize_net(net_new);
	// copy_network(net_new,net);
	// delete [] net_new;
}

void PairRANN::flatten_beta(RANN::Net ***nets,double *beta){
	int i,j;
	int count = 0;
	for (i=0;i<nelementsp;i++){
		for (j=0;j<netperelement[i];j++){
			nets[i][j]->flatten_beta(&beta[count]);
			count+=nets[i][j]->betalen;
		}
	}
}

void PairRANN::copy_network(RANN::Net ****net_newp,RANN::Net ***net_old){
	int i,j;
	int count = 0;
	RANN::Net ***net_new = new RANN::Net **[nelementsp];
	for (i=0;i<nelementsp;i++){
		net_new[i] = new RANN::Net *[netperelement[i]];
		for (j=0;j<netperelement[i];j++){
			net_new[i][j] = create_net(net_old[i][j]->style);
			// printf("test1\n");
			net_old[i][j]->copy_network(net_new[i][j]);
			// printf("test2\n");
		}
	}
	net_newp[0] = net_new;
}

void PairRANN::unnormalize_net(RANN::Net ****net_newp,RANN::Net ***net_old){
	int i,j;
	int count = 0;
	RANN::Net ***net_new = new RANN::Net **[nelementsp];
	for (i=0;i<nelementsp;i++){
		net_new[i] = new RANN::Net *[netperelement[i]];
		for (j=0;j<netperelement[i];j++){
			net_new[i][j] = create_net(net_old[i][j]->style);
			net_old[i][j]->unnormalize_net(net_new[i][j]);
		}
	}
	net_newp[0] = net_new;
}

void PairRANN::unflatten_beta(RANN::Net ***nets,double *beta){
	int i,j;
	int count=0;
	for (i=0;i<nelementsp;i++){
		for (j=0;j<netperelement[i];j++){
			nets[i][j]->unflatten_beta(&beta[count]);
			count+=nets[i][j]->betalen;
		}
	}
}

void PairRANN::separate_validation(){
	int n1,n2,i,vnum,len,startI,endI,j,t,k;
	char str[MAXLINE];
	int Iv[nsims];
	int Ir[nsims];
	bool w;
	n1=n2=0;
	sprintf(str,"finishing setup\n");
	std::cout<<str;
	for (i=0;i<nsims;i++)Iv[i]=-1;
	for (i=0;i<nsets;i++){
		startI=0;
		for (j=0;j<i;j++)startI+=Xset[j];
		endI = startI+Xset[i];
		len = Xset[i];
		// vnum = rand();
		// if (vnum<floor(RAND_MAX*validation)){
		// 	vnum = 1;
		// }
		// else{
		// 	vnum = 0;
		// }
		vnum = 0;// if Xset has only 1 entry, do not include it in validation ever. (Code above puts it randomly in validation or fit).
		vnum+=floor(len*validation);
		while (vnum>0){
			w = true;
			t = floor(rand() % len)+startI;
			for (j=0;j<n1;j++){
				if (t==Iv[j]){
					w = false;
					break;
				}
			}
			if (w){
				Iv[n1]=t;
				vnum--;
				n1++;
			}
		}
		for (j=startI;j<endI;j++){
			w = true;
			for (k=0;k<n1;k++){
				if (j==Iv[k]){
					w = false;
					break;
				}
			}
			if (w){
				Ir[n2]=j;
				n2++;
			}
		}
	}
	nsimr = n2;
	nsimv = n1;
	r = new int [n2];
	v = new int [n1];
	natomsr = 0;
	natomsv = 0;
	for (i=0;i<n1;i++){
		v[i]=Iv[i];
		natomsv += sims[v[i]].inum;
	}
	for (i=0;i<n2;i++){
		r[i]=Ir[i];
		natomsr += sims[r[i]].inum;
	}
	sprintf(str,"assigning %d simulations (%d atoms) for validation, %d simulations (%d atoms) for fitting\n",nsimv,natomsv,nsimr,natomsr);
	std::cout<<str;
	int na = 0;
	for (i=0;i<n2;i++){
		sims[r[i]].startrow=na;
		na+=1;
		if (doforces)na+=sims[r[i]].inum*3;
		if (dostresses)na+=6;
		// sims[r[i]].force_weight*=nsimr/natomsr;
	}
	jlen1=na;
	na=0;
	for (i=0;i<n1;i++){
		sims[v[i]].startrow=na;
		na+=1;
		if (doforces)na+=sims[v[i]].inum*3;
		if (dostresses)na+=6;
	}
}

void PairRANN::unpack_errors(double *tp, double *energy_fit,double *force_fit, double* stress_fit, double *reg_fit,int rows,int columns){
	int i,nn,j;
	int rrows;
	if (columns>1 && doregularizer){
		rrows = rows-columns;
	}
	else {
		rrows = rows;
	}
	if (columns>1) {
		natoms = natomsr;
		nsims = nsimr;
	}
	else {
		natoms=natomsv;
		nsims= nsimv;
	}
	*energy_fit = 0;
	*force_fit = 0;
	*stress_fit = 0;
	*reg_fit =0;
	if (doforces && dostresses){
		for (i=0;i<nsims;i++){
			if (columns>1)nn = r[i];
			else nn = v[i];
			int sr = sims[nn].startrow;
			for (j=0;j<sims[nn].inum*3;j++){
				*force_fit+=tp[sr+j]*tp[sr+j];
			}
			sr+=sims[nn].inum*3;
			*energy_fit+=tp[sr]*tp[sr];
			sr++;
			for (j=0;j<6;j++){
				*stress_fit+=tp[sr+j]*tp[sr+j];
			}
		}
		*force_fit/=nsims;
		*stress_fit/=nsims;
	}
	else if (dostresses){
		for (i=0;i<nsims;i++){
			if (columns>1)nn = r[i];
			else nn = v[i];
			int sr = sims[nn].startrow;
			*energy_fit+=tp[sr]*tp[sr];
			sr++;
			for (j=0;j<6;j++){
				*stress_fit+=tp[sr+j]*tp[sr+j];
			}
		}
		*stress_fit/=nsims;
	}
	else if (doforces){
		for (i=0;i<nsims;i++){
			if (columns>1)nn = r[i];
			else nn = v[i];
			int sr = sims[nn].startrow;
			for (j=0;j<sims[nn].inum*3;j++){
				*force_fit+=tp[sr+j]*tp[sr+j];
			}
			sr+=sims[nn].inum*3;
			*energy_fit+=tp[sr]*tp[sr];
		}
		*force_fit/=nsims;
	}
	else {
		for (i=0;i<nsims;i++){
			if (columns>1)nn = r[i];
			else nn = v[i];
			int sr = sims[nn].startrow;
			*energy_fit+=tp[sr]*tp[sr];
		}
	}
	*energy_fit/=nsims;
	if (doregularizer && columns>1){
		for (i=0;i<columns-1;i++){
			j = i+rrows;
			*reg_fit +=tp[j]*tp[j];
		}
		*reg_fit /= betalen;
	}
}

void PairRANN::do_debugs(int counter,double *tp,double *targetv,double *Jp,double *bp,double *dp,int rows,int columns){
	int count1,count2,count3,count4,count5,count5spin,count6,count7;
	count1=count2=count3=count4=count5=count5spin=count6=count7=-1;
	if (debug_level1_freq>0)count1 = counter%debug_level1_freq;
	if (debug_level2_freq>0)count2 = counter%debug_level2_freq;
	if (debug_level3_freq>0)count3 = counter%debug_level3_freq;
	if (debug_level4_freq>0)count4 = counter%debug_level4_freq;
	if (debug_level5_freq>0)count5 = counter%debug_level5_freq;
	if (debug_level5_spin_freq>0)count5spin = counter%debug_level5_spin_freq;
	if (debug_level6_freq>0)count6 = counter%debug_level6_freq;
	if (debug_level7_freq>0)count7 = counter%debug_level7_freq;
        if (count1==0){
			write_debug_level1(tp,targetv);
		}
		if (count2==0){
			write_debug_level2(tp,targetv);
		}
		if (count3==0){
			write_debug_level3(Jp,tp,bp,dp,rows,columns);
		}
		if (count4==0){
			write_debug_level4(tp,targetv);
		}
		if (count5==0){
			write_debug_level5(tp,targetv);
		}
		if (count5spin=0){
			write_debug_level5_spin(tp,targetv);
		}
		if (count6==0){
			write_debug_level6(tp,targetv);
		}
		if (count7==0){
			write_debug_level7(tp,targetv);
		}
}

//top level run function, calls compute_jacobian and qrsolve.
void PairRANN::levenburg_marquardt_ch(){
	char str[MAXLINE];
	int iter,i,j,nn;
	bool goodstep=true;
	double energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,force_fit1,reg_fit,reg_fit1,stress_fit,stress_fit1,stress_fitv;
	double lambda = lambda_initial;
	double vraise = lambda_increase;
	double vreduce = lambda_reduce;
	char line[MAXLINE];
    this->energy_fitv_best = 10^300;
	int i_off, j_off, j_offPi;
	double time1, time2;
	int rows = nsimr;
	int vrows = nsimv;
	int columns = betalen;
	if (doforces){
		rows +=natomsr*3;
		vrows += natomsv*3;
	}
	if (dostresses){
		rows +=nsimr*6;
		vrows +=nsimv*6;
	}
	int rmc = rows;
	if (doregularizer){
		rows += betalen;//do not regulate last bias
		rmc = rows-columns;
	}
	sprintf(str,"types=%d; betalen=columns=%d; rows=%d; rows-columns=%d, regularization:%d\n",nelementsp,betalen,rows,rmc, doregularizer);
	std::cout<<str;
	double J[rows*columns];
	double J1[rows*columns];
	double J2[columns*columns];
	double t2[columns];
	double target[rows];
	double target1[rows];
	double targetv[vrows];
	double beta[columns];
	double beta1[columns];
	double D[columns];
	double *dp;
	int rows1;
	if (columns>rows)rows1=columns;
	else rows1=rows;
	double delta[rows1];//extra length used internally in qrsolve
	dp = delta;
    double *tp,*tp1,*bp,*bp1,*Jp,*Jp1;
	tp = target;
	tp1 = target1;
	bp = beta;
	bp1 = beta1;
	Jp = J;
	Jp1 = J1;
	force_fit = energy_fit = reg_fit = energy_fit1 = force_fit1 = reg_fit1 = stress_fit = stress_fit1 = 0.0;
	//clock_t start1 = clock();
	double start_time_tot = omp_get_wtime();
	RANN::Net ***net1;
	copy_network(&net1,nets);
	if (doforces || dostresses){
		jacobian_force(Jp,tp,r,nsimr,natomsr,net1);
	}
	else{ 
		jacobian_convolution(Jp,tp,r,nsimr,natomsr,net1);
	}
	unpack_errors(tp,&energy_fit,&force_fit,&stress_fit,&reg_fit,rows,columns);
	double initial_reg = regularizer;
	double initial_eng = energy_fit;
	flatten_beta(nets,bp);
	force_fitv = energy_fitv = 0.0;
	int counter = 0;
	int count = 0;
	iter = 0;
	//write_debug_level5(tp,targetv);
	FILE *fid = fopen(log_file,"w");
	if (fid==NULL)errorf("couldn't open log file!");
	if (doforces && dostresses)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err: %.10e, s_err: %.10e, sv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,stress_fit,stress_fitv,reg_fit,lambda);
	else if (doforces)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,reg_fit,lambda);
	else if (dostresses)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, s_err: %.10e, sv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,stress_fit,stress_fitv,reg_fit,lambda);
	else sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,reg_fit,lambda);

	write_potential_file(true,line,0,initial_reg);
	double start2;
	while (iter<max_epochs){
		if (goodstep){
			if (nsimv>0){
				//do validation forward pass
				if (doforces || dostresses){
					forward_pass_force(targetv,v,nsimv,nets);
				}
				else {
					forward_pass(targetv,v,nsimv,nets);
				}
				unpack_errors(targetv,&energy_fitv,&force_fitv,&stress_fitv,&reg_fit1,vrows,1);
			}
			else{
				energy_fitv = 0.0;
				force_fitv = 0.0;
				stress_fitv = 0.0;
			}

			// clock_t start2 = clock();
			start2 = omp_get_wtime();

			for (i=0;i<betalen;i++){
				i_off = i*betalen;
				for (int k=0;k<=i;k++){
					J2[i_off+k] = 0.0;
				}
			}
			// #pragma omp parallel default(none) shared(J2,Jp,jlen2,betalen, doregularizer)
			#pragma omp parallel
			{
			// loop reordered to remove the dependancy. Single thread calculation would be slow. Observed gain when thread more than around 8
			#pragma omp for schedule(guided)
			for (int i=0;i<betalen;i++){
				int i_off = i*betalen;
				for (int k=0;k<=i;k++){
					for (int j=0;j<rmc;j++){
						int j_off = j*betalen;
						int j_offPi = j_off+i;
						J2[i_off+k] += Jp[j_offPi]*Jp[j_off+k];
					}
				}
			}
			if (doregularizer){
				#pragma omp for
				for (int i=0;i<betalen;i++){
					int	i_off = i*betalen;
					int ij_off = rmc*betalen + i_off;
					J2[i_off+i]+=Jp[ij_off+i]*Jp[ij_off+i];
				}
			}

			#pragma omp barrier
			#pragma omp single
			{
			for (int i=0;i<betalen;i++){
				D[i] = J2[i*betalen+i];
				if (D[i]==0){errorf(FLERR,"Jacobian is rank deficient!\n");}//one or more weight/bias has no effect on the computed energy of any of the atoms.
				if (doregularizer) {// t2 can be initialized with 0 or derivative w.r.t. weight}
					t2[i]=Jp[rmc*betalen+i*betalen+i]*tp[rmc+i];
				}
				else
				    t2[i]=0;
			}
			}
			// loop splitting for threading. Initialization for t2 is done above.
			#pragma omp for schedule(guided)
			for (int i=0;i<betalen;i++){
				for (int j=0;j<rmc;j++){
					t2[i]+=Jp[j*betalen+i]*tp[j];
				}
			}

			}
			double adexp = 0.1;
			if (adaptive_regularizer && doregularizer) {regularizer *= sqrt(initial_reg/reg_fit*energy_fit);
				if (regularizer<1e-8){
					doregularizer=false;
					rows -=betalen;
					rmc = rows;
				}
			}
			double time = (double) (omp_get_wtime() - start2)*1000.0;
			do_debugs(counter,tp,targetv,Jp,bp,dp,rows,columns);
		}
		else {
			do_debugs(counter,tp1,targetv,Jp1,bp1,dp,rows,columns);
		}
        bool is_write_potential = false;
        if (count == potential_output_freq){
			count = 0;
			if (((force_fitv+stress_fitv+energy_fitv)*natomsv+(energy_fit+force_fit+stress_fit)*natomsr)/natoms < this->energy_fitv_best) {
				this->energy_fitv_best = ((force_fitv+stress_fitv+energy_fitv)*natomsv+(energy_fit+force_fit+stress_fit)*natomsr)/natoms;
				is_write_potential = true;
			}
			else {
				count--;
			}
        }
		
		if (doforces && dostresses)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err: %.10e, s_err: %.10e, sv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,stress_fit,stress_fitv,reg_fit,lambda);
		else if (doforces)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,reg_fit,lambda);
		else if (dostresses)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, s_err: %.10e, sv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,stress_fit,stress_fitv,reg_fit,lambda);
		else sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,reg_fit,lambda);

		std::cout<<line;
		fprintf(fid,"%s",line);
		count++;
        if (is_write_potential){
            write_potential_file(true,line,iter,initial_reg);
        }
		counter++;
		#pragma omp for
		for (i=0;i<betalen;i++){
			J2[i*betalen+i]=D[i]+sqrt(D[i]*lambda);
		}
		chsolve(J2,betalen,t2,dp);
		#pragma omp for
		for (i=0;i<betalen;i++){
			bp1[i]=bp[i]+dp[i];
		}
		unflatten_beta(net1,bp1);
		if (doforces || dostresses){
			jacobian_force(Jp1,tp1,r,nsimr,natomsr,net1);
		}
		else{
			jacobian_convolution(Jp1,tp1,r,nsimr,natomsr,net1);
		}
		unpack_errors(tp1,&energy_fit1,&force_fit1,&stress_fit1,&reg_fit1,rows,columns);
		if (energy_fit1+stress_fit1+force_fit1+reg_fit1<energy_fit+stress_fit+force_fit+reg_fit){
			goodstep = true;
			lambda = lambda*vreduce;
			energy_fit = energy_fit1;
			force_fit = force_fit1;
			stress_fit = stress_fit1;
			reg_fit = reg_fit1;
			double *tempb;
			tempb = bp;
			bp = bp1;
			bp1= tempb;
			double *tempJ;
			tempJ = Jp;
			Jp = Jp1;
			Jp1 = tempJ;
			double *tempT = tp;
			tp = tp1;
			tp1 = tempT;
			unflatten_beta(nets,bp);
			iter++;
		}
		else {
			goodstep=false;
			lambda = lambda*vraise;
			if (lambda > 10e50){
				//write_potential_file(true,line,iter,initial_reg);
				errorf("Terminating because convergence is not making progress.\n");
			}
		}
    if (iter%10==0){
      if (PyErr_CheckSignals() != 0){
        throw py::error_already_set();
      }
    }
		if (energy_fit+force_fit<tolerance){
			std::cout<<"Terminating because reached convergence tolerance\n";
			write_potential_file(true,line,iter,initial_reg);
			break;
		}
	}
	//delete dynamic memory use
	for (int i=0;i<nelementsp;i++){
		for (int j=0;j<netperelement[i];j++){
			delete net1[i][j];
		}
		delete [] net1[i];
	}
	delete [] net1;
    double time = (double) (omp_get_wtime() - start_time_tot)*1000.0;
}

void PairRANN::conjugate_gradient(){
	//jlen is number of rows; betalen is number of columns of jacobian
	char str[MAXLINE];
	int iter,jlen,i,jlenv,j,jlen2,nn;
	bool goodstep=true;
	double energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,force_fit1,reg_fit,reg_fit1,stress_fit,stress_fit1,stress_fitv;
	double lambda = lambda_initial;
	double vraise = lambda_increase;
	double vreduce = lambda_reduce;
	char line[MAXLINE];
    this->energy_fitv_best = 10^300;
	int i_off, j_off, j_offPi;
	double time1, time2;
	int rows = nsimr;
	int vrows = nsimv;
	int columns = betalen;
	if (doforces){
		rows +=natomsr*3;
		vrows += natomsv*3;
	}
	if (dostresses){
		rows +=nsimr*6;
		vrows ==nsimv*6;
	}
	int rmc = rows;
	if (doregularizer){
		rows += betalen;//do not regulate last bias
		rmc = rows-columns;
	}
	sprintf(str,"types=%d; betalen=columns=%d; rows=%d; rows-columns=%d, regularization:%d\n",nelementsp,betalen,rows,rmc, doregularizer);
	std::cout<<str;
	double J[rows*columns];
	double target[rows];
	double target1[rows];
	double target2[rows];
	double targetv[vrows];
	double beta[columns];
	double beta1[columns];
	double beta2[columns];
	double dX[columns];
	double dX1[columns];
	double *dXp;
	double *dXp1;
	double s[columns];
	double bpr;
	double *dp;
	double delta[rows];//extra length used internally in qrsolve
	dp = delta;
    double *tp,*tp1,*tp2,*bp,*bp1,*bp2,*Jp,*Jp1,*sp;
	tp = target;
	tp1 = target1;
	tp2 = target2;
	bp = beta;
	bp1 = beta1;
	bp2 = beta2;
	Jp = J;
	dXp = dX;
	dXp1 = dX1;
	sp = s;
	energy_fit = reg_fit = energy_fit1 = reg_fit1 = 0.0;
	//clock_t start1 = clock();
	double start_time_tot = omp_get_wtime();
	if (doforces || dostresses){
		jacobian_force(Jp,tp,r,nsimr,natomsr,nets);
	}
	else{ 
		jacobian_convolution(Jp,tp,r,nsimr,natomsr,nets);
	}
	unpack_errors(tp,&energy_fit,&force_fit,&stress_fit,&reg_fit,rows,columns);
	RANN::Net ***net1;
	RANN::Net ***net2;
	copy_network(&net1,nets);
	copy_network(&net2,nets);
	double initial_reg = regularizer;
	double initial_eng = energy_fit;
	flatten_beta(nets,bp);
	energy_fitv = 0.0;
	int counter = 0;
	int count = 0;
	iter = 0;
	if (nsimv>0){
		//do validation forward pass
		if (doforces || dostresses){
			forward_pass_force(targetv,v,nsimv,nets);
		}
		else {
			forward_pass(targetv,v,nsimv,nets);
		}
		unpack_errors(targetv,&energy_fitv,&force_fitv,&stress_fitv,&reg_fit1,rows,1);	
	}
	else{
		energy_fitv = 0.0;
		force_fitv = 0.0;
		stress_fitv = 0.0;
	}
	FILE *fid = fopen(log_file,"w");
	if (fid==NULL)errorf("couldn't open log file!");
	if (doforces && dostresses)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err: %.10e, s_err: %.10e, sv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,stress_fit,stress_fitv,reg_fit,lambda);
	else if (doforces)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,reg_fit,lambda);
	else if (dostresses)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, s_err: %.10e, sv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,stress_fit,stress_fitv,reg_fit,lambda);
	else sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,reg_fit,lambda);

	write_potential_file(true,line,0,initial_reg);
	printf("%s",line);
	double start2;
	//get gradient direction
	#pragma omp parallel
	{
	#pragma omp for
	for (j=0;j<betalen;j++){
		dX[j]=0.0;
		for (i=0;i<rmc;i++){
			dXp[j]+=2*Jp[i*betalen+j]*tp[i];
		}
		if (doregularizer){
			dXp[j]+=2*Jp[(rmc+j)*betalen+j]*tp[rmc+j];
		}
	}
	}
	//set initial conjugate direction
	bpr = 0.0;
	for (j=0;j<columns;j++){
		sp[j]=dXp[j];
	}
	while (iter<max_epochs){
		//line search
		double e3 = energy_fit*nsimr+force_fit*natomsr+stress_fit*nsimr*6+reg_fit*betalen;
		double Gmag = 0.0;
		double G1mag = 0.0;
		#pragma omp parallel for reduction(+:Gmag,G1mag)
		for (j=0;j<columns;j++){
			Gmag += sp[j]*sp[j];
			G1mag += dXp[j]*dXp[j];
		}
		Gmag = sqrt(Gmag);
		G1mag = sqrt(G1mag);
		double R = sqrt(5.0)*0.5-0.5;
		double C = 1.0-R;
		double alpha_min = e3/G1mag;
		double alpha_max = alpha_min*(3-log(e3/rmc)); 
		if (alpha_max<alpha_min*3){alpha_max=alpha_min*3;}
		int nIter = ceil(-2.078087*log(tolerance/abs(4*alpha_min)));
		//int nIter = 5;
		double x1 = (R*alpha_min+C*alpha_max)/Gmag;
		double x2 = (C*alpha_min+R*alpha_max)/Gmag;
		//printf("range: %f %f %f %f\n",alpha_min,x1*Gmag,x2*Gmag,alpha_max);
		#pragma omp parallel for
		for (j=0;j<columns;j++){
			bp1[j] = bp[j]+x1*sp[j];
			bp2[j] = bp[j]+x2*sp[j];
		}
		unflatten_beta(net1,bp1);
		unflatten_beta(net2,bp2);
		if (doforces || dostresses){
			forward_pass_force(tp1,r,nsimr,net1);
			forward_pass_force(tp2,r,nsimr,net2);
		}
		else {
			forward_pass(tp1,r,nsimr,net1);
			forward_pass(tp2,r,nsimr,net2);
		}
		double e1;
		double e2;
		double *temp;
		e1 = 0.0;
		e2 = 0.0;
		#pragma omp parallel 
		{
		#pragma omp for reduction(+:e1,e2)
		for (i=0;i<rows;i++){
			e1 += tp1[i]*tp1[i];
			e2 += tp2[i]*tp2[i];
		}
		}
		for (i=0;i<nIter;i++){
			//printf("%d %f %f %f\n",nIter,e1/jlen2,e2/jlen2,e3/jlen2);
			//printf("range: %f %f %f %f\n",alpha_min,x1*Gmag,x2*Gmag,alpha_max);
			if (e1 > e2 && e3 > e2){
				alpha_min = x1*Gmag;
				x1 = x2;
				e1 = e2;
				temp = tp1;
				tp1 = tp2;
				tp2 = temp;
				temp = bp1;
				bp1 = bp2;
				bp2 = temp;
				x2 = (C*alpha_min+R*alpha_max)/Gmag;
				//#pragma omp for
				#pragma omp parallel for
				for (j=0;j<columns;j++){
					bp2[j] = bp[j]+x2*sp[j];
				}
				//#pragma omp single
				//{
				unflatten_beta(net2,bp2);
				if (doforces||dostresses){
					forward_pass_force(tp2,r,nsimr,net2);
				}
				else{
					forward_pass(tp2,r,nsimr,net2);
				}
				e2 = 0.0;
				#pragma omp parallel
				{
				#pragma omp for reduction(+:e2)
				for (j=0;j<rows;j++){
					e2+=tp2[j]*tp2[j];
				}
				}
			}
			else if (e2 > e1 && e3 > e1){
				alpha_max = x2*Gmag;
				x2 = x1;
				e2 = e1;
				temp = tp2;
				tp2 = tp1;
				tp1 = temp;
				temp = bp2;
				bp2 = bp1;
				bp1 = temp;
				x1 = (R*alpha_min+C*alpha_max)/Gmag;

				//#pragma omp for
				#pragma omp parallel for
				for (j=0;j<columns;j++){
					bp1[j] = bp[j]+x1*sp[j];
				}
				//#pragma omp single
				//{
				unflatten_beta(net1,bp1);
				if (doforces || dostresses){
					forward_pass_force(tp1,r,nsimr,net1);
				}
				else {
					forward_pass(tp1,r,nsimr,net1);
				}
				e1 = 0.0;
				#pragma omp parallel
				{
				#pragma omp for reduction(+ : e1)
				for (j=0;j<rows;j++){
					e1+=tp1[j]*tp1[j];
				}
				}
			}
			else {
				alpha_max = x1*Gmag;
				alpha_min = 0.0;
				x1 = (R*alpha_min+C*alpha_max)/Gmag;
				x2 = (C*alpha_min+R*alpha_max)/Gmag;

				//#pragma omp for
				#pragma omp parallel for
				for (j=0;j<columns;j++){
					bp1[j] = bp[j]+x1*sp[j];
					bp2[j] = bp[j]+x2*sp[j];
				}
				//#pragma omp single
				//{
				unflatten_beta(net1,bp1);
				unflatten_beta(net2,bp2);
				if (doforces || dostresses){
					forward_pass_force(tp1,r,nsimr,net1);
					forward_pass_force(tp2,r,nsimr,net2);
				}
				else {
					forward_pass(tp1,r,nsimr,net1);
					forward_pass(tp2,r,nsimr,net2);
				}
				e1 = 0.0;
				e2 = 0.0;
				#pragma omp parallel
				{
				#pragma omp for reduction(+ : e1,e2)
				for (j=0;j<rows;j++){
					e1 += tp1[j]*tp1[j];
					e2 += tp2[j]*tp2[j];
				}
				}
			}
		}
		if (e2<e1 && e2<e3) {
			temp = bp;
			bp = bp2;
			bp2 = temp;
		}
		else if (e1<e3 && e1<e3) {
			temp = bp;
			bp = bp1;
			bp1 = temp;
		}
		else {
			#pragma omp parallel for
			for (i=0;i<columns;i++){
				bp[i] = 0.5*(bp[i]+bp1[i]);
			}
		}
		unflatten_beta(nets,bp);
		if (doforces || dostresses){
			jacobian_force(Jp,tp,r,nsimr,natomsr,nets);
		}
		else {
			jacobian_convolution(Jp,tp,r,nsimr,natomsr,nets);
		}
		energy_fit = 0.0;
		unpack_errors(tp,&energy_fit,&force_fit,&stress_fit,&reg_fit,rows,columns);
		e3 = energy_fit*nsimr+force_fit*natomsr+stress_fit*nsimr*6+reg_fit*betalen;
		start2 = omp_get_wtime();
		if (nsimv>0){
			//do validation forward pass
			if (doforces || dostresses){
				forward_pass_force(targetv,v,nsimv,nets);
			}
			else {
				forward_pass(targetv,v,nsimv,nets);
			}
			unpack_errors(targetv,&energy_fitv,&force_fitv,&stress_fitv,&reg_fit1,rows,1);
		}
        bool is_write_potential = false;
        if (count == potential_output_freq){
			count = 0;
			if (((force_fitv+stress_fitv+energy_fitv)*natomsv+(energy_fit+force_fit+stress_fit)*natomsr)/natoms < this->energy_fitv_best) {
				this->energy_fitv_best = ((force_fitv+stress_fitv+energy_fitv)*natomsv+(energy_fit+force_fit+stress_fit)*natomsr)/natoms;
				is_write_potential = true;
			}
			else {
				count--;
			}
        }
		double adexp = 0.1;
		if (adaptive_regularizer) {regularizer *= sqrt(initial_reg/reg_fit*energy_fit);}
		if (regularizer<1e-8){
			doregularizer=false;
			rows -=betalen;
		 	rmc = 0;
		}
		if (doforces && dostresses)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err: %.10e, s_err: %.10e, sv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,stress_fit,stress_fitv,reg_fit,lambda);
		else if (doforces)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,reg_fit,lambda);
		else if (dostresses)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, s_err: %.10e, sv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,stress_fit,stress_fitv,reg_fit,lambda);
		else sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,reg_fit,lambda);

		std::cout<<line;
		fprintf(fid,"%s",line);
		count++;
        if (is_write_potential){
            write_potential_file(true,line,counter,initial_reg);
        }
		do_debugs(counter,tp,targetv,Jp,bp,dp,rows,columns);
		counter++;
		//get new gradient direction
		double *tempdX;
		tempdX = dXp;
		dXp = dXp1;
		dXp1 = tempdX;
		double numen;
		double denom;
		numen = 0.0;
		denom = 0.0;
		#pragma omp parallel
		{
		#pragma omp for
		for (j=0;j<columns;j++){
			dXp[j]=0.0;
			for (i=0;i<jlen2;i++){
				dXp[j]+=2*Jp[i*columns+j]*tp[i];
			}
			if (doregularizer){
				dXp[j]+=2*Jp[(jlen2+j)*columns+j]*tp[rmc+j];
			}
		}
		#pragma omp barrier
		//get betapr
		#pragma omp for reduction(+:numen,denom)
		for (j=0;j<columns;j++){
			numen += dXp[j]*(dXp[j]-dXp1[j]);
			denom += dXp[j]*dXp1[j];
		}
		#pragma omp single
		{
		bpr = numen/denom;
		}
		if (bpr<0.0){
			#pragma omp single
			{
			temp = sp;
			sp = dXp;
			dXp = temp;
			}
		}
		else{
			//update conjugate direction
			#pragma omp for
			for (j=0;j<columns;j++){
				sp[j] = dXp[j]+bpr*sp[j];
			}
		}
		}
		
		if (energy_fit<tolerance){
			std::cout<<"Terminating because reached convergence tolerance\n";
			write_potential_file(true,line,iter,initial_reg);
			break;
		}
	}
	//delete dynamic memory use
	for (int i=0;i<nelementsp;i++){
		for (int j=0;j<netperelement[i];j++){
			delete net1[i][j];
			delete net2[i][j];
		}
		delete [] net1[i];
		delete [] net2[i];
	}
	delete [] net1;
	delete [] net2;

    double time = (double) (omp_get_wtime() - start_time_tot)*1000.0;
    // printf("LM_ch(): %f ms\n",time);

}

//top level run function, calls compute_jacobian and qrsolve. Cannot be parallelized.
void PairRANN::levenburg_marquardt_linesearch(){
	//jlen is number of rows; betalen is number of columns of jacobian
	char str[MAXLINE];
	int iter,i,j,nn;
	bool goodstep=true;
	double energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,force_fit1,reg_fit,reg_fit1,stress_fit,stress_fit1,stress_fitv;
	double lambda = lambda_initial;
	double vraise = lambda_increase;
	double vreduce = lambda_reduce;
	char line[MAXLINE];
    this->energy_fitv_best = 10^300;
	int i_off, j_off, j_offPi;
	double time1, time2;
	int rows = nsimr;
	int vrows = nsimv;
	int columns = betalen;
	if (doforces){
		rows +=natomsr*3;
		vrows += natomsv*3;
	}
	if (dostresses){
		rows +=nsimr*6;
		vrows ==nsimv*6;
	}
	int rmc = rows;
	if (doregularizer){
		rows += betalen;//do not regulate last bias
		rmc = rows-columns;
	}
	sprintf(str,"types=%d; betalen=columns=%d; rows=%d; rows-columns=%d, regularization:%d\n",nelementsp,betalen,rows,rmc, doregularizer);
	std::cout<<str;
	double J[rows*columns];
	double J1[rows*columns];
	double J2[columns*columns];
	double t2[columns];
	double target[rows];
	double target1[rows];
	double target2[rows];
	double targetv[vrows];
	double beta[columns];
	double beta1[columns];
	double beta2[columns];
	double D[columns];
	double *dp;
	double delta[rows];//extra length used internally in qrsolve
	dp = delta;
    double *tp,*tp1,*tp2,*bp,*bp1,*bp2,*Jp,*Jp1;
	tp = target;
	tp1 = target1;
	tp2 = target2;
	bp = beta;
	bp1 = beta1;
	bp2 = beta2;
	Jp = J;
	Jp1 = J1;
	force_fit = energy_fit = reg_fit = energy_fit1 = force_fit1 = reg_fit1 = 0.0;
	//clock_t start1 = clock();
	double start_time_tot = omp_get_wtime();
	if (doforces || dostresses){
		jacobian_force(Jp,tp,r,nsimr,natomsr,nets);
	}
	else{ 
		jacobian_convolution(Jp,tp,r,nsimr,natomsr,nets);
	}
	RANN::Net ***net1;
	RANN::Net ***net2;
	copy_network(&net1,nets);
	copy_network(&net2,nets);
	unpack_errors(tp,&energy_fit,&force_fit,&stress_fit,&reg_fit,rows,columns);
	double e3 = energy_fit*nsimr+force_fit*natomsr+stress_fit*nsimr*6+reg_fit*betalen;
	double initial_reg = regularizer;
	double initial_eng = energy_fit;
	flatten_beta(nets,bp);
	force_fitv = energy_fitv = 0.0;
	int counter = 0;
	int count = 0;
	int count1 = 0;
	int count2 = 0;
	int count3 = 0;
	int count4 = 0;
	int count5 = 0;
	int count5spin = 0;
	int count6 = 0;
	iter = 0;
	FILE *fid = fopen(log_file,"w");
	if (fid==NULL)errorf("couldn't open log file!");
	if (doforces && dostresses)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err: %.10e, s_err: %.10e, sv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,stress_fit,stress_fitv,reg_fit,lambda);
	else if (doforces)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,reg_fit,lambda);
	else if (dostresses)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, s_err: %.10e, sv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,stress_fit,stress_fitv,reg_fit,lambda);
	else sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,reg_fit,lambda);

	write_potential_file(true,line,0,initial_reg);
	double start2;
	while (iter<max_epochs){
		if (nsimv>0){
			//do validation forward pass
			if (doforces || dostresses){
				forward_pass_force(targetv,v,nsimv,nets);
			}
			else {
				forward_pass(targetv,v,nsimv,nets);
			}
			unpack_errors(targetv,&energy_fitv,&force_fitv,&stress_fitv,&reg_fit1,rows,1);
		}
		else{
			energy_fitv = 0.0;
			force_fitv = 0.0;
			stress_fitv = 0.0;
		}

		for (i=0;i<betalen;i++){
			i_off = i*betalen;
			for (int k=0;k<=i;k++){
				J2[i_off+k] = 0.0;
			}
		}
		#pragma omp parallel
		{
		#pragma omp for
		for (int i=0;i<betalen;i++){
			int i_off = i*betalen;
			for (int k=0;k<=i;k++){
				for (int j=0;j<rmc;j++){
					int j_off = j*betalen;
					int j_offPi = j_off+i;
					J2[i_off+k] += Jp[j_offPi]*Jp[j_off+k];
				}
			}
		}
		if (doregularizer){
			#pragma omp for
			for (int i=0;i<betalen;i++){
				int	i_off = i*betalen;
				int ij_off = rmc*betalen + i_off;
				J2[i_off+i]+=Jp[ij_off+i]*Jp[ij_off+i];
			}
		}
		#pragma omp barrier
		#pragma omp single
		{
		for (int i=0;i<betalen;i++){
			D[i] = J2[i*betalen+i];
			if (D[i]==0){errorf(FLERR,"Jacobian is rank deficient!\n");}//one or more weight/bias has no effect on the computed energy of any of the atoms.
			if (doregularizer) // t2 can be initialized with 0 or derivative w.r.t. weight
				t2[i]=Jp[rmc*betalen+i*betalen+i]*tp[rmc+i];
			else
				t2[i]=0;
		}
		}
		// loop splitting for threading. Initialization for t2 is done above.
		#pragma omp for
		for (int i=0;i<betalen;i++){
			// t2[i]=0;
			for (j=0;j<rmc;j++){
				//printf("%d %f\n",j,tp[j]);
				t2[i]+=Jp[j*betalen+i]*tp[j];
			}
		}

		}
		double adexp = 0.1;
		if (adaptive_regularizer) {regularizer *= sqrt(initial_reg/reg_fit*energy_fit);}
		if (regularizer<1e-8){
			doregularizer=false;
			rows -=betalen;
			rmc = 0;
		}

        bool is_write_potential = false;
        if (count == potential_output_freq){
			count = 0;
			if (((force_fitv+stress_fitv+energy_fitv)*natomsv+(energy_fit+force_fit+stress_fit)*natomsr)/natoms < this->energy_fitv_best) {
				this->energy_fitv_best = ((force_fitv+stress_fitv+energy_fitv)*natomsv+(energy_fit+force_fit+stress_fit)*natomsr)/natoms;
				is_write_potential = true;
			}
			else {
				count--;
			}
        }
		if (doforces && dostresses)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err: %.10e, s_err: %.10e, sv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,stress_fit,stress_fitv,reg_fit,lambda);
		else if (doforces)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, f_err: %.10e, fv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,reg_fit,lambda);
		else if (dostresses)sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, s_err: %.10e, sv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,stress_fit,stress_fitv,reg_fit,lambda);
		else sprintf(line,"iter: %d, evals: %d, e_err: %.10e, e1: %.10e, ev_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fit1,energy_fitv,reg_fit,lambda);

		std::cout<<line;
		fprintf(fid,"%s",line);
		count++;

        if (is_write_potential){
            write_potential_file(true,line,iter,initial_reg);
        }
        do_debugs(counter,tp,targetv,Jp,bp,dp,rows,columns);
		counter++;
		double lambda_min = lambda_reduce*lambda;
		double lambda_max = lambda_increase*lambda;
		double R = sqrt(5.0)*0.5-0.5;
		double C = 1.0-R;
		int nIter = ceil(-2.078087*log(tolerance/(lambda_max-lambda_min)));
		//nIter = 7;
		double x1 = (R*lambda_min+C*lambda_max);
		double x2 = (C*lambda_min+R*lambda_max);

		#pragma omp for
		for (i=0;i<betalen;i++){
			J2[i*betalen+i]=D[i]+sqrt(D[i]*x1);
		}
		chsolve(J2,betalen,t2,dp);
		#pragma omp for
		for (i=0;i<betalen;i++){
			bp1[i]=bp[i]+dp[i];
		}
		unflatten_beta(net1,bp1);
		if (doforces || dostresses){
			forward_pass_force(tp1,r,nsimr,net1);
		}
		else {
			forward_pass(tp1,r,nsimr,net1);
		}
		#pragma omp for
		for (i=0;i<betalen;i++){
			J2[i*betalen+i]=D[i]+sqrt(D[i]*x2);
		}
		chsolve(J2,betalen,t2,dp);
		#pragma omp for
		for (i=0;i<betalen;i++){
			bp2[i]=bp[i]+dp[i];
		}
		unflatten_beta(net2,bp2);
		if (doforces||dostresses){
			forward_pass_force(tp2,r,nsimr,net2);
		}
		else{
			forward_pass(tp2,r,nsimr,net2);
		}
		double e1;
		double e2;
		double *temp;
		e1 = 0.0;
		e2 = 0.0;
		#pragma omp parallel 
		{
		#pragma omp for reduction(+:e1,e2)
		for (i=0;i<jlen1;i++){
			e1 += tp1[i]*tp1[i];
			e2 += tp2[i]*tp2[i];
		}
		}
		for (int ii=0;ii<nIter;ii++){
			//printf("range: %f %f %f %f\n",lambda_min,x1,x2,lambda_max);
			//printf("ii: %d es: e3 %f e2 %f e1 %f\n",ii,e3,e2,e1);
			if (e1 > e2 && e3 > e2){
				lambda_min = x1;
				x1 = x2;
				e1 = e2;
				temp = tp1;
				tp1 = tp2;
				tp2 = temp;
				temp = bp1;
				bp1 = bp2;
				bp2 = temp;
				x2 = C*lambda_min+R*lambda_max;
				#pragma omp for
				for (i=0;i<betalen;i++){
					J2[i*betalen+i]=D[i]+sqrt(D[i]*x2);
				}
				chsolve(J2,betalen,t2,dp);
				#pragma omp for
				for (i=0;i<betalen;i++){
					bp2[i]=bp[i]+dp[i];
				}
				unflatten_beta(net2,bp2);
				if (doforces||dostresses){
					forward_pass_force(tp2,r,nsimr,net2);
				}
				else{
					forward_pass(tp2,r,nsimr,net2);
				}
				e2 = 0.0;
				#pragma omp parallel
				{
				#pragma omp for reduction(+:e2)
				for (j=0;j<jlen1;j++){
					e2+=tp2[j]*tp2[j];
				}
				}
			}
			else if (e2 > e1 && e3 > e1){
				lambda_max = x2;
				x2 = x1;
				e2 = e1;
				temp = tp2;
				tp2 = tp1;
				tp1 = temp;
				temp = bp2;
				bp2 = bp1;
				bp1 = temp;
				x1 = R*lambda_min+C*lambda_max;
				#pragma omp for
				for (i=0;i<betalen;i++){
					J2[i*betalen+i]=D[i]+sqrt(D[i]*x1);
				}
				chsolve(J2,betalen,t2,dp);
				#pragma omp for
				for (i=0;i<betalen;i++){
					bp1[i]=bp[i]+dp[i];
				}
				unflatten_beta(net1,bp1);
				if (doforces || dostresses){
					forward_pass_force(tp1,r,nsimr,net1);
				}
				else {
					forward_pass(tp1,r,nsimr,net1);
				}
				e1 = 0.0;
				#pragma omp parallel
				{
				#pragma omp for reduction(+ : e1)
				for (j=0;j<rows;j++){
					e1+=tp1[j]*tp1[j];
				}
				}
			}
			else {
				lambda_max = lambda*lambda_increase*lambda_increase;
				lambda_min = lambda*lambda_increase;
				x1 = R*lambda_min+C*lambda_max;
				x2 = C*lambda_min+R*lambda_max;
				#pragma omp for
				for (i=0;i<betalen;i++){
					J2[i*betalen+i]=D[i]+sqrt(D[i]*x1);
				}
				chsolve(J2,betalen,t2,dp);
				#pragma omp for
				for (i=0;i<betalen;i++){
					bp1[i]=bp[i]+dp[i];
				}
				unflatten_beta(net1,bp1);
				if (doforces || dostresses){
					forward_pass_force(tp1,r,nsimr,net1);
				}
				else {
					forward_pass(tp1,r,nsimr,net1);
				}
				#pragma omp for
				for (i=0;i<betalen;i++){
					J2[i*betalen+i]=D[i]+sqrt(D[i]*x2);
				}
				chsolve(J2,betalen,t2,dp);
				#pragma omp for
				for (i=0;i<betalen;i++){
					bp2[i]=bp[i]+dp[i];
				}
				unflatten_beta(net2,bp2);
				if (doforces||dostresses){
					forward_pass_force(tp2,r,nsimr,net2);
				}
				else{
					forward_pass(tp2,r,nsimr,net2);
				}
				e1 = 0.0;
				e2 = 0.0;
				#pragma omp parallel
				{
				#pragma omp for reduction(+ : e1,e2)
				for (j=0;j<jlen1;j++){
					e1 += tp1[j]*tp1[j];
					e2 += tp2[j]*tp2[j];
				}
				}
			}
		}
		if (e2<e1 && e2<e3) {
			temp = bp;
			bp = bp2;
			bp2 = temp;
			lambda = x2;
			unflatten_beta(nets,bp);
			if (doforces || dostresses){
				jacobian_force(Jp,tp,r,nsimr,natomsr,nets);
			}
			else {
				jacobian_convolution(Jp,tp,r,nsimr,natomsr,nets);
			}
		}
		else if (e1<e3 && e1<e3) {
			temp = bp;
			bp = bp1;
			bp1 = temp;
			lambda = x1;
			unflatten_beta(nets,bp);
			if (doforces || dostresses){
				jacobian_force(Jp,tp,r,nsimr,natomsr,nets);
			}
			else {
				jacobian_convolution(Jp,tp,r,nsimr,natomsr,nets);
			}
		}
		else {
			lambda = lambda*lambda_increase;
		}
		unpack_errors(tp,&energy_fit,&force_fit,&stress_fit,&reg_fit,rows,columns);
		iter++;
	}
	//delete dynamic memory use
		//delete dynamic memory use
	for (int i=0;i<nelementsp;i++){
		for (int j=0;j<netperelement[i];j++){
			delete net1[i][j];
			delete net2[i][j];
		}
		delete [] net1[i];
		delete [] net2[i];
	}
	delete [] net1;
	delete [] net2;

	// clock_t end = clock();
    // time1 = (double) (end-start1) / CLOCKS_PER_SEC * 1000.0;
	// sprintf(str,"LM_ch(): %f ms\n",time1);
	// std::cout<<str;
    double time = (double) (omp_get_wtime() - start_time_tot)*1000.0;
    // printf("LM_ch(): %f ms\n",time);

}

void PairRANN::bfgs(){
	//jlen is number of rows; betalen is number of columns of jacobian
	char str[MAXLINE];
	int iter,i,j,nn;
	bool goodstep=true;
	double energy_fit,energy_fit1,energy_fitv,force_fit,force_fitv,force_fit1,reg_fit,reg_fit1,stress_fit,stress_fit1,stress_fitv;
	double lambda = lambda_initial;
	double vraise = lambda_increase;
	double vreduce = lambda_reduce;
	char line[MAXLINE];
    this->energy_fitv_best = 10^300;
	int i_off, j_off, j_offPi;
	double time1, time2;
	bool reset = false;
	int rows = nsimr;
	int vrows = nsimv;
	int columns = betalen;
	if (doforces){
		rows +=natomsr*3;
		vrows += natomsv*3;
	}
	if (dostresses){
		rows +=nsimr*6;
		vrows ==nsimv*6;
	}
	int rmc = rows;
	if (doregularizer){
		rows += betalen;//do not regulate last bias
		rmc = rows-columns;
	}
	sprintf(str,"types=%d; betalen=columns=%d; rows=%d; rows-columns=%d, regularization:%d\n",nelementsp,betalen,rows,rmc, doregularizer);
	std::cout<<str;
	double J[rows*columns];
	double target[rows];
	double target1[rows];
	double target2[rows];
	double targetv[vrows];
	double beta[columns];
	double beta1[columns];
	double beta2[columns];
	double dX[columns];
	double dX1[columns];
	double *dXp;
	double *dXp1;
	double s[columns];
	double y[columns];
	double Hy[columns];
	double H[columns*columns];
	double bpr;
	double *dp;
	double delta[rows];//extra length used internally in qrsolve
	dp = delta;
    double *tp,*tp1,*tp2,*bp,*bp1,*bp2,*Jp,*Jp1,*sp,*Hp;
	tp = target;
	tp1 = target1;
	tp2 = target2;
	bp = beta;
	bp1 = beta1;
	bp2 = beta2;
	Jp = J;
	dXp = dX;
	dXp1 = dX1;
	sp = s;
	Hp = H;
	energy_fit = reg_fit = energy_fit1 = reg_fit1 = 0.0;
	//clock_t start1 = clock();
	double start_time_tot = omp_get_wtime();
	if (doforces || dostresses){
		jacobian_force(Jp,tp,r,nsimr,natomsr,nets);
	}
	else{ 
		jacobian_convolution(Jp,tp,r,nsimr,natomsr,nets);
	}
	unpack_errors(tp,&energy_fit,&force_fit,&stress_fit,&reg_fit,rows,columns);
	RANN::Net ***net1;
	RANN::Net ***net2;
	copy_network(&net1,nets);
	copy_network(&net2,nets); 
	double initial_reg = regularizer;
	double initial_eng = energy_fit;
	flatten_beta(nets,bp);
	energy_fitv = 0.0;
	int counter = 0;
	int count = 0;
	iter = 0;
	if (nsimv>0){
		//do validation forward pass
		if (doforces || dostresses){
			forward_pass_force(targetv,v,nsimv,nets);
		}
		else {
			forward_pass(targetv,v,nsimv,nets);
		}
		unpack_errors(targetv,&energy_fitv,&force_fitv,&stress_fitv,&reg_fit1,rows,1);	

	}
	else{
		energy_fitv = 0.0;
		force_fitv = 0.0;
		stress_fitv = 0.0;
	}
	FILE *fid = fopen(log_file,"w");
	if (fid==NULL)errorf("couldn't open log file!");
	if (doforces && dostresses)sprintf(line,"Niter: %d, evals: %d, e_err: %.10e, ev_err %.10e, f_err: %.10e, fv_err: %.10e, s_err: %.10e, sv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fitv,force_fit,force_fitv,stress_fit,stress_fitv,reg_fit,lambda);
	else if (doforces)sprintf(line,"Niter: %d, evals: %d, e_err: %.10e, ev_err %.10e, f_err: %.10e, fv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fitv,force_fit,force_fitv,reg_fit,lambda);
	else if (dostresses)sprintf(line,"Niter: %d, evals: %d, e_err: %.10e, ev_err %.10e, s_err: %.10e, sv_err: %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fitv,stress_fit,stress_fitv,reg_fit,lambda);
	else sprintf(line,"Niter: %d, evals: %d, e_err: %.10e, ev_err %.10e, r_err: %.10e, lambda: %.10e\n",iter,counter,energy_fit,energy_fitv,reg_fit,lambda);
	write_potential_file(true,line,0,initial_reg);
	printf("%s",line);
	double start2;
	double x1 = 0.0;
	double x2 = 1.0;
	//get gradient direction
	#pragma omp parallel
	{
	#pragma omp for
	for (j=0;j<betalen;j++){
		dX[j]=0.0;
		for (i=0;i<rmc;i++){
			dXp[j]+=2*Jp[i*betalen+j]*tp[i];
		}
		if (doregularizer){
			dXp[j]+=2*Jp[(rmc+j)*betalen+j]*tp[rmc+j];
		}
	}
	}
	for (j=0;j<betalen;j++){
		for (int k=0;k<betalen;k++){
			H[j*betalen+k]=0.0;
		}
		H[j*betalen+j]=1.0;
	}
	//set initial conjugate direction
	for (j=0;j<betalen;j++){
		sp[j]=dXp[j];
	}
	while (iter<max_epochs){
		//line search
		double e3 = energy_fit*nsimr+force_fit*nsimr+stress_fit*nsimr+reg_fit*betalen;
		double Gmag = 0.0;
		double G1mag = 0.0;
		#pragma omp parallel for reduction(+:Gmag,G1mag)
		for (j=0;j<betalen;j++){
			Gmag += sp[j]*sp[j];
			G1mag += dXp[j]*dXp[j];
		}
		Gmag = sqrt(Gmag);
		G1mag = sqrt(G1mag);
		double R = sqrt(5.0)*0.5-0.5;
		double C = 1.0-R;
		double alpha_min = e3/G1mag;
		double alpha_max = alpha_min*(5-4*log(e3/rmc)); 
		if (alpha_max<alpha_min*5){alpha_max=alpha_min*5;}
		int nIter = ceil(-2.078087*log(tolerance/abs(4*alpha_min)));
		//int nIter = 5;
		x1 = (R*alpha_min+C*alpha_max)/Gmag;
		x2 = (C*alpha_min+R*alpha_max)/Gmag;
		//printf("range: %f %f %f %f\n",alpha_min,x1*Gmag,x2*Gmag,alpha_max);
		#pragma omp parallel for
		for (j=0;j<betalen;j++){
			bp1[j] = bp[j]+x1*sp[j];
			bp2[j] = bp[j]+x2*sp[j];
		}
		unflatten_beta(net1,bp1);
		unflatten_beta(net2,bp2);
		if (doforces || dostresses){
			forward_pass_force(tp1,r,nsimr,net1);
			forward_pass_force(tp2,r,nsimr,net2);
		}
		else {
			forward_pass(tp1,r,nsimr,net1);
			forward_pass(tp2,r,nsimr,net2);
		}
		double e1;
		double e2;
		double *temp;
		e1 = 0.0;
		e2 = 0.0;
		#pragma omp parallel 
		{
		#pragma omp for reduction(+:e1,e2)
		for (i=0;i<jlen1;i++){
			e1 += tp1[i]*tp1[i];
			e2 += tp2[i]*tp2[i];
		}
		}
		for (i=0;i<nIter;i++){
			//printf("%d %f %f %f\n",nIter,e1/jlen2,e2/jlen2,e3/jlen2);
			//printf("range: %f %f %f %f\n",alpha_min,x1*Gmag,x2*Gmag,alpha_max);
			if (e1 > e2 && e3 > e2){
				alpha_min = x1*Gmag;
				x1 = x2;
				e1 = e2;
				temp = tp1;
				tp1 = tp2;
				tp2 = temp;
				temp = bp1;
				bp1 = bp2;
				bp2 = temp;
				x2 = (C*alpha_min+R*alpha_max)/Gmag;
				//#pragma omp for
				#pragma omp parallel for
				for (j=0;j<betalen;j++){
					bp2[j] = bp[j]+x2*sp[j];
				}
				//#pragma omp single
				//{
				unflatten_beta(net2,bp2);
				if (doforces||dostresses){
					forward_pass_force(tp2,r,nsimr,net2);
				}
				else{
					forward_pass(tp2,r,nsimr,net2);
				}
				//}
				e2 = 0.0;
				#pragma omp parallel
				{
				#pragma omp for reduction(+:e2)
				for (j=0;j<jlen1;j++){
					e2+=tp2[j]*tp2[j];
				}
				}
			}
			else if (e2 > e1 && e3 > e1){
				alpha_max = x2*Gmag;
				x2 = x1;
				e2 = e1;
				temp = tp2;
				tp2 = tp1;
				tp1 = temp;
				temp = bp2;
				bp2 = bp1;
				bp1 = temp;
				x1 = (R*alpha_min+C*alpha_max)/Gmag;

				//#pragma omp for
				#pragma omp parallel for
				for (j=0;j<betalen;j++){
					bp1[j] = bp[j]+x1*sp[j];
				}
				//#pragma omp single
				//{
				unflatten_beta(net1,bp1);
				if (doforces || dostresses){
					forward_pass_force(tp1,r,nsimr,net1);
				}
				else {
					forward_pass(tp1,r,nsimr,net1);
				}
				//}
				e1 = 0.0;
				#pragma omp parallel
				{
				#pragma omp for reduction(+ : e1)
				for (j=0;j<jlen1;j++){
					e1+=tp1[j]*tp1[j];
				}
				}
			}
			else {
				alpha_max = x1*Gmag;
				alpha_min = 0.0;
				x1 = (R*alpha_min+C*alpha_max)/Gmag;
				x2 = (C*alpha_min+R*alpha_max)/Gmag;

				//#pragma omp for
				#pragma omp parallel for
				for (j=0;j<betalen;j++){
					bp1[j] = bp[j]+x1*sp[j];
					bp2[j] = bp[j]+x2*sp[j];
				}
				//#pragma omp single
				//{
				unflatten_beta(net1,bp1);
				unflatten_beta(net2,bp2);
				if (doforces || dostresses){
					forward_pass_force(tp1,r,nsimr,net1);
					forward_pass_force(tp2,r,nsimr,net2);
				}
				else {
					forward_pass(tp1,r,nsimr,net1);
					forward_pass(tp2,r,nsimr,net2);
				}
				//}
				e1 = 0.0;
				e2 = 0.0;
				#pragma omp parallel
				{
				#pragma omp for reduction(+ : e1,e2)
				for (j=0;j<jlen1;j++){
					e1 += tp1[j]*tp1[j];
					e2 += tp2[j]*tp2[j];
				}
				}
			}
		}
		if (e2<e1 && e2<e3) {
			temp = bp;
			bp = bp2;
			bp2 = temp;
			#pragma omp parallel for
			for (j=0;j<betalen;j++){
				sp[j]=sp[j]*x2;
			}
		}
		else if (e1<e3 && e1<e3) {
			temp = bp;
			bp = bp1;
			bp1 = temp;
			#pragma omp parallel for
			for (j=0;j<betalen;j++){
				sp[j]=sp[j]*x1;
			}
		}
		else {
			#pragma omp parallel for
			for (i=0;i<betalen;i++){
				bp[i] = 0.5*(bp[i]+bp1[i]);
				sp[i] = sp[i]*(x1+x2)*0.5;
			}
		}
		unflatten_beta(nets,bp);
		if (doforces || dostresses){
			jacobian_force(Jp,tp,r,nsimr,natomsr,nets);
		}
		else {
			jacobian_convolution(Jp,tp,r,nsimr,natomsr,nets);
		}
		unpack_errors(tp,&energy_fit,&force_fit,&stress_fit,&reg_fit,rows,columns);
		start2 = omp_get_wtime();
		if (nsimv>0){
			//do validation forward pass
			if (doforces || dostresses){
				forward_pass_force(targetv,v,nsimv,nets);
			}
			else {
				forward_pass(targetv,v,nsimv,nets);
			}
			unpack_errors(targetv,&energy_fitv,&force_fitv,&stress_fitv,&reg_fit1,rows,1);
		}
        bool is_write_potential = false;
        if (count == potential_output_freq){
			count = 0;
			if (((force_fitv+stress_fitv+energy_fitv)*natomsv+(energy_fit+force_fit+stress_fit)*natomsr)/natoms < this->energy_fitv_best) {
				this->energy_fitv_best = ((force_fitv+stress_fitv+energy_fitv)*natomsv+(energy_fit+force_fit+stress_fit)*natomsr)/natoms;
				is_write_potential = true;
			}
			else {
				count--;
			}
        }
		double adexp = 0.1;
		if (adaptive_regularizer) {regularizer *= sqrt(initial_reg/reg_fit*energy_fit);}
		if (regularizer<1e-8){
			doregularizer=false;
			rows -=betalen;
		 	rmc = 0;
		}
		if (doforces && dostresses)sprintf(line,"Niter: %d, evals: %d, e_err: %.10e, ev_err %.10e, f_err: %.10e, fv_err: %.10e, s_err: %.10e, sv_err: %.10e, r_err: %.10e, lambda: %.10e\n",nIter,counter,energy_fit,energy_fitv,force_fit,force_fitv,stress_fit,stress_fitv,reg_fit,lambda);
		else if (doforces)sprintf(line,"Niter: %d, evals: %d, e_err: %.10e, ev_err %.10e, f_err: %.10e, fv_err: %.10e, r_err: %.10e, lambda: %.10e\n",nIter,counter,energy_fit,energy_fitv,force_fit,force_fitv,reg_fit,lambda);
		else if (dostresses)sprintf(line,"Niter: %d, evals: %d, e_err: %.10e, ev_err %.10e, s_err: %.10e, sv_err: %.10e, r_err: %.10e, lambda: %.10e\n",nIter,counter,energy_fit,energy_fitv,stress_fit,stress_fitv,reg_fit,lambda);
		else sprintf(line,"Niter: %d, evals: %d, e_err: %.10e, ev_err %.10e, r_err: %.10e, lambda: %.10e\n",nIter,counter,energy_fit,energy_fitv,reg_fit,lambda);
		std::cout<<line;
		fprintf(fid,"%s",line);
		count++;
        if (is_write_potential){
            write_potential_file(true,line,counter,initial_reg);
        }
		do_debugs(counter,tp,targetv,Jp,bp,dp,rows,columns);
        counter++;
		//get new gradient direction
		double *tempdX;
		tempdX = dXp;
		dXp = dXp1;
		dXp1 = tempdX;
		double numen;
		double denom;
		numen = 0.0;
		denom = 0.0;
		#pragma omp parallel
		{
		#pragma omp for
		for (j=0;j<betalen;j++){
			dXp[j]=0.0;
			for (i=0;i<rmc;i++){
				dXp[j]+=2*Jp[i*betalen+j]*tp[i];
			}
			if (doregularizer){
				dXp[j]+=2*Jp[(rmc+j)*betalen+j]*tp[rmc+j];
			}
		}
		if (abs(x2)>1e-12){
			reset = false;
			#pragma omp for 
			for (j=0;j<betalen;j++){
				y[j] = dXp[j]-dXp1[j];
			}
			#pragma omp for reduction (+ : denom,numen)
			for (j=0;j<betalen;j++){
				denom+=sp[j]*y[j];
				Hy[j] = 0;
				for (int k=0;k<betalen;k++){
					Hy[j]+=Hp[j*betalen+k]*y[k];
				}
				numen+=y[j]*Hy[j]+sp[j]*y[j];
			}
			#pragma omp for
			for (j=0;j<betalen;j++){
				for (int k=0;k<betalen;k++){
					Hp[j*betalen+k]+=numen/denom/denom*sp[j]*sp[k]-(Hy[j]*sp[k]+sp[j]*Hy[k])/denom;
				}
			}
			#pragma omp for
			for (j=0;j<betalen;j++){
				sp[j]=0.0;
				for (int k=0;k<betalen;k++){
					sp[j]+=Hp[j*betalen+k]*dXp[k];
				}
			}
		}
		else {
			//if (reset == true){errorf(FLERR,"linesearch alpha is zero");}
			reset = true;
			for (j=0;j<betalen;j++){
				for (int k=0;k<betalen;k++){
					H[j*betalen+k]=0.0;
				}
				H[j*betalen+j]=1.0;
			}
			//set initial conjugate direction
			for (j=0;j<betalen;j++){
				sp[j]=dXp[j];
				//sp[j]=-sp[j];
			}
		}
		}
		if (energy_fit<tolerance){
			std::cout<<"Terminating because reached convergence tolerance\n";
			write_potential_file(true,line,iter,initial_reg);
			break;
		}
	}
	//delete dynamic memory use
	for (int i=0;i<nelementsp;i++){
		for (int j=0;j<netperelement[i];j++){
			delete net1[i][j];
			delete net2[i][j];
		}
		delete [] net1[i];
		delete [] net2[i];
	}
	delete [] net1;
	delete [] net2;

	// clock_t end = clock();
    // time1 = (double) (end-start1) / CLOCKS_PER_SEC * 1000.0;
	// sprintf(str,"LM_ch(): %f ms\n",time1);
	// std::cout<<str;
    double time = (double) (omp_get_wtime() - start_time_tot)*1000.0;
    // printf("LM_ch(): %f ms\n",time);

}


void PairRANN::jacobian_convolution(double *J,double *target,int *s,int sn,int natoms,RANN::Net ***nets){
	for (int i=0;i<sn;i++){
		target[i]=-sims[s[i]].energy*sims[s[i]].energy_weight;
	}
	for (int i=0;i<nelementsp;i++){
		for (int k=0;k<netperelement[i];k++){
			if (nets[i][k]->layers>0)nets[i][k]->jacobian_convolution(J,target,s,sn,natoms);
		}
	}
}

void PairRANN::jacobian_force(double *J,double *target,int *s,int sn,int natoms,RANN::Net ***nets){
	for (int i=0;i<sn;i++){
		int sr = sims[s[i]].startrow;
		if (doforces){
			for (int j=0;j<sims[s[i]].inum;j++){
				for (int v=0;v<3;v++){
					target[sr+j*3+v]=-sims[s[i]].f[j][v]*sims[s[i]].force_weight;
				}
			}
			sr+=sims[s[i]].inum*3;
		}
		target[sr]=-sims[s[i]].energy*sims[s[i]].energy_weight;
		if (dostresses){
			double volumei = 1/(sims[s[i]].box[0][0]*sims[s[i]].box[1][1]*sims[s[i]].box[2][2]);
			sr++;
			for (int j=0;j<3;j++){
				for (int m=j;m<3;m++){
					target[sr]=-sims[s[i]].stress[j][m]*sims[s[i]].force_weight*volumei;
					sr++;
				}
			}
		}
	}
	int rows = nsimr;
	int vrows = nsimv;
	int columns = betalen;
	if (doforces){
		rows +=natomsr*3;
		vrows += natomsv*3;
	}
	if (dostresses){
		rows +=nsimr*6;
		vrows +=nsimv*6;
	}
	int rmc = rows;
	if (doregularizer){
		rows += betalen;//do not regulate last bias
		rmc = rows-columns;
	}
	for (int i = 0;i<rows;i++){
		for (int j=0;j<columns;j++){
			J[i*columns+j]=0.0;
		}
	}
	for (int i=0;i<nelementsp;i++){
		for (int k=0;k<netperelement[i];k++){
			if (nets[i][k]->layers>0)nets[i][k]->force_fit(J,target,s,sn,natoms);
		}
	}
}

//finds total error from features
void PairRANN::forward_pass(double *target,int *s,int sn,RANN::Net ***nets){
	for (int i=0;i<sn;i++){
		target[i]=-sims[s[i]].energy*sims[s[i]].energy_weight;
	}
	for (int i=0;i<nelementsp;i++){
		for (int k=0;k<netperelement[i];k++){
			if (nets[i][k]->layers>0)nets[i][k]->forward_pass(target,s,sn);
		}
	}
}

//finds total error from features
void PairRANN::forward_pass_force(double *target,int *s,int sn,RANN::Net ***nets){
	for (int i=0;i<sn;i++){
		int sr = sims[s[i]].startrow;
		if (doforces){
			for (int j=0;j<sims[s[i]].inum;j++){
				for (int v=0;v<3;v++){
					target[sr+j*3+v]=-sims[s[i]].f[j][v]*sims[s[i]].force_weight;
				}
			}
			sr+=sims[s[i]].inum*3;
		}
		target[sr]=-sims[s[i]].energy*sims[s[i]].energy_weight;
		sr++;
		if (dostresses){
			double volumei = 1/(sims[s[i]].box[0][0]*sims[s[i]].box[1][1]*sims[s[i]].box[2][2]);
			for (int j=0;j<3;j++){
				for (int m=j;m<3;m++){
					target[sr]=-sims[s[i]].stress[j][m]*sims[s[i]].force_weight*volumei;
					sr++;
				}
			}
		}
	}
	for (int i=0;i<nelementsp;i++){
		for (int k=0;k<netperelement[i];k++){
			if (nets[i][k]->layers>0)nets[i][k]->forward_pass_force(target,s,sn);
		}
	}
}

//finds per atom energies from features
void PairRANN::get_per_atom_energy(double **energies,int *s,int sn,RANN::Net ***nets){
	for (int i=0;i<sn;i++){
		energies[i]=new double [sims[i].inum];
		for (int j=0;j<sims[i].inum;j++){
			energies[i][j]=0;
		}
	}
	for (int i=0;i<nelementsp;i++){
		for (int k=0;k<netperelement[i];k++){
			if (nets[i][k]->layers>0)nets[i][k]->get_per_atom_energy(energies,s,sn);
		}
	}
}

//finds total energy and per atom forces from features
void PairRANN::propagateforward(double *energy,double **force,int ii,int jnum,int itype,double *features, double *dfeaturesx,double *dfeaturesy, double *dfeaturesz,double *exfeatures, double *dexfeaturesx,double *dexfeaturesy, double *dexfeaturesz,int *jl,int nn, RANN::Net **nets, double *xn, double *yn, double *zn) {
	for (int k=0;k<netperelement[itype];k++){
		if (nets[k]->layers>0)nets[k]->propagateforward(energy,force,ii,jnum,itype,features, dfeaturesx,dfeaturesy,dfeaturesz,exfeatures, dexfeaturesx,dexfeaturesy,dexfeaturesz,jl,nn,xn,yn,zn);
	}
}

void PairRANN::propagateforwardspin(double *energy,double **force,double **fm,double **hm,int ii,int jnum,int itype,double *features, double *dfeaturesx,double *dfeaturesy, double *dfeaturesz,double *exfeatures, double *dexfeaturesx,double *dexfeaturesy, double *dexfeaturesz, double *sx, double *sy, double *sz, double *sxx, double *sxy, double *sxz, double *syy, double *syz, double *szz,int *jl,int nn, RANN::Net **nets) {
	for (int k=0;k<netperelement[itype];k++){
		if (nets[k]->layers>0)nets[k]->propagateforwardspin(energy,force,fm,hm,ii,jnum,itype,features,dfeaturesx,dfeaturesy,dfeaturesz,exfeatures,dexfeaturesx,dexfeaturesy,dexfeaturesz,sx,sy,sz,sxx,sxy,sxz,syy,syz,szz,jl,nn);
	}
}

void PairRANN::cull_neighbor_list(double *xn,double *yn, double *zn,int *tn, int* jnum,int *jl,int i,int sn,double cutmax){
	int *jlist,j,count,jj,*type,jtype;
	double xtmp,ytmp,ztmp,delx,dely,delz,rsq;
	double **x = sims[sn].x;
	xtmp = x[i][0];
	ytmp = x[i][1];
	ztmp = x[i][2];
	type = sims[sn].type;
	jlist = sims[sn].firstneigh[i];
	count = 0;
	for (jj=0;jj<jnum[0];jj++){
		j = jlist[jj];
		j &= NEIGHMASK;
		jtype = map[type[j]];
		delx = xtmp - x[j][0];
		dely = ytmp - x[j][1];
		delz = ztmp - x[j][2];
		rsq = delx*delx + dely*dely + delz*delz;
		if (rsq>cutmax*cutmax){
			continue;
		}
		xn[count]=delx;
		yn[count]=dely;
		zn[count]=delz;
		tn[count]=jtype;
		//jl[count]=sims[sn].id[j];
		jl[count]=j;
		//jl is currently only used to calculate spin dot products.
		//j includes ghost atoms. id maps back to atoms in the box across periodic boundaries.
		//lammps code uses id instead of j because spin spirals are not supported.
		count++;
	}
	jl[count]=i;
	jnum[0]=count+1;
}

void PairRANN::screen_neighbor_list(double *xn,double *yn, double *zn,int *tn, int* jnum,int *jl,int i,int sn,bool *Bij,double *Sik, double *dSikx, double*dSiky, double *dSikz, double *dSijkx, double *dSijky, double *dSijkz){
	double xnc[jnum[0]],ync[jnum[0]],znc[jnum[0]];
	double Sikc[jnum[0]];
	double dSikxc[jnum[0]];
	double dSikyc[jnum[0]];
	double dSikzc[jnum[0]];
	double dSijkxc[jnum[0]][jnum[0]];
	double dSijkyc[jnum[0]][jnum[0]];
	double dSijkzc[jnum[0]][jnum[0]];
	int jj,kk,count,count1,tnc[jnum[0]],jlc[jnum[0]];
	count = 0;
	for (jj=0;jj<jnum[0]-1;jj++){
		if (Bij[jj]){
			count1 = 0;
			xnc[count]=xn[jj];
			ync[count]=yn[jj];
			znc[count]=zn[jj];
			tnc[count]=tn[jj];
			jlc[count]=jl[jj];
			Sikc[count]=Sik[jj];
			dSikxc[count]=dSikx[jj];
			dSikyc[count]=dSiky[jj];
			dSikzc[count]=dSikz[jj];
			for (kk=0;kk<jnum[0]-1;kk++){
				if (Bij[kk]){
					dSijkxc[count][count1] = dSijkx[jj*(jnum[0]-1)+kk];
					dSijkyc[count][count1] = dSijky[jj*(jnum[0]-1)+kk];
					dSijkzc[count][count1] = dSijkz[jj*(jnum[0]-1)+kk];
					count1++;
				}
			}
			count++;
		}
	}
	jnum[0]=count+1;
	for (jj=0;jj<count;jj++){
		xn[jj]=xnc[jj];
		yn[jj]=ync[jj];
		zn[jj]=znc[jj];
		tn[jj]=tnc[jj];
		jl[jj]=jlc[jj];
		Bij[jj] = true;
		Sik[jj]=Sikc[jj];
		dSikx[jj]=dSikxc[jj];
		dSiky[jj]=dSikyc[jj];
		dSikz[jj]=dSikzc[jj];
		for (kk=0;kk<count;kk++){
			dSijkx[jj*count+kk] = dSijkxc[jj][kk];
			dSijky[jj*count+kk] = dSijkyc[jj][kk];
			dSijkz[jj*count+kk] = dSijkzc[jj][kk];
		}
	}
}

//adapted from public domain source at:  http://math.nist.gov/javanumerics/jama
//replaced with Cholesky solution for greater speed for finding solve step. Still used to process input data.
void PairRANN::qrsolve(double *A,int m,int n,double *b, double *x_){
	double QR_[m*n];
//	char str[MAXLINE];
	double Rdiag[n];
	int i=0, j=0, k=0;
	int j_off, k_off;
	double nrm;
    // loop to copy QR from A.
	for (k=0;k<n;k++){
		k_off = k*m;
		for (i=0;i<m;i++){
			QR_[k_off+i]=A[i*n+k];
		}
	}
    for (k = 0; k < n; k++) {
       // Compute 2-norm of k-th column.
       nrm = 0.0;
       k_off = k*m;
       for (i = k; i < m; i++) {
			nrm += QR_[k_off+i]*QR_[k_off+i];
       }
       if (nrm==0.0){
    	   errorf(FLERR,"Jacobian is rank deficient!\n");
       }
       nrm = sqrt(nrm);
	   // Form k-th Householder vector.
	   if (QR_[k_off+k] < 0) {
		 nrm = -nrm;
 	   }
	   for (i = k; i < m; i++) {
		 QR_[k_off+i] /= nrm;
	   }
	   QR_[k_off+k] += 1.0;

	   // Apply transformation to remaining columns.
	   for (j = k+1; j < n; j++) {
		 double s = 0.0;
		 j_off = j*m;
		 for (i = k; i < m; i++) {
			s += QR_[k_off+i]*QR_[j_off+i];
		 }
		 s = -s/QR_[k_off+k];
		 for (i = k; i < m; i++) {
			QR_[j_off+i] += s*QR_[k_off+i];
		 }
	   }
       Rdiag[k] = -nrm;
    }
    //loop to find least squares
    for (int j=0;j<m;j++){
    	x_[j] = b[j];
    }
    // Compute Y = transpose(Q)*b
	for (int k = 0; k < n; k++)
	{
		k_off = k*m;
		double s = 0.0;
		for (int i = k; i < m; i++)
		{
		   s += QR_[k_off+i]*x_[i];
		}
		s = -s/QR_[k_off+k];
		for (int i = k; i < m; i++)
		{
		   x_[i] += s*QR_[k_off+i];
		}
	}
	// Solve R*X = Y;
	for (int k = n-1; k >= 0; k--)
	{
		k_off = k*m;
		x_[k] /= Rdiag[k];
		for (int i = 0; i < k; i++) {
		   x_[i] -= x_[k]*QR_[k_off+i];
		}
	}
}

//adapted from public domain source at:  http://math.nist.gov/javanumerics/jama
void PairRANN::chsolve(double *A,int n,double *b, double *x){

	//clock_t start = clock();
	double start_time = omp_get_wtime();

	int	nthreads=omp_get_num_threads();

	double L_[n*n]; // was L_[n][n]
	int i,j,k;
	int iXn, jXn, kXn;
	double d, s;

	// initialize L
	for (k=0;k<n*n;k++){
		L_[k]=0.0;
	}

	// Cholesky-Crout decomposition
	#pragma omp parallel default(none) shared (A,L_,n,s)
	{
	for (int j = 0; j <n; j++) {
		int jXn = j*n;
		s = 0.0;
		// #pragma omp for schedule(static) reduction(+:s)
		for (int k = 0; k < j; k++) {
			s += L_[jXn + k] * L_[jXn + k];
		}
		#pragma omp barrier
		double d = A[jXn+j] - s;
		#pragma omp single
		{
		if (d>0){
			L_[jXn + j] = sqrt(d);
		}
		}
		//// #pragma omp parallel for schedule(static) default(none) shared (A,L_,n,j,jXn)
		////#pragma omp barrier
		#pragma omp for schedule(static)
		for (int i = j+1; i <n; i++) {
			int iXn = i * n;
			double sum = 0.0;
			for (int k = 0; k < j; k++) {
				sum += L_[iXn + k] * L_[jXn + k];
			}
			L_[iXn + j] =  (A[iXn + j] - sum) / L_[jXn + j];
		}
	}
	}
	// Solve L*
	// Forward substitution to solve L*y = b;
	// #pragma omp parallel default(none) shared (x,b,L_,n,s) private(i)
	// #pragma omp parallel
	{
	for (int k = 0; k < n; k++)
	{
		int kXn = k*n;
		s = 0.0;
		// #pragma omp parallel for default(none) reduction(+:s) schedule(static) shared (x,L_,kXn,k) private(i) if (nthreads>k)
		// #pragma omp for reduction(+:s) schedule(static)
		for (i = 0; i < k; i++) {
			s += x[i]*L_[kXn+i];
		}
		// #pragma omp single
		x[k] = (b[k] - s) / L_[kXn+k];
	}
	}
	// Backward substitution to solve L'*X = Y; omp does not work
	for (int k = n-1; k >= 0; k--)
	{
		double s = 0.0;
		for (int i = k+1; i < n; i++) {
			s += x[i]*L_[i*n+k];
		}
		x[k] = (x[k] - s)/L_[k*n+k];
	}

	//	clock_t end = clock();
//	double time = (double) (end-start) / CLOCKS_PER_SEC * 1000.0;
	double time = (double) (omp_get_wtime() - start_time)*1000.0;
	//printf(" - chsolve(): %f ms\n",time);

	return;
}

//writes files used for restarting and final output:
void PairRANN::write_potential_file(bool writeparameters, char *header,int iter, double reg){
	int i,j,k,l;
	char filename[strlen(potential_output_file)+10];
	if (overwritepotentials){
		sprintf(filename,"%s",potential_output_file);
	}
	else {
		sprintf(filename,"%s.%d",potential_output_file,iter);
	}
	FILE *fid = fopen(filename,"w");
	if (fid==NULL){
		printf("%s",filename);
		errorf("Invalid parameter file name");
	}
	RANN::Net ***net_out;
	if (normalizeinput){
		unnormalize_net(&net_out,nets);
	}
	else {
		copy_network(&net_out,nets);
	}
	fprintf(fid,"#");
	fprintf(fid,header);
	//atomtypes section
	fprintf(fid,"atomtypes:\n");
	for (i=0;i<nelements;i++){
		fprintf(fid,"%s ",elements[i]);
	}
	fprintf(fid,"\n");
	//mass section
	for (i=0;i<nelements;i++){
		fprintf(fid,"mass:%s:\n",elements[i]);
		fprintf(fid,"%f\n",mass[i]);
	}
	//fingerprints per element section
	for (i=0;i<nelementsp;i++){
		if (fingerprintperelement[i]>0){
			fprintf(fid,"fingerprintsperelement:%s:\n",elementsp[i]);
			fprintf(fid,"%d\n",fingerprintperelement[i]);
		}
	}
	//fingerprints section:
	for (i=0;i<nelementsp;i++){
		bool printheader = true;
		for (j=0;j<fingerprintperelement[i];j++){
			if (printheader){
				fprintf(fid,"fingerprints:");
				fprintf(fid,"%s",elementsp[fingerprints[i][j]->atomtypes[0]]);
				for (k=1;k<fingerprints[i][j]->n_body_type;k++){
					fprintf(fid,"_%s",elementsp[fingerprints[i][j]->atomtypes[k]]);
				}
				fprintf(fid,":\n");
			}
			else {fprintf(fid,"\t");}
			fprintf(fid,"%s_%d",fingerprints[i][j]->style,fingerprints[i][j]->id);
			printheader = true;
			if (j<fingerprintperelement[i]-1 && fingerprints[i][j]->n_body_type == fingerprints[i][j+1]->n_body_type){
				printheader = false;
				for (k=1;k<fingerprints[i][j]->n_body_type;k++){
					if (fingerprints[i][j]->atomtypes[k]!=fingerprints[i][j+1]->atomtypes[k]){
						printheader = true;
						fprintf(fid,"\n");
						break;
					}
				}
			}
			else fprintf(fid,"\n");
		}
	}
	//fingerprint contants section:
	for (i=0;i<nelementsp;i++){
		for (j=0;j<fingerprintperelement[i];j++){
			fingerprints[i][j]->write_values(fid);
		}
	}
	//screening section
	for (i=0;i<nelements;i++){
		for (j=0;j<nelements;j++){
			for (k=0;k<nelements;k++){
				fprintf(fid,"screening:%s_%s_%s:Cmax:\n",elements[i],elements[j],elements[k]);
				fprintf(fid,"%f\n",screening_max[i*nelements*nelements+j*nelements+k]);
				fprintf(fid,"screening:%s_%s_%s:Cmin:\n",elements[i],elements[j],elements[k]);
				fprintf(fid,"%f\n",screening_min[i*nelements*nelements+j*nelements+k]);
			}
		}
	}
	//network section
	for (i=0;i<nelements;i++){
		if (netperelement[i]>0){
			fprintf(fid,"netsperelement:%s:\n",elementsp[i]);
			fprintf(fid,"%d\n",netperelement[i]);
			fprintf(fid,"nets:%s:\n",elementsp[i]);
			for (j=0;j<netperelement[i];j++){
				fprintf(fid,"%s_%d\t",net_out[i][j]->style,net_out[i][j]->id);
			}
			fprintf(fid,"\n");
		}
	}
	for (i=0;i<nelementsp;i++){
		for (j=0;j<netperelement[i];j++){
			net_out[i][j]->write_values(fid);
		}
	}
	//state equation per element section
	for (i=0;i<nelementsp;i++){
		if (stateequationperelement[i]>0){
			fprintf(fid,"stateequationsperelement:%s:\n",elementsp[i]);
			fprintf(fid,"%d\n",stateequationperelement[i]);
		}
	}
	//state equations section:
	for (i=0;i<nelementsp;i++){
		bool printheader = true;
		for (j=0;j<stateequationperelement[i];j++){
			if (printheader){
				fprintf(fid,"stateequations:");
				fprintf(fid,"%s",elementsp[state[i][j]->atomtypes[0]]);
				for (k=1;k<state[i][j]->n_body_type;k++){
					fprintf(fid,"_%s",elementsp[state[i][j]->atomtypes[k]]);
				}
				fprintf(fid,":\n");
			}
			else {fprintf(fid,"\t");}
			fprintf(fid,"%s_%d",state[i][j]->style,state[i][j]->id);
			printheader = true;
			if (j<stateequationperelement[i]-1 && state[i][j]->n_body_type == state[i][j+1]->n_body_type){
				printheader = false;
				for (k=1;k<state[i][j]->n_body_type;k++){
					if (state[i][j]->atomtypes[k]!=state[i][j+1]->atomtypes[k]){
						printheader = true;
						fprintf(fid,"\n");
						break;
					}
				}
			}
			else fprintf(fid,"\n");
		}
	}
	//state equations contants section:
	for (i=0;i<nelementsp;i++){
		for (j=0;j<stateequationperelement[i];j++){
			state[i][j]->write_values(fid);
		}
	}
	//calibration parameters section
	if (writeparameters){
		fprintf(fid,"calibrationparameters:algorithm:\n");
		fprintf(fid,"%s\n",algorithm);
		fprintf(fid,"calibrationparameters:dumpdirectory:\n");
		fprintf(fid,"%s\n",dump_directory);
		fprintf(fid,"calibrationparameters:doforces:\n");
		fprintf(fid,"%d\n",doforces);
		fprintf(fid,"calibrationparameters:dostresses:\n");
		fprintf(fid,"%d\n",dostresses);
		fprintf(fid,"calibrationparameters:normalizeinput:\n");
		fprintf(fid,"%d\n",normalizeinput);
		fprintf(fid,"calibrationparameters:tolerance:\n");
		fprintf(fid,"%.10e\n",tolerance);
		fprintf(fid,"calibrationparameters:regularizer:\n");
		fprintf(fid,"%.10e\n",reg);
		fprintf(fid,"calibrationparameters:logfile:\n");
		fprintf(fid,"%s\n",log_file);
		fprintf(fid,"calibrationparameters:potentialoutputfile:\n");
		fprintf(fid,"%s\n",potential_output_file);
		fprintf(fid,"calibrationparameters:potentialoutputfreq:\n");
		fprintf(fid,"%d\n",potential_output_freq);
		fprintf(fid,"calibrationparameters:maxepochs:\n");
		fprintf(fid,"%d\n",max_epochs);
		// for (i=0;i<nelements;i++){
		// 	for (j=0;j<net_out[i].layers-1;j++){
		// 		for (int k=0;k<net_out[i].bundles[j];k++){
		// 			if (net_out[i].identitybundle[j][k])continue;
		// 			bool anyfrozen = false;
		// 			for (int l=0;l<net_out[i].bundleoutputsize[j][k]*net_out[i].bundleinputsize[j][k];l++){
		// 				if (net_out[i].freezeW[j][k][l]){
		// 					anyfrozen = true;
		// 					break;
		// 				}
		// 			}
		// 			if (anyfrozen){
		// 				fprintf(fid,"calibrationparameters:freezeW:%d:%d\n",j,k);
		// 				for (int l=0;l<net_out[i].bundleoutputsize[j][k];l++){
		// 					for (int m=0;m<net_out[i].bundleinputsize[j][k];m++){
		// 						fprintf(fid,"%d ",net_out[i].freezeW[j][k][l*net_out[i].bundleoutputsize[j][k]+m]);
		// 					}
		// 					fprintf(fid,"\n");
		// 				}
		// 			}
		// 			anyfrozen = false;
		// 			for (int l=0;l<net_out[i].bundleoutputsize[j][k];l++){
		// 				if (net_out[i].freezeB[j][k][l]){
		// 					anyfrozen = true;
		// 					break;
		// 				}
		// 			}
		// 			if (anyfrozen){
		// 				fprintf(fid,"calibrationparameters:freezeB:%d:%d\n",j,k);
		// 				for (int l=0;l<net_out[i].bundleoutputsize[j][k];l++){
		// 					fprintf(fid,"%d\n",net_out[i].freezeB[j][k][l]);
		// 				}
		// 			}
		// 		}
		// 	}
		// }
		fprintf(fid,"calibrationparameters:validation:\n");
		fprintf(fid,"%f\n",validation);
		fprintf(fid,"calibrationparameters:overwritepotentials:\n");
		fprintf(fid,"%d\n",overwritepotentials);
		fprintf(fid,"calibrationparameters:debug1freq:\n");
		fprintf(fid,"%d\n",debug_level1_freq);
		fprintf(fid,"calibrationparameters:debug2freq:\n");
		fprintf(fid,"%d\n",debug_level2_freq);
		fprintf(fid,"calibrationparameters:debug3freq:\n");
		fprintf(fid,"%d\n",debug_level3_freq);
		fprintf(fid,"calibrationparameters:debug4freq:\n");
		fprintf(fid,"%d\n",debug_level4_freq);
		fprintf(fid,"calibrationparameters:debug5freq:\n");
		fprintf(fid,"%d\n",debug_level5_freq);
		fprintf(fid,"calibrationparameters:debug5spinfreq:\n");
		fprintf(fid,"%d\n",debug_level5_spin_freq);
		fprintf(fid,"calibrationparameters:debug6freq:\n");
		fprintf(fid,"%d\n",debug_level6_freq);
		fprintf(fid,"calibrationparameters:adaptiveregularizer:\n");
		fprintf(fid,"%d\n",adaptive_regularizer);
		fprintf(fid,"calibrationparameters:lambdainitial:\n");
		fprintf(fid,"%f\n",lambda_initial);
		fprintf(fid,"calibrationparameters:lambdaincrease:\n");
		fprintf(fid,"%f\n",lambda_increase);
		fprintf(fid,"calibrationparameters:lambdareduce:\n");
		fprintf(fid,"%f\n",lambda_reduce);
		fprintf(fid,"calibrationparameters:inumweight:\n");
		fprintf(fid,"%f\n",inum_weight);
		fprintf(fid,"calibrationparameters:seed:\n");
		fprintf(fid,"%d\n",seed);
		fprintf(fid,"calibrationparameters:targettype:\n");
		fprintf(fid,"%d\n",targettype);
	}
	fclose(fid);
	delete [] net_out;
}

void PairRANN::screen(double *Sik, double *dSikx, double*dSiky, double *dSikz, double *dSijkx, double *dSijky, double *dSijkz, bool *Bij, int ii,int sid,double *xn,double *yn,double *zn,int *tn,int jnum)
{
	//#pragma omp parallel
	{
	//see Baskes, Materials Chemistry and Physics 50 (1997) 152-1.58
	int i,*jlist,jj,j,kk,k,itype,jtype,ktype;
	double Sijk,Cijk,Cn,Cd,Dij,Dik,Djk,C,dfc,dC,**x;
	PairRANN::Simulation *sim = &sims[sid];
	double xtmp,ytmp,ztmp,delx,dely,delz,rij,delx2,dely2,delz2,rik,delx3,dely3,delz3,rjk;
	i = sim->ilist[ii];
	itype = map[sim->type[i]];
	for (int jj=0;jj<jnum;jj++){
		Sik[jj]=1;
		Bij[jj]=true;
		dSikx[jj]=0;
		dSiky[jj]=0;
		dSikz[jj]=0;
	}
	for (int jj=0;jj<jnum;jj++)
		for (kk=0;kk<jnum;kk++)
			dSijkx[jj*jnum+kk]=0;
	for (int jj=0;jj<jnum;jj++)
		for (kk=0;kk<jnum;kk++)
			dSijky[jj*jnum+kk]=0;
	for (int jj=0;jj<jnum;jj++)
		for (kk=0;kk<jnum;kk++)
			dSijkz[jj*jnum+kk]=0;
	for (kk=0;kk<jnum;kk++){//outer sum over k in accordance with source, some others reorder to outer sum over jj
		//if (Bij[kk]==false){continue;}
		ktype = tn[kk];
		delx2 = xn[kk];
		dely2 = yn[kk];
		delz2 = zn[kk];
		rik = delx2*delx2+dely2*dely2+delz2*delz2;
		if (rik>cutmax*cutmax){
			//Bij[kk]= false;
			continue;
		}
		for (jj=0;jj<jnum;jj++){
			if (jj==kk){continue;}
			//if (Bij[jj]==false){continue;}
			jtype = tn[jj];
			delx = xn[jj];
			dely = yn[jj];
			delz = zn[jj];
			rij = delx*delx+dely*dely+delz*delz;
			if (rij>cutmax*cutmax){
				//Bij[jj] = false;
				continue;
			}
			delx3 = delx2-delx;
			dely3 = dely2-dely;
			delz3 = delz2-delz;
			rjk = delx3*delx3+dely3*dely3+delz3*delz3;
			if (rik+rjk-rij<1e-13){continue;}//bond angle > 90 degrees
			if (rik+rij-rjk<1e-13){continue;}//bond angle > 90 degrees
			double Cmax = screening_max[itype*nelements*nelements+jtype*nelements+ktype];
			double Cmin = screening_min[itype*nelements*nelements+jtype*nelements+ktype];
			double temp1 = rij-rik+rjk;
			Cn = temp1*temp1-4*rij*rjk;
			temp1 = rij-rjk;
			Cd = temp1*temp1-rik*rik;
			Cijk = Cn/Cd;
			C = (Cijk-Cmin)/(Cmax-Cmin);
			if (C>=1){continue;}
			else if (C<=0){
				//Bij[kk]=false;
				Sik[kk]=0.0;
				dSikx[kk]=0.0;
				dSiky[kk]=0.0;
				dSikz[kk]=0.0;
				break;
			}
			dC = Cmax-Cmin;
			dC *= dC;
			dC *= dC;
			temp1 = 1-C;
			temp1 *= temp1;
			temp1 *= temp1;
			Sijk = 1-temp1;
			Sijk *= Sijk;
			Dij = 4*rik*(Cn+4*rjk*(rij+rik-rjk))/Cd/Cd;
			Dik = -4*(rij*Cn+rjk*Cn+8*rij*rik*rjk)/Cd/Cd;
			Djk = 4*rik*(Cn+4*rij*(rik-rij+rjk))/Cd/Cd;
			temp1 = Cijk-Cmax;
			double temp2 = temp1*temp1;
			dfc = 8*temp1*temp2/(temp2*temp2-dC);
			Sik[kk] *= Sijk;
			dSijkx[kk*jnum+jj] = dfc*(delx*Dij-delx3*Djk);
			dSikx[kk] += dfc*(delx2*Dik+delx3*Djk);
			dSijky[kk*jnum+jj] = dfc*(dely*Dij-dely3*Djk);
			dSiky[kk] += dfc*(dely2*Dik+dely3*Djk);
			dSijkz[kk*jnum+jj] = dfc*(delz*Dij-delz3*Djk);
			dSikz[kk] += dfc*(delz2*Dik+delz3*Djk);
		}
	}
	}
}

//treats # as starting a comment to be ignored.
int PairRANN::count_words(char *line){
	return count_words(line,": ,\t_\n");
}

int PairRANN::count_words(char *line,char *delimiter){
	int n = strlen(line) + 1;
	char copy[n];
	strncpy(copy,line,n);
	char *ptr;
	if ((ptr = strchr(copy,'#'))) *ptr = '\0';
	if (strtok(copy,delimiter) == NULL) {
		return 0;
	}
	n=1;
	while ((strtok(NULL,delimiter))) n++;
	return n;
}

void PairRANN::errorf(const std::string &file, int line,const char *message){
	//see about adding message to log file
	printf("Error: file: %s, line: %d\n%s\n",file,line,message);
	exit(1);
}

void PairRANN::errorf(char *file, int line,const char *message){
	//see about adding message to log file
	printf("Error: file: %s, line: %d\n%s\n",file,line,message);
	exit(1);
}

void PairRANN::errorf(const char *message){
	//see about adding message to log file
	std::cout<<message;
	std::cout<<"\n";
	exit(1);
}


int PairRANN::factorial(int n) {
   if ((n==0)||(n==1))
      return 1;
   else
      return n*factorial(n-1);
}

int PairRANN::inumeric(const char *filename, int linenum, const std::string &str){
  return utils::inumeric(filename,linenum,str,true,lmp);
}

double PairRANN::numeric(const char *filename, int linenum, const std::string &str){
  return utils::numeric(filename,linenum,str,true,lmp);
}

std::vector<std::string> PairRANN::tokenmaker(std::string line,std::string delimiter){
	int nwords = count_words(const_cast<char *>(line.c_str()),const_cast<char *>(delimiter.c_str()));
	char **words=new char *[nwords+1];
	nwords = 0;
	words[nwords++]=strtok(const_cast<char *>(line.c_str()),const_cast<char *>(delimiter.c_str()));
	while ((words[nwords++] = strtok(NULL,const_cast<char *>(delimiter.c_str())))) continue;
	nwords--;
	std::vector<std::string> linev;
	for (int i=0;i<nwords;i++){
		linev.emplace_back(words[i]);
	}
	delete [] words;
	return linev;
}
