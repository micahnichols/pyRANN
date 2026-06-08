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

----------------*/

#ifndef LMP_RANN_NET_H
#define LMP_RANN_NET_H

#include <string>
#include <vector>

namespace LAMMPS_NS {
  class PairRANN;
namespace RANN {
    class Activation;
    class Fingerprint;

  class Net {
   public:
    Net(PairRANN *);
    ~Net();
    char *style;
    bool empty;
    bool fullydefined;
    int start_index;
    int end_index;
    int start_index_sf;
    int length;
    int id;    //based on ordering of fingerprints listed for i-j in potential file
    PairRANN *pair;
    int n_body_type; 
    int *atomtypes;

    int layers;
    int *dimensions;//vector of length layers with entries for neurons per layer
    //int *activations;//unused
    int maxlayer;//longest layer (for memory allocation)
    int sumlayers;
    int *startI;
    double **Weight;
    double **Bias;
    bool **freezeW;
    bool **freezeB;
    bool *freezebeta;
    bool exchange;

    double *normalshift;
    double *normalgain;
    bool *weightdefined;
    bool *biasdefined;
    bool *dimensiondefined;

    RANN::Activation ***activations;
    RANN::Fingerprint **fingerprint_map;
    int map_length;
    int *startingneuron;
    int *neuronmap;
    int betalen;
    int betalen_f;
    double *Eexchange;
    virtual void allocate();
    virtual void flatten_beta(double*);//fill beta vector from net structure
    virtual void unflatten_beta(double*);//fill net structure from beta vector
    virtual void copy_network(Net*);
    virtual void normalize_net(Net*);
    virtual void unnormalize_net(Net*);
    virtual void normalize_data();
    virtual void create_random_weights(int,int,int,int,int);
	  virtual void create_random_biases(int,int,int,int);
    virtual void init(int *, int);
    virtual bool check_net_completeness();//run after each constant is defined
    virtual bool check_net_correctness();//run after net is complete
    virtual bool parse_values(std::vector<std::string>, std::vector<std::string>, FILE *, char*,int*);
    virtual void parse_layer_size(std::vector<std::string>,std::vector<std::string>, char *,int);
    virtual void parse_weight(std::vector<std::string>,std::vector<std::string>, FILE *,char *,int *);
    virtual void parse_bias(std::vector<std::string>,std::vector<std::string>, FILE *,char *,int *);
    virtual void parse_activation_functions(std::vector<std::string>,std::vector<std::string>, char *,int);
    virtual void parse_fingerprint_map(std::vector<std::string>,std::vector<std::string>, char *,int);
    virtual void parse_freezeW(std::vector<std::string>,std::vector<std::string>, FILE *,char *,int *);
    virtual void parse_freezeB(std::vector<std::string>,std::vector<std::string>, FILE *,char *,int *);
  	virtual void jacobian_convolution(double *,double *,int *,int,int);
    virtual void force_fit(double *,double *,int *,int,int);
	  virtual void forward_pass(double *,int *,int);
    virtual void forward_pass_force(double *,int *,int);
	  virtual void get_per_atom_energy(double **,int *,int);
	  virtual void propagateforward(double *,double **,int ,int ,int, double*,double*,double*,double*,double*,double*,double*,double*,int *,int,double *, double *, double*);
	  virtual void propagateforwardspin(double *,double **,double **,double **,int ,int ,int, double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,double*,int *,int);   
    virtual void write_values(FILE *);
    virtual void updatespin(double *,double **,double**,int ,int ,int ,int *,int );
  };
}    // namespace RANN
}    // namespace LAMMPS_NS

#endif /* RANN_FINGERPRINT_H_ */