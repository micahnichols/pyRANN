#ifndef CREATE_STYLES_H_
#define CREATE_STYLES_H_

#include "pair_spin_rann.h"
#include "rann_activation_linear.h"
#include "rann_activation_tanh.h"
#include "rann_activation_ttanh.h"
#include "rann_activation_tanhtwist.h"
#include "rann_activation_sig_i.h"
#include "rann_activation_capped.h"
#include "rann_activation_sharpcapped.h"
#include "rann_activation_slowcapped.h"
#include "rann_activation_zero.h"
#include "rann_fingerprint_bond.h"
#include "rann_fingerprint_bondscreened.h"
#include "rann_fingerprint_bondscreenedspin.h"
#include "rann_fingerprint_bondspin.h"
#include "rann_fingerprint_radial.h"
#include "rann_fingerprint_radialscreened.h"
#include "rann_fingerprint_radialscreenedspin.h"
#include "rann_fingerprint_radialscreenedspinn.h"
#include "rann_fingerprint_radialspin.h"
#include "rann_fingerprint_torsion.h"
#include "rann_fingerprint_radialscreenedexchange.h"
#include "rann_fingerprint_radialexchange.h"
#include "rann_net_default.h"
#include "rann_net_exchange.h"
#include "rann_net_exchangespin.h"
#include "rann_state_rose.h"
#include "rann_state_rosescreened.h"
#include "rann_state_eshift.h"
#include "rann_state_spin_j.h"
#include "rann_state_spin_jscreened.h"
#include "rann_state_repulse.h"
#include "rann_state_spinbiquadratic.h"
#include "rann_state_covalent.h"
#include "rann_state_eamscreenedr.h"
#include "rann_state_eamscreenedsingle.h"

using namespace LAMMPS_NS;

RANN::Fingerprint *PairRANN::create_fingerprint(const char *style)
{
  if (strcmp(style,"radial")==0) {
          return new RANN::Fingerprint_radial(this);
  }
  else if (strcmp(style,"radialscreened")==0) {
          return new RANN::Fingerprint_radialscreened(this);
  }
  else if (strcmp(style,"radialscreenedspin")==0) {
          return new RANN::Fingerprint_radialscreenedspin(this);
  }
  else if (strcmp(style,"radialscreenedexchange")==0) {
          return new RANN::Fingerprint_radialscreenedexchange(this);
  }
  else if (strcmp(style,"radialexchange")==0) {
          return new RANN::Fingerprint_radialexchange(this);
  }
  else if (strcmp(style,"radialscreenedspinn")==0) {
          return new RANN::Fingerprint_radialscreenedspinn(this);
  }
  else if (strcmp(style,"radialspin")==0) {
          return new RANN::Fingerprint_radialspin(this);
  }
  else if (strcmp(style,"bond")==0) {
          return new RANN::Fingerprint_bond(this);
  }
  else if (strcmp(style,"bondscreened")==0) {
          return new RANN::Fingerprint_bondscreened(this);
  }
  else if (strcmp(style,"bondscreenedspin")==0) {
          return new RANN::Fingerprint_bondscreenedspin(this);
  }
  else if (strcmp(style,"bondspin")==0) {
          return new RANN::Fingerprint_bondspin(this);
  }
  else if (strcmp(style,"torsion")==0) {
	return new RANN::Fingerprint_torsion(this);
  }
  errorf(FLERR,"Unknown fingerprint style");
  return nullptr;
  
}


RANN::Activation *PairRANN::create_activation(const char *style)
{
  if (strcmp(style,"linear")==0) {
    return new RANN::Activation_linear(this);
  }
  else if (strcmp(style,"tanh")==0) {
    return new RANN::Activation_tanh(this);
  }
  else if (strcmp(style,"ttanh")==0) {
    return new RANN::Activation_ttanh(this);
  }
  else if (strcmp(style,"tanhtwist")==0) {
    return new RANN::Activation_tanhtwist(this);
  }
  else if (strcmp(style,"sigI")==0) {
    return new RANN::Activation_sigI(this);
  }
  else if (strcmp(style,"capped")==0) {
	return new RANN::Activation_capped(this);
  }
  else if (strcmp(style,"sharpcapped")==0) {
	return new RANN::Activation_sharpcapped(this);
  }
  else if (strcmp(style,"slowcapped")==0) {
	return new RANN::Activation_slowcapped(this);
  }
  else if (strcmp(style,"zero")==0) {
	return new RANN::Activation_zero(this);
  }
  errorf(FLERR,"Unknown activation style");
  return nullptr;
}

RANN::Net *PairRANN::create_net(const char *style)
{
  if (strcmp(style,"default")==0) {
    return new RANN::Net_default(this);
  }
  else if (strcmp(style,"exchange")==0) {
    return new RANN::Net_exchange(this);
  }
  else if (strcmp(style,"exchangespin")==0) {
    return new RANN::Net_exchangespin(this);
  }
  else if (strcmp(style,"empty")==0) {
    return new RANN::Net(this);
  }
  errorf(FLERR,"Unknown network style");
  return nullptr;
}


RANN::State *PairRANN::create_state(const char *style)
{
  if (strcmp(style,"rose")==0) {
          return new RANN::State_rose(this);
  }
  else if (strcmp(style,"rosescreened")==0) {
          return new RANN::State_rosescreened(this);
  }
  else if (strcmp(style,"eshift")==0) {
          return new RANN::State_eshift(this);
  }
  else if (strcmp(style,"spinj")==0) {
	  return new RANN::State_spinj(this);
  }
  else if (strcmp(style,"spinjscreened")==0) {
	  return new RANN::State_spinjscreened(this);
  }
  else if (strcmp(style,"repulse")==0) {
	  return new RANN::State_repulse(this);
  }
  else if (strcmp(style,"spinbiquadratic")==0) {
	  return new RANN::State_spinbiquadratic(this);
  }
  else if (strcmp(style,"covalent")==0) {
	  return new RANN::State_covalent(this);
  }
  else if (strcmp(style,"eamscreenedr")==0) {
	  return new RANN::State_eamscreenedr(this);
  }
  else if (strcmp(style,"eamscreenedsingle")==0) {
          return new RANN::State_eamscreenedsingle(this);
  }
  errorf(FLERR,"Unknown state style");
  return nullptr;
}

#endif
