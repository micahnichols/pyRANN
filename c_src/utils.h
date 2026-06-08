#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <sstream>


#ifndef UTILS_H_
#define UTILS_H_

namespace LAMMPS_NS{

namespace utils {
  double numeric(const char *,int,std::string line,bool,char *);
  double inumeric(const char *,int,std::string line,bool,char *);
  std::string trim_comment(const std::string &line);
  FILE * open_potential(char *filename,char *,std::nullptr_t);
  char *strdup(const std::string &text);
};





}
#endif
