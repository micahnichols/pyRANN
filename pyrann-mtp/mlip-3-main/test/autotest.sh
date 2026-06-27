#!/bin/bash
# 
# MLIP automatic testing script
# 
# Logic description of this script:
#  1. Upload MLIP project from the git repository.
#  2. Cleanup the LAMMPS directory if it already exsists in the specified 
#     directory overwise upload LAMMPS from the git repository.
#  3. Run ./configure script.
#  4. Compile binary.
#  5. Run intergration testing.
#  6. Send notification.
#  

# Base directory
BASE_DIR="${BASE_DIR:-/tmp}"

# project deploy directory
MLIP_DEV_DIR="${BASE_DIR}/mlip-dev-autotest"
LAMMPS_DIR="${BASE_DIR}/lammps-autotest"

# git repository
MLIP_DEV_GIT="git@gitlab.skoltech.ru:shapeev/mlip-dev.git"

# lammps git repository
LAMMPS_GIT="https://github.com/lammps/lammps.git"

# from address 
FROM_EMAIL=${FROM_EMAIL:-"MLIP Testing <$(id -u -n)@$(hostname)>"}

# email addresses to send notification
MAILTO_EMAIL="r.nabiev@skoltech.ru"

# error log file
ERRLOG="${BASE_DIR}/mlip-dev-autotest-stderr.log"

# pid file
# to prevent duplicate cron job executions
PIDFILE="${BASE_DIR}/mlip-dev-autotest.pid"

# mailing function
#   $1 - subject
Send() {
  if [ -n ${MAILTO_EMAIL} ]; then 
    local BODY=/dev/null
    if [ -f ${ERRLOG} ] ; then 
		head -n 25 ${ERRLOG} > "${ERRLOG}.mail"
		BODY="${ERRLOG}.mail" 
	fi
    mail -s "*Automatic Notification* $*" -r ${FROM_EMAIL} ${MAILTO_EMAIL} < ${BODY}
  fi
}

# function to terminate
Halt() {
  echo -e "ERROR: $*" 1>&2
  Send "$*"
  exit 1
}

# Pid file create function
PidFile() {
  echo $$ > ${PIDFILE}
  if [ $? -ne 0 ]
  then
	Halt "Could not create PID file."
  fi
}

Usage() {
 cat << _USAGE
Usage:	${0##*/} [-sph] [-l <lammps path>]
Options:
	-s          silent mode
	-p          testing of public domain version
	-l <path>   path to lammps
	-h	        display help
_USAGE
 exit 1
}

# script options
SILENT_FLG=0
MK_OPT="-j8"
GIT_OPT=""

# options related to the project
CONFIGURE_OPT=""
PUBLIC_FLG=0

# parsing options
while getopts ":l:sph" opt
  do
	case $opt in
      s  ) SILENT=1 ;;
      p  ) PUBLIC=1 ;;
	  l  ) LAMMPS_DIR=${OPTARG} ;;
	  :  ) echo "missing argument for option -${OPTARG}" | tee ${ERRLOG} 1>&2
		   Halt "Script running failed." ;;
      h  ) Usage ;;
	  \? ) echo "unknown option -${OPTARG}" | tee ${ERRLOG} 1>&2
		   Halt "Script running failed." ;;
    esac
  done

if [ ${SILENT} == 1]; then
  GIT_OPT="${GIT_OPT} -q"
  MK_OPT="${MK_OPT} --no-print-directory --silent"
fi

if [ -f ${PIDFILE} ]; then
  PID=$(cat ${PIDFILE})
  ps -p ${PID} > /dev/null 2>${ERRLOG}
  if [ $? -eq 0 ]; then
    Halt "Job already running."
  else
    # job not found assume not running
    PidFile
  fi
else
  PidFile
fi

# upload project
rm -rf ${MLIP_DEV_DIR}
git clone ${GIT_OPT} ${MLIP_DEV_GIT} ${MLIP_DEV_DIR} 2>${ERRLOG} || Halt "MLIP uploading failed."

if [ ! -f ${LAMMPS_DIR}/src/lammps.h ];  then
# upload lammps
  rm -rf ${LAMMPS_DIR}
  git clone ${GIT_OPT} ${LAMMPS_GIT} ${LAMMPS_DIR} 2>${ERRLOG} || Halt "LAMMPS uploading failed."
else
  make ${MK_OPT} -C ${LAMMPS_DIR}/src/ clean-all
fi

cd ${MLIP_DEV_DIR}

if [ ${PUBLIC} == 1 ];  then
  CONFIGURE_OPT="${CONFIGURE_OPT} --publicdomain"
fi

# create configuration
./configure ${CONFIGURE_OPT} --lammps=${LAMMPS_DIR} 2>${ERRLOG} || Halt "MLIP configuration failed."
# build mlip program
make ${MK_OPT} mlp 2>${ERRLOG} || Halt "MLIP compilation failed."
# build lammps programs
make ${MK_OPT} lammps 2>${ERRLOG} || Halt "MLIP compilation failed."

# run integration testing
make test 2>${ERRLOG} || Halt "MLIP Integration testing failed."

# notification
Send "MLIP Integration testing successfully passed."

# Remove pid file
rm $PIDFILE
