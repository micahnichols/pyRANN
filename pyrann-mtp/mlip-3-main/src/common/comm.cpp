#include "comm.h"


//#ifdef MLIP_MPI
MPI_data mpi;

void MPI_data::InitComm(MPI_Comm _comm)
{
    comm = _comm;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
#ifdef MLIP_MPI
    fnm_ending = "." + std::to_string(rank);
#else
    fnm_ending = "";
#endif
}

//#endif
