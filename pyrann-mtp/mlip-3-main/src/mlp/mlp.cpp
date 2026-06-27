/*   MLIP is a software for Machine Learning Interatomic Potentials
 *   MLIP is released under the "New BSD License", see the LICENSE file.
 *   Contributors: Alexander Shapeev, Evgeny Podryabinkin, Ivan Novikov
 */


#include "../common/mpi_stubs.h"
#include "../common/comm.h"
#include "../common/stdafx.h"
#include "../common/utils.h"
#include "mlp.h"
#include "mlp_commands.h"


int ExecuteCommand(const std::string& command,
    std::vector<std::string>& args,
    std::map<std::string, std::string>& opts)
{
    bool is_command_not_found = true;
    is_command_not_found = !Commands(command, args, opts);

    if (is_command_not_found)
#ifndef MLIP_MLIP
        std::cout << "Error: command " << command << " does not exist." << std::endl;
#else
        if (mpi.rank == 0) std::cout << "Error: command " << command << " does not exist." << std::endl;
#endif

    return 1;
}

int main(int argc, char *argv[])
{
#ifdef MLIP_MPI
    MPI_Init(&argc, &argv);
    mpi.InitComm(MPI_COMM_WORLD);
#endif

    std::vector<std::string> args;
    std::map<std::string, std::string> opts;

    // parse argv to extract args and opts
    std::string command = "help";
    if (argc >= 2) command = argv[1];
    ParseOptions(argc-1, argv+1, args, opts);

    try {
        ExecuteCommand(command, args, opts);
    }
    catch (const MlipException& e) 
    { 
        std::string mess;
#ifdef MLIP_MPI
        if (((std::string)e.What()).size() > 0)
            mess += "Rank " + std::to_string(mpi.rank) + ", ";
#endif
        mess += e.What();
        std::cerr << mess << std::endl;
        exit(777);
    }

#ifdef MLIP_MPI
    MPI_Finalize();
#endif

    return 0;
}
