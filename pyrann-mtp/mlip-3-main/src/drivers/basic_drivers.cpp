/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#include "basic_drivers.h"
#include <sstream>


using namespace std;


const char* CfgReader::tagname = {"cfg_reader"};

CfgReader::CfgReader(string _filename, Wrapper * _p_mlip, AnyPotential* _p_potential)
{
    p_mlip = _p_mlip;
    filename = _filename;
    p_potential = _p_potential;
}

CfgReader::CfgReader(const Settings& settings, Wrapper * _p_mlip, AnyPotential* _p_potential)
{
    InitSettings();
    ApplySettings(settings);
    PrintSettings();

    p_mlip = _p_mlip;
    p_potential = _p_potential;

    SetTagLogStream("cfg_reader", log_output); // 
}


// tag name of object


void CfgReader::Run()
{
#ifdef MLIP_MPI
    if (!log_output.empty() && 
        log_output != "stdout" && 
        log_output != "stderr")
            log_output += '_' + to_string(mpi.rank);
#endif // MLIP_MPI

    std::ifstream ifs;

    if (filename.find(' ', 0) != string::npos)
        ERROR("Whitespaces in filename restricted");
    ifs.open(filename, ios::binary);
    if (!ifs.is_open())
        ERROR("Can't open file \"" + filename + "\" for reading configurations");
//  else
//      Message("Configurations will be read from file \"" + filename + "\"");

//  SetLogStream(log_output);

    int cfgcntr = 0;
    while (cfg.Load(ifs) && cfgcntr<max_cfg_cnt)
    {
        if (cfgcntr++ % mpi.size != mpi.rank)
            continue;

        cfg.features["from"] = "database:" + filename;

        std::stringstream logstrm1;
        logstrm1 << "Cfg reading: #" << cfgcntr 
                << " size: " << cfg.size()
                << ", EFS data: " 
                << (cfg.has_energy() ? 'E' : ' ')
                << (cfg.has_forces() ? 'F' : ' ') 
                << (cfg.has_stresses() ? 'S' : '-')
                << (cfg.has_site_energies() ? 'V' : '-')
                << (cfg.has_charges() ? 'Q' : '-')
                << std::endl;
        MLP_LOG(tagname,logstrm1.str());

        CalcEFS(cfg);
    }

    if ((cfgcntr % mpi.size != 0) && mpi.rank >= (cfgcntr % mpi.size))
    {
        Configuration cfg;  // empty cfg submitted. Important for parallel selection 
        CalcEFS(cfg);
    }

    Message("Reading configuration complete");
}
