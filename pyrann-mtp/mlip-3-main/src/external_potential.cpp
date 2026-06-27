/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   Contributors: Evgeny Podryabinkin
 */


#include "external_potential.h"


using namespace std;


bool ExternalPotential::CheckCfgNotChanged(Configuration& cfg_vrf, Configuration& cfg, string err_fnm)
{
    bool answer = true;

    if (norm(cfg.lattice - cfg_vrf.lattice) > 1.0e-4)
    {
        cfg_vrf.AppendToFile(err_fnm + "_inp");
        cfg.AppendToFile(err_fnm + "_out");
        Warning("Lattice changed after EFS calculation!");
        answer &= false;
    }
    if (cfg.size() != cfg_vrf.size())
    {
        cfg_vrf.AppendToFile(err_fnm + "_inp");
        cfg.AppendToFile(err_fnm + "_out");
        Warning("Atoms count changed after EFS calculation!");
        answer &= false;
    }
    for (int i=0; i<cfg.size(); i++)
    {
        Vector3 shift((cfg_vrf.pos(i) - cfg.pos(i)) * cfg_vrf.lattice.inverse());   // shift must has components close to 0 or 1
        if ((fabs(shift[0]) > 1.0e-4 && fabs(shift[0])-1.0 > 1.0e-4) ||
            (fabs(shift[1]) > 1.0e-4 && fabs(shift[1])-1.0 > 1.0e-4) ||
            (fabs(shift[2]) > 1.0e-4 && fabs(shift[2])-1.0 > 1.0e-4))
        {
            cfg_vrf.AppendToFile(err_fnm + "_inp");
            cfg.AppendToFile(err_fnm + "_out");
            Warning("Position of atom#" + to_string(i+1) + " changed after EFS calculation!");
            answer &= false;
        }
    }
    for (int i=0; i<cfg.size(); i++)
    {
        if (cfg_vrf.type(i) != cfg.type(i))
        {
            cfg_vrf.AppendToFile(err_fnm + "_inp");
            cfg.AppendToFile(err_fnm + "_out");
            Warning("Type of atom#" + to_string(i+1) + " changed after EFS calculation!");
            answer &= false;
        }
    }

    return answer;
}

void ExternalPotential::CopyEFS(Configuration & from, Configuration & to)// Copying EFS data from one configuration to another
{
    to.has_energy(from.has_energy());
    to.has_forces(from.has_forces());
    to.has_stresses(from.has_stresses());
    to.has_site_energies(from.has_site_energies());

    if (from.size() != to.size())
    {
        to.resize(from.size());
        memcpy(&to.pos(0), &from.pos(0), from.size() * sizeof(Vector3));
    }

    if (from.has_forces())
        memcpy(&to.force(0, 0), &from.force(0, 0), from.size() * sizeof(Vector3));
    if (from.has_stresses())
        to.stresses = from.stresses;
    if (from.has_energy())
        to.energy = from.energy;
    if (from.has_site_energies())
        memcpy(&to.site_energy(0), &from.site_energy(0), from.size() * sizeof(double));

    if (from.features.count("EFS_by") > 0)
        to.features["EFS_by"] = from.features["EFS_by"];
}

void ExternalPotential::CalcEFS(Configuration & conf)
{

    Configuration& cfg = conf;

    //save configuration to input file
    ofstream ofs(input_filename, ios::binary);
    cfg.Save(ofs);

    // starting external potential
    int res = system(start_command.c_str());
    if (res != 0)
        Message("External potential exits with code " + to_string(res));

    //load configuration from output file
    ifstream ifs(output_filename, ios::binary);
    cfg.Load(ifs);

#ifdef MLIP_DEBUG
        if (!CheckCfgNotChanged(cfg, conf)) 
            ERROR("Configuration changed after calculation on abinitio model");
#endif

    CopyEFS(conf, cfg);
}
