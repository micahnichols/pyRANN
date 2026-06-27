/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   Contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_EXTERNAL_POTENTIAL_INTERFACE_H
#define MLIP_EXTERNAL_POTENTIAL_INTERFACE_H


#include "basic_potentials.h"


 // class allows one to use potentials embedded in LAMMPS. It is implemented via files exchange with LAMMPS that should be launched by MLIP for EFS calculation
class ExternalPotential : public AnyPotential, protected InitBySettings
{
public:
    const int io_max_attemps = 5; // number of attempts to read/write files 

    std::string input_filename = "";   // input filename for external potential with configuration 
    std::string output_filename = "";  // output filename for external potential with configuration 
    std::string start_command = "";    // command to OS environment launching external potential

    void InitSettings()           // Sets correspondence between variables and setting names in settings file
    {
        MakeSetting(input_filename, "input_file");
        MakeSetting(output_filename, "output_file");
        MakeSetting(start_command, "start_command");
    }

    ExternalPotential(const Settings& settings)
    {
        InitSettings();
        ApplySettings(settings);
        PrintSettings();
        if ((input_filename == "") || (output_filename == "") || (start_command == "")) {
            ERROR("No input file name, or output file name or launch commad provided!");
        }
    }
    /*ExternalPotential(   const std::string& _input_filename,
                        const std::string& _output_filename,
                        const std::string& _start_command) :
                        input_filename(_input_filename),
                        output_filename(_output_filename),
                        start_command(_start_command) 
    {}*/

    bool CheckCfgNotChanged(Configuration& cfg_vrf, Configuration& cfg, std::string="Error.cfg");
    void CopyEFS(Configuration & from, Configuration & to);
    void CalcEFS(Configuration& conf);
};

#endif //#ifndef MLIP_EXTERNAL_POTENTIAL_INTERFACE_H

