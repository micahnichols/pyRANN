/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#ifndef MLIP_BASIC_DRIVERS_H
#define MLIP_BASIC_DRIVERS_H



/////

#include "../wrapper.h"
#include "../basic_potentials.h"
#include <iostream>


// Basic class for all drivers (algorithms requiring potential and modifing configurations, e.g. MD, Relaxation, etc.)
//  For EFS calculation the driver uses its own CalcEFS(cfg) mehod that invokes either AnyPotential->CalcEFS or Wrapper->CalcEFS
class AnyDriver // : protected InitBySettings//, protected LogWriting 
{
protected:
    void CalcEFS(Configuration & cfg)
    {
        if (p_mlip != nullptr)
            p_mlip->Process(cfg);
        else if (p_potential != nullptr)
            p_potential->CalcEFS(cfg);
        else
            ERROR("No potential specified");
    }

public:
    Wrapper* p_mlip = nullptr;
    AnyPotential* p_potential = nullptr;// pointer to potential driver operates with
    virtual void Run() = 0;             // function starting the driving proces
    virtual ~AnyDriver() {};
};


// Driver taking configurations from file.
class CfgReader : public AnyDriver, protected InitBySettings
{
private:
    Configuration cfg;

protected:
    static const char* tagname;         // tag name of object

public:
    // default settings
    std::string filename = "";          //  filename
    int max_cfg_cnt = HUGE_INT;         //  limit of configurations to be read
    std::string log_output = "";        //  log

    CfgReader(std::string _filename, 
              Wrapper *_p_mlip=nullptr, 
              AnyPotential *_p_potential=nullptr);
    CfgReader(const Settings& settings, 
              Wrapper* _p_mlip=nullptr, 
              AnyPotential * _p_potential=nullptr);

    void InitSettings()
    {
        MakeSetting(filename, "cfg_filename");
        MakeSetting(max_cfg_cnt, "limit");
        MakeSetting(log_output, "log");
    }

    
    void Run();                         //  starts reading configuations from file
};


#endif //#ifndef MLIP_BASIC_DRIVERS_H
