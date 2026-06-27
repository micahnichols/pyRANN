/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#include "basic_trainer.h"


const char* AnyTrainer::tagname = {"fit"};

AnyTrainer::AnyTrainer(AnyLocalMLIP * _p_mlip, const Settings & settings) :
    p_mlip(_p_mlip)
{
    InitSettings();
    ApplySettings(settings);

    if (mpi.rank == 0)
    {
        std::ifstream ifs(mlip_fitted_fnm);
        if (ifs.is_open())
            Warning("File " + mlip_fitted_fnm + " already exists and will be overwritten during the training procedure!");
    }

    SetTagLogStream("fit", fit_log);

    Message("Basic trainer initialization complete");
}


