/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */


#include "lotf_linear.h"
#include <sstream>


using namespace std;


const char* LOTF_linear::tagname = {"Lotf"};


void LOTF_linear::Retrain()
{
    if (p_linreg == nullptr)
        p_trainer->Train(p_selector->selected_cfgs);
    else
    {
        for (auto& changed_item : p_selector->last_updated)
        {
            // weight calculation + decoding equation numbers 
            set<int> decoded_eqns;
            int wgt=0;
            for (int eqn : changed_item.second)
            {
                wgt += (eqn > 0) ? 1 : -1;
                decoded_eqns.insert((eqn > 0) ? eqn-1 : -eqn+1);
            }
            Configuration cfg(*changed_item.first);
            cfg.fitting_items = decoded_eqns;

            if (ignore_entry_count) // selecting cfg are trained independed on number of entries in selected set
            {
                if (changed_item.first->fitting_items.empty()) // cfg outsorted
                    p_linreg->AddToSLAE(cfg, -1);
                else if (changed_item.first->fitting_items == decoded_eqns) // new cfg
                    p_linreg->AddToSLAE(cfg, 1);
                // else already trained on this cfg
            }
            else
                p_linreg->AddToSLAE(cfg, wgt);
        }

        p_linreg->Train();
    }

    train_count++;
}

void LOTF_linear::CopyEFS(Configuration & from, Configuration & to)// Copying EFS data from one configuration to another
{
    to.has_energy(from.has_energy());
    to.has_forces(from.has_forces());
    to.has_stresses(from.has_stresses());
    to.has_site_energies(from.has_site_energies());

    if (from.size() != to.size())
        to.resize(from.size());

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

void LOTF_linear::DoSingleAbinitioCalc(Configuration& cfg)
{
    for (auto& item : p_selector->last_updated)
    {
        bool is_new = true;
        for (auto tmp_ptr = item.second.begin(); tmp_ptr != item.second.end(); ++tmp_ptr)
            if (*tmp_ptr < 0)   // last_updated.second contain negative updated indexes <=> configuration is leaving the training set and not freshely added
                is_new = false;

        if (is_new)     // this means *item.first - entered configuration. Otherwise if (item.second[0] < 0) *item.first - ousorted configuration, and no EFS recalculation is required
        {
            p_abinitio->CalcEFS(*item.first);
            pot_calcs_count++;

            // collect abinitio_cfgs if neccessary
            if (collect_abinitio_cfgs)
                abinitio_cfgs.emplace_back(*item.first);

            CopyEFS(*item.first, cfg);

            break;  // only one cfg may enter to selected set, so we can break loop after finding it
        }
    }
}

void LOTF_linear::CalcEFS(Configuration & cfg)
{
    bool selected = (p_selector->Select(cfg) > 0);

    if (selected)
    {
        // querry abinitio EFS
        try
        {
#ifdef MLIP_MPI
            MPI_Barrier(mpi.comm);

            if (sequentional_abinitio_treatment) // configurations on difeerent processes are treated by abinitio potential one by one 
                for (int prank=0; prank < mpi.size; prank++)
                {
                    DoSingleAbinitioCalc(cfg);
                    MPI_Barrier(mpi.comm);
                }
            else    // simultanious abinitio calculations
                DoSingleAbinitioCalc(cfg);

            MPI_Barrier(mpi.comm);
#else
            DoSingleAbinitioCalc(cfg);
#endif
        }
        catch (...)
        {
            ERROR("Unable to perform ab-initio EFS calculation");
        }

        // retrain mlip
        Retrain();

        // calculate EFS via mlip if necesary
        if (EFS_via_MTP)
        {
            p_trainer->p_mlip->CalcEFS(cfg);
            MTP_calcs_count++;
        }
    }
    else
    {
        p_trainer->p_mlip->CalcEFS(cfg);
        MTP_calcs_count++;
    }

    std::stringstream logstrm1;
    logstrm1<< "\tLOTF: cfg# "      << p_selector->cfg_eval_count
            << "\tAbinitio_calcs "  << pot_calcs_count
            << "\tMTP_calcs "       << MTP_calcs_count
            << std::endl;
    MLP_LOG(tagname, logstrm1.str());
}

