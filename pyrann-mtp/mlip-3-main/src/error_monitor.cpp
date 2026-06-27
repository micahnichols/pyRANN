/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Evgeny Podryabinkin
 */

#include "error_monitor.h"

#include <iostream>
#include <fstream>
#include <sstream>


const char* ErrorMonitor::tagname = {"errors"};


ErrorMonitor::ErrorMonitor(Settings settings, double _relfrc_regparam)
{
    relfrc_regparam = _relfrc_regparam;
    Reset();

    InitSettings();
    ApplySettings(settings);

    if (!log.empty() && log != "stdout" && log != "stderr")
    {
        log += mpi.fnm_ending;
        // clean output files
        std::ofstream ofs(log);
        if (!ofs.is_open())
            ERROR("Can not open file \""+log+"\" for output");
    }

    SetTagLogStream("errors", log);
}

ErrorMonitor::ErrorMonitor()
{
    Reset();

    report_to = "stdout";
}

ErrorMonitor::~ErrorMonitor()
{
}

void ErrorMonitor::Compare(Configuration & cfg_valid, Configuration & cfg_check, double wgt)
{
    if (cfg_valid.size() != cfg_check.size())
    {
        Warning("ErrorMonitor: attempting to compare configurations of different sizes");
        return;
    }
    int size = cfg_valid.size();

    ene_cfg.clear();
    epa_cfg.clear();
    if (cfg_valid.has_energy() && cfg_check.has_energy())
    {
        ene_cfg.delta = wgt * fabs(cfg_valid.energy - cfg_check.energy);
        ene_cfg.dltsq = ene_cfg.delta*ene_cfg.delta;
        ene_cfg.value = cfg_valid.energy;
        ene_cfg.reltv = ene_cfg.delta / ene_cfg.value;

        epa_cfg.delta = ene_cfg.delta / cfg_valid.size(); // ene_cfg.delta has already multiplied by wgt
        epa_cfg.dltsq = epa_cfg.delta*epa_cfg.delta;
        epa_cfg.value = ene_cfg.value / cfg_valid.size();
        epa_cfg.reltv = ene_cfg.reltv;
    }

    frc_cfg.clear();
    if (cfg_valid.has_forces() && cfg_check.has_forces())
    {
        for (int i=0; i<(int)cfg_check.nbhs.size(); i++)
        {
            Vector3 &f = cfg_valid.force(i);

            Quantity frc_one;

            frc_one.valsq = f.NormSq();
            frc_one.value = sqrt(frc_one.valsq);
            frc_one.dltsq = (cfg_check.force(i) - f).NormSq();
            frc_one.delta = sqrt(frc_one.dltsq);
            frc_one.reltv = frc_one.delta / (frc_one.value + relfrc_regparam + 1.0e-300);

            frc_cfg.accumulate_serial(frc_one);
        }

        frc_cfg.max.delta *= wgt;
        frc_cfg.max.dltsq *= wgt*wgt;
        frc_cfg.max.reltv *= wgt;
        frc_cfg.sum.reltv *= wgt / size;
        frc_cfg.sum.delta *= wgt;
        frc_cfg.sum.dltsq *= wgt*wgt;
    }

    str_cfg.clear();
    if (cfg_valid.has_stresses() && cfg_check.has_stresses())
    {
        str_cfg.valsq = cfg_valid.stresses.NormFrobeniusSq();
        str_cfg.value = sqrt(str_cfg.valsq);
        str_cfg.dltsq = (cfg_check.stresses - cfg_valid.stresses).NormFrobeniusSq();
        str_cfg.delta = sqrt(str_cfg.dltsq);
        str_cfg.reltv = str_cfg.delta / (str_cfg.value + 1.0e-300);
    }

    vir_cfg.clear();
    if (cfg_valid.has_stresses() && cfg_check.has_stresses())
    {
        const double eVA3_to_GPa = 160.2176487;
        double volume = fabs(cfg_valid.lattice.det()) / eVA3_to_GPa;

        Matrix3 valid_virial(cfg_valid.stresses);
        Matrix3 check_virial(cfg_check.stresses);
        valid_virial *= 1.0 / volume;
        check_virial *= 1.0 / volume;

        vir_cfg.valsq = valid_virial.NormFrobeniusSq();
        vir_cfg.value = sqrt(vir_cfg.valsq);
        vir_cfg.dltsq = (check_virial - valid_virial).NormFrobeniusSq();
        vir_cfg.delta = sqrt(vir_cfg.dltsq);
        vir_cfg.reltv = vir_cfg.delta / (vir_cfg.value + 1.0e-300);
    }

    std::stringstream logstrm1;
    if (cfg_check.size() > 0)
        logstrm1<< "Errors: cfg# " << ++cfg_count
                << "\t#atoms: " << cfg_check.nbhs.size();
        if (cfg_check.has_energy())
        logstrm1<< "\tDiff(ene):" << cfg_check.energy - cfg_valid.energy
                << "\tDiff(epa):" << fabs(cfg_check.energy - cfg_valid.energy) / cfg_check.nbhs.size();
        if (cfg_check.has_forces())
        logstrm1<< "\tRMS(force): " << sqrt(frc_cfg.sum.dltsq / size)
                << "\tMAX(force): " << frc_cfg.max.delta
                << "\tRel(force): " << frc_cfg.sum.reltv;
        if (cfg_check.has_stresses())
        logstrm1<< "\tDiff(stress): " << vir_cfg.delta
                << "\tRel(stress): " << str_cfg.reltv
                << std::endl;
    MLP_LOG(tagname,logstrm1.str());
}

void ErrorMonitor::AddToCompare(Configuration& cfg_valid, Configuration& cfg_check, double wgt)
{
    Compare(cfg_valid, cfg_check, wgt);

    bool energy_Ok = (cfg_valid.size() != 0 && cfg_valid.has_energy() && cfg_check.has_energy());
    bool stress_Ok = (cfg_valid.size() != 0 && cfg_valid.has_stresses() && cfg_check.has_stresses());

    ene_all.accumulate(ene_cfg, energy_Ok ? 1 : 0);
    epa_all.accumulate(epa_cfg, energy_Ok ? 1 : 0);
    frc_all.accumulate(frc_cfg);
    str_all.accumulate(str_cfg, stress_Ok ? 1 : 0);
    vir_all.accumulate(vir_cfg, stress_Ok ? 1 : 0);
}

void ErrorMonitor::GetReport()
{
    if (mpi.rank == 0)
    {
        std::stringstream logstrm1;
        logstrm1<< "_________________Errors report_________________\n"
                << "Energy:\n"
                << "\tErrors checked for " << ene_all.count << " configurations\n"
                << "\tMaximal absolute difference = " << ene_all.max.delta << '\n'
                << "\tAverage absolute difference = " << ene_aveabs() << '\n'
                << "\tRMS     absolute difference = " << ene_rmsabs() << '\n'
                << "\n"
                << "Energy per atom:\n"
                << "\tErrors checked for " << epa_all.count << " configurations\n"
                << "\tMaximal absolute difference = " << epa_all.max.delta << '\n'
                << "\tAverage absolute difference = " << epa_aveabs() << '\n'
                << "\tRMS     absolute difference = " << epa_rmsabs() << '\n'
                << "\n"
                << "Forces:\n"
                << "\tErrors checked for " << frc_all.count << " atoms\n"
                << "\tMaximal absolute difference = " << frc_all.max.delta << '\n'
                << "\tAverage absolute difference = " << frc_aveabs() << '\n'
                << "\tRMS     absolute difference = " << frc_rmsabs() << '\n'
                //<< "\tAverage relative difference = " << frc_averel() << '\n'
                << "\tMax(ForceDiff) / Max(Force) = " << frc_all.max.delta / 
                                                         (frc_all.max.value + 1.0e-300) << '\n'
                << "\tRMS(ForceDiff) / RMS(Force) = " << frc_rmsrel() << '\n'
                << "\n"
                << "Stresses (in energy units):\n"
                << "\tErrors checked for " << str_all.count << " configurations\n"
                << "\tMaximal absolute difference = " << str_all.max.delta << '\n'
                << "\tAverage absolute difference = " << str_aveabs() << '\n'
                << "\tRMS     absolute difference = " << str_rmsabs() << '\n'
                //<< "\tAverage relative difference = " << str_averel() << '\n'
                << "\tMax(StresDiff) / Max(Stres) = " << str_all.max.delta / 
                                                         (str_all.max.value + 1.0e-300) << '\n'
                << "\tRMS(StresDiff) / RMS(Stres) = " << str_rmsrel() << '\n'
                << "\n"
                << "Virial stresses (in pressure units):\n"
                << "\tErrors checked for " << vir_all.count << " configurations\n"
                << "\tMaximal absolute difference = " << vir_all.max.delta << '\n'
                << "\tAverage absolute difference = " << vir_aveabs() << '\n'
                << "\tRMS     absolute difference = " << vir_rmsabs() << '\n'
                //<< "\tAverage relative difference = " << vir_averel() << '\n'
                << "\tMax(StresDiff) / Max(Stres) = " << vir_all.max.delta / 
                                                         (vir_all.max.value + 1.0e-300) << '\n'
                << "\tRMS(StresDiff) / RMS(Stres) = " << vir_rmsrel() << '\n'
                << "_______________________________________________\n";
        if (report_to == "stdout")
            std::cout << logstrm1.str() << std::endl;
        else if (report_to == "stderr")
            std::cerr << logstrm1.str() << std::endl;
        else
        {
            std::ofstream ofs(report_to);
            if (!ofs.is_open())
                ERROR("Can not open file \""+report_to+"\" for output");
            else
                ofs << logstrm1.str() << std::endl;
        }
    }
}

void ErrorMonitor::Reset()
{
    cfg_count = 0;  

    ene_cfg.clear();
    epa_cfg.clear();
    str_cfg.clear();
    frc_cfg.clear();

    ene_all.clear();
    epa_all.clear();
    frc_all.clear();
    str_all.clear();
}

