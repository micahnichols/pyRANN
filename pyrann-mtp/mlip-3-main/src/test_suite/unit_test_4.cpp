#include <fstream>
#include <iostream>
#include <algorithm>

#include "../common/utils.h"
#include "../mtpr.h"
#include "../non_linear_regression.h"
#include "../sample_potentials.h"
#include "../mtpr_trainer.h"
#include "../error_monitor.h"
#include "../eam.h"
#include "../sw.h"

using namespace std;

#define FAIL(...) \
{std::string str("" __VA_ARGS__); \
std::cout << " failed!" << std::endl; \
if(!str.empty()) std::cout << std::endl << "Error message: " << str << std::endl; \
return false;}

// "EAMSimple CalcEnergyGrad test by finite difference"
bool utest_eamsceg()
{
    Configuration cfg, cfg1, cfg2;
    double diffstep = 1e-5;
    double relErrMax = 0.0, absErrMax = 0.0;
    EAMSimple eam(8, 5, 1.9);
    std::vector<double> energy_grad(eam.CoeffCount());

    for (int i = 0; i < eam.CoeffCount(); i++) {
        //eam.Coeff()[i] = 20.0 / (i + 1);
        eam.Coeff()[i] = 0.1*sin(i);
    }

    ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);

    for (int i = 0; cfg.Load(ifs); i++) {
        FillWithZero(energy_grad);
        eam.CalcEnergyGrad(cfg, energy_grad);
        for (int j = 0; j < eam.CoeffCount(); j++) {
            eam.Coeff()[j] -= diffstep;
            eam.CalcEFS(cfg); double energy_m = cfg.energy;
            eam.Coeff()[j] += 2 * diffstep;
            eam.CalcEFS(cfg); double energy_p = cfg.energy;
            eam.Coeff()[j] -= diffstep;
            double cdiff = (energy_p - energy_m) / (2 * diffstep);
            //std::cout << energy_p << " " << energy_m << std::endl;
            //std::cout << energy_grad[j] << " " << cdiff  << " "  << fabs(energy_grad[j]-cdiff) << " ";
            //std::cout << fabs((energy_grad[j]-cdiff)/cdiff) << std::endl;
            double absErr = fabs(energy_grad[j] - cdiff);
            double relErr = fabs((energy_grad[j] - cdiff) / cdiff);
            if (relErr > relErrMax) relErrMax = relErr;
            if (absErr > absErrMax) absErrMax = absErr;
            //std::cout << cdiffForce << " " << cfg.force(i)[l] << std::endl;
        }
    }

    if (relErrMax > 0.02) {
        std::cout << "relErrMax is " << relErrMax;
        FAIL()
    }

    ifs.close();
    return true;
} 

// "EAMSimple CalcEFS check with finite differences"
bool utest_eamscefs()
{
    Configuration cfg, cfg1, cfg2;
    double displacement = 1e-4;
    double control_delta = 1E-3;
    EAMSimple eam(4, 4, 1.6);

    for (int i = 0; i < eam.CoeffCount(); i++) {
        //eam.Coeff()[i] = 1.0 / pow(2, i);
        //eam.Coeff()[i] = 0.25/(i+1);
        eam.Coeff()[i] = 0.1*sin(i + 1);
    }

    ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);

    cfg.Load(ifs);
    if (!eam.CheckEFSConsistency_debug(cfg, displacement, control_delta))
        FAIL()

        ifs.close();
    return true;
}

// "EAMSimple AddLossGrad test by finite difference"
bool utest_eamsalg()
{
    Configuration cfg, cfg1, cfg2;
    double displacement = 1e-5;
    double control_delta = 0.001;
    EAMSimple eam(8, 5, 1.6);

    for (int i = 0; i < eam.CoeffCount(); i++) {
        //eam.Coeff()[i] = 1.0 / pow(2, i);
        eam.Coeff()[i] = 2.1*sin(i + 1);
    }

    Settings settings;
    settings["fit:energy_weight"] = "1.0";
    settings["fit:force_weight"] = "1.0";
    settings["fit:stress_weight"] = "0.1";
    settings["fit:scale_by_force"] = "0.0";

    NonLinearRegression dtr(&eam, settings.ExtractSubSettings("fit"));
    //NonLinearRegression dtr(&eam, 1.0, 1.0, 0.1, 0.0, 1.0);

    //ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);
    ifstream ifs("NaCl_small.cfgs", ios::binary);

    for (int i=0; cfg.Load(ifs); i++) {
        if (!dtr.CheckLossConsistency_debug(cfg, displacement, control_delta))
            FAIL();
    }

    ifs.close();

    return true;
}

// "EAMMult CalcEFS check with finite differences"
bool utest_eammefs()
{
    Configuration cfg, cfg1, cfg2;
    double displacement = 1e-5;
    double control_delta = 1E-3;
    EAMMult eam(4, 4, 2, 1.6);

    for (int i = 0; i < eam.CoeffCount(); i++) {
        //eam.Coeff()[i] = 1.0 / pow(2, i);
        //eam.Coeff()[i] = 0.25/(i+1);
        eam.Coeff()[i] = 2.1*sin(i + 1);
    }

    ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);

    cfg.Load(ifs);
    if (!eam.CheckEFSConsistency_debug(cfg, displacement, control_delta))
        FAIL()

        ifs.close();
    return true;
}

// "EAMMult AddLossGrad test by finite difference"
bool utest_eammalg()
{
    Configuration cfg, cfg1, cfg2;
    double displacement = 1e-6;
    double control_delta = 0.001;
    EAMMult eam(8, 5, 2, 1.6);

    for (int i = 0; i < eam.CoeffCount(); i++) {
        //eam.Coeff()[i] = 1.0 / pow(2, i);
        eam.Coeff()[i] = 1.1*sin(i + 1);
    }

    Settings settings;
    settings["fit:energy_weight"] = "1.0";
    settings["fit:force_weight"] = "1.0";
    settings["fit:stress_weight"] = "0.1";
    settings["fit:scale_by_force"] = "0.0";

    NonLinearRegression dtr(&eam, settings.ExtractSubSettings("fit"));

    //NonLinearRegression dtr(&eam, 1.0, 1.0, 0.1, 0.0, 1.0);

    //ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);
    ifstream ifs("NaCl_small.cfgs", ios::binary);

    //vector<Configuration> train_set;
    //for (int i = 0; cfg.Load(ifs); i++) {
    //  train_set.push_back(cfg);
    //}

    //unsigned int start_time = clock();
    //  dtr.CalcObjectiveFunctionGrad(train_set);
    //unsigned int end_time = clock();
    //cout << "LossGrad time = " << (end_time - start_time) / 1E+6 << endl;

    //unsigned int start_time2 = clock();
    //  dtr.ObjectiveFunction(train_set);
    //unsigned int end_time2 = clock();
    //cout << "Loss time = " << (end_time2 - start_time2) / 1E+6 << endl;

    //exit(1);

    for (int i=0; cfg.Load(ifs); i++) {
        // FD test
        if (!dtr.CheckLossConsistency_debug(cfg, displacement, control_delta))
            FAIL()
    }

    ifs.close();

    return true;
}

// "Stillinger-Weber CalcEFS check with finite differences"
bool utest_swcefs()
{
    Configuration cfg, cfg1, cfg2;
    double displacement = 1e-5;
    double control_delta = 1E-3;
    StillingerWeberRadialBasis sw("unlearned.sw");

    for (int i = 0; i < sw.CoeffCount(); i++) {
        sw.Coeff()[i] = sqrt(5.5*fabs(sin(i+1)));
        //cout << sw.Coeff()[i] << " ";
    }
    //cout << endl;

    ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);

    cfg.Load(ifs);
    if (!sw.CheckEFSConsistency_debug(cfg, displacement, control_delta))
        FAIL()

        ifs.close();

    return true;
}

// "Stillinger-Weber CalcEFSvsCalcEFSGrad EFS comparison"
bool utest_swcefsc()
{
    Configuration cfg;
    double control_delta = 1E-3;
    StillingerWeberRadialBasis sw("unlearned.sw");
    double absEnErr, absForcesErr, absStressesErr;

    for (int i = 0; i < sw.CoeffCount(); i++) {
        sw.Coeff()[i] = sqrt(5.5*fabs(sin(i+1)));
        //cout << sw.Coeff()[i] << " ";
    }
    //cout << endl;

    ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);

    cfg.Load(ifs);

    Configuration cfg1 = cfg;
    Configuration cfg2 = cfg;

    vector<double> energy_grad;
    Array3D forces_grad;
    Array3D stress_grad;

    sw.CalcEFS(cfg1);
    sw.CalcEFSGrads(cfg2, energy_grad, forces_grad, stress_grad);

    absEnErr = fabs(cfg1.energy-cfg2.energy);

    if (absEnErr > control_delta) FAIL();

    absForcesErr = 0;
    for (int i = 0; i < cfg.size(); i++) {
        for (int l = 0; l < 3; l++) {
            double curr_err = fabs(cfg1.force(i)[l]-cfg2.force(i)[l]);
            if (curr_err > absForcesErr) absForcesErr = curr_err;
        }
    }

    if (absForcesErr > control_delta) FAIL();

    absStressesErr = 0;
    for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
            double curr_err = fabs(cfg1.stresses[a][b]-cfg2.stresses[a][b]);
            if (curr_err > absStressesErr) absStressesErr = curr_err;
        }
    }

    if (absForcesErr > control_delta) FAIL();

    ifs.close();

    return true;
}

// "Stillinger-Weber CalcEFSGrad check with central differences"
bool utest_swcefscd()
{

    ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);

    Configuration cfg;
    cfg.Load(ifs);
    Configuration cfg1 = cfg;
    Configuration cfg2 = cfg;

    StillingerWeberRadialBasis sw("unlearned.sw");

    for (int i = 0; i < sw.CoeffCount(); i++) {
        sw.Coeff()[i] = sqrt(5.5*fabs(sin(i+1)));
        //cout << sw.Coeff()[i] << " ";
    }
    //cout << endl;

    vector<double> energy_grad;
    Array3D forces_grad;
    Array3D stress_grad;

    vector<double> energy_grad_dummy;
    Array3D forces_grad_dummy;
    Array3D stress_grad_dummy;

    sw.CalcEFSGrads(cfg, energy_grad, forces_grad, stress_grad);

    double delta = 1e-5;
    double control_delta = 1E-3;
    int natom = cfg.size();
    Array2D forces_p;
    forces_p.resize(natom, 3);
    Array2D forces_m;
    forces_m.resize(natom, 3);
    vector<double> stress_p(9);
    vector<double> stress_m(9);

    double relEnErr = 0;
    double relForcesErr = 0;
    double relStressesErr = 0;

    for (int i = 0; i < sw.CoeffCount(); i++) {
        //cout << "coeff = " << i << endl;
        sw.Coeff()[i] += delta;
        sw.CalcEFS(cfg1);
        sw.Coeff()[i] -= 2*delta;
        sw.CalcEFS(cfg2);
        sw.Coeff()[i] += delta;
        //cout << "energy:" << endl;
        double cdiff = (cfg1.energy - cfg2.energy) / (2*delta);
        double curr_err = fabs((cdiff - energy_grad[i])/cdiff);
        if (curr_err > relEnErr) relEnErr = curr_err;
        //if (curr_err > 1e-5) 
        //  cout << cdiff << " " << energy_grad[i] << endl;

        //cout << "forces:" << endl;
        for (int j = 0; j < cfg.size(); j++) {
            for (int l = 0; l < 3; l++) {
                //cout << cfg1.force(j, l) << " " << cfg2.force(j, l) << " ";
                double cdiff = (cfg1.force(j, l) - cfg2.force(j, l)) / (2*delta);
                double curr_err = fabs((cdiff - forces_grad(j, i, l))/cdiff);
                if (curr_err > relForcesErr) relForcesErr = curr_err;
                //if (curr_err > 1e-5)
                //  cout << cdiff << " " << forces_grad(j, i, l) << endl;
                //cout << j << " " << l << endl;
            }
        }

        //cout << "stresses:" << endl;
        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                double cdiff = (cfg1.stresses[a][b] - cfg2.stresses[a][b]) / (2*delta);
                double curr_err = fabs((cdiff - stress_grad(a, b, i))/cdiff);
                if (curr_err > relStressesErr) relStressesErr = curr_err;
                //if (curr_err > 1e-5)
                //  cout << cdiff << " " << stress_grad(a, b, i) << endl;
            }
        }

    }

    //cout << relEnErr << endl;
    //cout << relForcesErr << endl;
    //cout << relStressesErr << endl;

    if (relEnErr > control_delta) FAIL();
    if (relForcesErr > control_delta) FAIL();
    if (relStressesErr > control_delta) FAIL();

    return true;
}

// "Stillinger-Weber CalcEnergyGradvsCalcEFSGrad energy grad comparison"
bool utest_swcefsgc()
{
    ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);

    Configuration cfg;
    cfg.Load(ifs);

    StillingerWeberRadialBasis sw("unlearned.sw");

    for (int i = 0; i < sw.CoeffCount(); i++) {
        sw.Coeff()[i] = sqrt(5.5*fabs(sin(i+1)));
        //cout << sw.Coeff()[i] << " ";
    }
    //cout << endl;

    vector<double> energy_grad1;
    Array3D forces_grad;
    Array3D stress_grad;

    vector<double> energy_grad2;

    sw.CalcEFSGrads(cfg, energy_grad1, forces_grad, stress_grad);
    sw.CalcEnergyGrad(cfg, energy_grad2);

    double control_delta = 1e-4;
    double absEnErr = 0;

    for (int i = 0; i < sw.CoeffCount(); i++) {
        if (fabs(energy_grad1[i]-energy_grad2[i]) > absEnErr) {
            absEnErr = fabs(energy_grad1[i]-energy_grad2[i]);
        }
    }

    //cout << absEnErr << endl;

    if (absEnErr > control_delta) FAIL();

    return true;
}

// "Lennard-Jones CalcEFS check with finite differences"
bool utest_ljcefs()
{
    Configuration cfg, cfg1, cfg2;
    double displacement = 1e-5;
    double control_delta = 1E-3;
    LennardJones lj("unlearned_lj_lin.mlip", 1.6, 5);

    for (int i = 0; i < lj.CoeffCount(); i++) {
        //lj.Coeff()[i] = 1.0 / pow(2, i);
        //lj.Coeff()[i] = 0.25/(i+1);
        lj.Coeff()[i] = 12.1*sin(i + 1);
    }

    ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);

    cfg.Load(ifs);
    if (!lj.CheckEFSConsistency_debug(cfg, displacement, control_delta))
        FAIL()

        ifs.close();

    return true;
}

// "Lennard-Jones AddLossGrad test by finite difference"
bool utest_ljagl()
{
    Configuration cfg, cfg1, cfg2;
    double displacement = 1e-5;
    double control_delta = 0.001;
    LennardJones lj("unlearned_lj_lin.mlip", 1.6, 5);

    for (int i = 0; i < lj.CoeffCount(); i++) {
        //lj.Coeff()[i] = 1.0 / pow(2, i);
        lj.Coeff()[i] = 121.1*(i + 1);
    }
    //cout << endl;

    Settings settings;
    settings["fit:energy_weight"] = "1.0";
    settings["fit:force_weight"] = "1.0";
    settings["fit:stress_weight"] = "0.1";
    settings["fit:scale_by_force"] = "0.0";

    NonLinearRegression dtr(&lj, settings.ExtractSubSettings("fit"));

    //NonLinearRegression dtr(&lj, 1.0, 1.0, 0.1, 0.0, 1.0);

    //ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);
    ifstream ifs("NaCl_small.cfgs", ios::binary);

    for (int i = 0; cfg.Load(ifs); i++) {
        // FD test
        if (!dtr.CheckLossConsistency_debug(cfg, displacement, control_delta))
            FAIL()
    }

    ifs.close();

    return true;
}

// "NonlinearLennard-Jones CalcEFS check with finite differences"
bool utest_nlljcefs()
{
    Configuration cfg, cfg1, cfg2;
    double displacement = 1e-5;
    double control_delta = 1E-3;
    NonlinearLennardJones lj("unlearned_lj_nonlin.mlip", 1.6, 5);

    for (int i = 0; i < lj.CoeffCount(); i++) {
        //lj.Coeff()[i] = 1.0 / pow(2, i);
        //lj.Coeff()[i] = 0.25/(i+1);
        lj.Coeff()[i] = pow(fabs(1.5*sin(i + 1)), 0.25);
    }

    ifstream ifs("NaCl_small.cfgs", ios::binary);

    cfg.Load(ifs);
    if (!lj.CheckEFSConsistency_debug(cfg, displacement, control_delta))
        FAIL()

        ifs.close();

    return true;
}

// "NonlinearLennard-Jones AddLossGrad test by finite difference"
bool utest_nlljalg()
{
    Configuration cfg, cfg1, cfg2;
    double displacement = 1e-5;
    double control_delta = 0.001;
    NonlinearLennardJones lj("unlearned_lj_nonlin.mlip", 1.6, 5);

    for (int i = 0; i < lj.CoeffCount(); i++) {
        //lj.Coeff()[i] = 1.0 / pow(2, i);
        lj.Coeff()[i] = fabs(2.5*sin(i + 1));
    }

    Settings settings;
    settings["fit:energy_weight"] = "1000.0";
    settings["fit:force_weight"] = "1.0";
    settings["fit:stress_weight"] = "0.1";
    settings["fit:scale_by_force"] = "0.0";

    NonLinearRegression dtr(&lj, settings.ExtractSubSettings("fit"));

    //NonLinearRegression dtr(&lj, 1000.0, 1.0, 0.1, 0.0, 1.0);

    //ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);
    ifstream ifs("NaCl_small.cfgs", ios::binary);

    for (int i = 0; cfg.Load(ifs); i++) {
        // FD test
        if (!dtr.CheckLossConsistency_debug(cfg, displacement, control_delta))
            FAIL()
    }

    ifs.close();

    return true;
}

// "PairPot AddLossGrad test by finite difference"
bool utest_ppalg()
{
    Configuration cfg, cfg1, cfg2;
    double displacement = 1e-5;
    double control_delta = 0.001;
    PairPotentialNonlinear pair(8, 1.6);

    Settings settings;
    settings["fit:energy_weight"] = "1.0";
    settings["fit:force_weight"] = "1.0";
    settings["fit:stress_weight"] = "10.0";
    settings["fit:scale_by_force"] = "0.0";

    NonLinearRegression dtr(&pair, settings.ExtractSubSettings("fit"));

    //NonLinearRegression dtr(&pair, 1.0, 1.0, 10.0, 0.0, 1.0);

    for (int i = 0; i < pair.CoeffCount(); i++) {
        //pair.Coeff()[i] = 1.0 / pow(2, i);
        pair.Coeff()[i] = 1*sin(i + 1);
    }

    //ifstream ifs("Al_Ni4cdiffs.cfgs", ios::binary);
    ifstream ifs("NaCl_small.cfgs", ios::binary);

    for (int i = 0; cfg.Load(ifs); i++) {
        // FD test
        if (!dtr.CheckLossConsistency_debug(cfg, displacement, control_delta))
            FAIL()
    }

    ifs.close();

    return true;
}

// "MTPR CalcEFS (check forces correctness by using of central differences)"
bool utest_mtprcefs()
{
    //Configuration cfg, cfg1, cfg2;

    //double diffstep = 1E-4;
    //double relErrMax = 0.0, absErrMax = 0.0;
    //double frob_norm2 = 0.0, forces_prec2 = 0.0;

    //ifstream ifs("2comp.cfgs", ios::binary);

    //MLMTPR pot("learned.mtp");

    //for (int n = 0; cfg.load(ifs); n++) {
    //pot.CalcEFS(cfg);

    //for (int i = 0; i < cfg.size(); i++) {
    //for (int l = 0; l < 3; l++) {
    //cfg1 = cfg;
    //cfg2 = cfg;
    //cfg1.pos(i)[l] += diffstep;
    //cfg2.pos(i)[l] -= diffstep;
    //pot.CalcEFS(cfg1);
    //pot.CalcEFS(cfg2);
    //double cdiffForce = (cfg2.energy - cfg1.energy) / (2 * diffstep);
    //double absErr = fabs(cfg.force(i)[l] - cdiffForce);
    //double relErr = fabs((cfg.force(i)[l] - cdiffForce) / cdiffForce);
    //if (relErr > relErrMax) relErrMax = relErr;
    //if (absErr > absErrMax) absErrMax = absErr;
    //}
    //}
    //if (relErrMax > 1E-2) {
    //cout << relErrMax << endl;
    //FAIL()
    //}
    //if (absErrMax > 1E-2) {
    //cout << absErrMax << endl;
    //FAIL()
    //}
    //}

    //ifs.close();

    Configuration cfg;
    double diffstep = 1E-4;

    ifstream ifs("2comp.cfgs", ios::binary);
    cfg.Load(ifs);

    MLMTPR pot("learned.mtp");

    //cout << pot.CoeffCount() << " " << pot.alpha_count-1 << endl;

    if (!pot.CheckEFSConsistency_debug(cfg, diffstep))
        FAIL();

    ifstream ifs1;
    ifs1.open("fe_magmom.cfg");
    cfg.Load(ifs1);

    MLMTPR pot_mag("fe_mag.mtp");

    std::random_device random_device;
    std::default_random_engine eng(random_device());
    std::uniform_real_distribution<double> distr(-1, 1);

    for (int i = 0; i < pot_mag.CoeffCount(); i++) 
        pot_mag.Coeff()[i] = 0.01*distr(eng);

    if (!pot_mag.CheckEFSConsistency_debug(cfg, diffstep))
        FAIL();

    return true;
}

// "MTPR CalcEFSGrads (Compare gradients calculated by potential and pointer)"
bool utest_mtprcefsgrads()
{
    Configuration cfg;
    //ifstream ifs("2comp.cfgs", ios::binary);
    ifstream ifs("fe_magmom.cfg", ios::binary);
    cfg.Load(ifs);
    //MLMTPR pot("learned.mtp");
    MLMTPR pot("fe_mag.mtp");
    AnyLocalMLIP* p_any = &pot;
    
    Array1D ene_grad;
    Array3D frc_grad;
    Array3D str_grad;
    Array1D ene_grad2;
    Array3D frc_grad2;
    Array3D str_grad2;

    pot.CalcEFSGrads(cfg, ene_grad, frc_grad, str_grad);
    pot.AnyLocalMLIP::CalcEFSGrads(cfg, ene_grad2, frc_grad2, str_grad2);

    if (ene_grad.size() != pot.CoeffCount() ||
        ene_grad2.size() != pot.CoeffCount()) 
        FAIL();
    
    if (frc_grad.size1 != cfg.size() ||
        frc_grad.size2 != 3 ||
        frc_grad.size3 != pot.CoeffCount() ||
        frc_grad2.size1 != cfg.size() ||
        frc_grad2.size2 != 3 ||
        frc_grad2.size3 != pot.CoeffCount()) {
        FAIL();
    }

    for (int i=0; i<frc_grad.size1; i++)
        for (int j=0; j<frc_grad.size2; j++)
            for (int k=0; k<frc_grad.size3; k++)
                if (frc_grad(i, j, k) != frc_grad2(i, j, k))
                    FAIL();

    for (int i=0; i<str_grad.size1; i++)
        for (int j=0; j<str_grad.size2; j++)
            for (int k=0; k<str_grad.size3; k++)
                if (str_grad(i, j, k) != str_grad2(i, j, k)) 
                    FAIL();

    pot.AnyLocalMLIP::CalcForcesGrad(cfg, frc_grad2);

    for (int i=0; i<frc_grad.size1; i++)
        for (int j=0; j<frc_grad.size2; j++)
            for (int k=0; k<frc_grad.size3; k++)
                if (frc_grad(i, j, k) != frc_grad2(i, j, k)) 
                    FAIL();

    pot.AnyLocalMLIP::CalcStressesGrads(cfg, str_grad2);

    for (int i=0; i<str_grad.size1; i++)
        for (int j=0; j<str_grad.size2; j++)
            for (int k=0; k<str_grad.size3; k++)
                if (str_grad(i, j, k) != str_grad2(i, j, k))
                    FAIL();

    pot.AnyLocalMLIP::CalcEnergyGrad(cfg, ene_grad2);

    for (int i=0; i<ene_grad.size(); i++)
        if (ene_grad[i] != ene_grad2[i])
            FAIL();

    return true;
}

// "MTPR CalcEFSGrads2 (Check energies, forces and stresses gradients using central differences)"
bool utest_mtprcefsgrads2()
{
    Configuration cfg;
    //ifstream ifs("2comp.cfgs", ios::binary);
    ifstream ifs("al_fe_mag.cfg", ios::binary);
    cfg.Load(ifs);
    //MLMTPR pot("learned.mtp");
    MLMTPR pot("al_fe_mag.mtpr");

    /*std::random_device random_device;
    std::default_random_engine eng(random_device());
    std::uniform_real_distribution<double> distr(-1, 1);

    for (int i = 0; i < pot.CoeffCount(); i++) 
        pot.Coeff()[i] = 0.1*distr(eng);*/

    Array1D ene_grad;
    Array3D frc_grad;
    Array3D str_grad;

    pot.CalcEFSGrads(cfg, ene_grad, frc_grad, str_grad);
    double delta = 1e-4;
    double rel_err_en_max = 0;
    double rel_err_force_max = 0;
    double rel_err_stress_max = 0;

    //cout.precision(12);
    for (int i=0; i<pot.CoeffCount(); i++)
    {
        Configuration cfg1 = cfg;
        Configuration cfg2 = cfg;
        pot.Coeff()[i] += delta;
        pot.CalcEFS(cfg1);
        pot.Coeff()[i] -= 2*delta;
        pot.CalcEFS(cfg2);
        pot.Coeff()[i] += delta;
        double cdiff = (cfg1.energy - cfg2.energy) / (2*delta);

        if (abs((cdiff - ene_grad[i]) / cdiff)>rel_err_en_max) 
            rel_err_en_max = fabs((cdiff - ene_grad[i]) / cdiff);

        for (int j = 0; j < cfg.size(); j++) {
            for (int a = 0; a < 3; a++) {
                double cdiff = (cfg1.force(j)[a] - cfg2.force(j)[a]) / (2*delta);
                if (abs((cdiff - frc_grad(j,a,i)) / cdiff)>rel_err_force_max) {
                    //cout << cdiff << " " << frc_grad(j,a,i) << endl;
                    rel_err_force_max = fabs((cdiff - frc_grad(j,a,i)) / cdiff);
                }
            }
        }

        for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
                double cdiff = (cfg1.stresses[a][b] - cfg2.stresses[a][b]) / (2*delta);
                if (abs((cdiff - str_grad(a,b,i)) / cdiff)>rel_err_stress_max) {
                    //cout << cdiff << " " << str_grad(a,b,i) << endl;
                    rel_err_stress_max = fabs((cdiff - str_grad(a,b,i)) / cdiff);
                }
            }
        }

    }

        //cout << rel_err_en_max << endl;
        //cout << rel_err_force_max << endl;
        //cout << rel_err_stress_max << endl;

    if (rel_err_en_max > 1e-3)
        FAIL();
    if (rel_err_force_max > 1e-3)
        FAIL();
    if (rel_err_stress_max > 1e-3)
        FAIL();

    return true;
}

// NLR weighting test
bool utest_nlrw0()
{	
	
    const int ncut=20;
    const int nend=20;
    map<string,string> mapp;

    mapp["energy_weight"] = "1.0";
    mapp["force_weight"] = "0.01";
    mapp["stress_weight"] = "0.001";
	mapp["init_random"] = "FALSE";
	mapp["skip_preinit"] = "TRUE";
	mapp["iteration_limit"] = "123";
	mapp["save_to"] = "temp/Trained.mtp_";

    double e_vibr,e_struct,e_mol,f_vibr,f_struct,f_mol,s_vibr,s_struct,s_mol;

    //SetTagLogStream("dev",&std::cout);
    // vibration
   {
	  #ifndef MLIP_MPI
      auto cfgs = LoadCfgs("CuPd.cfg",ncut);
#else
      auto cfgs = MPI_LoadCfgs("CuPd.cfg",ncut);
#endif

      MLMTPR mtpr0("NaClunlearned.mtp");

      MTPR_trainer nlr0(&mtpr0, mapp); // 1, 1.0e-3, 1.0e-3);
	  nlr0.AddSpecies(cfgs);

	  for (int i = 0; i < mtpr0.CoeffCount(); i++)
         mtpr0.Coeff()[i] = 1e-2*sin(99*(i+2) + i*i);

	  mtpr0.inited=true;

      Configuration cfg;
      ErrorMonitor errmon;
      
      nlr0.bfgs_conv_tol=1.0e-7;
//       nlr0.weighting = "vibrations";
      nlr0.wgt_scale_power_energy = 1;
      nlr0.wgt_scale_power_forces = 0;
      nlr0.wgt_scale_power_stress = 1;

      nlr0.Train(cfgs);
	
      ifstream ifs("CuPd.cfg", ios::binary);
      for (int i = 0; i < nend && cfg.Load(ifs); i++) {
          //if (i < ncut) continue;
          Configuration cfg1(cfg);
          mtpr0.CalcEFS(cfg1);
          errmon.AddToCompare(cfg, cfg1);
      }

        e_vibr = errmon.ene_rmsabs();
        f_vibr = errmon.frc_rmsabs();
        s_vibr = errmon.str_rmsabs();

       //printf("\n>> %e %e %e\n",errmon.ene_rmsabs(),errmon.frc_rmsabs(),errmon.str_rmsabs());

    }
    // structures
    {

 #ifndef MLIP_MPI
      auto cfgs = LoadCfgs("CuPd.cfg",ncut);
#else
      auto cfgs = MPI_LoadCfgs("CuPd.cfg",ncut);
#endif

      MLMTPR mtpr1("NaClunlearned.mtp");
      MTPR_trainer nlr1(&mtpr1, mapp); // 1, 1.0e-3, 1.0e-3);
	  nlr1.AddSpecies(cfgs);

	for (int i = 0; i < mtpr1.CoeffCount(); i++)
        mtpr1.Coeff()[i] = 1e-2*sin(99*(i+2) + i*i);

	  mtpr1.inited=true;

      Configuration cfg;
      ErrorMonitor errmon;

      nlr1.bfgs_conv_tol = 1.0e-7;
//    nlr1.weighting = "structures";
      nlr1.wgt_scale_power_energy = 2;
      nlr1.wgt_scale_power_forces = 1;
      nlr1.wgt_scale_power_stress = 2;
      nlr1.Train(cfgs);

      ifstream ifs("CuPd.cfg", ios::binary);
      for (int i = 0; i < nend && cfg.Load(ifs); i++) {
         //if (i < ncut) continue;
          Configuration cfg1(cfg);
          mtpr1.CalcEFS(cfg1);
          errmon.AddToCompare(cfg, cfg1);
      }

        e_struct = errmon.ene_rmsabs();
        f_struct = errmon.frc_rmsabs();
        s_struct = errmon.str_rmsabs();

      //printf("\n>> %e %e %e\n",errmon.ene_rmsabs(),errmon.frc_rmsabs(),errmon.str_rmsabs());
             
    } 
    // molecules
    {

#ifndef MLIP_MPI
      auto cfgs = LoadCfgs("CuPd.cfg",ncut);
#else
      auto cfgs = MPI_LoadCfgs("CuPd.cfg",ncut);
#endif

      MLMTPR mtpr2("NaClunlearned.mtp");
      MTPR_trainer nlr2(&mtpr2, mapp); // 1, 1.0e-3, 1.0e-3);
	  nlr2.AddSpecies(cfgs);

	for (int i = 0; i < mtpr2.CoeffCount(); i++)
        mtpr2.Coeff()[i] = 1e-2*sin(99*(i+2) + i*i);

	  mtpr2.inited=true;

      Configuration cfg;
      ErrorMonitor errmon;
   
      nlr2.bfgs_conv_tol = 1.0e-7;
//       nlr0.weighting = "molecules";
      nlr2.wgt_scale_power_energy = 0;
      nlr2.wgt_scale_power_forces = 0;
      nlr2.wgt_scale_power_stress = 0;
      nlr2.Train(cfgs);

      ifstream ifs("CuPd.cfg", ios::binary);
      for (int i = 0; i < nend && cfg.Load(ifs); i++) {
          //if (i < ncut) continue;
          Configuration cfg1(cfg);
          mtpr2.CalcEFS(cfg1);
          errmon.AddToCompare(cfg, cfg1);
      }

        e_mol = errmon.ene_rmsabs();
        f_mol = errmon.frc_rmsabs();
        s_mol = errmon.str_rmsabs();

       //printf("\n>> %e %e %e\n",errmon.ene_rmsabs(),errmon.frc_rmsabs(),errmon.str_rmsabs());
    }

    //dividing the weights by greater powers of N's (configs.' sizes) leads to bigger RMS errors of corresponding quantities - this what is checked below
    if (e_mol >= e_vibr)
       FAIL()
    if (f_vibr >= f_mol)
       FAIL()
    if (e_vibr <= e_mol)
       FAIL()
    if (s_mol >= s_vibr)
       FAIL()
    if (s_mol >= s_struct)
       FAIL()           

  return true;
}

