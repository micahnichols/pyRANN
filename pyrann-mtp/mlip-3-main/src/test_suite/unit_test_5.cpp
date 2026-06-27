#include <fstream>
#include <iostream>
#include <algorithm>

#include "../common/utils.h"
#include "../sample_potentials.h"
#include "../non_linear_regression.h"
#include "../mtpr.h"
#include "../mtpr_trainer.h"
#include "../eam.h"

using namespace std;


#define FAIL(...) \
{std::string str("" __VA_ARGS__); \
std::cout << " failed!" << std::endl; \
if(!str.empty()) std::cout << std::endl << "Error message: " << str << std::endl; \
return false;}


// "Training PairPotentialNonlinear"
bool utest_ppnl()
{
    PairPotentialNonlinear pair_exact(6, 1.9);
    PairPotentialNonlinear pair(6, 1.9);

    for (int i = 0; i < pair.CoeffCount(); i++) {
        pair_exact.Coeff()[i] = 1.0 / pow(2, i);
        pair.Coeff()[i] = 0.25;
    }

    Settings settings;
    settings["fit:energy_weight"] = "1.0";
    settings["fit:force_weight"] = "1.0";
    settings["fit:stress_weight"] = "1.0";

    NonLinearRegression dtr(&pair, settings.ExtractSubSettings("fit"));

    //NonLinearRegression dtr(&pair, 1.0, 1.0, 1.0, 0.0, 1.0);

    Configuration cfg;
    vector<Configuration> training_set;
    //ifstream ifs1("Ti4MVtest.cfg", ios::binary);
    ifstream ifs1("NaCl_eam.cfgs", ios::binary);
    for (int i = 0; cfg.Load(ifs1); i++) {
        pair_exact.CalcEFS(cfg);

#ifdef MLIP_MPI
        if (i % mpi.size == mpi.rank)
#endif
            training_set.push_back(cfg);
    }
    ifs1.close();

    dtr.Train(training_set);

    // comparing
    double max_coeff_difference = 0;
    for (int i = 0; i < pair.CoeffCount(); i++) {
        max_coeff_difference = std::max(max_coeff_difference,
            fabs((pair.Coeff()[i] - pair_exact.Coeff()[i])));
        //std::cout << pair.Coeff()[i] << " " << pair_exact.Coeff()[i] << std::endl;
    }
    if (max_coeff_difference > 0.02) {
        std::cerr << "(error: max_coeff_difference is " << max_coeff_difference << ") ";
        FAIL()
    }
    double max_en_difference = 0;

    for (const Configuration& cfg_orig : training_set) {
        cfg = cfg_orig;
        pair.CalcEFS(cfg);
        max_en_difference = std::max(max_en_difference,
            fabs((cfg.energy - cfg_orig.energy) / cfg_orig.energy));
        //std::cout << cfg.energy << " " << pair.train_set[i].energy << " ";
        //std::cout << fabs((cfg.energy - pair.train_set[i].energy)/pair.train_set[i].energy) << std::endl;
    }
    if (max_en_difference > 0.02) {
        std::cerr << "(error: max_en_difference is " << max_en_difference << ") ";
        FAIL()
    }
    return true;
}

// "EAMSimple fitting test")
bool utest_eams1()
{
    EAMSimple eam_exact(8, 5, 1.6, 5.0);
    EAMSimple eam(8, 5, 1.6, 5.0);

    for (int i = 0; i < eam.CoeffCount(); i++) {
        //eam_exact.Coeff()[i] = 0.5 * i / eam.CoeffCount();
        //eam.Coeff()[i] = 1.0 / (i + 1);
        eam_exact.Coeff()[i] = 1.3*sin(i+1);
        eam.Coeff()[i] = 0.1;
    }

    Settings settings;
    settings["fit:energy_weight"] = "1000.0";
    settings["fit:force_weight"] = "1.0";
    settings["fit:stress_weight"] = "0.1";

    NonLinearRegression trainer(&eam, settings.ExtractSubSettings("fit"));

    //NonLinearRegression trainer(&eam, 1000.0, 1.0, 0.1, 0.0, 1.0);

    Configuration cfg;
    vector<Configuration> training_set;
    ifstream ifs1("NaCl_small.cfgs", ios::binary);
    for (int i = 0; cfg.Load(ifs1); i++) {
        //for (int i = 0; i < 1; i++) {
        //cfg.load(ifs1);
        eam_exact.CalcEFS(cfg);
        //cout << cfg.energy << endl;
        //for (int j = 0; j < cfg.size(); j++)
        //  cout << cfg.force(j)[0] << " " << cfg.force(j)[1] << " " << cfg.force(j)[2] << endl;
#ifdef MLIP_MPI
        if (i % mpi.size == mpi.rank)
#endif
            training_set.push_back(cfg);
    }
    ifs1.close();

    trainer.Train(training_set);

    // comparing
    //double max_coeff_difference = 0;
    //for (int i = 0; i < eam.CoeffCount(); i++) {
    //  std::cout << eam.Coeff()[i] << " " << eam_exact.Coeff()[i] << std::endl;
    //  max_coeff_difference = std::max(max_coeff_difference,
    //      fabs((eam.Coeff()[i] - eam_exact.Coeff()[i])));
    //}
    //if (max_coeff_difference > 0.02) {
    //  std::cerr << "(error: max_coeff_difference is " << max_coeff_difference << ") ";
    //  //FAIL()
    //}

    //double sum_coeffs2 = 0;
    //for (int i = 8; i < 2*8; i++) {
    //  sum_coeffs2 += eam.Coeff()[i]*eam.Coeff()[i];
    //  //std::cout << eam.Coeff()[i] << std::endl;
    //} 
    //std::cout << "sum_coeffs2 = " << sum_coeffs2 << std::endl;

    // comparing
    double max_en_difference = 0;
    for (const Configuration& cfg_orig : training_set) {
        cfg = cfg_orig;
        eam.CalcEFS(cfg);
        max_en_difference = std::max(max_en_difference,
            fabs((cfg.energy - cfg_orig.energy) / cfg_orig.energy));
        //std::cout << "energy: " << std::endl;
        //std::cout << cfg.energy << " " << cfg_orig.energy << " ";
        //std::cout << fabs((cfg.energy - cfg_orig.energy)/cfg_orig.energy) << std::endl;
        //std::cout << "forces: " << std::endl;
        //for (int ii = 0; ii < cfg_orig.size(); ii++) {
        //  std::cout << cfg.force(ii)[0] << " " << cfg_orig.force(ii)[0] << " ";
        //  std::cout << cfg.force(ii)[1] << " " << cfg_orig.force(ii)[1] << " ";
        //  std::cout << cfg.force(ii)[2] << " " << cfg_orig.force(ii)[2] << std::endl;
        //}
        //std::cout << "stresses: " << std::endl;
        //for (int a = 0; a < 3; a++)
        //  for (int b = 0; b < 3; b++)
        //      std::cout << cfg.stresses[a][b] << " " << eam.train_set[i].stresses[a][b] << std::endl;
    }
    if (max_en_difference > 0.02) {
        std::cerr << "(error: max_en_difference is " << max_en_difference << ") ";
        FAIL()
    }

    return true;
}

// "EAMMult fitting test"
bool utest_eamm1()
{
    EAMMult eam_exact(5, 3, 2, 1.6, 5.0);
    EAMMult eam(5, 3, 2, 1.6, 5.0);

    for (int i = 0; i < eam.CoeffCount(); i++) {
        //eam_exact.Coeff()[i] = 0.5 * i / eam.CoeffCount();
        //eam.Coeff()[i] = 1.0 / (i + 1);
        eam_exact.Coeff()[i] = 0.6*sin(i+1);
        eam.Coeff()[i] = 0.1;
    }

    Settings settings;
    settings["fit:energy_weight"] = "1.0";
    settings["fit:force_weight"] = "1.0";
    settings["fit:stress_weight"] = "0.1";

    NonLinearRegression trainer(&eam, settings.ExtractSubSettings("fit"));

    //NonLinearRegression trainer(&eam, 1.0, 1.0, 0.1, 0.0, 1.0);

    Configuration cfg;
    vector<Configuration> training_set;
    ifstream ifs1("NaCl_small.cfgs", ios::binary);
    for (int i = 0; cfg.Load(ifs1); i++) {
        //for (int i = 0; i < 1; i++) {
        //cfg.load(ifs1);
        eam_exact.CalcEFS(cfg);
        //cout << cfg.energy << endl;
        //for (int j = 0; j < cfg.size(); j++)
        //  cout << cfg.force(j)[0] << " " << cfg.force(j)[1] << " " << cfg.force(j)[2] << endl;
#ifdef MLIP_MPI
        if (i % mpi.size == mpi.rank)
#endif
            training_set.push_back(cfg);
    }
    ifs1.close();

    trainer.Train(training_set);

    // comparing
    double max_en_difference = 0;
    for (const Configuration& cfg_orig : training_set) {
        cfg = cfg_orig;
        eam.CalcEFS(cfg);
        max_en_difference = std::max(max_en_difference,
            fabs((cfg.energy - cfg_orig.energy) / cfg_orig.energy));
        //std::cout << "energy: " << std::endl;
        //std::cout << cfg.energy << " " << cfg_orig.energy << " ";
        //std::cout << fabs((cfg.energy - cfg_orig.energy)/cfg_orig.energy) << std::endl;
        //std::cout << "forces: " << std::endl;
        //for (int ii = 0; ii < cfg_orig.size(); ii++) {
        //  std::cout << cfg.force(ii)[0] << " " << cfg_orig.force(ii)[0] << " ";
        //  std::cout << cfg.force(ii)[1] << " " << cfg_orig.force(ii)[1] << " ";
        //  std::cout << cfg.force(ii)[2] << " " << cfg_orig.force(ii)[2] << std::endl;
        //}
        //std::cout << "stresses: " << std::endl;
        //for (int a = 0; a < 3; a++)
        //  for (int b = 0; b < 3; b++)
        //      std::cout << cfg.stresses[a][b] << " " << cfg_orig.stresses[a][b] << std::endl;
    }
    if (max_en_difference > 0.02) {
        std::cerr << "(error: max_en_difference is " << max_en_difference << ") ";
        FAIL()
    }
    return true;
}
//#endif

// "Lennard-Jones fitting test"
bool utest_ljp1()
{
    LennardJones LJexact("unlearned_lj_lin.mlip", 1.6);
    LennardJones LJ("unlearned_lj_lin.mlip", 1.6);

    for (int i = 0; i < LJ.CoeffCount(); i++) {
        LJexact.Coeff()[i] = sin(i) / 5;
        //LJexact.Coeff()[i] = 0.1 * i / LJ.CoeffCount();
        //LJ.Coeff()[i] = 1.0 / (i + 1);
        LJ.Coeff()[i] = 5;
    }

    Settings settings;
    settings["fit:energy_weight"] = "1.0";
    settings["fit:force_weight"] = "1.0";
    settings["fit:stress_weight"] = "1.0";

    NonLinearRegression trainer(&LJ, settings.ExtractSubSettings("fit"));

    //NonLinearRegression trainer(&LJ, 1.0, 1.0, 1.0, 0.0, 1.0);

    vector<Configuration> training_set;
    Configuration cfg;
    ifstream ifs1("NaCl_small.cfgs", ios::binary);
    for (int i = 0; cfg.Load(ifs1); i++) {
        //for (int i = 0; i < 1; i++) {
        //cfg.load(ifs1);
        LJexact.CalcEFS(cfg);
#ifdef MLIP_MPI
        if (i % mpi.size == mpi.rank)
#endif
            training_set.push_back(cfg);
    }
    ifs1.close();

    trainer.Train(training_set);

    // comparing
    double max_en_difference = 0;
    for (const Configuration& cfg_orig : training_set) {
        cfg = cfg_orig;
        LJ.CalcEFS(cfg);
        max_en_difference = std::max(max_en_difference,
            fabs(cfg.energy - cfg_orig.energy));
        //std::cout << "energy: " << std::endl;
        //std::cout << cfg.energy << " " << LJ.train_set[i].energy << " ";
        //std::cout << fabs((cfg.energy - LJ.train_set[i].energy)/LJ.train_set[i].energy) << std::endl;
        //std::cout << "forces: " << std::endl;
        //for (int ii = 0; ii < LJ.train_set[i].size(); ii++) {
        //  std::cout << cfg.force(ii)[0] << " " << LJ.train_set[i].force(ii)[0] << " ";
        //  std::cout << cfg.force(ii)[1] << " " << LJ.train_set[i].force(ii)[1] << " ";
        //  std::cout << cfg.force(ii)[2] << " " << LJ.train_set[i].force(ii)[2] << std::endl;
        //}
        //std::cout << "stresses: " << std::endl;
        //for (int a = 0; a < 3; a++)
        //  for (int b = 0; b < 3; b++)
        //      std::cout << cfg.stresses[a][b] << " " << LJ.train_set[i].stresses[a][b] << std::endl;
    }
    if (max_en_difference > 0.02) {
        std::cerr << "(error: max_en_difference is " << max_en_difference << ") ";
        FAIL()
    }
    return true;
}

// "NonlinearLennard-Jones fitting test"
bool utest_nlljp1()
{
    NonlinearLennardJones LJexact("unlearned_lj_nonlin.mlip", 1.6, 5);
    NonlinearLennardJones LJ("unlearned_lj_nonlin.mlip", 1.6, 5);

    for (int i = 0; i < LJ.CoeffCount(); i++) {
        LJexact.Coeff()[i] = 2*i;
        //LJexact.Coeff()[i] = 0.1 * i / LJ.CoeffCount();
        //LJ.Coeff()[i] = 1.0 / (i + 1);
        LJ.Coeff()[i] = 1.5*fabs(sin(i));
    }

    Settings settings;
    settings["fit:energy_weight"] = "100.0";
    settings["fit:force_weight"] = "1.0";
    settings["fit:stress_weight"] = "0.1";

    NonLinearRegression trainer(&LJ, settings.ExtractSubSettings("fit"));
    trainer.wgt_eqtn_constr = 0.0;

    //NonLinearRegression trainer(&LJ, 100.0, 1.0, 0.1, 0.0, 0.0);

    vector<Configuration> training_set;
    Configuration cfg;
    ifstream ifs1("NaCl_small.cfgs", ios::binary);
    for (int i = 0; cfg.Load(ifs1); i++) {
        //for (int i = 0; i < 1; i++) {
        //cfg.load(ifs1);
        LJexact.CalcEFS(cfg);
#ifdef MLIP_MPI
        if (i % mpi.size == mpi.rank)
#endif
            training_set.push_back(cfg);
    }
    ifs1.close();

    trainer.Train(training_set);

    // comparing
    double max_en_difference = 0;
    int i = 0;
    for (const Configuration& cfg_orig : training_set) {
        cfg = cfg_orig;
        LJ.CalcEFS(cfg);
        max_en_difference = std::max(max_en_difference,
            fabs(cfg.energy - cfg_orig.energy));
        //std::cout << "energy: " << std::endl;
        //std::cout << cfg.energy << " " << training_set[i].energy << " ";
        //std::cout << fabs((cfg.energy - training_set[i].energy)/training_set[i].energy) << std::endl;
        //std::cout << "forces: " << std::endl;
        //for (int ii = 0; ii < training_set[i].size(); ii++) {
        //  std::cout << cfg.force(ii)[0] << " " << training_set[i].force(ii)[0] << " ";
        //  std::cout << cfg.force(ii)[1] << " " << training_set[i].force(ii)[1] << " ";
        //  std::cout << cfg.force(ii)[2] << " " << training_set[i].force(ii)[2] << std::endl;
        //}
        //std::cout << "stresses: " << std::endl;
        //for (int a = 0; a < 3; a++)
        //  for (int b = 0; b < 3; b++)
        //      std::cout << cfg.stresses[a][b] << " " << training_set[i].stresses[a][b] << std::endl;
        i++;
    }
    if (max_en_difference > 0.02) {
        std::cerr << "(error: max_en_difference is " << max_en_difference << ") ";
        FAIL()
    }
    return true;
}

//MTPR CalcEFS with absolute atom numeration test
bool utest_mtprp21()
{

    MLMTPR mtpr_relative("learned.mtp");
    MLMTPR mtpr_absolute("learned_abs.mtp");

    Configuration cfg;

    ifstream ifs("NaCl_absolute.cfgs", ios::binary);
    cfg.Load(ifs);
    ifs.close();

    Configuration cfg2;

    ifstream ifs2("NaCl_small.cfgs", ios::binary);
    cfg2.Load(ifs2);
    ifs2.close();

    try
    {
    mtpr_absolute.CalcEFS(cfg);
    }
    catch (...) {
    FAIL();
    }

    try
    {
    mtpr_relative.CalcEFS(cfg2);
    }
    catch (...) {
    FAIL();
    }

    try
    {
    mtpr_absolute.CalcEFS(cfg2);
    FAIL();
    }
    catch (...) {
    }

    try
    {
    mtpr_relative.CalcEFS(cfg);
    FAIL();
    }
    catch (...) {
    }

    return true;
}

//MTPR species_count extension test
bool utest_mtprp11()
{

    MLMTPR mtpr_unfit("unlearned_0spec.mtp");
    //MLMTPR mtpr_fit("learned.mtp");

	map<string,string> mapp;
	mapp["save_to"] = "temp/Trained.mtp_";

    MTPR_trainer nlr_unfit(&mtpr_unfit,mapp);//, 1, 0.1, 1e-3, 0, 1e-8);
    //MTPR_trainer nlr_fit(&mtpr_fit,mapp);//, 1, 0.1, 1e-3, 0, 1e-8);

    int old_coeff_count = mtpr_unfit.CoeffCount();

    if (mtpr_unfit.species_count!=0)
        FAIL("f1");
    if (mtpr_unfit.LinSize()!=8)
        FAIL("f2");

#ifdef MLIP_MPI
    vector<Configuration> training_set = MPI_LoadCfgs("NaCl_absolute.cfgs");
#else
    vector<Configuration> training_set = LoadCfgs("NaCl_absolute.cfgs");
#endif

    //mtpr_fit.Save("fit_new_spec.mtp");
    //mtpr_unfit.Save("unfit_new_spec.mtp");

    nlr_unfit.maxits = 1;
    nlr_unfit.Train(training_set);

    //nlr_fit.maxits = 1;
   // nlr_fit.Train(training_set);

    if (mtpr_unfit.species_count!=2)
        FAIL("f3");
    if (mtpr_unfit.LinSize()!=10)
        FAIL("f4");
        
   // if (mtpr_fit.species_count!=4)
   //     FAIL("f5");
   // if (mtpr_fit.LinSize()!=12)
   //     FAIL("f6");

    int new_coeff_count = mtpr_unfit.CoeffCount();

    if ( new_coeff_count - old_coeff_count !=  4*16 + 2)
        FAIL("f7");
    
    return true;
}

// "MTPR fitting test"
bool utest_mtprp1()
{
    MLMTPR mtpr_trained("learned.mtp");

    for (int i = 0; i < mtpr_trained.CoeffCount(); i++)
        mtpr_trained.Coeff()[i] = 1e-2*rand() / RAND_MAX;

#ifdef MLIP_MPI
    MPI_Bcast(&mtpr_trained.Coeff()[0], mtpr_trained.CoeffCount(), MPI_DOUBLE, 0, mpi.comm);
#endif

    MLMTPR mtpr_fit("unlearned.mtp");
	map<string,string> mapp;
	mapp["save_to"] = "temp/Trained.mtp_";
	mapp["init_random"] = "FALSE";

	for (int i = 0; i < mtpr_fit.CoeffCount(); i++)
        mtpr_fit.Coeff()[i] = 1e-2*sin(99*(i+2) + i*i);

    //std::cout << mtpr_fit.CoeffCount() << std::endl;
    MTPR_trainer nlr(&mtpr_fit,mapp);//, 1, 1e-3, 1e-3, 0, 1e-8);

#ifdef MLIP_MPI
    vector<Configuration> training_set = MPI_LoadCfgs("NaCl_small.cfgs");
#else
    vector<Configuration> training_set = LoadCfgs("NaCl_small.cfgs");
#endif

    for (Configuration cfg: training_set)
        if (cfg.size()!=0)
            mtpr_trained.CalcEFS(cfg);

   Configuration cfg;

    nlr.wgt_eqtn_energy = 1;
    nlr.wgt_eqtn_forces = 1e-2;
	nlr.wgt_eqtn_stress = 1e-3;
    nlr.wgt_eqtn_constr = 1e-6;
    nlr.bfgs_conv_tol=1e-12;
    nlr.Train(training_set);

    // comparing
    double max_en_difference = 0;
    for (Configuration& conf : training_set) {
        if (conf.size()!=0){
            cfg = conf;
            mtpr_fit.CalcEFS(cfg);
            max_en_difference = std::max(max_en_difference,fabs(cfg.energy - conf.energy));
        }
    }
    double total_max = 0;
#ifdef MLIP_MPI
    MPI_Allreduce(&max_en_difference, &total_max, 1, MPI_DOUBLE, MPI_MAX, mpi.comm);
#else
    total_max = max_en_difference;
#endif

      if (total_max > 1e-2) {
#ifdef MLIP_MPI
        if (mpi.rank==0)
#endif
            std::cerr << "(error: max_en_difference is " << max_en_difference << ") ";
        FAIL()
        }
    return true;
}

// "MTPR loss gradient test"
bool utest_mtprp2()
{
    MLMTPR mtpr("learned.mtp");
	map<string,string> mapp;
	mapp["save_to"] = "temp/Trained.mtp_";
    MTPR_trainer nlr(&mtpr,mapp);//, 1, 1, 0);
    ifstream ifs("2comp.cfgs", ios::binary);
    Configuration cfg;
    int fail = 0;

    //if (mpi.rank == 0) cout << mtpr.CoeffCount() << " " << mtpr.alpha_count << endl;

    for (int i = 0; cfg.Load(ifs); i++)
#ifdef MLIP_MPI
        if (i % mpi.size == mpi.rank)
#endif
        {
            if (!nlr.CheckLossConsistency_debug(cfg, 1e-8))
                fail = 1;
        }
    ifs.close();

    int failed_any = 0;

#ifdef MLIP_MPI
    MPI_Reduce(&fail, &failed_any, 1, MPI_INT, MPI_MAX, 0, mpi.comm);
#else
    failed_any = fail;
#endif

    if (failed_any > 0) {
        std::cerr << "Somewhere the gradient is wrong";
        FAIL()
    }

    MLMTPR pot_mag("fe_mag.mtp");
    std::random_device random_device;
    std::default_random_engine eng(random_device());
    std::uniform_real_distribution<double> distr(-1, 1);

    for (int i = 0; i < pot_mag.CoeffCount(); i++) 
        pot_mag.Coeff()[i] = 0.01*distr(eng);

	map<string,string> mapp1;
	//mapp1["save_to"] = "temp/Trained.mtp_";
    MTPR_trainer nlr1(&pot_mag,mapp1);

    ifstream ifs1("fe_magmom.cfg", ios::binary);

    for (int i = 0; cfg.Load(ifs1); i++)
#ifdef MLIP_MPI
        if (i % mpi.size == mpi.rank)
#endif
        {
            if (!nlr1.CheckLossConsistency_debug(cfg, 1e-6))
                fail = 1;
        }
    ifs1.close();

    failed_any = 0;

#ifdef MLIP_MPI
    MPI_Reduce(&fail, &failed_any, 1, MPI_INT, MPI_MAX, 0, mpi.comm);
#else
    failed_any = fail;
#endif

    if (failed_any > 0) {
        std::cerr << "Somewhere the gradient is wrong";
        FAIL()
    }

    return true;
}
