#include <fstream>
#include <iostream>

#include "../common/utils.h"
#include "../common/bfgs.h"
#include "../configuration.h"
#include "../mtp.h"
#include "../drivers/relaxation.h"
#include "../linear_regression.h"
#include "../error_monitor.h"
#include "../basis.h"


using namespace std;


#define FAIL(...) \
{std::string str("" __VA_ARGS__); \
std::cout << " failed!" << std::endl; \
if (!str.empty()) std::cout << std::endl << "Error message: " << str << std::endl; \
return false;}

// "Linesearch"
bool utest_ls()
{
    {
        // minimize f(x) = -x - x^2 + x^3
        const double alpha_0[] = { 0.12,0.234,0.33,0.44,0.55,0.66,0.77,0.88,2.1,1000,1e30 };
        const int alpha_0_count = sizeof(alpha_0) / sizeof(double);
        for (int i = 0; i < alpha_0_count; i++) {
            Linesearch ls;
            ls.Reset(alpha_0[i]);
            int count;
            for (count = 0; count < 30; count++) {
                double x = ls.x();
                ls.Iterate(-x - x*x + x*x*x, -1 - 2 * x + 3 * x*x);
                if (std::abs(x - 1) < 1e-5) break;
            }
            if (count == 30) FAIL("no convergence after 30 iterations, alpha_0 = " + std::to_string(alpha_0[i]));
        }
    }
    {
        class Linesearch_mod : public Linesearch {
        public:
            void SetX(double _x) { curr_x = _x; }
        };

        // minimize f(x) = x/2-sin(x)
        Linesearch_mod ls;
        ls.Reset(10.0);
        {
            double x = ls.x();
            ls.Iterate(x / 2 - sin(x), 0.5 - cos(x));
        }
        {
            double x = ls.x();
            ls.Iterate(x / 2 - sin(x), 0.5 - cos(x));
        }
        ls.SetX(6.28);

        int count;
        for (count = 0; count < 30; count++) {
            double x = ls.x();
            ls.Iterate(x / 2 - sin(x), 0.5 - cos(x));
            if (std::abs(x - 1.04719755) < 1e-5) break;
        }
        if (count == 30) FAIL("no convergence after 30 iterations on Linesearch_mod");
    }
    {
        // minimize f(x) = x/2-sin(x)
        const double alpha_0[] = { 6.28,0.12,3.14, 1000, 1e30 };
        const int alpha_0_count = sizeof(alpha_0) / sizeof(double);
        for (int i = 0; i < alpha_0_count; i++) {
            Linesearch ls;
            ls.Reset(alpha_0[i]);
            int count;
            for (count = 0; count < 300; count++) {
                double x = ls.x();
                ls.Iterate(x/2 - sin(x), 0.5 - cos(x));
                if (std::abs(x - 1.04719755) < 1e-5) break;
            }
            if (count == 300) FAIL("no convergence after 300 iterations, alpha_0 = " + std::to_string(alpha_0[i]));
        }
    }
    return true;
}

// "checks minimal distance in configurations"
bool utest_cmd()
{
    int count = 0;
    int status = 0;

    // read the database
    std::ifstream ifs("configurations/reading_test.cfg", ios::binary);
    Configuration cfg;
    for (count = 0; (status = cfg.Load(ifs)); count++) {
        cfg.InitNbhs(6.0);
        double min_dist = 123e123;
        for (int i = 0; i < (int)cfg.nbhs.size(); i++)
            for (int j = 0; j < (int)cfg.nbhs[i].dists.size(); j++)
                min_dist = __min(min_dist, cfg.nbhs[i].dists[j]);
        double cfg_min_dist = cfg.MinDist();
        if (fabs(min_dist - cfg_min_dist) > 1e-6)
            FAIL()
    }
    return true;
}

// "Supercell rotation test"
bool utest_sr()
{
    ifstream ifs("4relax.cfg", ios::binary);
    Configuration cfg;
    cfg.Load(ifs);
    Matrix3 rot_mat(-0.686602, -0.340459, -0.642391, 
                    -0.126594, -0.814094, 0.566767,
                    -0.715927, 0.470466, 0.515858   );
    MTP mtp("MTP4relax.mtp");
    cfg.Deform(rot_mat);
    mtp.CalcEFS(cfg);
    cfg.CorrectSupercell();
    Configuration cfg2(cfg);
    mtp.CalcEFS(cfg2);

    for (int i=0; i<cfg.size(); i++)
        if ((cfg2.force(i) - cfg.force(i)).Norm() > 1.0e-7)
        //{
        //  cout << cfg2.force(i, 0) << ' ' << cfg2.force(i, 1) << ' ' << cfg2.force(i, 2) << endl;
        //  cout << cfg.force(i, 0) << ' ' << cfg.force(i, 1) << ' ' << cfg.force(i, 2) << endl;
        //}
                FAIL();

    if ((cfg2.stresses - cfg.stresses).NormFrobeniusSq() > 1.0e-7)
        FAIL();

    return true;
}

// "Relaxation driver test"
bool utest_rd()
{
    ifstream ifs("4relax.cfg", ios::binary);
    Configuration cfg;
    cfg.Load(ifs);
    MTP mtp("MTP4relax.mtp");
    Relaxation rlx(&mtp, cfg);
    rlx.init_mindist = 1.1;
    rlx.tol_force = 0.0001;
    rlx.tol_stress = 0.001;
    rlx.itr_limit = 500;
    //rlx.p_log = &cout;

    rlx.Run();
    cfg = rlx.cfg;

    //cout<< cfg.energy << ' ' << rlx.step;
    double mf = 0;
    for (int i=0; i<cfg.size(); i++)
     mf = __max(mf, cfg.force(i).Norm());

    if (fabs(cfg.energy + 25.339994) > 0.0001)
        FAIL("Energy of the relaxed configuration is "+to_string(cfg.energy)+". Should be 25.34");
    if (mf > rlx.tol_force)
        FAIL("Max force is "+to_string(mf)+" (too big)");
    if (cfg.stresses.MaxAbs() > rlx.tol_stress)
        FAIL("Max stress is " + to_string(cfg.stresses.MaxAbs()) + " (too big)");
    return true;
}

// "Robustness of relaxation with changing potential"
bool utest_rr()
{
    ifstream ifs("4relax.cfg", ios::binary);
    Configuration cfg;
    cfg.Load(ifs);
    MTP mtp("MTP4relax.mtp");

    class CrazyPotential : public AnyPotential
    {
    public:
        int call_cntr = 0;
        AnyPotential* ppot;
        CrazyPotential(AnyPotential* _pot) : ppot(_pot) {};
        void CalcEFS(Configuration& cfg) override
        {
            ppot->CalcEFS(cfg);
            if (call_cntr<100) cfg.energy += 100 * (call_cntr++ % 7);
        }
    };
    CrazyPotential crazy_pot(&mtp);

    map<string, string> settings;
    settings["tol_force"] = "0.0001";
    settings["tol_stress"] = "0.001";
    settings["itr_limit"] = "500";
    //settings["log"] = "stdout";
    Relaxation rlx(settings, nullptr, &crazy_pot);
    rlx.cfg = cfg;

    rlx.Run();
    cfg = rlx.cfg;

    while (cfg.energy > 100)
        cfg.energy -= 100;
    //cout<< cfg.energy << ' ' << rlx.step;

    double mf = 0;
    for (int i=0; i<cfg.size(); i++)
        mf = __max(mf, cfg.force(i).Norm());
    if (fabs(cfg.energy + 25.339994) > 0.0001)
        FAIL("Energy of the relaxed configuration is " + to_string(cfg.energy) +
            " but should be 25.34.");
    if (mf > rlx.tol_force)
        FAIL("Max force is " + to_string(mf) + " (too big)");
    if (cfg.stresses.MaxAbs() > rlx.tol_stress)
        FAIL("Max stress is " + to_string(cfg.stresses.MaxAbs()) + " (too big)");

    return true;
}

// "repulsion in relaxation"
bool utest_reprel()
{
    ifstream ifs("4repuls.cfg", ios::binary);
    Configuration cfg;
    cfg.Load(ifs);
    MTP foo("MTP4relax.mtp");


    Settings settings;
    settings["tol_force"] = "0.0001";
    settings["tol_stress"] = "0.001";
    settings["iteration_limit"] = "0";
    settings["init_mindist"] = "{3:1.3,6:0.7,19:2.0}";
    //settings["log"] = "stdout";
    Relaxation rlx(settings, nullptr, &foo);
    rlx.cfg = cfg;

    rlx.Run();
    auto md = rlx.cfg.GetTypeMindists();
    if (md[make_pair(3, 3)]<1.3 || 
        md[make_pair(3, 6)]<2.0 || 
        md[make_pair(3, 19)]<3.3 || 
        md[make_pair(6, 6)]<1.4 || 
        md[make_pair(6, 19)]<2.7 || 
        md[make_pair(19, 19)]<4.0)
        FAIL();

    return true;
}

// "Another MTP learning + ErrorMonitor"
bool utest_mtpl()
{
    MTP mtp("MTP2_61.params.mtp");
    Settings setup;
    setup["energy_weight"] = to_string(500);
    setup["force_weight"] = to_string(1);
    setup["stress_weight"] = to_string(10);
    LinearRegression regr(&mtp, setup);
    std::ifstream ifs("Si1.cfg", ios::binary);
    Configuration cfg;
    Settings foo;
    ErrorMonitor errmon(foo);

    vector<Configuration> training_set;
    for (int count = 0; cfg.Load(ifs); count++)
#ifndef MLIP_MPI
        training_set.push_back(cfg);
#else
        if (count % mpi.size == mpi.rank) 
            training_set.push_back(cfg);
#endif

    regr.Train(training_set);
    ifs.close();

    // the MTP that is already trained
    MTP mtp_ready("MTP_testsuite_test.params.mtp");
    ifs.open("Si2.cfg", ios::binary);
#ifndef MLIP_MPI
    for (int count = 0; cfg.Load(ifs); count++) {
        Configuration cfg2(cfg);
        mtp.CalcEFS(cfg);
        errmon.AddToCompare(cfg2, cfg);
    }
#else
    auto cfgs = MPI_LoadCfgs("Si2.cfg");
    for (int i=0; i<cfgs.size(); i++)
    {
        Configuration cfg = cfgs[i];
        mtp.CalcEFS(cfg);
        errmon.AddToCompare(cfgs[i], cfg);
    }
#endif
    
    ifs.close();

#ifdef GEN_TESTS
    std::ofstream ofs("test_errlearning.txt");
    ofs << errmon.cfg_cntr << std::endl;
    ofs << errmon.ene_aveabs() << std::endl;
    ofs << errmon.ene_averel() << std::endl;
    ofs << errmon.epa_aveabs() << std::endl;
    ofs << errmon.epa_averel() << std::endl;
    ofs << errmon.frc_aveabs() << std::endl;
    ofs << errmon.frc_rmsrel() << std::endl;
    ofs << errmon.str_rmsrel() << std::endl;
    ofs.close();
#endif

#ifdef MLIP_MPI
    if( 0 == mpi.rank) 
#endif
    {
    ifs.open("test_errlearning.txt");
    double foo;
    int bar;
    ifs >> bar;
//     if (bar != errmon.ene_cfg.count)
//         FAIL("")
        ifs >> foo;
    if (fabs(foo - errmon.ene_aveabs()) > 1.0e-6)
        FAIL()
        ifs >> foo;
    if (fabs(foo - errmon.ene_averel()) > 1.0e-6)
        FAIL()
        ifs >> foo;
    if (fabs(foo - errmon.epa_aveabs()) > 1.0e-6)
        FAIL()
        ifs >> foo;
    if (fabs(foo - errmon.epa_averel()) > 1.0e-6)
        FAIL()
        ifs >> foo;
    if (fabs(foo - errmon.frc_aveabs()) > 1.0e-6)
        FAIL()
        ifs >> foo;
    if (fabs(foo - errmon.frc_rmsrel()) > 1.0e-6)
        FAIL()
        ifs >> foo;
    if (fabs(foo - errmon.str_rmsrel()) > 1.0e-6)
        FAIL()
        ifs.close();
    }
    return true;
}

// "MTP fitting on insufficient data"
bool utest_mtpf()
{

    MTP mtp("MTP100_3.mtp");
#ifdef MLIP_MPI
    auto cfgs = MPI_LoadCfgs("configurations/reading_test.cfg",5);
#else
    auto cfgs = LoadCfgs("configurations/reading_test.cfg",5);
#endif
    Settings foo;
    LinearRegression lr(&mtp, foo);
    ErrorMonitor errmon(foo);

#ifndef MLIP_MPI
    lr.AddToSLAE(cfgs[2]);
#else
    if (2 % mpi.size == mpi.rank)
        lr.AddToSLAE(cfgs[2/mpi.size]);
#endif
    lr.Train();
#ifndef MLIP_MPI
    {
        Configuration cfg(cfgs[2]);
        mtp.CalcEFS(cfg);
        errmon.Compare(cfgs[2], cfg);
    }
#else
    if (2 % mpi.size == mpi.rank) {
        Configuration cfg(cfgs[2/mpi.size]);
        mtp.CalcEFS(cfg);
        errmon.Compare(cfgs[2/mpi.size], cfg);
    }

//     errmon.MPI_Synchronize();
#endif
    if (errmon.epa_cfg.delta > 10e-9 ||
        errmon.frc_cfg.max.delta > 10e-9 ||
        errmon.vir_cfg.delta > 10e-9)
        FAIL();
    //errmon.report(&cout);

    for (int i=0; i<cfgs.size(); i++)
    {
        auto ccfg(cfgs[i]);
        lr.AddToSLAE(ccfg);
        lr.Train();
        Configuration cfg = ccfg;
        mtp.CalcEFS(cfg);
        errmon.AddToCompare(ccfg, cfg);
        //errmon.report(&cout);
//         printf("\n%d: %e %e %e\n",mpi.rank,errmon.epa_all.max.delta,errmon.frc_all.max.delta,errmon.vir_all.max.delta  );
        if (errmon.epa_all.max.delta > 10e-7 ||
            errmon.frc_all.max.delta > 10e-6 ||
            errmon.vir_all.max.delta > 10e-4)
            FAIL();
    }
    return true;
}

// "Check radial basis derivatives with finite differences"
bool utest_rbd()
{
    double diffstep = 1e-4;
    int basis_function_count = 10;
    double r_min = 1.6;
    double r_cut = 5.2;
    double r = 3.5;

    Basis_Shapeev bas(r_min, r_cut, basis_function_count);
    bas.CalcDers(r);

    Basis_Shapeev bas_p(r_min, r_cut, basis_function_count);
    bas_p.CalcDers(r + diffstep);

    Basis_Shapeev bas_m(r_min, r_cut, basis_function_count);
    bas_m.CalcDers(r - diffstep);
    for (int i = 0; i < basis_function_count; i++) {
        double cdiff = (bas_p.vals[i] - bas_m.vals[i]) / (2 * diffstep);
        if (fabs((cdiff - bas.ders[i]) / cdiff) > 2e-4) {
            std::cout << fabs((cdiff - bas.ders[i]) / cdiff) << std::endl;
            FAIL()
        }
        //std::cout << cdiff << " " << bas.ders[i] << std::endl;
    }

    Basis_Chebyshev bas_ch(r_min, r_cut, basis_function_count);
    bas_ch.CalcDers(r);

    Basis_Chebyshev bas_ch_p(r_min, r_cut, basis_function_count);
    bas_ch_p.CalcDers(r + diffstep);

    Basis_Chebyshev bas_ch_m(r_min, r_cut, basis_function_count);
    bas_ch_m.CalcDers(r - diffstep);
    for (int i = 0; i < basis_function_count; i++) {
        double cdiff = (bas_ch_p.vals[i] - bas_ch_m.vals[i]) / (2 * diffstep);
        if (fabs((cdiff - bas_ch.ders[i]) / cdiff) > 2e-4) {
            std::cout << fabs((cdiff - bas_ch.ders[i]) / cdiff) << std::endl;
            FAIL()
        }
        //std::cout << cdiff << " " << bas_ch.ders[i] << std::endl;
    }

    Basis_Chebyshev_repuls bas_ch_repuls(r_min, r_cut, basis_function_count);
    bas_ch_repuls.CalcDers(r);

    Basis_Chebyshev_repuls bas_ch_repuls_p(r_min, r_cut, basis_function_count);
    bas_ch_repuls_p.CalcDers(r + diffstep);

    Basis_Chebyshev_repuls bas_ch_repuls_m(r_min, r_cut, basis_function_count);
    bas_ch_repuls_m.CalcDers(r - diffstep);
    for (int i = 0; i < basis_function_count; i++) {
        double cdiff = (bas_ch_repuls_p.vals[i] - bas_ch_repuls_m.vals[i]) / (2 * diffstep);
        if (fabs((cdiff - bas_ch_repuls.ders[i]) / cdiff) > 2e-4) {
            std::cout << fabs((cdiff - bas_ch_repuls.ders[i]) / cdiff) << std::endl;
            FAIL()
        }
        //std::cout << cdiff << " " << bas_ch_repuls.ders[i] << std::endl;
    }
    return true;
}

// "Check EFS Grads calculation by MTP"
bool utest_efsgrad()
{
    MTP mtp("MTP9.mtp");
    AnyLocalMLIP* p_any = &mtp;
    ifstream ifs("B.cfgs", ios::binary);
    Configuration cfg;
    cfg.Load(ifs);
    
    Array1D ene_grad;
    Array3D frc_grad;
    Array3D str_grad;
    Array1D ene_grad2;
    Array3D frc_grad2;
    Array3D str_grad2;

    mtp.CalcEFSGrads(cfg, ene_grad, frc_grad, str_grad);
    mtp.AnyLocalMLIP::CalcEFSGrads(cfg, ene_grad2, frc_grad2, str_grad2);

    if (ene_grad.size() != mtp.alpha_count ||
        ene_grad2.size() != mtp.alpha_count)
        FAIL();

    for (int i=0; i<ene_grad.size(); i++)
        if (ene_grad[i] != ene_grad2[i])
            FAIL();

    if (frc_grad.size1 != cfg.size() ||
        frc_grad.size2 != 3 ||
        frc_grad.size3 != mtp.alpha_count ||
        frc_grad2.size1 != cfg.size() ||
        frc_grad2.size2 != 3 ||
        frc_grad2.size3 != mtp.alpha_count)
        FAIL();

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

    mtp.AnyLocalMLIP::CalcForcesGrad(cfg, frc_grad2);

    for (int i=0; i<frc_grad.size1; i++)
        for (int j=0; j<frc_grad.size2; j++)
            for (int k=0; k<frc_grad.size3; k++)
                if (frc_grad(i, j, k) != frc_grad2(i, j, k))
                    FAIL();

    mtp.AnyLocalMLIP::CalcStressesGrads(cfg, str_grad2);

    for (int i=0; i<str_grad.size1; i++)
        for (int j=0; j<str_grad.size2; j++)
            for (int k=0; k<str_grad.size3; k++)
                if (str_grad(i, j, k) != str_grad2(i, j, k))
                    FAIL();

    mtp.AnyLocalMLIP::CalcEnergyGrad(cfg, ene_grad2);

    for (int i=0; i<ene_grad.size(); i++)
        if (ene_grad[i] != ene_grad2[i])
            FAIL();

    return true;
}

// LR weighting test
bool utest_lrw0()
{
    // vibration
    {
        MTP mtp0("MTP9prev.mtp");
        Settings setup;
        setup["force_weight"] = to_string(0.01);
        setup["stress_weight"] = to_string(0.1);
        LinearRegression lr0(&mtp0, setup);
        Configuration cfg;
        ErrorMonitor errmon(setup);

#ifndef MLIP_MPI
        auto train_set = LoadCfgs("TSB60_train.cfgs");
        auto valid_set = LoadCfgs("TSB40_valid.cfgs");
#else
        auto train_set = MPI_LoadCfgs("TSB60_train.cfgs");
        auto valid_set = MPI_LoadCfgs("TSB40_valid.cfgs");
#endif

        lr0.wgt_scale_power_energy = 1;
        lr0.wgt_scale_power_forces = 0;
        lr0.wgt_scale_power_stress = 1;
        lr0.Train(train_set);

        for (int i = 0; i < valid_set.size(); i++) {
            Configuration cfg1(valid_set[i]);
            mtp0.CalcEFS(cfg1);
            errmon.AddToCompare(valid_set[i], cfg1);
        }

//             printf(">> %e %e %e %e\n",errmon.epa_all.max.delta,
//                   errmon.frc_all.max.delta,
//                   errmon.vir_all.max.delta);
        if (fabs(errmon.epa_all.max.delta - 1.49252476444) > 1.0e-4 ||
            fabs(errmon.frc_all.max.delta - 12.7142916172 ) > 1.0e-4 ||
            fabs(errmon.vir_all.max.delta - 1239.3425052 ) > 1.0e-3)
//          cout << "1 - " << errmon.epa_all.max.delta << ' ' << errmon.frc_all.max.delta << ' ' << errmon.vir_all.max.delta << endl;
            FAIL();
    }
    // structures
    {
        MTP mtp0("MTP9prev.mtp");
        Settings setup;
        setup["force_weight"] = to_string(0.01);
        setup["stress_weight"] = to_string(0.1);
        LinearRegression lr0(&mtp0, setup);
        Configuration cfg;
        ErrorMonitor errmon(setup);

#ifndef MLIP_MPI
        auto train_set = LoadCfgs("TSB60_train.cfgs");
        auto valid_set = LoadCfgs("TSB40_valid.cfgs");
#else
        auto train_set = MPI_LoadCfgs("TSB60_train.cfgs");
        auto valid_set = MPI_LoadCfgs("TSB40_valid.cfgs");
#endif

        lr0.wgt_scale_power_energy = 2;
        lr0.wgt_scale_power_forces = 1;
        lr0.wgt_scale_power_stress = 2;
        lr0.Train(train_set);

        for (int i = 0; i < valid_set.size(); i++) {
            Configuration cfg1(valid_set[i]);
            mtp0.CalcEFS(cfg1);
            errmon.AddToCompare(valid_set[i], cfg1);
        }

//             printf(">> %e %e %e\n",errmon.epa_all.max.delta,
//                   errmon.frc_all.max.delta,
//                   errmon.vir_all.max.delta);
        if (fabs(errmon.epa_all.max.delta - 1.25020950612) > 1.0e-4 ||
            fabs(errmon.frc_all.max.delta - 12.633767615) > 1.0e-4 ||
            fabs(errmon.vir_all.max.delta - 1117.63601107) > 1.0e-3)
//          cout << "2 - " << errmon.epa_all.max.delta << ' ' << errmon.frc_all.max.delta << ' ' << errmon.vir_all.max.delta << endl;
            FAIL();
    }
    // molecules
    {
        MTP mtp0("MTP9prev.mtp");
        Settings setup;
        setup["force_weight"] = to_string(0.01);
        setup["stress_weight"] = to_string(0.1);
        LinearRegression lr0(&mtp0, setup);
        Configuration cfg;
        ErrorMonitor errmon(setup);

#ifndef MLIP_MPI
        auto train_set = LoadCfgs("TSB60_train.cfgs");
        auto valid_set = LoadCfgs("TSB40_valid.cfgs");
#else
        auto train_set = MPI_LoadCfgs("TSB60_train.cfgs");
        auto valid_set = MPI_LoadCfgs("TSB40_valid.cfgs");
#endif

        lr0.wgt_scale_power_energy = 0;
        lr0.wgt_scale_power_forces = 0;
        lr0.wgt_scale_power_stress = 0;
        lr0.Train(train_set);

        for (int i = 0; i < valid_set.size(); i++) {
            Configuration cfg1(valid_set[i]);
            mtp0.CalcEFS(cfg1);
            errmon.AddToCompare(valid_set[i], cfg1);
        }

//             printf(">>2: %e %e %e\n",errmon.epa_all.max.delta,
//                   errmon.frc_all.max.delta,
//                   errmon.vir_all.max.delta);
        if (fabs(errmon.epa_all.max.delta - 1.30491550962) > 1.0e-4 ||
            fabs(errmon.frc_all.max.delta - 13.7758870568) > 1.0e-3 ||
            fabs(errmon.vir_all.max.delta - 1182.34427787) > 1.0e-3)
//          cout << "3 - " << errmon.epa_all.max.delta << ' ' << errmon.frc_all.max.delta << ' ' << errmon.vir_all.max.delta << endl;
            FAIL();
    }
    return true;
}
