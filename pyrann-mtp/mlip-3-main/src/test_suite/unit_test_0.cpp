#include "../common/blas.h"
#include "../common/comm.h"
#include "../wrapper.h"
#include "../mtpr.h"

using namespace std;


#define FAIL(...) \
{std::string str("" __VA_ARGS__); \
std::cout << " failed!" << std::endl; \
if (!str.empty()) std::cout << std::endl << "Error message: " << str << std::endl; \
return false;}

// "configurations reading"
bool utest_cr()
{
#ifdef MLIP_MPI
    if (mpi.rank == 0) 
#endif
    {
        int count = 0;
        int status = 0;

        // read the database
        std::ifstream ifs("configurations/reading_test.cfg", ios::binary);
        Configuration cfg;
        for (count = 0; (status = cfg.Load(ifs)); count++);
        if ((status != false) || (count != 7)) 
            FAIL();
    }

    return true;

}

// "Binary and normal configurations reading"
bool utest_bnr()
{
    if (mpi.rank == 0) 
    {
        int count = 0;
        int status = 0;

        Configuration cfg;
        vector<Configuration> set_bin;
        vector<Configuration> set;

        std::ifstream ifs("configurations/reading_test.cfg", ios::binary);
        std::ofstream ofs("temp/configurations_test1.bin.cfgs", ios::binary);

        for (int i = 0; cfg.Load(ifs); i++)
            cfg.SaveBin(ofs);

        ifs.close();
        ofs.close();

        // read the database
        std::ifstream ifs1("temp/configurations_test1.bin.cfgs", ios::binary);
        std::ifstream ifs2("configurations/reading_test.cfg", ios::binary);
        std::ifstream ifs3("configurations/binary_sample.cfg", ios::binary);

        for (; (status = cfg.Load(ifs1)); count++) {
            set_bin.push_back(cfg);
        }
        for (; (status = cfg.Load(ifs2)); count++) {
            set.push_back(cfg);
        }
        for (; (status = cfg.Load(ifs3)); count++);

        bool flag = true;
        for (int i = 0; i < set.size(); i++) {
            if (set[i] != set_bin[i]) {
                flag = false;
                break;
            }
        }

        if ((status != false) || (count != 15) || !flag) FAIL()

            ifs1.close();
        ifs2.close();
        ifs3.close();
    }

    return true;
} 


// "Loading configuration database to a vector") 
bool utest_ldv()
{
    if (mpi.rank == 0) 
    {
        vector<Configuration> cfgs(LoadCfgs("configurations/reading_test.cfg"));
        vector<Configuration> cfgs2 = LoadCfgs("configurations/reading_test.cfg");
        Configuration cfg;
        ifstream ifs("configurations/reading_test.cfg", ios::binary);
        for (int i = 0; cfg.Load(ifs); i++)
            if (cfg != cfgs[i] || cfg != cfgs2[i])
                FAIL();
    }

    return true;

}

// "Binary configurations file format and old format r/w"
bool utest_rw()
{
    if (mpi.rank == 0) 
    {
        std::ifstream ifs("configurations/reading_test.cfg", ios::binary);
        std::ofstream ofs_bin("temp/test1.bin.cfgs", std::ios::binary);
        Configuration cfg;

        for (int count = 0; cfg.Load(ifs); count++)
            cfg.SaveBin(ofs_bin);

        ifs.close();
        ofs_bin.close();

        ifs.open("configurations/reading_test.cfg", ios::binary);
        std::ifstream ifs_bin("temp/test1.bin.cfgs", std::ios::binary);
        Configuration cfgb;

        while (cfgb.Load(ifs_bin)) {
            cfg.Load(ifs);
            if (cfg.size() != cfgb.size())
                FAIL();

            if (!(cfg.lattice == cfgb.lattice))
                FAIL();

            for (int i = 0; i < cfg.size(); i++)
                if (!(cfg.pos(i) == cfgb.pos(i)))
                    FAIL();

            if ((cfg.has_forces() != cfgb.has_forces()) ||
                (cfg.has_energy() != cfgb.has_energy()) ||
                (cfg.has_stresses() != cfgb.has_stresses()) ||
                (cfg.has_site_energies() != cfgb.has_site_energies()))
                FAIL();

            if (cfg.has_energy())
                if (cfg.energy != cfgb.energy)
                    FAIL();

            if (cfg.has_forces())
                for (int i = 0; i < cfg.size(); i++)
                    if (!(cfg.force(i) == cfgb.force(i)))
                        FAIL();

            if (cfg.has_stresses())
                if (!(cfg.stresses == cfgb.stresses))
                    FAIL();

            if (cfg.has_site_energies())
                for (int i = 0; i < cfg.size(); i++)
                    if (cfg.site_energy(i) != cfgb.site_energy(i))
                        FAIL();
        }

        ifs.close();
        ifs_bin.close();
    }

    return true;
} 

// "New format configurations reading/writing test" 
bool utest_cfgrw()
{
    if (mpi.rank == 0) 
    {
        int count = 0;
        int status = 0;

        // read the database
        std::ifstream ifs("configurations/reading_test.cfg", ios::binary);
        Configuration cfg;
        vector<Configuration> cfgvec;
        ofstream ofs("temp/cfg_rw_test.cfgs", ios::binary);
        for (count = 0; (status = cfg.Load(ifs)); count++) {
            cfgvec.push_back(cfg);
            cfg.Save(ofs, Configuration::SAVE_NO_LOSS);
        }
        if ((status != false) || (count != 7)) FAIL();
        ifs.close();
        ofs.close();

        ofs.open("temp/cfg_rw_test2.cfgs", ios::binary);
        ifs.open("temp/cfg_rw_test.cfgs", ios::binary);
        for (count = 0; (status = cfg.Load(ifs)); count++) {
            if (cfg != cfgvec[count])
                FAIL();
            cfg.Save(ofs);
        }
        if ((status != false) || (count != 7)) FAIL();
        ofs.close();
        ifs.close();
    }

    return true;
} 

// "Configuration with ghost atoms reading/writing test" 
bool utest_cfgrwgst()
{
    if (mpi.rank == 0)
    {
        int count = 0;
        int status = 0;

        ofstream ofs("temp/cfg_ghost.cfg", ios::binary);

        Configuration cfg(CrystalSystemType::BCC, 1.0, 2);
        cfg.has_site_energies(true);
        cfg.has_forces(true);
        cfg.has_charges(true);
        cfg.has_magmom(true);
        cfg.Save(ofs);
        for (int i=2; i<cfg.size(); i++)
            cfg.ghost_inds.insert(i);
        cfg.lattice *= 0.5;
        cfg.Save(ofs, Configuration::SAVE_GHOST_ATOMS);
        cfg.Save(ofs);

        ofs.close();
        ifstream ifs("temp/cfg_ghost.cfg", ios::binary);
        auto cfgs = LoadCfgs("temp/cfg_ghost.cfg");

        if (cfgs.size()!=3 || 
            cfgs[0].size()!=cfgs[1].size() || 
            cfgs[2].size()!=2 ||
            cfgs[1].ghost_inds.size() != 14 ||
            !cfgs[2].ghost_inds.empty())
            FAIL();
        if (cfgs[0].pos(0) != cfgs[1].pos(0) || cfgs[0].pos(0) != cfgs[2].pos(0) ||
            cfgs[0].pos(1) != cfgs[1].pos(1) || cfgs[0].pos(1) != cfgs[2].pos(1) ||
            cfgs[0].pos(2) != cfgs[1].pos(2) ||
            cfgs[0].pos(3) != cfgs[1].pos(3) ||
            cfgs[0].pos(4) != cfgs[1].pos(4) ||
            cfgs[0].pos(5) != cfgs[1].pos(5) ||
            cfgs[0].pos(6) != cfgs[1].pos(6) ||
            cfgs[0].pos(7) != cfgs[1].pos(7) ||
            cfgs[0].pos(8) != cfgs[1].pos(8) ||
            cfgs[0].pos(9) != cfgs[1].pos(9) ||
            cfgs[0].pos(10) != cfgs[1].pos(10) ||
            cfgs[0].pos(11) != cfgs[1].pos(11) ||
            cfgs[0].pos(12) != cfgs[1].pos(12) ||
            cfgs[0].pos(13) != cfgs[1].pos(13) ||
            cfgs[0].pos(14) != cfgs[1].pos(14) ||
            cfgs[0].pos(15) != cfgs[1].pos(15))
            FAIL();
        if (cfgs[1].ghost_inds != cfg.ghost_inds)
            FAIL();
    }

    return true;
}

//"Identical configurations reading test" 
bool utest_icrt()
{
    if (mpi.rank == 0) 
    {
        int count = 0;
        bool status = 0;

        for (int i = 1; i <= 3; i++) {
            // read the database
            std::ifstream ifs("configurations/identical/all_identical" + to_string(i) + ".cfg", std::ios::binary);
            Configuration cfg_init, cfg;
            status = cfg_init.Load(ifs);
            if (status != true) FAIL();
            for (count = 1; (status = cfg.Load(ifs)); count++)
                if (cfg != cfg_init) FAIL();
            ifs.close();
        }
    }

    return true;
} 

//"Testing Configuration periodic extension"
bool utest_pe()
{
    if (mpi.rank == 0) 
    {
        std::ifstream ifs("configurations/period_ext_test.cfg", std::ios::binary);
        MTP mtp("mlips/fitted.mtp");
        for (int i = 0; i < 3; i++) {
            Configuration cfg1, cfg2;
            cfg1.Load(ifs);
            cfg2.Load(ifs);
            mtp.CalcEFS(cfg1);
            mtp.CalcEFS(cfg2);
            if (cfg1.stresses != cfg2.stresses) FAIL();
        }
    }

    return true;
} 

// "Testing reading configuration with fractional coordinates"
bool utest_frac()
{
    if (mpi.rank == 0) 
    {
        // read the database
        std::ifstream ifs("configurations/reading_test.cfg", std::ios::binary);
        Configuration cfg1, cfg2;
        cfg1.Load(ifs);
        ifs.close();
        ofstream ofs("temp/frac_coord.cfg");
        cfg1.Save(ofs, Configuration::SAVE_DIRECT_COORDS | Configuration::SAVE_NO_LOSS);
        ofs.close();
        ifs.open("temp/frac_coord.cfg", std::ios::binary);
        cfg2.Load(ifs);
        for (int i=0; i<cfg1.size(); i++)
            if ((cfg1.pos(i) - cfg2.pos(i)).Norm() > 1.0e-13)
                FAIL();
    }

    return true;
} 

// "Testing Configuration::features"
bool utest_ftrs()
{
    if (mpi.rank == 0) 
    {
        std::ofstream ofs;
        Configuration cfg2;

        std::ifstream ifs("configurations/reading_test.cfg", ios::binary);
        Configuration cfg;
        if (!cfg.Load(ifs)) FAIL()
            ifs.close();
        cfg.features["test_feature1"] = "ABC";
        cfg.features["test feature2"] = "DEF";
        cfg.features["test_feature3"] = "line with spaces";
        cfg.features["test_feature4"] = "line with spaces and\nnewline";
        cfg.features["test feature5"] = "line with \" \" - space";
        cfg.features["test_feature5"] = "line with spaces and newline at end\n";
        cfg.features["\n"] = "\n   mo\re \trash i\n fea\tu\res \n";
        //  cfg.features[" "] = "\n";
        //  cfg.features[" "] = "";

        ofs.open("temp/temp.cfgs", ios::binary);
        cfg.Save(ofs);
        ofs.close();
        ifs.open("temp/temp.cfgs", ios::binary);
        if (!cfg2.Load(ifs)) FAIL();
        ifs.close();
        if (cfg.features != cfg2.features) FAIL();

        ofs.open("temp/temp.cfgs", std::ios::binary);
        cfg.SaveBin(ofs);
        ofs.close();
        ifs.open("temp/temp.cfgs", std::ios::binary);
        if (!cfg2.Load(ifs)) FAIL();
        ifs.close();
        if (cfg.features != cfg2.features) FAIL();
    }

    return true;
} 

// "VASP OUTCAR iterations count"
bool utest_vaspr()
{
    if (mpi.rank == 0)
    {
        Configuration cfg;
        cfg.LoadFromOUTCAR("configurations/VASP/OUTCAR_converged");

        if (cfg.features["EFS_by"] != "VASP")
            FAIL();

        cfg.LoadFromOUTCAR("configurations/VASP/OUTCAR_not_converged");

        if (cfg.features["EFS_by"] != "VASP_not_converged")
            FAIL();
    }

    return true;
} 

// "BLAS_test"
bool utest_blas()
{
    if (mpi.rank == 0) 
    {
        double A[6] ={ 1.0,2.0,1.0,-3.0,4.0,-1.0 };
        double B[6] ={ 1.0,2.0,1.0,-3.0,4.0,-1.0 };
        double C[9] ={ .5,.5,.5,.5,.5,.5,.5,.5,.5 };
        BLAS_M3_equal_M1_mult_M2T(A, B, C, 3, 3, 2);

        if (fabs(C[0] - 5) > 1e-6 ||
            fabs(C[1] - -5) > 1e-6 ||
            fabs(C[2] -  2) > 1e-6 ||
            fabs(C[3] - -5) > 1e-6 ||
            fabs(C[4] - 10) > 1e-6 ||
            fabs(C[5] - 7) > 1e-6 ||
            fabs(C[6] - 2) > 1e-6 ||
            fabs(C[7] - 7) > 1e-6 ||
            fabs(C[8] - 17) > 1e-6)
            FAIL();
    }

    if (mpi.rank == 0)
    {
        Array2D A(3,4);
        A(0, 0)=1; A(0, 1)=2; A(0, 2)=3; A(0, 3)=4;
        A(1, 0)=5; A(1, 1)=6; A(1, 2)=7; A(1, 3)=8;
        A(2, 0)=9; A(2, 1)=-1;A(2, 2)=-2;A(2, 3)=-3;
        Array2D B(5,4);
        B(0, 0)=9; B(0, 1)=8; B(0, 2)=7; B(0, 3)=6;
        B(1, 0)=5; B(1, 1)=4; B(1, 2)=3; B(1, 3)=2;
        B(2, 0)=1; B(2, 1)=0; B(2, 2)=-1; B(2, 3)=-2;
        B(3, 0)=-3;B(3, 1)=-4; B(3, 2)=-5; B(3, 3)=-6;
        B(4, 0)=-7;B(4, 1)=-8; B(4, 2)=-9; B(4, 3)=10;
        Array2D C(3,5);
        C.set(0);
        BLAS_M3_equal_M1_mult_M2T(&A(0,0), &B(0,0), &C(0,0), 3, 5, 4);

        if (fabs(C(0, 0) - 70) > 1e-6 ||
            fabs(C(0, 1) - 30) > 1e-6 ||
            fabs(C(0, 2) - -10) > 1e-6 ||
            fabs(C(0, 3) - -50) > 1e-6 ||
            fabs(C(0, 4) - -10) > 1e-6 ||
            fabs(C(1, 0) - 190) > 1e-6 ||
            fabs(C(1, 1) - 86) > 1e-6 ||
            fabs(C(1, 2) - -18) > 1e-6 ||
            fabs(C(1, 3) - -122) > 1e-6 ||
            fabs(C(1, 4) - -66) > 1e-6 ||
            fabs(C(2, 0) - 41) > 1e-6 ||
            fabs(C(2, 1) - 29) > 1e-6 ||
            fabs(C(2, 2) - 17) > 1e-6 ||
            fabs(C(2, 3) - 5) > 1e-6 ||
            fabs(C(2, 4) - -67) > 1e-6)  
            FAIL();
    }

    return true;
} 

// "Errorneus configurations reading test"
bool utest_errr()
{
    if (mpi.rank == 0) 
    {
        // read the database
        for (int i=1; i<=9; i++) {
            std::ifstream ifs((std::string)"configurations/errorneous/" + to_string(i) + ".cfg");
            if (!ifs.is_open()) FAIL();
            Configuration cfg;
            bool is_valid = true;
            try { cfg.Load(ifs); }
            catch (const MlipException&) {
                is_valid = false;
            }
            if (is_valid) FAIL();
            ifs.close();
        }
    }

    return true;
} 
/*
// "VASP loading" 
bool utest_vaspl()
{
{
#ifdef GEN_TESTS
Configuration cfg;
std::ofstream ofs;

cfg.LoadFromOUTCAR("configurations/VASP/OUTCAR_usual");
ofs.open("configurations/VASP/cfg_usual");
cfg.Save(ofs);
ofs.close();

cfg.LoadFromOUTCAR("configurations/VASP/OUTCAR_no_stresses");
ofs.open("configurations/VASP/cfg_no_stresses");
cfg.Save(ofs);
ofs.close();
#endif
}
Configuration cfg1, cfg2;
std::ifstream ifs;

cfg1.LoadFromOUTCAR("configurations/VASP/OUTCAR_usual");
ifs.open("configurations/VASP/cfg_usual"); cfg2.Load(ifs); ifs.close();
if (cfg1 != cfg2) FAIL();

cfg1.LoadFromOUTCAR("configurations/VASP/OUTCAR_no_stresses");
ifs.open("configurations/VASP/cfg_no_stresses"); cfg2.Load(ifs); ifs.close();
if (cfg1 != cfg2) FAIL();

std::vector<Configuration> db;
if (!Configuration::LoadDynamicsFromOUTCAR(db, "configurations/VASP/OUTCAR_relax"))
FAIL();

if (!cfg1.LoadLastFromOUTCAR("configurations/VASP/OUTCAR_relax"))
FAIL("Valid OUTCAR dynamics read, but returned false");
if (cfg1 != db.back())
FAIL();

if (Configuration::LoadDynamicsFromOUTCAR(db, "configurations/VASP/OUTCAR_broken"))
FAIL("Broken OUTCAR read, but returned true");

reurn true;
} 
*/
// "Construction of neighborhoods for non-periodic (i.e. with ghost atoms) configuration" 
bool utest_gstnbh()
{
    if (mpi.rank == 0)
    {
        int count = 0;
        int status = 0;

        Configuration cfg(CrystalSystemType::BCC, 1.0, 2);
        for (int i=2; i<cfg.size(); i++)
            cfg.ghost_inds.insert(i);
        cfg.lattice *= 0.5;

        cfg.InitNbhs(sqrt(3)/2 + 0.000001);

        if (cfg.nbhs.size() != cfg.size()-(int)cfg.ghost_inds.size())
            FAIL();
        Configuration cfg_check(CrystalSystemType::BCC, 1.0, 2);
        cfg_check.InitNbhs(sqrt(3)/2 + 0.000001);
        if (cfg.nbhs[1].count != cfg_check.nbhs[1].count ||
            cfg.nbhs[1].vecs != cfg_check.nbhs[1].vecs ||
            cfg.nbhs[1].types != cfg_check.nbhs[1].types ||
            cfg.nbhs[1].my_type != cfg_check.nbhs[1].my_type ||
            cfg.nbhs[1].my_ind != 1)
            FAIL();
    }

    return true;
}
//Check MTP and magnetic MTPR Load and Save functions
bool utest_mtp()
{
    if (mpi.rank == 0) 
    {
        try {
            MTP mtp102("mlips/fitted.mtp");
            mtp102.Save("temp/MTPv1.1.0test.mtp");
            MTP mtp103("temp/MTPv1.1.0test.mtp");

            Configuration cfg1;
            Configuration cfg2;
            ifstream ifs("configurations/reading_test.cfg", ios::binary);
            cfg1.Load(ifs);
            cfg2 = cfg1;
            mtp103.CalcEFS(cfg1);
            mtp102.CalcEFS(cfg2);

            if (cfg1.energy != cfg2.energy)
                FAIL();
            for (int i=0; i<cfg1.size(); i++)
                if (cfg1.force(i) != cfg2.force(i))
                    FAIL();
            if (cfg1.stresses[0][0] != cfg2.stresses[0][0] ||
                cfg1.stresses[1][1] != cfg2.stresses[1][1] ||
                cfg1.stresses[2][2] != cfg2.stresses[2][2] ||
                cfg1.stresses[0][1] != cfg2.stresses[0][1] ||
                cfg1.stresses[0][2] != cfg2.stresses[0][2] ||
                cfg1.stresses[1][2] != cfg2.stresses[1][2])
                FAIL();

            MLMTPR mtp104("fe_mag.mtp");

            std::random_device random_device;
            std::default_random_engine eng(random_device());
            std::uniform_real_distribution<double> distr(-1, 1);

            for (int i = 0; i < mtp104.CoeffCount(); i++) 
                mtp104.Coeff()[i] = 0.01*distr(eng);
            mtp104.inited = true;

            mtp104.Save("temp/MTP_mag_test.mtp");
            MLMTPR mtp105("temp/MTP_mag_test.mtp");

            ifstream ifs1("fe_magmom.cfg", ios::binary);
            cfg1.Load(ifs1);
            cfg2 = cfg1;
            mtp105.CalcEFS(cfg1);
            mtp104.CalcEFS(cfg2);

            double delta = 1e-6;

            if (fabs(cfg1.energy - cfg2.energy) > delta)
                FAIL();
            for (int i=0; i<cfg1.size(); i++) {
                for (int a = 0; a < 3; a++) {
                    if (fabs(cfg1.force(i)[a] - cfg2.force(i)[a]) > delta) {
                        FAIL();
                    }
                }
            }
            if (fabs(cfg1.stresses[0][0] - cfg2.stresses[0][0]) > delta ||
                fabs(cfg1.stresses[1][1] - cfg2.stresses[1][1]) > delta ||
                fabs(cfg1.stresses[2][2] - cfg2.stresses[2][2]) > delta ||
                fabs(cfg1.stresses[0][1] - cfg2.stresses[0][1]) > delta ||
                fabs(cfg1.stresses[0][2] - cfg2.stresses[0][2]) > delta ||
                fabs(cfg1.stresses[1][2] - cfg2.stresses[1][2]) > delta) 
                FAIL();

       
        }
        catch (...)
            FAIL();
    }

    return true;
} 

// "MTP file reading 1"
bool utest_mtpfr1()
{
    if (mpi.rank == 0) 
        MTP basis("mlips/lightweight.mtp");

    return true;
} 

// "MTP file reading 2"
bool utest_mtpfr2()
{
    if (mpi.rank == 0) 
    {
        MTP mtp("mlips/lightweight.mtp");
        for (double i : mtp.regress_coeffs)
            if (i != 0.0)
                FAIL();
    }

    return true;
} 

// "MTP file reading 3" 
bool utest_mtpfr3()
{
    if (mpi.rank == 0) 
    {
        MTP mtp("mlips/fitted.mtp");
        double res=0;
        for (double i : mtp.regress_coeffs)
            res += i;
        if (res == 0.0)
            FAIL();
    }

    return true;
} 

//PAR_TEST("MTP learning") 
bool utest_mlpl()
{
    MTP mtp("mlips/unfitted.mtp");
    Settings setup;
    setup["energy_weight"] = to_string(500);
    setup["force_weight"] = to_string(1);
    setup["stress_weight"] = to_string(10);
    LinearRegression regr(&mtp, setup);
    Configuration cfg;
    vector<Configuration> training_set;

#ifndef MLIP_MPI
    training_set =  LoadCfgs("configurations/reading_test.cfg");
#else
    training_set =  MPI_LoadCfgs("configurations/reading_test.cfg");
#endif

    regr.Train(training_set);

#ifdef GEN_TESTS
    if (mpi.rank==0)
        mtp.Save("mlips/fitted.mtp");
#endif
    if (mpi.rank==0)
    {
        // the MTP that is already trained
        MTP mtp_ready("mlips/fitted.mtp");
        std::ifstream ifs("configurations/reading_test.cfg", ios::binary);
        for (int count=0; cfg.Load(ifs); count++) {
            Configuration cfg2(cfg);

            // check computing energy with trained vs stored coefficients
            mtp.CalcEFS(cfg);
            mtp_ready.CalcEFS(cfg2);
            if (abs(cfg2.energy - cfg.energy) > 1e-6) {
                std::cerr << "training produced wrong coefficients (error " << abs(cfg2.energy - cfg.energy) << ")";
                FAIL()
            }
            //std::cout << cfg2.energy << " " << cfg.energy << std::endl;

            // check that CalcEFS and calc_E give the same energy
            mtp_ready.CalcEFS(cfg);
            mtp_ready.CalcE(cfg2);
            if (abs(cfg2.energy - cfg.energy) > 1e-6) {
                std::cerr << "CalcEFS differs from calc_E (error " << abs(cfg2.energy - cfg.energy) << ")";
                FAIL()
            }
            // check the optimized (backpropagation) procedure
            mtp.CalcEFSDebug(cfg);
            mtp.CalcEFS(cfg2);
            for (int i=0; i<cfg.size(); i++)
                if (distance(cfg2.force(i), cfg.force(i)) > 1e-6) {
                    std::cerr << "CalcEFS differs from CalcEFS_opt (force error " << distance(cfg2.force(i), cfg.force(i)) << ")";
                    FAIL()
                }

            if (norm(cfg2.stresses - cfg.stresses) > 1e-6) {
                std::cerr << "CalcEFS differs from CalcEFS_opt (stress error " << norm(cfg2.stresses - cfg.stresses) << ")";
                FAIL()
            }

        }
        ifs.close();
    }

    MPI_Barrier(mpi.comm);

    return true;
}

/*PAR_TEST("parallel trainer test")
{
{
MTP mtp("paralel_fitting_test.mtp");
mtp.Save("temp/paralel_fitting_test.mtp");
}
MTP mtp("temp/paralel_fitting_test.mtp");
LinearRegression regr(&mtp, 500, 1, 10);
auto cfgs = LoadCfgs("configurations/reading_test.cfg");
regr.Train(cfgs);

train("temp/paralel_fitting_test.mtp", "configurations/reading_test.cfg", 500,1,10,0, "", nullptr);

for (auto& cfg : cfgs)
mtp.CalcEFS(cfg);

MTP mtpp("temp/paralel_fitting_test.mtp");

ErrorMonitor errmon;
for (auto& cfg : cfgs)
{
Configuration cfgg(cfg);
errmon.compare(cfg, cfgg);
if (errmon.ene_cfg.delta > 1.0e-13 ||
errmon.frc_cfg.max.delta > 1.0e-12 ||
errmon.str_all.max.delta > 1.0e-13)
FAIL( + to_string(errmon.ene_cfg.delta) + " " +
to_string(errmon.frc_cfg.max.delta) + " " +
to_string(errmon.str_all.max.delta));
}

} END_TEST;*/
