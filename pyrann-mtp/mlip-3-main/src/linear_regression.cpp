/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   This file contributors: Alexander Shapeev, Evgeny Podryabinkin
 */


#include "linear_regression.h"
#include <sstream>


using namespace std;


const char* LinearRegression::tagname = {"fit"};

LinearRegression::LinearRegression(AnyLocalMLIP* p_mlip, const Settings settings)
    : AnyTrainer(p_mlip, settings)
{
    int n = p_mlip->CoeffCount();        // Matrix size and basis functions count

    matrix1.resize(n, n);
    matrix2.resize(n, n);
    vector1.resize(n);
    vector2.resize(n);

    quad_opt_vec = vector1.data();
    quad_opt_matr = matrix1.data();

    quad_opt_eqn_count = 0;

    ClearSLAE();
}

LinearRegression::~LinearRegression()
{
    if (added_count != 0)
        Train();
}

void LinearRegression::ClearSLAE()
{
    const int n = p_mlip->CoeffCount();      // Matrix size 

    quad_opt_eqn_count = 0;
    quad_opt_scalar = 0.0;
    memset(quad_opt_vec, 0, n*sizeof(double));
    memset(quad_opt_matr, 0, n*n*sizeof(double));
}

void LinearRegression::CopySLAE()
{
    int n = p_mlip->CoeffCount();    // Matrix size 

    if (quad_opt_matr == matrix1.data())
    {
        memcpy(matrix2.data(), matrix1.data(), n*n*sizeof(double));
        memcpy(vector2.data(), vector1.data(), n*sizeof(double));
    }
    else if (quad_opt_matr == matrix2.data())
    {
        memcpy(matrix1.data(), matrix2.data(), n*n*sizeof(double));
        memcpy(vector1.data(), vector2.data(), n*sizeof(double));
    }
    else
        ERROR("Invalid internal state");
}

void LinearRegression::RestoreSLAE()
{
    if (quad_opt_matr == matrix1.data())
    {
        quad_opt_matr = matrix2.data();
        quad_opt_vec = vector2.data();
    }
    else if (quad_opt_matr == matrix2.data())
    {
        quad_opt_matr = matrix1.data();
        quad_opt_vec = vector1.data();
    }
    else
        ERROR("Invalid internal state");
}

void LinearRegression::AddToSLAE(Configuration& cfg, double weight)
{
    if (cfg.nbh_cutoff != p_mlip->CutOff())
        cfg.InitNbhs(p_mlip->CutOff());

    if (cfg.nbhs.size() == 0 || weight == 0)             // 
        return;

    int n = p_mlip->CoeffCount();        // Matrix size and basis functions count

    auto itr_f_begin = cfg.fitting_items.lower_bound(1);
    auto itr_s_begin = cfg.fitting_items.lower_bound(3*(int)cfg.nbhs.size()+1);
 
    bool forces_eq_to_fit_not_empty = (itr_f_begin == cfg.fitting_items.end()) ?
                                      false : (*itr_f_begin < 3*(int)cfg.nbhs.size()+1);
    bool stress_eq_to_fit_not_empty = (itr_s_begin == cfg.fitting_items.end()) ?
                                      false : (*itr_s_begin < 1+3*(int)cfg.nbhs.size()+9);
    bool fit_to_energy = (wgt_eqtn_energy > 0.0 &&
                          cfg.has_energy() &&
                          (selection_factor > 0.0 || cfg.fitting_items.count(0) > 0));
    bool fit_to_forces = (wgt_eqtn_forces > 0.0 &&
                          cfg.has_forces() &&
                          (selection_factor > 0.0 || forces_eq_to_fit_not_empty));
    bool fit_to_stress = (wgt_eqtn_stress > 0.0 &&
                          cfg.has_stresses() &&
                          (selection_factor > 0.0 || stress_eq_to_fit_not_empty));

    if (fit_to_forces || fit_to_stress)
        p_mlip->CalcEFSGrads(cfg, energy_cmpnts, forces_cmpnts, stress_cmpnts);
    else if (fit_to_energy)
        p_mlip->CalcEnergyGrad(cfg, energy_cmpnts);

    if (fit_to_energy)
    {
        double wgt = weight * wgt_energy(cfg);

        for (int i = 0; i < n; i++)
            for (int j = i; j < n; j++)
                quad_opt_matr[i*n + j] += wgt * energy_cmpnts[i] * energy_cmpnts[j];

        for (int i = 0; i < n; i++)
            quad_opt_vec[i] += wgt * energy_cmpnts[i] * cfg.energy;

        quad_opt_scalar += wgt * cfg.energy * cfg.energy;

        quad_opt_eqn_count += (int)weight;
    }

    if (fit_to_forces)
    {
        for (Neighborhood& nbh : cfg.nbhs)
        {
            int ind=nbh.my_ind; 

            for (int a=0; a<3; a++)
            {
                double wgt = weight * wgt_forces(cfg, ind, a);

                for (int i=0; i<n; i++)
                {
                    for (int j=i; j<n; j++)
                        quad_opt_matr[i*n + j] +=
                        wgt * forces_cmpnts(ind, a, i) * forces_cmpnts(ind, a, j);

                    quad_opt_vec[i] += wgt * forces_cmpnts(ind, a, i) * cfg.force(ind, a);
                }

                quad_opt_scalar += wgt * cfg.force(ind, a) * cfg.force(ind, a);

                quad_opt_eqn_count += (int)weight;
            }
        }
    }

    if (fit_to_stress)
    {
        for (int a=0; a<3; a++)
            for (int b=0; b<3; b++)
            {
                double wgt = weight * wgt_stress(cfg, a, b);

                for (int i=0; i<n; i++)
                {
                    for (int j=i; j<n; j++)
                        quad_opt_matr[i*n + j] +=
                        wgt * stress_cmpnts(a, b, i) * stress_cmpnts(a, b, j);

                    quad_opt_vec[i] += wgt * stress_cmpnts(a, b, i) * cfg.stresses[a][b];
                }

                quad_opt_scalar += wgt * cfg.stresses[a][b] * cfg.stresses[a][b];
            }

        quad_opt_eqn_count += 6 * (int)weight;
    }

    added_count++;
}

void LinearRegression::RemoveFromSLAE(Configuration & cfg)
{
    AddToSLAE(cfg, -1);
}

void LinearRegression::SymmetrizeSLAE()
{
    int n = p_mlip->CoeffCount();        // Matrix size

    for (int i = 0; i < n; i++) 
        for (int j = i + 1; j < n; j++) 
            quad_opt_matr[j*n + i] = quad_opt_matr[i*n + j];
}

void LinearRegression::SolveSLAE()
{
    const double reg_param_soft = 1e-13;
    const double reg_param_hard = 1e-13;

    const int n = p_mlip->CoeffCount();      // Matrix size
    double* regress_coeffs = p_mlip->Coeff();

    // Regularization (soft)
    for (int i=0; i<n; i++)
        quad_opt_matr[i*n + i] += reg_param_soft*quad_opt_matr[i*n + i];

    // Gaussian Elimination with leading row swaps
    for (int i=0; i<n-1; i++) 
    {
        double& m_ii = quad_opt_matr[i*n + i];

        // find row j0 with maximal in modulus element max_el in i-th column
        int j0 = i;
        double max_el = m_ii;
        for (int j=i+1; j<n; j++)
        {
            double foo = fabs(quad_opt_matr[j*n + i]);
            if (foo > max_el)
            {
                max_el = foo;
                j0 = j;
            }
        }
        if (j0 != i)// swap rows
        {
            for (int k=0; k<n; k++)
                swap(quad_opt_matr[i*n + k], quad_opt_matr[j0*n + k]);
            swap(quad_opt_vec[i], quad_opt_vec[j0]);
        }
        // strong regularization
        if (fabs(m_ii) < reg_param_hard)
            m_ii = reg_param_hard;  
        
        // Gsussian elimination
        for (int j=i+1; j<n; j++) {
            double ratio = quad_opt_matr[j*n + i] / m_ii;
            for (int k=i; k<n; k++) 
                quad_opt_matr[j*n + k] -= ratio * quad_opt_matr[i*n + k];
            quad_opt_vec[j] -= ratio * quad_opt_vec[i];
        }
    }
    // Calculating regression coefficients
    regress_coeffs[n - 1] = quad_opt_vec[n - 1] / quad_opt_matr[(n - 1)*n + (n - 1)];
    for (int i=n-2; i>=0; i--) {
        double temp = quad_opt_vec[i];
        for (int j=i+1; j<n; j++) {
            temp -= (quad_opt_matr[i*n + j] * regress_coeffs[j]);
        }
        regress_coeffs[i] = temp / quad_opt_matr[i*n + i];
    }
}

void LinearRegression::Train(std::vector<Configuration>& train_set)
{
    //cout << train_set.size() << endl;
    std::stringstream logstrm1;
    logstrm1 << "Fitting: training over " << train_set.size() << " configurations" << endl;
    MLP_LOG(tagname,logstrm1.str());

    ClearSLAE();

    int cntr = 1;
    for (Configuration& cfg : train_set)
    {
        AddToSLAE(cfg);
        logstrm1 << "Fitting: Adding to SLAE cfg#" << cntr++ << endl;
        MLP_LOG(tagname,logstrm1.str()); logstrm1.str("");
    }

#ifdef MLIP_MPI
    int quad_opt_eqn_count_ = 0;
    MPI_Allreduce(&quad_opt_eqn_count, &quad_opt_eqn_count_, 1, MPI_INT, MPI_SUM, mpi.comm);
    quad_opt_eqn_count = quad_opt_eqn_count_;
#endif

    logstrm1 << "Fitting: training over " << quad_opt_eqn_count << " equations" << endl;
    MLP_LOG(tagname,logstrm1.str()); logstrm1.clear();
    Train();
    logstrm1 << "Fitting: training complete" << endl; 
    MLP_LOG(tagname,logstrm1.str());

    if (mpi.rank == 0)
        if (!mlip_fitted_fnm.empty())
            p_mlip->Save(mlip_fitted_fnm);
}

void LinearRegression::Train()
{
    SymmetrizeSLAE();
    CopySLAE();

    //if (quad_opt_eqn_count < p_mlip->CoeffCount())
    //  Warning("Fitting: Not enough equations for training. Eq. count = " + 
    //          std::to_string(quad_opt_eqn_count) + " , whereas number of MTP parameters = " +
    //          std::to_string(p_mlip->CoeffCount()));

#ifdef MLIP_MPI     // here we sum SLAE from all processors 
    // quad_opt_scalar summation
    double buff_opt_scalar;
    MPI_Allreduce(&quad_opt_scalar, &buff_opt_scalar, 1, MPI_DOUBLE, MPI_SUM, mpi.comm);
    quad_opt_scalar = buff_opt_scalar;

    // taking pointers to buffer arrays for sending (receiving in quad_opt_... arrays)
    int n = p_mlip->CoeffCount();        // Matrix size 
    double* buff_opt_matr = nullptr;
    double* buff_opt_vec = nullptr;
    if (quad_opt_matr == matrix1.data())
    {
        buff_opt_matr = matrix2.data();
        buff_opt_vec = vector2.data();
    }
    else
    {
        buff_opt_matr = matrix1.data();
        buff_opt_vec = vector1.data();
    }

    // vec summation
    MPI_Allreduce(buff_opt_vec, quad_opt_vec, n, MPI_DOUBLE, MPI_SUM, mpi.comm);

    // matrix summation
    MPI_Allreduce(buff_opt_matr, quad_opt_matr, n*n, MPI_DOUBLE, MPI_SUM, mpi.comm);
#endif

    SolveSLAE();
    RestoreSLAE();

    added_count = 0;
}
