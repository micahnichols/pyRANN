/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#ifndef MLIP_MTPR_TRAINER_H
#define MLIP_MTPR_TRAINER_H

#include "common/stdafx.h"
#include "non_linear_regression.h"
#include "mtpr.h"
#include "common/bfgs.h"

class MTPR_trainer : public NonLinearRegression
{
private:
    //LINEAR REGRESSION
    std::vector<double> lin_matrix; // matrix of the quadratic optimization problem A*A^T = b*A^T
    std::vector<double> lin_vector; // vector of the quadratic optimization problem A*A^T = b*A^T
    double lin_scalar;         // scalar of the quadratic optimization problem
    int lin_eqn_count;         // number of equation in the quadratic optimization problem
    std::vector<double> lin_matrix_mpi; // temporary container to add "lin_matrix" from different cores
    std::vector<double> lin_vector_mpi; // temporary container to add "lin_vector" from different cores	
	
    MLMTPR* p_mlmtpr = nullptr;                         // Pointer to the instance of MTPR potential to be trained
    BFGS bfgs;                                          // Instance of BFGS object for optimization 
    double bfgs_f;                                      // Holder of loss_function value in BFGS
    Array1D bfgs_g;                                     // Holder of loss_function gradient in BFGS
    bool reg_init=true;                       			// Whether the reg_vector needs to be rebuilt (based on changing of diagonal elements of SLAE)
	
public: 
    int maxits = 1000;                          		// Max. number of steps in BFGS
    int random_perturb = 0;                             // If >0, than simulatied annealing (random shifting of coefficients) is enabled
    bool skip_preinit = false;							// If false, than 75 BFGS iterations are done before the actual training 									
    double bfgs_conv_tol = 1e-3;						// Convergence criterion for fucntion decrease in BFGS					
    bool init_random=false;						// Random initialization of parameters for an uninitialized potential
    bool no_mindist_update=false;							// Automatically adjust mindist in the potential according to the training set
    bool auto_minmax_magmom=false;							// Automatically adjust min_magmom and max_magmom in the potential according to the training set
    const double reg_param = 1e-10; 					// Regularization parameter for SLAE solving


    void InitSettings()                        // Sets correspondence between variables and setting names in settings file
    {
        MakeSetting(maxits,             "iteration_limit");  
        MakeSetting(skip_preinit,       "skip_preinit"); 
        MakeSetting(bfgs_conv_tol,      "tolerance");
        MakeSetting(init_random,        "init_random");
        MakeSetting(no_mindist_update,  "no_mindist_update");  
        MakeSetting(auto_minmax_magmom, "auto-minmax-magmom");
    };
    
    MTPR_trainer(MLMTPR* _p_mlmtpr,Settings settings):
        NonLinearRegression(_p_mlmtpr, settings), p_mlmtpr(_p_mlmtpr)     // Initialization of the settings
    {
        InitSettings();
        ApplySettings(settings);
        PrintSettings();
		
        int n = p_mlmtpr->alpha_count + p_mlmtpr->species_count - 1;        // Matrix size
        for (int i=0; i<n; i++)
            p_mlmtpr->reg_vector[i] = reg_param;							//Initialize the regularization vector

    };          
    ~MTPR_trainer() {};

    void ClearSLAE();                                           // Setting SLAE Matrix and right part to zero
    void SymmetrizeSLAE();                                      // Symmetrization of the SLAE matrix before solving (only upper right part is filled during adding or removing configuration to regression)
    void SolveSLAE(int TS_size);                                           // Find the corresponding linear coefficients
    void AddToSLAE(Configuration& cfg, double weight = 1);  // Adds configuration to regression SLAE. If weight = -1 removes from regression

    void AddSpecies(std::vector<Configuration>& training_set);      //Extend the species in the MTPR potential if needed
    void LinOptimize(std::vector<Configuration>& training_set); // Solve the linear SLAE to find the linear coefficients
    void Train(std::vector<Configuration>& training_set) override; //"Main" training function
    void Rescale(std::vector<Configuration>& training_set);            //Find the optimal scaling parameter
    void NonLinOptimize(std::vector<Configuration>& training_set, int max_iter);     //launch the nonlinear optimization procedure, with Shapeev BFGS

};
#endif // MLIP_MTPR_TRAINER_H
