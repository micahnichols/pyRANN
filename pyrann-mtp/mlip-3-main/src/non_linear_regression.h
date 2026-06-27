/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */


#ifndef MLIP_NON_LINEAR_REGRESSION_H
#define MLIP_NON_LINEAR_REGRESSION_H

#include "basic_mlip.h"
#include "basic_trainer.h"


class NonLinearRegression : public AnyTrainer//, protected LogWriting 
{
protected:
    std::vector<Vector3> dLdF;
    Matrix3 dLdS;
    void AddLoss(Configuration &orig);
    double loss_=0;                                          // result of AddLoss and AddLossGrad
    std::vector<double> loss_grad_;                         // result of AddLossGrad

public:
    int norm_by_forces = 0;                                 // whether to scale weight of E&F in configurations depending on the abs(F)
    void AddLossGrad(Configuration &orig);

    NonLinearRegression(AnyLocalMLIP* _p_mlip,              // Constructor requires MTP basis
                        double _wgt_energy = 1.0,           
                        double _wgt_forces = 1.0,
                        double _wgt_stress = 1.0,
                        double _wgt_relfrc = 0.0,
                        double _wgt_constr = 1.0) 
    {
        wgt_eqtn_energy = 1.0;
        wgt_eqtn_forces = 1.0;
        wgt_eqtn_stress = 1.0;
        wgt_rel_forces = 0.0;
        wgt_eqtn_constr = 1.0;
    };

    NonLinearRegression(AnyLocalMLIP* _p_mlip, const Settings& _settings) :
        AnyTrainer(_p_mlip, _settings) {};
                                                        
    double ObjectiveFunction(std::vector<Configuration>& train_set);            // Calculates objective function summed over train_set
    void CalcObjectiveFunctionGrad(std::vector<Configuration>& train_set);      // Calculates objective function summed over train_set with their gradients

    virtual void Train(std::vector<Configuration>& train_set) override;

    bool CheckLossConsistency_debug(Configuration cfg,
                                    double displacement = 0.001,
                                    double control_delta = 0.01);

};

#endif // MLIP_NON_LINEAR_REGRESSION_H
