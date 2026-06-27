/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 */

#include "basic_mlip.h"
#include "common/multidimensional_arrays.h"

#ifndef MLIP_MTPR_H
#define MLIP_MTPR_H

class MLMTPR : virtual public AnyLocalMLIP
{
protected:
    int alpha_moments_count;                        //  /=================================================================================================
    int alpha_index_basic_count;                    //  |                                                                                                |
    int(*alpha_index_basic)[4];                     //  |   Internal representation of Moment Tensor Potential basis                                     |
    int alpha_index_times_count;                    //  |   These items is required for calculation of basis functions values and their derivatives      |
    int(*alpha_index_times)[4];                     //  |                                                                                                |
    int *alpha_moment_mapping;                      //  \=================================================================================================
    std::string pot_desc;
    std::string rbasis_type;                        //radial basis type (Chebyshev/Shapeev/Taylor polynomials + Chebyshev_repuls for artificial repulsion to min_dist)
    std::string mbasis_type = "";                   //magnetic basis type

    std::vector<double> moment_vals; //!< Array of basis function values calculated for certain atomic neighborhood
    Array3D moment_ders;//!< Array of basis function derivatives w.r.t. motion of neighboring atoms calculated for certain atomic neighborhood
    std::vector<double> basis_vals; //!< Array of the basis functions values calculated for certain neighbor    hood of the certain configuration
    Array3D basis_ders; //!< Array derivatives w.r.t. each atom in neighborhood of the basis functions values calculated for certain neighborhood of the certain configuration

    Array3D moment_jacobian_;                                               //Temporary vector for site_energy calculation
    std::vector<double> site_energy_ders_wrt_moments_;                      //Temporary vector for site_energy calculation
    std::vector<double> dist_powers_;                                       //Temporary vector for site_energy calculation
    std::vector<Vector3> coords_powers_;                                    //Temporary vector for site_energy calculation
    void CalcSiteEnergyDers(const Neighborhood& nbh) override;              //Site_energy calculation

    Array2D moment_jacobian_wrt_magmom_neigh; //!<Temporary vector for calculation of site energy  derivatives w.r.t. magnetic moments of neighboring atoms  
    Array1D moment_jacobian_wrt_magmom_central; //!<Temporary vector for calculation of site energy  derivatives w.r.t. magnetic moments of central atom

public:
    void MemAlloc();                                //Memory allocation
    double scaling = 1.0;                           //How to scale moments
    std::vector<double> regression_coeffs;          //Regression coefficients
    Array3D radial_coeffs;                          //Array of radial coefficients found by BFGS
    int mbasis_size = 1;

    bool inited = false;                                //Whether potential has already been trained

    std::vector<int> atomic_numbers;                    // Array of atomic numbers present in current potential. These can be relative and absolute ones

    std::vector<double> energy_cmpnts;                  // Energy components for SLAE matrix
    Array3D forces_cmpnts;                              // Force components for SLAE matrix
    Array3D stress_cmpnts;                              // Stress components for SLAE matrix

    void CalcBasisFuncs(Neighborhood& Neighborhood, std::vector<double>& bf_vals); //Linear basic functions calculation
    void CalcBasisFuncsDers(const Neighborhood& Neighborhood);      //Linear basic functions and their derivatives calculation
    void CalcEFSComponents(Configuration& cfg);                     //Calculate the components for linear regression matrix
    void CalcEComponents(Configuration& cfg);           //Calculate the components for linear regression matrix
    void CalcEFS(Configuration& cfg) override
    {
        AnyLocalMLIP::CalcEFS(cfg);
        cfg.features["EFS_by"] = "MultiMTP" + to_string(alpha_count);
    }

    void AccumulateCombinationGrad( const Neighborhood& nbh,
                                    std::vector<double>& out_grad_accumulator,
                                    const double se_weight = 0.0,
                                    const Vector3* se_ders_weights = nullptr) override;
    MLMTPR():
        AnyLocalMLIP() {
    }
    MLMTPR(const std::string& mtpfnm) :
        AnyLocalMLIP() {
        Load(mtpfnm);
    }

    ~MLMTPR();

    void Load(const std::string& filename) override;                //Load potential from the file
    void Save(const std::string& filename) override;                //Save potential to the file
	
    int CoeffCount() //!< number of coefficients
    {
        return (int)regression_coeffs.size();
    }
    double* Coeff() //!< coefficients themselves
    {
        return &regression_coeffs[0];
    }
	int Rsize()  //<! returns the amount of radial coefficients in a regression_coeffs array
	{
		return p_RadialBasis->size*mbasis_size*mbasis_size*species_count*species_count*radial_func_count;
	}
	int LinSize()  //<! amount of linear coefficients in a regression_coeffs array
	{
		return CoeffCount() - Rsize();
	}
	inline double& linear_coeffs(int ind) //!< Pointer to linear coefficients (inside the global coefficients vector) found by SLAE solving
    {
        return regression_coeffs[Rsize()+ind];
    }
    int Get_RB_size() 
    {
        return p_RadialBasis->size;
    }
    int type_from_atomic_number(int a_num) {                //!< returns the relative type (0,1,2,..) in the potential of the atomic number (33,47,..). Returns -1 if the atomic number is not present
        for (int i = 0; i < species_count; i++)
        {
            if (atomic_numbers[i] == a_num)
                return i;
        }
            
        throw MlipException("Atomic number "+to_string(a_num) +" is not present in the MTP potential!");                    
    }

    void AddPenaltyGrad(const double coeff, 
                        double& out_penalty_accumulator, 
                        Array1D* out_penalty_grad_accumulator = nullptr) override;     //!< add the penalty and its gradient to the loss function

    int alpha_count;                            //!< Basis functions count 
    int alpha_scalar_moments;                   //!< = alpha_count-1 (MTP-basis except constant function)
    int radial_func_count;                      //!< number of radial basis functions used
    int species_count;                          //!< number of components present in the potential

    std::vector<double> reg_vector;             //!< this will be added to the diagonal of SLAE matrix during training
    void Orthogonalize();                       //!<Orthogonalize the basic functions

    void AddSpecies(std::set<int> species, bool init_random=true);
};



#endif
