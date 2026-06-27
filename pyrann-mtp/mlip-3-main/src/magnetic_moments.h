/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   Contributors: Ivan Novikov
 */

#ifndef MLIP_MAGNETIC_MOMENTS_H
#define MLIP_MAGNETIC_MOMENTS_H

#include "common/multidimensional_arrays.h"
#include "configuration.h"

//This class calculates magnetic moment for each atom
//INPUT: CHGCAR and OUTCAR files
//OUTPUT: Configuration with the magnetic moments calculated 

class MagneticMoments
{
private:
    Vector3int mesh_size; //mesh size along each direction
    Array1D nearest_neigh_dist; //distance between the nearest neighbors of the same type

    bool is_collinear = true; //parameter indicates if we process the results of non collinear calculations 
    int n_directions = 1; //number of directions in magnetic moment, i.e. the parameter equals to 3 for non-collinear calculations and 1 otherwise
    Array1D density; //magnetization density
    Array2D weight; //weight (infuence of each atom on each node in mesh) in integral for each atom

    void SkipAugmentationOccupancies(std::ifstream&, int); //skipping reading of augmentation occupancies and initial magnetic moments in CHGCAR

    bool IsPointInsideExtendedRectangularParallelepiped(Vector3&, std::vector<Vector3>&); //check whether a point is inside exteneded rectangular parallelepiped
    void GenerateRectangularParallelepiped(Configuration&, std::vector<Vector3>&); //creation of rectangular parallelepiped from lattice
    void ExtendRectangularParallelepiped(Configuration&, std::vector<Vector3>&); //extension of rectangular parallelepiped
    void CreateGhostAtoms(Configuration&, Neighborhood&); //creation of ghost atoms inside rectangular parallelepiped

public:
    void LoadDensityFromCHGCAR(std::string&, int); //loading of magnetic density from chgcar
    void CalculateWeights(Configuration&); //calculation of weights (influence of each atom on each node in mesh)
    void Calculate(Configuration&); //calculation of magnetic moments

    MagneticMoments(){};
    MagneticMoments(bool is_collinear_, Array1D& nearest_neigh_dist_) :
         is_collinear(is_collinear_)
    {
        for (int i = 0; i < nearest_neigh_dist_.size(); i++)
            nearest_neigh_dist.push_back(nearest_neigh_dist_[i]);
    };
    ~MagneticMoments(){};

};

#endif //#ifndef MLIP_MAGNETIC_MOMENTS_H


