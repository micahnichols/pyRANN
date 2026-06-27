/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   Contributors: Alexander Shapeev, Evgeny Podryabinkin, Konstantin Gubaev
 */

#ifndef MLIP_BASIS_H
#define MLIP_BASIS_H


#include "common/utils.h"


class AnyBasis
{   
public:
    int size;

    double min_val;
    double max_val;
    double scaling = 1.0; // all functions are multiplied by scaling 

    // values and derivatives, set by calc(val)
    std::vector<double> vals;
    std::vector<double> ders;

public:
    AnyBasis(double _min_val, double _max_val, int _size);
    AnyBasis(std::ifstream& ifs);
    virtual ~AnyBasis() {};

    virtual std::string GetRBTypeString()
    {
        return "RBAny";
    }

    void ReadBasis(std::ifstream& ifs);
    void WriteBasis(std::ofstream& ofs);

    virtual void Calc(double val) = 0; // calculates values
    virtual void CalcDers(double val) = 0; // calculates values and derivatives
};



class Basis_Shapeev : public AnyBasis
{
private:
    static const int ALLOCATED_DEGREE = 11;
    double coeffs[ALLOCATED_DEGREE + 1][ALLOCATED_DEGREE + 3];
    void InitShapeevRB();

public:
    Basis_Shapeev(double _min_val, double _max_val, int _size);
    Basis_Shapeev(std::ifstream& ifs);

    std::string GetRBTypeString() override
    {
        return "RBShapeev";
    }

    void Calc(double val) override;
    void CalcDers(double val) override;
};



class Basis_Chebyshev : public AnyBasis
{
public:
    Basis_Chebyshev(double _min_val, double _max_val, int _size)
        : AnyBasis(_min_val, _max_val, _size) {};
    Basis_Chebyshev(std::ifstream& ifs)
        : AnyBasis(ifs) 
    {}

    std::string GetRBTypeString() override
    {
        return "BChebyshev";
    }

    void Calc(double val) override;
    void CalcDers(double val) override;
};


class Basis_Chebyshev_repuls : public AnyBasis
{
public:
    Basis_Chebyshev_repuls(double _min_val, double _max_val, int _size)
        : AnyBasis(_min_val, _max_val, _size) {};
    Basis_Chebyshev_repuls(std::ifstream& ifs)
        : AnyBasis(ifs) 
    {}

    std::string GetRBTypeString() override
    {
        return "BChebyshev_repuls";
    }

    void Calc(double val) override;
    void CalcDers(double val) override;
};

class RB_Chebyshev : public AnyBasis
{
public:
    RB_Chebyshev(double _min_val, double _max_val, int _size)
        : AnyBasis(_min_val, _max_val, _size) {};
    RB_Chebyshev(std::ifstream& ifs)
        : AnyBasis(ifs) 
    {}

    std::string GetRBTypeString() override
    {
        return "RBChebyshev";
    }

    void Calc(double val) override;
    void CalcDers(double val) override;
};

class RB_Chebyshev_repuls : public AnyBasis
{
public:
    RB_Chebyshev_repuls(double _min_val, double _max_val, int _size)
        : AnyBasis(_min_val, _max_val, _size) {};
    RB_Chebyshev_repuls(std::ifstream& ifs)
        : AnyBasis(ifs) 
    {}

    std::string GetRBTypeString() override
    {
        return "RBChebyshev_repuls";
    }

    void Calc(double val) override;
    void CalcDers(double val) override;
};


class Basis_Taylor : public AnyBasis
{
public: 
    Basis_Taylor(double _min_val, double _max_val, int _size)
        : AnyBasis(_min_val, _max_val, _size) {};
    Basis_Taylor(std::ifstream& ifs)
        : AnyBasis(ifs) 
    {}

    std::string GetRBTypeString() override
    {
        return "RBTaylor";
    }

    void Calc(double val) override;
    void CalcDers(double val) override;
};

#endif // MLIP_BASIS_H
