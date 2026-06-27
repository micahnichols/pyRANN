/*   This software is called MLIP for Machine Learning Interatomic Potentials.
 *   MLIP can only be used for non-commercial research and cannot be re-distributed.
 *   The use of MLIP must be acknowledged by citing approriate references.
 *   See the LICENSE file for details.
 *
 *   Contributors: Alexander Shapeev, Evgeny Podryabinkin, Konstantin Gubaev
 */


#include "basis.h"
#include <cmath>


using namespace std;


void AnyBasis::ReadBasis(std::ifstream & ifs)
{
    if ((!ifs.is_open()) || (ifs.eof()))
        ERROR("AnyBasis::ReadBasis: Can't load basis");

    string tmpstr;

    // reading min_val / scaling
    ifs >> tmpstr;
    if (tmpstr == "scaling") {
        ifs.ignore(2);
        ifs >> scaling;
        if (ifs.fail())
            ERROR("Error reading .mtp file");
        ifs >> tmpstr;
    }

    if (tmpstr != "min_val" && tmpstr != "min_dist")
        ERROR("Error reading .mtp file");
    ifs.ignore(2);
    ifs >> min_val;
    if (ifs.fail())
        ERROR("Error reading .mtp file");

    // reading max_val 
    ifs >> tmpstr;
    if (tmpstr != "max_val" && tmpstr != "max_dist")
        ERROR("Error reading .mtp file");
    ifs.ignore(2);
    ifs >> max_val;
    if (ifs.fail())
        ERROR("Error reading .mtp file");

    // reading size 
    ifs >> tmpstr;
    if (tmpstr != "basis_size" && tmpstr != "radial_basis_size")
        ERROR("Error reading .mtp file");
    ifs.ignore(2);
    ifs >> size;
    if (ifs.fail())
        ERROR("Error reading .mtp file");

    vals.resize(size);
    ders.resize(size);
}

void AnyBasis::WriteBasis(std::ofstream & ofs)
{
    if (!ofs.is_open())
        ERROR("AnyBasis::WriteBasis: Output stream isn't open");

    ofs << "radial_basis_type = " << GetRBTypeString() << '\n';
    if (scaling != 1.0)
        ofs << "\tscaling = " << scaling << '\n';
    ofs << "\tmin_dist = " << min_val << '\n'
        << "\tmax_dist = " << max_val << '\n'
        << "\tradial_basis_size = " << size << '\n';
}

AnyBasis::AnyBasis(double _min_val, double _max_val, int _size)
    : size(_size), min_val(_min_val), max_val(_max_val)
{
    vals.resize(size);
    ders.resize(size);
}

AnyBasis::AnyBasis(std::ifstream & ifs) 
{
    ReadBasis(ifs);
}

void Basis_Shapeev::InitShapeevRB()
{

#ifdef MLIP_DEBUG
    if (size > ALLOCATED_DEGREE) {
        ERROR("error: AnyBasis::MAX_DEGREE > AnyBasis::ALLOCATED_DEGREE");
    }
#endif

    coeffs[0][0] = (sqrt(21)*pow(max_val, 2)*pow(min_val, 2)) / (2.*pow(max_val - min_val, 2));
    coeffs[0][1] = -((sqrt(21)*max_val*pow(min_val, 2)) / pow(max_val - min_val, 2));
    coeffs[0][2] = (sqrt(21)*pow(min_val, 2)) / (2.*pow(max_val - min_val, 2));
    coeffs[0][3] = 0;
    coeffs[0][4] = 0;
    coeffs[0][5] = 0;
    coeffs[0][6] = 0;
    coeffs[0][7] = 0;
    coeffs[0][8] = 0;
    coeffs[0][9] = 0;
    coeffs[0][10] = 0;
    coeffs[0][11] = 0;
    coeffs[0][12] = 0;
    coeffs[0][13] = 0;
    coeffs[1][0] = (-3 * sqrt(7)*pow(max_val, 2)*pow(min_val, 2)*(max_val + 3 * min_val)) / (4.*pow(max_val - min_val, 3));
    coeffs[1][1] = (9 * sqrt(7)*max_val*pow(min_val, 2)*(max_val + min_val)) / (2.*pow(max_val - min_val, 3));
    coeffs[1][2] = (9 * sqrt(7)*pow(min_val, 2)*(3 * max_val + min_val)) / (4.*pow(-max_val + min_val, 3));
    coeffs[1][3] = (3 * sqrt(7)*pow(min_val, 2)) / pow(max_val - min_val, 3);
    coeffs[1][4] = 0;
    coeffs[1][5] = 0;
    coeffs[1][6] = 0;
    coeffs[1][7] = 0;
    coeffs[1][8] = 0;
    coeffs[1][9] = 0;
    coeffs[1][10] = 0;
    coeffs[1][11] = 0;
    coeffs[1][12] = 0;
    coeffs[1][13] = 0;
    coeffs[2][0] = (sqrt(3.6666666666666665)*pow(max_val, 2)*pow(min_val, 2)*(pow(max_val, 2) + 7 * max_val*min_val + 7 * pow(min_val, 2))) / pow(max_val - min_val, 4);
    coeffs[2][1] = -((sqrt(3.6666666666666665)*max_val*pow(min_val, 2)*(11 * pow(max_val, 2) + 35 * max_val*min_val + 14 * pow(min_val, 2))) / pow(max_val - min_val, 4));
    coeffs[2][2] = (sqrt(3.6666666666666665)*pow(min_val, 2)*(34 * pow(max_val, 2) + 49 * max_val*min_val + 7 * pow(min_val, 2))) / pow(max_val - min_val, 4);
    coeffs[2][3] = -((sqrt(33)*pow(min_val, 2)*(13 * max_val + 7 * min_val)) / pow(max_val - min_val, 4));
    coeffs[2][4] = (5 * sqrt(33)*pow(min_val, 2)) / pow(max_val - min_val, 4);
    coeffs[2][5] = 0;
    coeffs[2][6] = 0;
    coeffs[2][7] = 0;
    coeffs[2][8] = 0;
    coeffs[2][9] = 0;
    coeffs[2][10] = 0;
    coeffs[2][11] = 0;
    coeffs[2][12] = 0;
    coeffs[2][13] = 0;
    coeffs[3][0] = (-3 * sqrt(6.5)*pow(max_val, 2)*pow(min_val, 2)*(pow(max_val, 3) + 12 * pow(max_val, 2)*min_val + 28 * max_val*pow(min_val, 2) + 14 * pow(min_val, 3))) / (4.*pow(max_val - min_val, 5));
    coeffs[3][1] = (3 * sqrt(6.5)*max_val*pow(min_val, 2)*(17 * pow(max_val, 3) + 104 * pow(max_val, 2)*min_val + 126 * max_val*pow(min_val, 2) + 28 * pow(min_val, 3))) / (4.*pow(max_val - min_val, 5));
    coeffs[3][2] = (3 * sqrt(6.5)*pow(min_val, 2)*(43 * pow(max_val, 3) + 141 * pow(max_val, 2)*min_val + 84 * max_val*pow(min_val, 2) + 7 * pow(min_val, 3))) / (2.*pow(-max_val + min_val, 5));
    coeffs[3][3] = (15 * sqrt(6.5)*pow(min_val, 2)*(18 * pow(max_val, 2) + 30 * max_val*min_val + 7 * pow(min_val, 2))) / (2.*pow(max_val - min_val, 5));
    coeffs[3][4] = (165 * sqrt(6.5)*pow(min_val, 2)*(3 * max_val + 2 * min_val)) / (4.*pow(-max_val + min_val, 5));
    coeffs[3][5] = (165 * sqrt(6.5)*pow(min_val, 2)) / (4.*pow(max_val - min_val, 5));
    coeffs[3][6] = 0;
    coeffs[3][7] = 0;
    coeffs[3][8] = 0;
    coeffs[3][9] = 0;
    coeffs[3][10] = 0;
    coeffs[3][11] = 0;
    coeffs[3][12] = 0;
    coeffs[3][13] = 0;
    coeffs[4][0] = (sqrt(0.6)*pow(max_val, 2)*pow(min_val, 2)*(5 * pow(max_val, 4) + 90 * pow(max_val, 3)*min_val + 360 * pow(max_val, 2)*pow(min_val, 2) + 420 * max_val*pow(min_val, 3) + 126 * pow(min_val, 4))) / (2.*pow(max_val - min_val, 6));
    coeffs[4][1] = (-3 * sqrt(0.6)*max_val*pow(min_val, 2)*(20 * pow(max_val, 4) + 195 * pow(max_val, 3)*min_val + 450 * pow(max_val, 2)*pow(min_val, 2) + 294 * max_val*pow(min_val, 3) + 42 * pow(min_val, 4))) / pow(max_val - min_val, 6);
    coeffs[4][2] = (3 * sqrt(0.6)*pow(min_val, 2)*(295 * pow(max_val, 4) + 1680 * pow(max_val, 3)*min_val + 2232 * pow(max_val, 2)*pow(min_val, 2) + 756 * max_val*pow(min_val, 3) + 42 * pow(min_val, 4))) / (2.*pow(max_val - min_val, 6));
    coeffs[4][3] = (-22 * sqrt(0.6)*pow(min_val, 2)*(65 * pow(max_val, 3) + 216 * pow(max_val, 2)*min_val + 153 * max_val*pow(min_val, 2) + 21 * pow(min_val, 3))) / pow(max_val - min_val, 6);
    coeffs[4][4] = (33 * sqrt(0.6)*pow(min_val, 2)*(137 * pow(max_val, 2) + 246 * max_val*min_val + 72 * pow(min_val, 2))) / (2.*pow(max_val - min_val, 6));
    coeffs[4][5] = (-429 * sqrt(0.6)*pow(min_val, 2)*(4 * max_val + 3 * min_val)) / pow(max_val - min_val, 6);
    coeffs[4][6] = (1001 * sqrt(0.6)*pow(min_val, 2)) / (2.*pow(max_val - min_val, 6));
    coeffs[4][7] = 0;
    coeffs[4][8] = 0;
    coeffs[4][9] = 0;
    coeffs[4][10] = 0;
    coeffs[4][11] = 0;
    coeffs[4][12] = 0;
    coeffs[4][13] = 0;
    coeffs[5][0] = -(sqrt(62.333333333333336)*pow(max_val, 2)*pow(min_val, 2)*(pow(max_val, 5) + 25 * pow(max_val, 4)*min_val + 150 * pow(max_val, 3)*pow(min_val, 2) + 300 * pow(max_val, 2)*pow(min_val, 3) + 210 * max_val*pow(min_val, 4) + 42 * pow(min_val, 5))) / (4.*pow(max_val - min_val, 7));
    coeffs[5][1] = (sqrt(62.333333333333336)*max_val*pow(min_val, 2)*(16 * pow(max_val, 5) + 225 * pow(max_val, 4)*min_val + 825 * pow(max_val, 3)*pow(min_val, 2) + 1020 * pow(max_val, 2)*pow(min_val, 3) + 420 * max_val*pow(min_val, 4) + 42 * pow(min_val, 5))) / (2.*pow(max_val - min_val, 7));
    coeffs[5][2] = -(sqrt(561)*pow(min_val, 2)*(107 * pow(max_val, 5) + 925 * pow(max_val, 4)*min_val + 2120 * pow(max_val, 3)*pow(min_val, 2) + 1580 * pow(max_val, 2)*pow(min_val, 3) + 350 * max_val*pow(min_val, 4) + 14 * pow(min_val, 5))) / (4.*pow(max_val - min_val, 7));
    coeffs[5][3] = (5 * sqrt(62.333333333333336)*pow(min_val, 2)*(73 * pow(max_val, 4) + 397 * pow(max_val, 3)*min_val + 555 * pow(max_val, 2)*pow(min_val, 2) + 228 * max_val*pow(min_val, 3) + 21 * pow(min_val, 4))) / pow(max_val - min_val, 7);
    coeffs[5][4] = (-65 * sqrt(62.333333333333336)*pow(min_val, 2)*(53 * pow(max_val, 3) + 177 * pow(max_val, 2)*min_val + 138 * max_val*pow(min_val, 2) + 24 * pow(min_val, 3))) / (4.*pow(max_val - min_val, 7));
    coeffs[5][5] = (91 * sqrt(561)*pow(min_val, 2)*(8 * pow(max_val, 2) + 15 * max_val*min_val + 5 * pow(min_val, 2))) / (2.*pow(max_val - min_val, 7));
    coeffs[5][6] = (-91 * sqrt(62.333333333333336)*pow(min_val, 2)*(31 * max_val + 25 * min_val)) / (4.*pow(max_val - min_val, 7));
    coeffs[5][7] = (182 * sqrt(62.333333333333336)*pow(min_val, 2)) / pow(max_val - min_val, 7);
    coeffs[5][8] = 0;
    coeffs[5][9] = 0;
    coeffs[5][10] = 0;
    coeffs[5][11] = 0;
    coeffs[5][12] = 0;
    coeffs[5][13] = 0;
    coeffs[6][0] = (sqrt(4.071428571428571)*pow(max_val, 2)*pow(min_val, 2)*(pow(max_val, 6) + 33 * pow(max_val, 5)*min_val + 275 * pow(max_val, 4)*pow(min_val, 2) + 825 * pow(max_val, 3)*pow(min_val, 3) + 990 * pow(max_val, 2)*pow(min_val, 4) + 462 * max_val*pow(min_val, 5) + 66 * pow(min_val, 6))) / pow(max_val - min_val, 8);
    coeffs[6][1] = -((sqrt(4.071428571428571)*max_val*pow(min_val, 2)*(41 * pow(max_val, 6) + 781 * pow(max_val, 5)*min_val + 4125 * pow(max_val, 4)*pow(min_val, 2) + 8085 * pow(max_val, 3)*pow(min_val, 3) + 6270 * pow(max_val, 2)*pow(min_val, 4) + 1782 * max_val*pow(min_val, 5) + 132 * pow(min_val, 6))) / pow(max_val - min_val, 8));
    coeffs[6][2] = (3 * sqrt(16.285714285714285)*pow(min_val, 2)*(89 * pow(max_val, 6) + 1078 * pow(max_val, 5)*min_val + 3740 * pow(max_val, 4)*pow(min_val, 2) + 4785 * pow(max_val, 3)*pow(min_val, 3) + 2310 * pow(max_val, 2)*pow(min_val, 4) + 363 * max_val*pow(min_val, 5) + 11 * pow(min_val, 6))) / pow(max_val - min_val, 8);
    coeffs[6][3] = (-13 * sqrt(16.285714285714285)*pow(min_val, 2)*(124 * pow(max_val, 5) + 990 * pow(max_val, 4)*min_val + 2255 * pow(max_val, 3)*pow(min_val, 2) + 1815 * pow(max_val, 2)*pow(min_val, 3) + 495 * max_val*pow(min_val, 4) + 33 * pow(min_val, 5))) / pow(max_val - min_val, 8);
    coeffs[6][4] = (65 * sqrt(4.071428571428571)*pow(min_val, 2)*(161 * pow(max_val, 4) + 847 * pow(max_val, 3)*min_val + 1221 * pow(max_val, 2)*pow(min_val, 2) + 561 * max_val*pow(min_val, 3) + 66 * pow(min_val, 4))) / pow(max_val - min_val, 8);
    coeffs[6][5] = (-39 * sqrt(4.071428571428571)*pow(min_val, 2)*(497 * pow(max_val, 3) + 1661 * pow(max_val, 2)*min_val + 1375 * max_val*pow(min_val, 2) + 275 * pow(min_val, 3))) / pow(max_val - min_val, 8);
    coeffs[6][6] = (26 * sqrt(16.285714285714285)*pow(min_val, 2)*(394 * pow(max_val, 2) + 759 * max_val*min_val + 275 * pow(min_val, 2))) / pow(max_val - min_val, 8);
    coeffs[6][7] = (-442 * sqrt(16.285714285714285)*pow(min_val, 2)*(13 * max_val + 11 * min_val)) / pow(max_val - min_val, 8);
    coeffs[6][8] = (1326 * sqrt(16.285714285714285)*pow(min_val, 2)) / pow(max_val - min_val, 8);
    coeffs[6][9] = 0;
    coeffs[6][10] = 0;
    coeffs[6][11] = 0;
    coeffs[6][12] = 0;
    coeffs[6][13] = 0;
    coeffs[7][0] = -(sqrt(273)*pow(max_val, 2)*pow(min_val, 2)*(pow(max_val, 7) + 42 * pow(max_val, 6)*min_val + 462 * pow(max_val, 5)*pow(min_val, 2) + 1925 * pow(max_val, 4)*pow(min_val, 3) + 3465 * pow(max_val, 3)*pow(min_val, 4) + 2772 * pow(max_val, 2)*pow(min_val, 5) + 924 * max_val*pow(min_val, 6) + 99 * pow(min_val, 7))) / (8.*pow(max_val - min_val, 9));
    coeffs[7][1] = (3 * sqrt(273)*max_val*pow(min_val, 2)*(17 * pow(max_val, 7) + 420 * pow(max_val, 6)*min_val + 3003 * pow(max_val, 5)*pow(min_val, 2) + 8470 * pow(max_val, 4)*pow(min_val, 3) + 10395 * pow(max_val, 3)*pow(min_val, 4) + 5544 * pow(max_val, 2)*pow(min_val, 5) + 1155 * max_val*pow(min_val, 6) + 66 * pow(min_val, 7))) / (8.*pow(max_val - min_val, 9));
    coeffs[7][2] = (-3 * sqrt(273)*pow(min_val, 2)*(278 * pow(max_val, 7) + 4473 * pow(max_val, 6)*min_val + 21714 * pow(max_val, 5)*pow(min_val, 2) + 41965 * pow(max_val, 4)*pow(min_val, 3) + 34650 * pow(max_val, 3)*pow(min_val, 4) + 11781 * pow(max_val, 2)*pow(min_val, 5) + 1386 * max_val*pow(min_val, 6) + 33 * pow(min_val, 7))) / (8.*pow(max_val - min_val, 9));
    coeffs[7][3] = (7 * sqrt(273)*pow(min_val, 2)*(917 * pow(max_val, 6) + 10038 * pow(max_val, 5)*min_val + 33495 * pow(max_val, 4)*pow(min_val, 2) + 43780 * pow(max_val, 3)*pow(min_val, 3) + 23265 * pow(max_val, 2)*pow(min_val, 4) + 4554 * max_val*pow(min_val, 5) + 231 * pow(min_val, 6))) / (8.*pow(max_val - min_val, 9));
    coeffs[7][4] = (-105 * sqrt(273)*pow(min_val, 2)*(259 * pow(max_val, 5) + 1953 * pow(max_val, 4)*min_val + 4422 * pow(max_val, 3)*pow(min_val, 2) + 3740 * pow(max_val, 2)*pow(min_val, 3) + 1155 * max_val*pow(min_val, 4) + 99 * pow(min_val, 5))) / (8.*pow(max_val - min_val, 9));
    coeffs[7][5] = (21 * sqrt(273)*pow(min_val, 2)*(1624 * pow(max_val, 4) + 8328 * pow(max_val, 3)*min_val + 12243 * pow(max_val, 2)*pow(min_val, 2) + 6050 * max_val*pow(min_val, 3) + 825 * pow(min_val, 4))) / (4.*pow(max_val - min_val, 9));
    coeffs[7][6] = (-119 * sqrt(273)*pow(min_val, 2)*(436 * pow(max_val, 3) + 1455 * pow(max_val, 2)*min_val + 1254 * max_val*pow(min_val, 2) + 275 * pow(min_val, 3))) / (4.*pow(max_val - min_val, 9));
    coeffs[7][7] = (153 * sqrt(273)*pow(min_val, 2)*(307 * pow(max_val, 2) + 602 * max_val*min_val + 231 * pow(min_val, 2))) / (4.*pow(max_val - min_val, 9));
    coeffs[7][8] = (-2907 * sqrt(273)*pow(min_val, 2)*(8 * max_val + 7 * min_val)) / (4.*pow(max_val - min_val, 9));
    coeffs[7][9] = (4845 * sqrt(273)*pow(min_val, 2)) / (4.*pow(max_val - min_val, 9));
    coeffs[7][10] = 0;
    coeffs[7][11] = 0;
    coeffs[7][12] = 0;
    coeffs[7][13] = 0;
    coeffs[8][0] = (sqrt(161)*pow(max_val, 2)*pow(min_val, 2)*(pow(max_val, 8) + 52 * pow(max_val, 7)*min_val + 728 * pow(max_val, 6)*pow(min_val, 2) + 4004 * pow(max_val, 5)*pow(min_val, 3) + 10010 * pow(max_val, 4)*pow(min_val, 4) + 12012 * pow(max_val, 3)*pow(min_val, 5) + 6864 * pow(max_val, 2)*pow(min_val, 6) + 1716 * max_val*pow(min_val, 7) + 143 * pow(min_val, 8))) / (6.*pow(max_val - min_val, 10));
    coeffs[8][1] = -(sqrt(161)*max_val*pow(min_val, 2)*(31 * pow(max_val, 8) + 962 * pow(max_val, 7)*min_val + 8918 * pow(max_val, 6)*pow(min_val, 2) + 34034 * pow(max_val, 5)*pow(min_val, 3) + 60060 * pow(max_val, 4)*pow(min_val, 4) + 50622 * pow(max_val, 3)*pow(min_val, 5) + 19734 * pow(max_val, 2)*pow(min_val, 6) + 3146 * max_val*pow(min_val, 7) + 143 * pow(min_val, 8))) / (3.*pow(max_val - min_val, 10));
    coeffs[8][2] = (sqrt(161)*pow(min_val, 2)*(1241 * pow(max_val, 8) + 25532 * pow(max_val, 7)*min_val + 164528 * pow(max_val, 6)*pow(min_val, 2) + 444444 * pow(max_val, 5)*pow(min_val, 3) + 553410 * pow(max_val, 4)*pow(min_val, 4) + 320892 * pow(max_val, 3)*pow(min_val, 5) + 81224 * pow(max_val, 2)*pow(min_val, 6) + 7436 * max_val*pow(min_val, 7) + 143 * pow(min_val, 8))) / (6.*pow(max_val - min_val, 10));
    coeffs[8][3] = (-10 * sqrt(161)*pow(min_val, 2)*(591 * pow(max_val, 7) + 8463 * pow(max_val, 6)*min_val + 38675 * pow(max_val, 5)*pow(min_val, 2) + 73931 * pow(max_val, 4)*pow(min_val, 3) + 63635 * pow(max_val, 3)*pow(min_val, 4) + 24167 * pow(max_val, 2)*pow(min_val, 5) + 3575 * max_val*pow(min_val, 6) + 143 * pow(min_val, 7))) / (3.*pow(max_val - min_val, 10));
    coeffs[8][4] = (10 * sqrt(161)*pow(min_val, 2)*(3150 * pow(max_val, 6) + 32032 * pow(max_val, 5)*min_val + 103792 * pow(max_val, 4)*pow(min_val, 2) + 137566 * pow(max_val, 3)*pow(min_val, 3) + 77935 * pow(max_val, 2)*pow(min_val, 4) + 17446 * max_val*pow(min_val, 5) + 1144 * pow(min_val, 6))) / (3.*pow(max_val - min_val, 10));
    coeffs[8][5] = (-68 * sqrt(161)*pow(min_val, 2)*(1498 * pow(max_val, 5) + 10816 * pow(max_val, 4)*min_val + 24349 * pow(max_val, 3)*pow(min_val, 2) + 21307 * pow(max_val, 2)*pow(min_val, 3) + 7150 * max_val*pow(min_val, 4) + 715 * pow(min_val, 5))) / (3.*pow(max_val - min_val, 10));
    coeffs[8][6] = (34 * sqrt(161)*pow(min_val, 2)*(6102 * pow(max_val, 4) + 30654 * pow(max_val, 3)*min_val + 45656 * pow(max_val, 2)*pow(min_val, 2) + 23738 * max_val*pow(min_val, 3) + 3575 * pow(min_val, 4))) / (3.*pow(max_val - min_val, 10));
    coeffs[8][7] = (-1292 * sqrt(161)*pow(min_val, 2)*(207 * pow(max_val, 3) + 689 * pow(max_val, 2)*min_val + 611 * max_val*pow(min_val, 2) + 143 * pow(min_val, 3))) / (3.*pow(max_val - min_val, 10));
    coeffs[8][8] = (1615 * sqrt(161)*pow(min_val, 2)*(131 * pow(max_val, 2) + 260 * max_val*min_val + 104 * pow(min_val, 2))) / (3.*pow(max_val - min_val, 10));
    coeffs[8][9] = (-3230 * sqrt(161)*pow(min_val, 2)*(29 * max_val + 26 * min_val)) / (3.*pow(max_val - min_val, 10));
    coeffs[8][10] = (17765 * sqrt(161)*pow(min_val, 2)) / (3.*pow(max_val - min_val, 10));
    coeffs[8][11] = 0;
    coeffs[8][12] = 0;
    coeffs[8][13] = 0;
    coeffs[9][0] = -(sqrt(3)*pow(max_val, 2)*pow(min_val, 2)*(5 * pow(max_val, 9) + 315 * pow(max_val, 8)*min_val + 5460 * pow(max_val, 7)*pow(min_val, 2) + 38220 * pow(max_val, 6)*pow(min_val, 3) + 126126 * pow(max_val, 5)*pow(min_val, 4) + 210210 * pow(max_val, 4)*pow(min_val, 5) + 180180 * pow(max_val, 3)*pow(min_val, 6) + 77220 * pow(max_val, 2)*pow(min_val, 7) + 15015 * max_val*pow(min_val, 8) + 1001 * pow(min_val, 9))) / (4.*pow(max_val - min_val, 11));
    coeffs[9][1] = (sqrt(3)*max_val*pow(min_val, 2)*(185 * pow(max_val, 9) + 7035 * pow(max_val, 8)*min_val + 81900 * pow(max_val, 7)*pow(min_val, 2) + 405132 * pow(max_val, 6)*pow(min_val, 3) + 966966 * pow(max_val, 5)*pow(min_val, 4) + 1171170 * pow(max_val, 4)*pow(min_val, 5) + 720720 * pow(max_val, 3)*pow(min_val, 6) + 214500 * pow(max_val, 2)*pow(min_val, 7) + 27027 * max_val*pow(min_val, 8) + 1001 * pow(min_val, 9))) / (2.*pow(max_val - min_val, 11));
    coeffs[9][2] = -(sqrt(3)*pow(min_val, 2)*(8885 * pow(max_val, 9) + 227115 * pow(max_val, 8)*min_val + 1870596 * pow(max_val, 7)*pow(min_val, 2) + 6703788 * pow(max_val, 6)*pow(min_val, 3) + 11657646 * pow(max_val, 5)*pow(min_val, 4) + 10180170 * pow(max_val, 4)*pow(min_val, 5) + 4384380 * pow(max_val, 3)*pow(min_val, 6) + 859716 * pow(max_val, 2)*pow(min_val, 7) + 63063 * max_val*pow(min_val, 8) + 1001 * pow(min_val, 9))) / (4.*pow(max_val - min_val, 11));
    coeffs[9][3] = (6 * sqrt(3)*pow(min_val, 2)*(4265 * pow(max_val, 8) + 77196 * pow(max_val, 7)*min_val + 461188 * pow(max_val, 6)*pow(min_val, 2) + 1206296 * pow(max_val, 5)*pow(min_val, 3) + 1516515 * pow(max_val, 4)*pow(min_val, 4) + 930930 * pow(max_val, 3)*pow(min_val, 5) + 266266 * pow(max_val, 2)*pow(min_val, 6) + 30888 * max_val*pow(min_val, 7) + 1001 * pow(min_val, 8))) / pow(max_val - min_val, 11);
    coeffs[9][4] = (-51 * sqrt(3)*pow(min_val, 2)*(3274 * pow(max_val, 7) + 43022 * pow(max_val, 6)*min_val + 187824 * pow(max_val, 5)*pow(min_val, 2) + 355810 * pow(max_val, 4)*pow(min_val, 3) + 315315 * pow(max_val, 3)*pow(min_val, 4) + 129129 * pow(max_val, 2)*pow(min_val, 5) + 22022 * max_val*pow(min_val, 6) + 1144 * pow(min_val, 7))) / pow(max_val - min_val, 11);
    coeffs[9][5] = (714 * sqrt(3)*pow(min_val, 2)*(942 * pow(max_val, 6) + 9054 * pow(max_val, 5)*min_val + 28665 * pow(max_val, 4)*pow(min_val, 2) + 38350 * pow(max_val, 3)*pow(min_val, 3) + 22737 * pow(max_val, 2)*pow(min_val, 4) + 5577 * max_val*pow(min_val, 5) + 429 * pow(min_val, 6))) / pow(max_val - min_val, 11);
    coeffs[9][6] = (-6783 * sqrt(3)*pow(min_val, 2)*(258 * pow(max_val, 5) + 1800 * pow(max_val, 4)*min_val + 4030 * pow(max_val, 3)*pow(min_val, 2) + 3614 * pow(max_val, 2)*pow(min_val, 3) + 1287 * max_val*pow(min_val, 4) + 143 * pow(min_val, 5))) / pow(max_val - min_val, 11);
    coeffs[9][7] = (1938 * sqrt(3)*pow(min_val, 2)*(1545 * pow(max_val, 4) + 7630 * pow(max_val, 3)*min_val + 11466 * pow(max_val, 2)*pow(min_val, 2) + 6188 * max_val*pow(min_val, 3) + 1001 * pow(min_val, 4))) / pow(max_val - min_val, 11);
    coeffs[9][8] = (-969 * sqrt(3)*pow(min_val, 2)*(6905 * pow(max_val, 3) + 22911 * pow(max_val, 2)*min_val + 20748 * max_val*pow(min_val, 2) + 5096 * pow(min_val, 3))) / (2.*pow(max_val - min_val, 11));
    coeffs[9][9] = (3553 * sqrt(3)*pow(min_val, 2)*(661 * pow(max_val, 2) + 1323 * max_val*min_val + 546 * pow(min_val, 2))) / pow(max_val - min_val, 11);
    coeffs[9][10] = (-81719 * sqrt(3)*pow(min_val, 2)*(23 * max_val + 21 * min_val)) / (2.*pow(max_val - min_val, 11));
    coeffs[9][11] = (163438 * sqrt(3)*pow(min_val, 2)) / pow(max_val - min_val, 11);
    coeffs[9][12] = 0;
    coeffs[9][13] = 0;
    coeffs[10][0] = (3 * sqrt(0.5454545454545454)*pow(max_val, 2)*pow(min_val, 2)*(pow(max_val, 10) + 75 * pow(max_val, 9)*min_val + 1575 * pow(max_val, 8)*pow(min_val, 2) + 13650 * pow(max_val, 7)*pow(min_val, 3) + 57330 * pow(max_val, 6)*pow(min_val, 4) + 126126 * pow(max_val, 5)*pow(min_val, 5) + 150150 * pow(max_val, 4)*pow(min_val, 6) + 96525 * pow(max_val, 3)*pow(min_val, 7) + 32175 * pow(max_val, 2)*pow(min_val, 8) + 5005 * max_val*pow(min_val, 9) + 273 * pow(min_val, 10))) / pow(max_val - min_val, 12);
    coeffs[10][1] = (-9 * sqrt(0.5454545454545454)*max_val*pow(min_val, 2)*(29 * pow(max_val, 10) + 1325 * pow(max_val, 9)*min_val + 18900 * pow(max_val, 8)*pow(min_val, 2) + 117390 * pow(max_val, 7)*pow(min_val, 3) + 363090 * pow(max_val, 6)*pow(min_val, 4) + 594594 * pow(max_val, 5)*pow(min_val, 5) + 525525 * pow(max_val, 4)*pow(min_val, 6) + 246675 * pow(max_val, 3)*pow(min_val, 7) + 57915 * pow(max_val, 2)*pow(min_val, 8) + 5915 * max_val*pow(min_val, 9) + 182 * pow(min_val, 10))) / pow(max_val - min_val, 12);
    coeffs[10][2] = (9 * sqrt(0.5454545454545454)*pow(min_val, 2)*(822 * pow(max_val, 10) + 25525 * pow(max_val, 9)*min_val + 261135 * pow(max_val, 8)*pow(min_val, 2) + 1195740 * pow(max_val, 7)*pow(min_val, 3) + 2757300 * pow(max_val, 6)*pow(min_val, 4) + 3360357 * pow(max_val, 5)*pow(min_val, 5) + 2177175 * pow(max_val, 4)*pow(min_val, 6) + 725010 * pow(max_val, 3)*pow(min_val, 7) + 113490 * pow(max_val, 2)*pow(min_val, 8) + 6825 * max_val*pow(min_val, 9) + 91 * pow(min_val, 10))) / pow(max_val - min_val, 12);
    coeffs[10][3] = (-255 * sqrt(0.5454545454545454)*pow(min_val, 2)*(397 * pow(max_val, 9) + 8847 * pow(max_val, 8)*min_val + 66780 * pow(max_val, 7)*pow(min_val, 2) + 228228 * pow(max_val, 6)*pow(min_val, 3) + 392301 * pow(max_val, 5)*pow(min_val, 4) + 351351 * pow(max_val, 4)*pow(min_val, 5) + 162162 * pow(max_val, 3)*pow(min_val, 6) + 36270 * pow(max_val, 2)*pow(min_val, 7) + 3393 * max_val*pow(min_val, 8) + 91 * pow(min_val, 9))) / pow(max_val - min_val, 12);
    coeffs[10][4] = (2295 * sqrt(0.5454545454545454)*pow(min_val, 2)*(345 * pow(max_val, 8) + 5676 * pow(max_val, 7)*min_val + 32004 * pow(max_val, 6)*pow(min_val, 2) + 81627 * pow(max_val, 5)*pow(min_val, 3) + 103285 * pow(max_val, 4)*pow(min_val, 4) + 66066 * pow(max_val, 3)*pow(min_val, 5) + 20566 * pow(max_val, 2)*pow(min_val, 6) + 2769 * max_val*pow(min_val, 7) + 117 * pow(min_val, 8))) / pow(max_val - min_val, 12);
    coeffs[10][5] = (-8721 * sqrt(0.5454545454545454)*pow(min_val, 2)*(444 * pow(max_val, 7) + 5460 * pow(max_val, 6)*min_val + 22995 * pow(max_val, 5)*pow(min_val, 2) + 43225 * pow(max_val, 4)*pow(min_val, 3) + 39130 * pow(max_val, 3)*pow(min_val, 4) + 16926 * pow(max_val, 2)*pow(min_val, 5) + 3185 * max_val*pow(min_val, 6) + 195 * pow(min_val, 7))) / pow(max_val - min_val, 12);
    coeffs[10][6] = (20349 * sqrt(0.5454545454545454)*pow(min_val, 2)*(612 * pow(max_val, 6) + 5625 * pow(max_val, 5)*min_val + 17475 * pow(max_val, 4)*pow(min_val, 2) + 23530 * pow(max_val, 3)*pow(min_val, 3) + 14430 * pow(max_val, 2)*pow(min_val, 4) + 3783 * max_val*pow(min_val, 5) + 325 * pow(min_val, 6))) / pow(max_val - min_val, 12);
    coeffs[10][7] = (-8721 * sqrt(0.5454545454545454)*pow(min_val, 2)*(3099 * pow(max_val, 5) + 21025 * pow(max_val, 4)*min_val + 46830 * pow(max_val, 3)*pow(min_val, 2) + 42770 * pow(max_val, 2)*pow(min_val, 3) + 15925 * max_val*pow(min_val, 4) + 1911 * pow(min_val, 5))) / pow(max_val - min_val, 12);
    coeffs[10][8] = (43605 * sqrt(0.5454545454545454)*pow(min_val, 2)*(913 * pow(max_val, 4) + 4444 * pow(max_val, 3)*min_val + 6720 * pow(max_val, 2)*pow(min_val, 2) + 3731 * max_val*pow(min_val, 3) + 637 * pow(min_val, 4))) / pow(max_val - min_val, 12);
    coeffs[10][9] = (-111435 * sqrt(0.5454545454545454)*pow(min_val, 2)*(352 * pow(max_val, 3) + 1164 * pow(max_val, 2)*min_val + 1071 * max_val*pow(min_val, 2) + 273 * pow(min_val, 3))) / pow(max_val - min_val, 12);
    coeffs[10][10] = (334305 * sqrt(0.5454545454545454)*pow(min_val, 2)*(74 * pow(max_val, 2) + 149 * max_val*min_val + 63 * pow(min_val, 2))) / pow(max_val - min_val, 12);
    coeffs[10][11] = (-334305 * sqrt(0.5454545454545454)*pow(min_val, 2)*(27 * max_val + 25 * min_val)) / pow(max_val - min_val, 12);
    coeffs[10][12] = (1448655 * sqrt(0.5454545454545454)*pow(min_val, 2)) / pow(max_val - min_val, 12);
    coeffs[10][13] = 0;
    coeffs[11][0] = -(sqrt(82.16666666666667)*pow(max_val, 2)*pow(min_val, 2)*(pow(max_val, 11) + 88 * pow(max_val, 10)*min_val + 2200 * pow(max_val, 9)*pow(min_val, 2) + 23100 * pow(max_val, 8)*pow(min_val, 3) + 120120 * pow(max_val, 7)*pow(min_val, 4) + 336336 * pow(max_val, 6)*pow(min_val, 5) + 528528 * pow(max_val, 5)*pow(min_val, 6) + 471900 * pow(max_val, 4)*pow(min_val, 7) + 235950 * pow(max_val, 3)*pow(min_val, 8) + 62920 * pow(max_val, 2)*pow(min_val, 9) + 8008 * max_val*pow(min_val, 10) + 364 * pow(min_val, 11))) / (4.*pow(max_val - min_val, 13));
    coeffs[11][1] = (sqrt(82.16666666666667)*max_val*pow(min_val, 2)*(101 * pow(max_val, 11) + 5456 * pow(max_val, 10)*min_val + 93500 * pow(max_val, 9)*pow(min_val, 2) + 711480 * pow(max_val, 8)*pow(min_val, 3) + 2762760 * pow(max_val, 7)*pow(min_val, 4) + 5861856 * pow(max_val, 6)*pow(min_val, 5) + 7002996 * pow(max_val, 5)*pow(min_val, 6) + 4719000 * pow(max_val, 4)*pow(min_val, 7) + 1746030 * pow(max_val, 3)*pow(min_val, 8) + 331760 * pow(max_val, 2)*pow(min_val, 9) + 28028 * max_val*pow(min_val, 10) + 728 * pow(min_val, 11))) / (4.*pow(max_val - min_val, 13));
    coeffs[11][2] = -(sqrt(82.16666666666667)*pow(min_val, 2)*(1667 * pow(max_val, 11) + 61754 * pow(max_val, 10)*min_val + 767360 * pow(max_val, 9)*pow(min_val, 2) + 4363590 * pow(max_val, 8)*pow(min_val, 3) + 12852840 * pow(max_val, 7)*pow(min_val, 4) + 20762742 * pow(max_val, 6)*pow(min_val, 5) + 18762744 * pow(max_val, 5)*pow(min_val, 6) + 9390810 * pow(max_val, 4)*pow(min_val, 7) + 2492490 * pow(max_val, 3)*pow(min_val, 8) + 318890 * pow(max_val, 2)*pow(min_val, 9) + 16016 * max_val*pow(min_val, 10) + 182 * pow(min_val, 11))) / (2.*pow(max_val - min_val, 13));
    coeffs[11][3] = (11 * sqrt(739.5)*pow(min_val, 2)*(809 * pow(max_val, 10) + 21740 * pow(max_val, 9)*min_val + 201990 * pow(max_val, 8)*pow(min_val, 2) + 871920 * pow(max_val, 7)*pow(min_val, 3) + 1957410 * pow(max_val, 6)*pow(min_val, 4) + 2395484 * pow(max_val, 5)*pow(min_val, 5) + 1611610 * pow(max_val, 4)*pow(min_val, 6) + 580840 * pow(max_val, 3)*pow(min_val, 7) + 104520 * pow(max_val, 2)*pow(min_val, 8) + 8060 * max_val*pow(min_val, 9) + 182 * pow(min_val, 10))) / (2.*pow(max_val - min_val, 13));
    coeffs[11][4] = (-1045 * sqrt(739.5)*pow(min_val, 2)*(157 * pow(max_val, 9) + 3156 * pow(max_val, 8)*min_val + 22272 * pow(max_val, 7)*pow(min_val, 2) + 73332 * pow(max_val, 6)*pow(min_val, 3) + 124852 * pow(max_val, 5)*pow(min_val, 4) + 113932 * pow(max_val, 4)*pow(min_val, 5) + 55328 * pow(max_val, 3)*pow(min_val, 6) + 13572 * pow(max_val, 2)*pow(min_val, 7) + 1482 * max_val*pow(min_val, 8) + 52 * pow(min_val, 9))) / (4.*pow(max_val - min_val, 13));
    coeffs[11][5] = (209 * sqrt(739.5)*pow(min_val, 2)*(4569 * pow(max_val, 8) + 69792 * pow(max_val, 7)*min_val + 375900 * pow(max_val, 6)*pow(min_val, 2) + 939400 * pow(max_val, 5)*pow(min_val, 3) + 1193920 * pow(max_val, 4)*pow(min_val, 4) + 787696 * pow(max_val, 3)*pow(min_val, 5) + 260988 * pow(max_val, 2)*pow(min_val, 6) + 39000 * max_val*pow(min_val, 7) + 1950 * pow(min_val, 8))) / (4.*pow(max_val - min_val, 13));
    coeffs[11][6] = (-1463 * sqrt(739.5)*pow(min_val, 2)*(633 * pow(max_val, 7) + 7383 * pow(max_val, 6)*min_val + 30200 * pow(max_val, 5)*pow(min_val, 2) + 56385 * pow(max_val, 4)*pow(min_val, 3) + 51870 * pow(max_val, 3)*pow(min_val, 4) + 23387 * pow(max_val, 2)*pow(min_val, 5) + 4732 * max_val*pow(min_val, 6) + 325 * pow(min_val, 7))) / pow(max_val - min_val, 13);
    coeffs[11][7] = (209 * sqrt(739.5)*pow(min_val, 2)*(11814 * pow(max_val, 6) + 104698 * pow(max_val, 5)*min_val + 320155 * pow(max_val, 4)*pow(min_val, 2) + 433020 * pow(max_val, 3)*pow(min_val, 3) + 272545 * pow(max_val, 2)*pow(min_val, 4) + 75166 * max_val*pow(min_val, 5) + 7007 * pow(min_val, 6))) / pow(max_val - min_val, 13);
    coeffs[11][8] = (-4807 * sqrt(739.5)*pow(min_val, 2)*(3817 * pow(max_val, 5) + 25300 * pow(max_val, 4)*min_val + 56080 * pow(max_val, 3)*pow(min_val, 2) + 51940 * pow(max_val, 2)*pow(min_val, 3) + 20020 * max_val*pow(min_val, 4) + 2548 * pow(min_val, 5))) / (4.*pow(max_val - min_val, 13));
    coeffs[11][9] = (24035 * sqrt(82.16666666666667)*pow(min_val, 2)*(2959 * pow(max_val, 4) + 14224 * pow(max_val, 3)*min_val + 21604 * pow(max_val, 2)*pow(min_val, 2) + 12264 * max_val*pow(min_val, 3) + 2184 * pow(min_val, 4))) / (4.*pow(max_val - min_val, 13));
    coeffs[11][10] = (-24035 * sqrt(82.16666666666667)*pow(min_val, 2)*(1303 * pow(max_val, 3) + 4294 * pow(max_val, 2)*min_val + 4000 * max_val*pow(min_val, 2) + 1050 * pow(min_val, 3))) / (2.*pow(max_val - min_val, 13));
    coeffs[11][11] = (28405 * sqrt(82.16666666666667)*pow(min_val, 2)*(631 * pow(max_val, 2) + 1276 * max_val*min_val + 550 * pow(min_val, 2))) / (2.*pow(max_val - min_val, 13));
    coeffs[11][12] = (-85215 * sqrt(739.5)*pow(min_val, 2)*(47 * max_val + 44 * min_val)) / (4.*pow(max_val - min_val, 13));
    coeffs[11][13] = (596505 * sqrt(739.5)*pow(min_val, 2)) / (4.*pow(max_val - min_val, 13));
}

Basis_Shapeev::Basis_Shapeev(double _min_val, double _max_val, int _size)
    : AnyBasis(_min_val, _max_val, _size)
{
    if (size > ALLOCATED_DEGREE + 1)
        ERROR("AnyBasis error: allocated degree ecceded.");
    InitShapeevRB();
}

Basis_Shapeev::Basis_Shapeev(std::ifstream & ifs)
    : AnyBasis(ifs)
{
    if (size > ALLOCATED_DEGREE + 1)
        ERROR("AnyBasis error: allocated degree ecceded.");
    InitShapeevRB();
}

void Basis_Shapeev::Calc(double val)
{

#ifdef MLIP_DEBUG
    if (val < min_val) {
        Warning("AnyBasis: val<min_val. val = " + to_string(val) +
            ", min_val = " + to_string(min_val) + '\n');
    }
    if (val > max_val) {
        ERROR("AnyBasis: val>max_val !!!. val = " + to_string(val) +
            ", max_val = " + to_string(max_val) + '\n');
    }
#endif

    for (int xi = 0; xi < size; xi++) {
        vals[xi] = 0;
        for (int i = -2; i <= xi; i++) {
            vals[xi] += scaling * coeffs[xi][i + 2] * pow(val, i);
        }
    }
}

void Basis_Shapeev::CalcDers(double val)
{
    Basis_Shapeev::Calc(val);

    for (int xi = 0; xi < size; xi++) {
        ders[xi] = 0;
        for (int i = -2; i <= xi; i++) {
            ders[xi] += scaling * coeffs[xi][i + 2] * i * pow(val, i - 1);
        }
    }
}

void Basis_Chebyshev::Calc(double val)
{

#ifdef MLIP_DEBUG
    if (val < min_val) {
        Warning("AnyBasis: val<min_val. val = " + to_string(val) +
            ", min_val = " + to_string(min_val) + '\n');
    }
    if (val > max_val) {
        ERROR("AnyBasis: val>max_val !!!. val = " + to_string(val) +
            ", max_val = " + to_string(max_val) + '\n');
    }
#endif

    double ksi = (2 * val - (min_val + max_val)) / (max_val - min_val);

    vals[0] = scaling * 1;
    vals[1] = scaling * ksi;
    for (int i = 2; i < size; i++) {
        vals[i] = 2 * ksi * vals[i - 1] - vals[i - 2];
    }
}

void Basis_Chebyshev::CalcDers(double val)
{

    Basis_Chebyshev::Calc(val);

    double mult = 2.0 / (max_val - min_val);
    double ksi = (2 * val - (min_val + max_val)) / (max_val - min_val);

    ders[0] = scaling * 0;
    ders[1] = scaling * mult;
    for (int i = 2; i < size; i++) {
        ders[i] = 2 * (mult * vals[i - 1] + ksi * ders[i - 1]) - ders[i - 2];
    }
}

void RB_Chebyshev::Calc(double val)
{

#ifdef MLIP_DEBUG
    if (val < min_val) {
        Warning("AnyBasis: val<min_val. val = " + to_string(val) +
            ", min_val = " + to_string(min_val) + '\n');
    }
    if (val > max_val) {
        ERROR("AnyBasis: val>max_val !!!. val = " + to_string(val) +
            ", max_val = " + to_string(max_val) + '\n');
    }
#endif

    double mult = 2.0 / (max_val - min_val);
    double ksi = (2 * val - (min_val + max_val)) / (max_val - min_val);

    vals[0] = scaling * (1 * (val - max_val)*(val - max_val));
    vals[1] = scaling * (ksi*(val - max_val)*(val - max_val));
    for (int i = 2; i < size; i++) {
        vals[i] = 2 * ksi*vals[i - 1] - vals[i - 2];
    }
}

void RB_Chebyshev::CalcDers(double val)
{

    RB_Chebyshev::Calc(val);

    double mult = 2.0 / (max_val - min_val);
    double ksi = (2 * val - (min_val + max_val)) / (max_val - min_val);

    ders[0] = scaling * (0 * (val - max_val)*(val - max_val) + 2 * (val - max_val));
    ders[1] = scaling * (mult * (val - max_val)*(val - max_val) + 2 * ksi*(val - max_val));
    for (int i = 2; i < size; i++) {
        ders[i] = 2 * (mult * vals[i - 1] + ksi * ders[i - 1]) - ders[i - 2];
    }
}

void Basis_Chebyshev_repuls::Calc(double val)
{

#ifdef MLIP_DEBUG
    if (val < min_val) {
        Warning("AnyBasis: val<min_val. val = " + to_string(val) +
            ", min_val = " + to_string(min_val) + '\n');
    }
    if (val > max_val) {
        ERROR("AnyBasis: val>max_val !!!. val = " + to_string(val) +
            ", max_val = " + to_string(max_val) + '\n');
    }
#endif

    if (val < min_val)
        val = min_val;

    double ksi = (2 * val - (min_val + max_val)) / (max_val - min_val);

    vals[0] = scaling * 1;
    vals[1] = scaling * ksi;
    for (int i = 2; i < size; i++) {
        vals[i] = 2 * ksi * vals[i - 1] - vals[i - 2];
    }
}

void Basis_Chebyshev_repuls::CalcDers(double val)
{

    Basis_Chebyshev_repuls::Calc(val);

    if (val < min_val)
        val = min_val;

    double mult = 2.0 / (max_val - min_val);
    double ksi = (2 * val - (min_val + max_val)) / (max_val - min_val);

    ders[0] = scaling * 0;
    ders[1] = scaling * mult;
    for (int i = 2; i < size; i++) {
        ders[i] = 2 * (mult * vals[i - 1] + ksi * ders[i - 1]) - ders[i - 2];
    }
    if (val == min_val)
        for (int i = 0; i < size; i++)
            ders[i] = 0.0;
}

void RB_Chebyshev_repuls::Calc(double val)
{

#ifdef MLIP_DEBUG
    if (val < min_val) {
        Warning("AnyBasis: val<min_val. val = " + to_string(val) +
            ", min_val = " + to_string(min_val) + '\n');
    }
    if (val > max_val) {
        ERROR("AnyBasis: val>max_val !!!. val = " + to_string(val) +
            ", max_val = " + to_string(max_val) + '\n');
    }
#endif

    if (val < min_val)
        val = min_val;

    double mult = 2.0 / (max_val - min_val);
    double ksi = (2 * val - (min_val + max_val)) / (max_val - min_val);

    vals[0] = scaling * (1 * (val - max_val)*(val - max_val));
    vals[1] = scaling * (ksi*(val - max_val)*(val - max_val));
    for (int i = 2; i < size; i++) {
        vals[i] = 2 * ksi*vals[i - 1] - vals[i - 2];
    }
}

void RB_Chebyshev_repuls::CalcDers(double val)
{

    RB_Chebyshev_repuls::Calc(val);

    if (val < min_val)
        val = min_val;

    double mult = 2.0 / (max_val - min_val);
    double ksi = (2 * val - (min_val + max_val)) / (max_val - min_val);

    ders[0] = scaling * (0 * (val - max_val)*(val - max_val) + 2 * (val - max_val));
    ders[1] = scaling * (mult * (val - max_val)*(val - max_val) + 2 * ksi*(val - max_val));
    for (int i = 2; i < size; i++) 
        ders[i] = 2 * (mult * vals[i - 1] + ksi * ders[i - 1]) - ders[i - 2];
    
    if (val == min_val)
        for (int i = 0; i < size; i++)
            ders[i] = 0.0;
}

void Basis_Taylor::Calc(double val)
{

#ifdef MLIP_DEBUG
    if (val < min_val) {
        Warning("AnyBasis: val<min_val. val = " + to_string(val) +
            ", min_val = " + to_string(min_val) + '\n');
    }
    if (val > max_val) {
        ERROR("AnyBasis: val>max_val !!!. val = " + to_string(val) +
            ", max_val = " + to_string(max_val) + '\n');
    }
#endif

    vals[0] = scaling * 1;
    for (int i = 1; i < size; i++)
    {
        vals[i] = scaling * val * vals[i - 1];
    }
}

void Basis_Taylor::CalcDers(double val)
{

    Basis_Taylor::Calc(val);

    ders[0] = scaling * 0;
    for (int i = 1; i < size; i++)
    {
        ders[i] = scaling * i * vals[i - 1];
    }
}


