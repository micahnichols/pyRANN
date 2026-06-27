/*
 * pybind11 bindings for MLIP MLMTPR (mtpr.h version)
 * Drop-in replacement for previous MTP bindings.
 */

#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <memory>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "mtpr.h"
#include "configuration.h"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// MLMTPR accessor (fix protected members)
// ---------------------------------------------------------------------------
struct MLMTPRAccessor : public MLMTPR {
    using MLMTPR::CalcBasisFuncs;
    using MLMTPR::CalcBasisFuncsDers;
    using MLMTPR::CalcSiteEnergyDers;
    using MLMTPR::SiteEnergy;

    using MLMTPR::basis_vals;
    using MLMTPR::basis_ders;
    using MLMTPR::alpha_scalar_moments;
    /* using MLMTPR::p_RadialBasis->max_val; */

    MLMTPRAccessor(const std::string& fn) : MLMTPR(fn) {}
};

// ---------------------------------------------------------------------------
// numpy helper
// ---------------------------------------------------------------------------
static py::array_t<double>
make_array(const double* data, std::vector<ssize_t> shape)
{
    ssize_t total = 1;
    for (auto s : shape) total *= s;

    auto arr = py::array_t<double>(shape);
    std::memcpy(arr.mutable_data(), data, sizeof(double) * total);
    return arr;
}

// ---------------------------------------------------------------------------
// PyNeighborhood
// ---------------------------------------------------------------------------
struct PyNeighborhood {
    Neighborhood nbh;

    explicit PyNeighborhood(const Neighborhood& src) : nbh(src) {}
};

// ---------------------------------------------------------------------------
// PyConfiguration (MULTI-CONFIG FIX)
// ---------------------------------------------------------------------------
struct PyConfiguration {
    std::vector<Configuration> cfgs;

    explicit PyConfiguration(const std::string& filename) {
        constexpr int MAX_COUNT = 1'000'000;
        cfgs = LoadCfgs(filename, MAX_COUNT);

        if (cfgs.empty())
            throw std::runtime_error("No configurations loaded.");
    }

    int ncfg() const { return (int)cfgs.size(); }

    int size(int i = 0) const { return cfgs[i].size(); }

    void init_nbhs(double cutoff) {
        for (auto& c : cfgs)
            c.InitNbhs(cutoff);
    }

    // ---------------- positions (BATCH SAFE) ----------------
    py::list positions() const {
        py::list out;

        for (const auto& cfg : cfgs) {
            int n = cfg.size();
            auto arr = py::array_t<double>({n, 3});
            auto r = arr.mutable_unchecked<2>();

            for (int i = 0; i < n; i++)
                for (int a = 0; a < 3; a++)
                    r(i, a) = cfg.pos(i, a);

            out.append(arr);
        }
        return out;
    }

    // ---------------- types ----------------
    py::list types() const {
        py::list out;

        for (const auto& cfg : cfgs) {
            int n = cfg.size();
            auto arr = py::array_t<int>(n);
            auto r = arr.mutable_unchecked<1>();

            for (int i = 0; i < n; i++)
                r(i) = cfg.type(i);

            out.append(arr);
        }
        return out;
    }

    // ---------------- nbhs ----------------
    py::list nbhs(int cfg_index = 0) const {
        const auto& cfg = cfgs[cfg_index];

        py::list out;
        for (const auto& nbh : cfg.nbhs)
            out.append(PyNeighborhood(nbh));

        return out;
    }

    std::string repr() const {
        return "<Configuration dataset ncfg=" + std::to_string(cfgs.size()) + ">";
    }
};

// ---------------------------------------------------------------------------
// PyMTPR (UPDATED FOR MLMTPR)
// ---------------------------------------------------------------------------
struct PyMTP {
    MLMTPRAccessor mtp;

    explicit PyMTP(const std::string& fn) : mtp(fn) {}

    double cutoff() const { return (double)mtp.p_RadialBasis->max_val; }
    // ---------------- basis values ----------------
    py::array_t<double> calc_basis_funcs(PyNeighborhood& pnbh)
    {
        std::vector<double> out;
	int nbf = mtp.alpha_count;
	/* std::vector<double> out(nbf); */
	out.resize(nbf);
	/* for (const auto& val : out) { */
	/* 	std::cout << val << " "; */
	/* } */

        mtp.CalcBasisFuncs(pnbh.nbh, out);

        return py::array_t<double>(
            nbf, out.data()
        );
    }

    // ---------------- basis + derivatives ----------------
    py::tuple calc_basis_funcs_ders(PyNeighborhood& pnbh)
    {
        mtp.CalcBasisFuncsDers(pnbh.nbh);

        auto& vals = mtp.basis_vals;
        auto& ders = mtp.basis_ders;

        int alpha = (int)vals.size();
        int n = pnbh.nbh.count;

        py::array_t<double> v({alpha});
        std::memcpy(v.mutable_data(), vals.data(), sizeof(double) * alpha);

        py::array_t<double> d({alpha, n, 3});
        std::memcpy(d.mutable_data(), ders.data(), sizeof(double) * alpha * n * 3);

        return py::make_tuple(v, d);
    }

    // ---------------- energy + forces ----------------
};

// ---------------------------------------------------------------------------
// MODULE
// ---------------------------------------------------------------------------
PYBIND11_MODULE(mtp_bindings, m)
{
    m.doc() = "MLIP MLMTPR bindings (mtpr.h version)";

    // ---------------- Neighborhood ----------------
    py::class_<PyNeighborhood>(m, "Neighborhood")
        .def(py::init<const Neighborhood&>());

    // ---------------- Configuration ----------------
    py::class_<PyConfiguration>(m, "Configuration")
        .def(py::init<const std::string&>())
        .def("init_nbhs", &PyConfiguration::init_nbhs)
        .def_property_readonly("positions", &PyConfiguration::positions)
        .def_property_readonly("types", &PyConfiguration::types)
        .def("nbhs", &PyConfiguration::nbhs)
	.def("ncfg", &PyConfiguration::ncfg) 
        .def("__repr__", &PyConfiguration::repr);

    // ---------------- MLMTPR ----------------
    py::class_<PyMTP>(m, "MTP")
        .def(py::init<const std::string&>())
	.def("cutoff", &PyMTP::cutoff)
        .def("calc_basis_funcs", &PyMTP::calc_basis_funcs)
        .def("calc_basis_funcs_ders", &PyMTP::calc_basis_funcs_ders);
}
