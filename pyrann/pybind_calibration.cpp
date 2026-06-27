#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "pair_spin_rann.h"

#include <cstring>

namespace py = pybind11;
using namespace LAMMPS_NS;

PYBIND11_MODULE(calibration, m) {
    m.doc() = "Calibration module. Run a = pyrann.calibration.PairRANN('input_file') then a.setup() to have the c++ code calculate fingerprints for all dump files in input file's dumpdirectory. The fingerprints can then be accessed by using the following classes and methods.";

    /* =========================
       Simulation
       ========================= */

    py::class_<PairRANN::Simulation>(m, "Simulation", "Class for accessing some per-simulation properties. Once setup() has been run, this can be used. setup() produces a list of simulation objects")
        .def_readwrite("inum", &PairRANN::Simulation::inum, "Number of atoms. a[sim_num].inum is the number of atoms for global simulation number sim_num.")
        .def_readwrite("energy", &PairRANN::Simulation::energy, "Energy of a simulation. a[sim_num].energy is the energy of the global simulation number sim_num.")
	/* .def_readwrite("type", &PairRANN::Simulation::type) */
	/* .def_readwrite("natoms", &PairRANN::Simulation::natoms) */

        .def_property_readonly("filename", [](PairRANN::Simulation &s) {
            return s.filename ? py::cast(s.filename) : py::none();
        },
	"Filename for a simulation number. a[sim_num].filename is the dump file filename for global simulation number sim_num.")

        /* Safe per-atom feature copy */
        .def("feature",
             [](PairRANN::Simulation &s, PairRANN &p, int atom) {
                 if (!s.features)
                     throw std::runtime_error("Simulation.features is null");

                 if (atom < 0 || atom >= s.inum)
                     throw py::index_error("Atom index out of range");

                 int type = s.type[atom];
                 int elem = p.map[type];
                 int nfeatures = p.fingerprintlength[elem];

                 if (nfeatures <= 0)
                     throw std::runtime_error("Invalid fingerprint length");

                 py::array_t<double> arr({nfeatures});
                 std::memcpy(
                     arr.mutable_data(),
                     s.features[atom],
                     nfeatures * sizeof(double)
                 );
                 return arr;
             },
             py::arg("pair"),
             py::arg("atom"),
	     "Array of fingerprint (non-normalized) values for a given atom. a[sim_num].feature(atom_id) is an array of fingerprints for atom atom_id (local) in simulation sim_num (global)\n\n"
	     ":param atom: Local atom id to retrieve fingerprint values.\n"
	     ":type atom: int\n"
	     ":return: Array of non-normalized fingerprint values for given local atom id in given glocal simulation number.\n"
	     ":usage: a = calibration.PairRANN.setup('input_file') -> a[global_sim_num].feature(local_atom_id).")
	.def_property_readonly("types",
	     [](PairRANN::Simulation &s) {
	         if (!s.type)
	     	     throw std::runtime_error("Simulation.type is null");
	         int type_len = s.inum;
	         if (type_len <= 0)
	             throw std::runtime_error("Invalid type length");

	         py::array_t<int> arr({type_len});
                 std::memcpy(
                     arr.mutable_data(),
                     s.type,
                     type_len * sizeof(int)
                 );
                 return arr;
             },
	     R"(Array of atom types for a simulation. a[sim_num].types is the array of atom types for global simulation number sim_num)"
	     );

    
        /* .def("types", */
        /*      [](PairRANN::Simulation &s, PairRANN &p) { */
        /*          if (!s.type) */
        /*              throw std::runtime_error("Simulation.type is null"); */

        /*          /1* if (atom < 0 || atom >= s.inum) *1/ */
        /*          /1*     throw py::index_error("Atom index out of range"); *1/ */

        /*          /1* int type = s.type[atom]; *1/ */
        /*          /1* int elem = p.map[type]; *1/ */
        /*          /1* int nfeatures = p.fingerprintlength[elem]; *1/ */
		 /* int type_len = s.inum; */

        /*          if (type_len <= 0) */
        /*              throw std::runtime_error("Invalid type length"); */

        /*          py::array_t<int> arr({type_len}); */
        /*          std::memcpy( */
        /*              arr.mutable_data(), */
        /*              s.type, */
        /*              type_len * sizeof(int) */
        /*          ); */
        /*          return arr; */
        /*      }, */
        /*      py::arg("pair")); */
             /* py::arg("atom")); */

    /* =========================
       PairRANN
       ========================= */

    /* py::class_<PairRANN>(m, "PairRANN") */
    py::class_<PairRANN, std::unique_ptr<PairRANN, py::nodelete>>(m, "PairRANN", "Class for accessing RANN calibration code. To use methods, you must set a variable equal to calibration.PairRANN('input_file') first. Then run methods on the chosen variable.")
        /* Safer constructor: no raw allocation */
        .def(py::init([](const std::string &arg) {
            return new PairRANN(const_cast<char*>(arg.c_str()));
        }), py::arg("arg"))

        .def("setup", &PairRANN::setup, py::call_guard<py::gil_scoped_release>(), "Set up RANN calibration code. Use this to obtain fingerprint values. calibration.PairRANN.setup().")
        .def("run", &PairRANN::run, "Run Calibration code to train a RANN potential. Ending training early using ctrl+c. The code checks for this command every 10 epochs. calibration.PairRANN.run().")
        .def("finish", &PairRANN::finish, "Clean up after running calibration code. calibration.PairRANN.finish().")

        .def_property_readonly("nsims", [](PairRANN &p) {
            return p.nsims;
        },
	"Total number of simulations read. calibration.PairRANN.nsims.")
	.def_property_readonly("natoms", [](PairRANN &p) {
	    return p.natoms;
	},
	"Total number of atoms read. calibration.PairRANN.natoms.")

        .def("__len__", [](PairRANN &p) {
            return p.nsims;
        })

        /* Return Simulation by reference, but only for immediate use */
        .def("__getitem__",
             [](PairRANN &p, int i) -> PairRANN::Simulation& {
                 if (i < 0 || i >= p.nsims)
                     throw py::index_error("Simulation index out of range");
                 return p.sims[i];
             },
             py::return_value_policy::reference_internal)

	.def_property_readonly("sims_per_file",
	     [](PairRANN &p) {
	         /* if (!s.type) */
	     	     /* throw std::runtime_error("Simulation.type is null"); */
	         /* int type_len = s.inum; */
		 int sim_len = p.nsets;
	         if (sim_len <= 0)
	             throw std::runtime_error("Invalid type length");

	         py::array_t<int> arr({sim_len});
                 std::memcpy(
                     arr.mutable_data(),
                     p.Xset,
                     sim_len * sizeof(int)
                 );
                 return arr;
             },
	     "An array of simulations per dump file. Array is ordered in the order the calibration code reads dump files (see calibration code std::out or Simulation class filename property for order). calibration.PairRANN.sims_per_file."
	     )

        /* Safe convenience feature access */
        .def("feature",
             [](PairRANN &p, int sim, int atom) {
                 if (sim < 0 || sim >= p.nsims)
                     throw py::index_error("Simulation index out of range");

                 auto &s = p.sims[sim];

                 if (!s.features)
                     throw std::runtime_error("Simulation.features is null");

                 if (atom < 0 || atom >= s.inum)
                     throw py::index_error("Atom index out of range");

                 int type = s.type[atom];
                 int elem = p.map[type];
                 int nfeatures = p.fingerprintlength[elem];

                 py::array_t<double> arr({nfeatures});
                 std::memcpy(
                     arr.mutable_data(),
                     s.features[atom],
                     nfeatures * sizeof(double)
                 );
                 return arr;
             },
             py::arg("sim"),
             py::arg("atom"),
	     "Array of fingerprint (non-normalized) values for a given simulation (global) for a given atom id (local). PairRANN.feature(sim_num, atom_id).\n\n"
	     ":param sim: Global simulation number.\n"
	     ":type sim: int\n"
	     ":param atom: Local atom id.\n"
	     ":type atom: int\n"
	     ":return: Array of non-normalized fingerprint values for a given local atom id in a given global simulation number.\n"
	     ":usage: calibration.PairRANN.feature(global_sim_num, local_atom_id)")
	     /* R"(Array of fingerprint (non-normalized) values for a given simulation (global) for a given atom id (local). PairRANN.feature(sim_num, atom_id). */
	/* Args: */
	/* :param sim (int): Global simulation number. */
	/* :param atom (int): Local atom id. */
	/* Returns: */
	/* Array of non-normalized fingerprint values for given local atom id in given global simulation number.)") */

        .def("force",
             [](PairRANN &p, int sim, int atom) {
                 if (sim < 0 || sim >= p.nsims)
                     throw py::index_error("Simulation index out of range");

                 auto &s = p.sims[sim];

                 if (!s.features)
                     throw std::runtime_error("Simulation.f is null");

                 if (atom < 0 || atom >= s.inum)
                     throw py::index_error("Atom index out of range");

                 int type = s.type[atom];
                 int elem = p.map[type];
                 /* int nfeatures = p.fingerprintlength[elem]; */
		 int nforces = 3;

                 py::array_t<double> arr({nforces});
                 std::memcpy(
                     arr.mutable_data(),
                     s.f[atom],
                     nforces * sizeof(double)
                 );
                 return arr;
             },
             py::arg("sim"),
             py::arg("atom"),
	     "Array of forces for a given simulation (global) for a given atom id (local). PairRANN.force(sim_num, atom_id).\n\n"
	     ":param sim: Global simulation number.\n"
	     ":type sim: int\n"
	     ":param atom: Local atom id.\n"
	     ":type atom: int\n"
	     ":return: Array of forces for a given local atom id in a given global simulation number.\n"
	     ":usage: calibration.PairRANN.force(global_sim_num, local_atom_id)")

        .def("types",
             [](PairRANN &p, int sim) {
                 if (sim < 0 || sim >= p.nsims)
                     throw py::index_error("Simulation index out of range");

                 auto &s = p.sims[sim];

                 if (!s.type)
                     throw std::runtime_error("Simulation.type is null");

                 /* if (atom < 0 || atom >= s.inum) */
                 /*     throw py::index_error("Atom index out of range"); */

                 /* int type = s.type[atom]; */
                 /* int elem = p.map[type]; */
                 /* int nfeatures = p.fingerprintlength[elem]; */
		 int type_len = s.inum;

                 py::array_t<int> arr({type_len});
                 std::memcpy(
                     arr.mutable_data(),
                     s.type,
                     type_len * sizeof(int)
                 );
                 return arr;
             },
             py::arg("sim"),
	     "Array of atom types for a given simulation (global). PairRANN.types(sim_num).\n\n"
	     ":param sim: Global simulation number.\n"
	     ":type sim: int\n"
	     ":return: Array of atom types in a given global simulation number.\n"
	     ":usage: calibration.PairRANN.types(global_sim_num)")
             /* py::arg("atom")); */

        .def("firstneigh",
             [](PairRANN &p, int sim, int atom) {
                 if (sim < 0 || sim >= p.nsims)
                     throw py::index_error("Simulation index out of range");

                 auto &s = p.sims[sim];

                 if (!s.features)
                     throw std::runtime_error("Simulation.firstneigh is null");

                 if (atom < 0 || atom >= s.inum)
                     throw py::index_error("Atom index out of range");

                 int type = s.type[atom];
                 int elem = p.map[type];
		 int numneigh = s.numneigh[atom];
                 /* int nfeatures = p.fingerprintlength[elem]; */

                 py::array_t<int> arr({numneigh});
                 std::memcpy(
                     arr.mutable_data(),
                     s.firstneigh[atom],
                     numneigh * sizeof(int)
                 );
                 return arr;
             },
             py::arg("sim"),
             py::arg("atom"),
	     "Array of neighbors for a given atom id (local) in a given simulation (global). PairRANN.firstneigh(sim_num, atom_id).\n\n"
	     ":param sim: Global simulation number.\n"
	     ":type sim: int\n"
	     ":param atom: Local atom id.\n"
	     ":type atom: int\n"
	     ":return: Array of atom id's for the neighbors for a given local atom id in a given global simulation number.\n"
	     ":usage: calibration.PairRANN.firstneigh(global_sim_num, local_atom_id)")

        .def("id",
             [](PairRANN &p, int sim) {
                 if (sim < 0 || sim >= p.nsims)
                     throw py::index_error("Simulation index out of range");

                 auto &s = p.sims[sim];

                 if (!s.id)
                     throw std::runtime_error("Simulation.id is null");

                 /* if (atom < 0 || atom >= s.inum) */
                 /*     throw py::index_error("Atom index out of range"); */

                 /* int type = s.type[atom]; */
                 /* int elem = p.map[type]; */
		 int id_len = s.inum;
		 /* int numneigh = s.numneigh[atom]; */
                 /* int nfeatures = p.fingerprintlength[elem]; */

                 py::array_t<int> arr({id_len});
                 std::memcpy(
                     arr.mutable_data(),
                     s.id,
                     id_len * sizeof(int)
                 );
                 return arr;
             },
             py::arg("sim"),
	     "Array of atom id's (local) for a given simulation number (global). PairRANN.id(sim_num).\n\n"
	     ":param sim: Global simulation number.\n"
	     ":type sim: int\n"
	     ":return: Array of local atom id's in a given global simulation number.\n"
	     ":usage: calibration.PairRANN.id(global_sim_num)");
             /* py::arg("atom")) */
}

