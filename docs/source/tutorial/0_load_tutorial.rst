Introduction to pyRANN
======================


Micah Nichols, mwn45@msstate.edu

0) pyRANN Basics
----------------

pyRANN version 1.0.0a1 is the alpha verion and is still actively under
development. Please report any bugs or suggestions to Micah Nichols at 
mwn45@msstate.edu or micahwnichols45@gmail.com. 

pyRANN is divided into 3 main classes and a helper function. 

The helper function, pyrann.load(), serves to load in atomic structures 
from files. The current accepted filetypes are POSCAR, LAMMPS data, 
LAMMPS dump, and cfg. More filetypes may be added later.

The first class is pyrann.system(). This class initializes singular
atomic structures. pyrann.load(filename, series=False) initializes
an instance of the system class. The methods of this class perform
operations to alter the system. Some directly change the system, and
some create new system instances. The export method will export the 
system to any of the filetypes mentioned above. Please see the full
documentation for more information.

The second class is pyrann.series(). This class is made of a list of
pyrann.system() instances. pyrann.load(filename, series=True) 
initializes an instance of the series class. The only method of this
class is export. If the filetype is POSCAR or LAMMPS data, export will
produce a separate file for each system instance in the series. If the
filetype is LAMMPS dump or cfg, export will produce one file with multiple
timesteps. Please see the full documentation and tutorials for more information.

The third class is pyrann.processing(). This class is used to calculate
atomic descriptors, visualize descriptor space, and reduce redundant
databases using atomic descriptors. The current machine learning 
interatomic potential (MLIP) formalisms supported for atomic descriptor
calculation are RANN, MTP, and GAP. More formalisms may be added later.
If the user wishes to use another formalism and can access the atomic
descriptors, this class supports manually defining the descriptors 
using class_instance.features = user_features. Visualization makes use
of Uniform Manifold Approximation & Projection (UMAP) and bokeh
interactive plots. Please see the full documentation and tutorials
for more information.

1) Loading a File
-----------------

Let’s start by getting a POSCAR file for an FCC Al system from Materials
Project. You can either download the poscar directly or use the Python
API:

.. code:: ipython3

    from mp_api.client import MPRester
    %load_ext dotenv
    %dotenv
    import os
    
    api_key = os.getenv('MY_API_KEY')
    with MPRester(api_key) as mpr:
        search = mpr.materials.summary.search(
            formula = 'Al',
            theoretical = False,
            is_stable = True,
        )
        for i in search:
            print(f'Material Id: {i.material_id}')
            print(f'Theoretical: {i.theoretical}')
            print(f'Is Stable: {i.is_stable}')
            print(f'Symmetry: {i.symmetry}')



.. parsed-literal::

    Retrieving SummaryDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]


.. parsed-literal::

    Material Id: mp-134
    Theoretical: False
    Is Stable: True
    Symmetry: crystal_system=<CrystalSystem.cubic: 'Cubic'> symbol='Fm-3m' hall='-F 4 2 3' number=225 point_group='m-3m' symprec=0.1 angle_tolerance=5.0 version='2.7.0'


This returned one structure that is stable (on the convex hull). We can
now save it to a POSCAR file.

.. code:: ipython3

    sys = mpr.get_structure_by_material_id('mp-134')
    sys.to(filename='Al.POSCAR')



.. parsed-literal::

    Retrieving MaterialsDoc documents:   0%|          | 0/1 [00:00<?, ?it/s]




.. parsed-literal::

    'Al1\n1.0\n   2.4733290000000001    0.0000000000000000    1.4279770000000001\n   0.8244430000000000    2.3318770000000000    1.4279770000000001\n   0.0000000000000000    0.0000000000000000    2.8559549999999998\nAl\n1\ndirect\n   0.0000000000000000    0.0000000000000000    0.0000000000000000 Al\n'



Now, let’s import pyRANN and load the system from the file.

.. code:: ipython3

    import pyrann as rann
    
    fcc = rann.load('Al.POSCAR')
    print(fcc)


.. parsed-literal::

    Initializing pyrann version 1.0.0.a1
    box and atoms are column vectors
    System created from Al.POSCAR
    box = 
    [[2.473329 0.824443 0.      ]
     [0.       2.331877 0.      ]
     [1.427977 1.427977 2.855955]]
    natoms = 1
    elements = ['Al']
    timestep = None
    types = [1]
    atoms = 
    [[0.]
     [0.]
     [0.]]
    energy = 0
    forces = 
    None
    stress = 
    None


We now have an instance of the system class stored in the variable
“fcc”.

pyRANN supports loading in POSCAR, LAMMPS data, LAMMPS dump, and cfg
files. The load function attempts to determine the filetype by looking
at the extension. If it does not correctly assume the filetype, the load
function takes in an argument “filetype”.

