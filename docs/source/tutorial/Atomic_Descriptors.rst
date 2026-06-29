MLIP Atomic Descriptors with pyRANN
===================================

We can access the atomic descriptors of the RANN, MTP, and GAP MLIP
formalism with pyRANN. In order to access the MTP descriptors, please
make sure to install the optional pyrann_mtp package using one of the
following commands.

.. code:: bash

   pip install "pyrann[mtp]"
   pip install "pyrann[all]"

To access the descriptors, we first need to initialize a processing
class instance with the filename of each of the 3 formalism potential
files. For the RANN formalism, the file can be a completed potential
file or an input file that has yet to be trained.

.. code:: ipython3

    import pyrann as rann
    import numpy as np
    
    rann_pot = rann.processing('Mg.input')
    mtp_pot = rann.processing('pot4_6.0.mtp')
    gap_pot = rann.processing('GAP_Mg_7.0_4.0_3556_gs.xml')


.. parsed-literal::

    Initializing pyrann version 1.0.0.a1


6) Getting Atomic Descriptors
-----------------------------

Now, we can get the atomic descriptors for each of the formalisms. Let’s
start with RANN. 4 Mg LAMMPS dump files have been provided for this
tutorial to act as a psuedo-database. Make sure that only the dump files
you want to calculate the descriptors for are in the working directory.
The RANN formalism scans the entire working directory for any file or
directory with the lowercase word “dump”.

.. code:: ipython3

    rann_pot.get_features()
    print(rann_pot.features)


.. parsed-literal::

    
    # Number Threads     : 16
    [[ 3.38935976  3.02644257  2.25109709 ...  1.61972099  1.95561874
       1.68242089]
     [ 6.37313423  5.05768164  3.35657445 ...  8.02213011  8.21899046
       8.04288521]
     [ 1.16156767  1.51729725  1.84310613 ... -0.03339917  0.02580062
      -0.024193  ]
     ...
     [-1.07721352 -1.49743447 -1.95603472 ... -0.04121257 -0.11339892
      -0.04301757]
     [-0.32591448 -0.50494778 -0.74019567 ... -0.03560158 -0.06744371
      -0.03037547]
     [-0.6583386  -0.85380547 -1.04831626 ... -0.05904069 -0.10222123
      -0.05760668]]


Running the get_features method also sets the filename, global_sim_num,
local_sim_num, and local_id attributes of the processing class. These
are all arrays that describe the filename, global simulation number,
local (within-file) simulation number, and local (within-simulation) id
for each atom.

Now, let’s get the atomic descriptors for the same database using the
MTP formalism. We need to set the file_fmt argument of the get_features
method to ‘dump’ so that pyRANN knows to translate the LAMMPS dump files
to MTP’s native cfg format.

.. code:: ipython3

    mtp_pot.get_features(file_fmt='dump')
    print(mtp_pot.features)


.. parsed-literal::

    [[ 0.         -2.62030552 -2.9269448  ...  3.58905704  3.54220025
       3.41129791]
     [ 0.         -1.09789216 -1.64585973 ... -0.41706344 -0.69452364
      -0.57435477]
     [ 0.         -2.07964607 -2.00114841 ... -0.29337904 -0.45432535
      -0.38497632]
     ...
     [ 0.          1.57324701  1.55826347 ...  0.01165797 -0.00835375
      -0.04001837]
     [ 0.          2.17987415  2.19832042 ... -0.52227954 -0.54389097
      -0.55966454]
     [ 0.          1.53174407  1.4282988  ... -0.28838971 -0.29020163
      -0.29370493]]


Now we can do the same for the GAP (or SOAP) formalism. For this
formalism, we also need to specify the element symbol.

.. code:: ipython3

    gap_pot.get_features(file_fmt='dump', symbol='Mg')
    print(gap_pot.features)


.. parsed-literal::

    [[ 3.36212173e+00  4.02145133e+00  7.07412320e+00 ... -6.60682039e-01
      -6.03285115e-01  0.00000000e+00]
     [ 4.61939006e+00  6.29585082e+00  8.61582969e+00 ... -8.81417063e-01
      -4.70978834e-01  0.00000000e+00]
     [ 8.33206505e-01 -2.05046993e-01 -5.75611097e-03 ... -4.99541177e-01
      -7.24089657e-01  0.00000000e+00]
     ...
     [-5.55010506e-01  1.27548385e+00  9.35055344e-01 ...  2.06545172e+00
      -1.05686924e+00  0.00000000e+00]
     [ 3.53201390e-01  1.00131400e+00  1.06164819e+00 ...  2.03311307e+00
      -4.74448290e-01  0.00000000e+00]
     [-3.18842598e-01  1.48647634e-01  1.69396143e+00 ...  1.25940406e+00
       2.10516594e-01  0.00000000e+00]]


7) Visualizing Descriptor Space with UMAP
-----------------------------------------

Initializing UMAP
~~~~~~~~~~~~~~~~~

Now that the features have been calculated for each formalism, we can
visualize the descriptor space using UMAP. First, we need to initialize
UMAP for each formalism.

.. code:: ipython3

    rann_mapper = rann_pot.get_umap()
    mtp_mapper = mtp_pot.get_umap()
    gap_mapper = gap_pot.get_umap()



.. parsed-literal::

    UMAP(init='pca', n_neighbors=54, verbose=True)
    Sat Jun 27 11:08:29 2026 Construct fuzzy simplicial set
    Sat Jun 27 11:08:29 2026 Finding Nearest Neighbors
    Sat Jun 27 11:08:29 2026 Building RP forest with 20 trees
    Sat Jun 27 11:08:33 2026 NN descent for 16 iterations
    	 1  /  16
    	 2  /  16
    	Stopping threshold met -- exiting after 2 iterations
    Sat Jun 27 11:08:50 2026 Finished Nearest Neighbor Search
    Sat Jun 27 11:08:52 2026 Construct embedding



.. parsed-literal::

    Epochs completed:   0%|            0/200 [00:00]


.. parsed-literal::

    	completed  0  /  200 epochs
    	completed  20  /  200 epochs
    	completed  40  /  200 epochs
    	completed  60  /  200 epochs
    	completed  80  /  200 epochs
    	completed  100  /  200 epochs
    	completed  120  /  200 epochs
    	completed  140  /  200 epochs
    	completed  160  /  200 epochs
    	completed  180  /  200 epochs
    Sat Jun 27 11:09:01 2026 Finished embedding
    UMAP(init='pca', n_neighbors=54, verbose=True)
    Sat Jun 27 11:09:01 2026 Construct fuzzy simplicial set
    Sat Jun 27 11:09:01 2026 Finding Nearest Neighbors
    Sat Jun 27 11:09:01 2026 Building RP forest with 20 trees


.. parsed-literal::

    Sat Jun 27 11:09:02 2026 NN descent for 16 iterations
    	 1  /  16
    	 2  /  16
    	Stopping threshold met -- exiting after 2 iterations
    Sat Jun 27 11:09:13 2026 Finished Nearest Neighbor Search
    Sat Jun 27 11:09:14 2026 Construct embedding



.. parsed-literal::

    Epochs completed:   0%|            0/200 [00:00]


.. parsed-literal::

    	completed  0  /  200 epochs
    	completed  20  /  200 epochs
    	completed  40  /  200 epochs
    	completed  60  /  200 epochs
    	completed  80  /  200 epochs
    	completed  100  /  200 epochs
    	completed  120  /  200 epochs
    	completed  140  /  200 epochs
    	completed  160  /  200 epochs
    	completed  180  /  200 epochs
    Sat Jun 27 11:09:23 2026 Finished embedding


.. parsed-literal::

    UMAP(init='pca', n_neighbors=54, verbose=True)
    Sat Jun 27 11:09:24 2026 Construct fuzzy simplicial set
    Sat Jun 27 11:09:24 2026 Finding Nearest Neighbors
    Sat Jun 27 11:09:24 2026 Building RP forest with 20 trees
    Sat Jun 27 11:09:27 2026 NN descent for 16 iterations
    	 1  /  16
    	 2  /  16
    	 3  /  16
    	 4  /  16
    	Stopping threshold met -- exiting after 4 iterations
    Sat Jun 27 11:09:55 2026 Finished Nearest Neighbor Search
    Sat Jun 27 11:09:57 2026 Construct embedding



.. parsed-literal::

    Epochs completed:   0%|            0/200 [00:00]


.. parsed-literal::

    	completed  0  /  200 epochs
    	completed  20  /  200 epochs
    	completed  40  /  200 epochs
    	completed  60  /  200 epochs
    	completed  80  /  200 epochs
    	completed  100  /  200 epochs
    	completed  120  /  200 epochs
    	completed  140  /  200 epochs
    	completed  160  /  200 epochs
    	completed  180  /  200 epochs
    Sat Jun 27 11:10:16 2026 Finished embedding


Interactive Plotting - Categorical Coloring by Filename
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, we can produce interactive plots of the descriptor space for each
formalism colored by filename.

.. code:: ipython3

    rann_pot.plot(rann_mapper, color_data=rann_pot.filenames,
                  hover_info={'Global Simulation Number': rann_pot.global_sim_num,
                              'Local Simulation Number': rann_pot.local_sim_num,
                              'Local ID': rann_pot.local_id},
                  key_str='rann')
    mtp_pot.plot(mtp_mapper, color_data=mtp_pot.filenames,
                 hover_info={'Global Simulation Number': mtp_pot.global_sim_num,
                             'Local Simulation Number': mtp_pot.local_sim_num,
                             'Local ID': mtp_pot.local_id},
                 key_str='mtp')
    gap_pot.plot(gap_mapper, color_data=gap_pot.filenames,
                 hover_info={'Global Simulation Number': gap_pot.global_sim_num,
                             'Local Simulation Number': gap_pot.local_sim_num,
                             'Local ID': gap_pot.local_id},
                 key_str='gap')

RANN
^^^^

Click the colored circles on the legend to hide atoms from that specific
filename. Hover over data points to see information on that particular
atom.

.. code:: ipython3

    from IPython.display import IFrame
    
    IFrame(src='rann.html', width='100%', height=1000)




.. raw:: html

    
    <iframe
        width="100%"
        height="1000"
        src="_static/rann.html"
        frameborder="0"
        allowfullscreen
    
    ></iframe>




MTP
^^^

.. code:: ipython3

    IFrame(src='mtp.html', width='100%', height=1000)




.. raw:: html

    
    <iframe
        width="100%"
        height="1000"
        src="_static/mtp.html"
        frameborder="0"
        allowfullscreen
    
    ></iframe>




GAP
^^^

.. code:: ipython3

    IFrame(src='gap.html', width='100%', height=1000)




.. raw:: html

    
    <iframe
        width="100%"
        height="1000"
        src="_static/gap.html"
        frameborder="0"
        allowfullscreen
    
    ></iframe>




Interactive Plotting - Linear Coloring by energy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, we can produce interactive plots of the descriptor space for each
formalism colored by the energy per atom :math:`(\frac{eV}{atom})`. The
energy per atom is calculated by dividing the energy of each simulation
by the number of atoms in that simulation.

To ensure that the array of energies is in the correct order for each
formalism, we must first load in the database using pyrann.load in the
same order that each formalism reads the files for feature calculation.

.. code:: ipython3

    # Get unique filenames and the index the first unique instance appears
    rann_filenames, rann_index = np.unique(rann_pot.filenames, return_index=True)
    mtp_filenames, mtp_index = np.unique(mtp_pot.filenames, return_index=True)
    gap_filenames, gap_index = np.unique(gap_pot.filenames, return_index=True)
    
    # Sort the index in ascending order. This is to get the unique filenames in the order the formalism reads them.
    rann_index = np.argsort(rann_index)
    mtp_index = np.argsort(mtp_index)
    gap_index = np.argsort(gap_index)
    
    # Get unique filenames in the order that the formalism reads them.
    rann_unique_filenames = rann_filenames[rann_index]
    mtp_unique_filenames = mtp_filenames[mtp_index]
    gap_unique_filenames = gap_filenames[gap_index]
    
    print(f'{rann_unique_filenames = }\n{mtp_unique_filenames = }\n{gap_unique_filenames = }')


.. parsed-literal::

    rann_unique_filenames = array(['18_hcp_tetra.dump', '07_fcc_thermal.dump', '06_bcc_thermal.dump',
           '05_hcp_thermal.dump'], dtype='<U19')
    mtp_unique_filenames = array(['18_hcp_tetra.cfg', '07_fcc_thermal.cfg', '05_hcp_thermal.cfg',
           '06_bcc_thermal.cfg'], dtype='<U18')
    gap_unique_filenames = array(['18_hcp_tetra.dump', '07_fcc_thermal.dump', '06_bcc_thermal.dump',
           '05_hcp_thermal.dump'], dtype='<U19')


**As you can see, MTP differs in the order that it reads the input
files. This step is very important to ensure all analysis is done
correctly**

Now we can load in the database in the correct order and calculate the
energy per atom for each atom for each formalism.

.. code:: ipython3

    # Loading in the database in formalism specific orders
    rann_series = [rann.load(i, series=True) for i in rann_unique_filenames]
    mtp_series = [rann.load(i, series=True) for i in mtp_unique_filenames]
    gap_series = [rann.load(i, series=True) for i in gap_unique_filenames]
    
    # Calculating the energy per atom for each formalism specific order using list comprehension
    rann_epa = np.array([j.energy / j.natoms for i in rann_series for j in i.systems for _ in range(j.natoms)])
    mtp_epa = np.array([j.energy / j.natoms for i in mtp_series for j in i.systems for _ in range(j.natoms)])
    gap_epa = np.array([j.energy / j.natoms for i in gap_series for j in i.systems for _ in range(j.natoms)])

We can now plot the interactive plots colored by energy per atom by
setting the color_type argument to “linear” and setting the color_data
argument to the epa array for each formalism.

.. code:: ipython3

    rann_pot.plot(rann_mapper, color_data=rann_epa,
                  color_type='linear',
                  hover_info={'Filename': rann_pot.filenames,
                              'Global Simulation Number': rann_pot.global_sim_num,
                              'Local Simulation Number': rann_pot.local_sim_num,
                              'Local ID': rann_pot.local_id},
                  key_str='rann_epa')
    mtp_pot.plot(mtp_mapper, color_data=mtp_epa,
                  color_type='linear',
                  hover_info={'Filename': mtp_pot.filenames,
                              'Global Simulation Number': mtp_pot.global_sim_num,
                              'Local Simulation Number': mtp_pot.local_sim_num,
                              'Local ID': mtp_pot.local_id},
                  key_str='mtp_epa')
    gap_pot.plot(gap_mapper, color_data=gap_epa,
                  color_type='linear',
                  hover_info={'Filename': gap_pot.filenames,
                              'Global Simulation Number': gap_pot.global_sim_num,
                              'Local Simulation Number': gap_pot.local_sim_num,
                              'Local ID': gap_pot.local_id},
                  key_str='gap_epa')

RANN
^^^^

The interactive plots with linear coloring do not have an interactive
legend like the previous plots. Instead, they feature a colorbar on the
left and a dual slider at the bottom left of the figure. You can adjust
the dual slider to show only the energies within the slider range on the
plot. When you move the dual slider, the default option is for the
colorbar to not be rescaled. In order to more accurately see the
differences in the visible data, this can be turned on by setting the
rescale_colorbar argument to True. There are also seven colormaps to
choose from. The default is Viridis256. The other options include
Greys256, Inferno256, Magma256, Plasma256, Cividis256, and Turbo256.
Change the colormap by setting the cmap argument to one of the listed
colormaps.

.. code:: ipython3

    IFrame(src='rann_epa.html', width='100%', height=1000)




.. raw:: html

    
    <iframe
        width="100%"
        height="1000"
        src="_static/rann_epa.html"
        frameborder="0"
        allowfullscreen
    
    ></iframe>




MTP
^^^

The interactive plots with linear coloring do not have an interactive
legend like the previous plots. Instead, they feature a colorbar on the
left and a dual slider at the bottom left of the figure. You can adjust
the dual slider to show only the energies within the slider range on the
plot. When you move the dual slider, the default option is for the
colorbar to not be rescaled. In order to more accurately see the
differences in the visible data, this can be turned on by setting the
rescale_colorbar argument to True. There are also seven colormaps to
choose from. The default is Viridis256. The other options include
Greys256, Inferno256, Magma256, Plasma256, Cividis256, and Turbo256.
Change the colormap by setting the cmap argument to one of the listed
colormaps.

.. code:: ipython3

    IFrame(src='mtp_epa.html', width='100%', height=1000)




.. raw:: html

    
    <iframe
        width="100%"
        height="1000"
        src="_static/mtp_epa.html"
        frameborder="0"
        allowfullscreen
    
    ></iframe>




GAP
^^^

The interactive plots with linear coloring do not have an interactive
legend like the previous plots. Instead, they feature a colorbar on the
left and a dual slider at the bottom left of the figure. You can adjust
the dual slider to show only the energies within the slider range on the
plot. When you move the dual slider, the default option is for the
colorbar to not be rescaled. In order to more accurately see the
differences in the visible data, this can be turned on by setting the
rescale_colorbar argument to True. There are also seven colormaps to
choose from. The default is Viridis256. The other options include
Greys256, Inferno256, Magma256, Plasma256, Cividis256, and Turbo256.
Change the colormap by setting the cmap argument to one of the listed
colormaps.

.. code:: ipython3

    IFrame(src='gap_epa.html', width='100%', height=1000)




.. raw:: html

    
    <iframe
        width="100%"
        height="1000"
        src="_static/gap_epa.html"
        frameborder="0"
        allowfullscreen
    
    ></iframe>




