pyRANN TUTORIAL
================
This is a tutorial for plotting the descriptor space of a RANN database with UMAP as well as reducing the RANN database.

.. code-block:: python
   :linenos:

   import pyrann as rann
   import numpy as np
   
   # Create processing class instance with Mg RANN input file
   processing = rann.processing('Mg.input')
   
   # Set processing instance features to RANN potential atomic descriptors
   # Because the formalism is RANN, it also sets the Global Simulation Number,
   #   Local Simulation Number, and Local ID for each atom
   processing.get_features()
   
   # Initialize UMAP instance for processing instance and return mapper
   mapper = processing.get_umap()
   
   # Plot UMAP embedding of atomic descriptor space with each atom colored by
   #   filename and rename the plot to "full_database.html"
   processing.plot(mapper, color_data=processing.filenames,
                   hover_info={'Global Simulation Number': processing.global_sim_num,
                               'Local Simulation Number': processing.local_sim_num,
                               'Local ID': processing.local_id},
                   key_str='full_database')
   
   # Reducing the database by 50%
   reduced = processing.reduce(percent=50.0)
   
   # Creating a boolean mask to only plot the reduced database
   subset_points = [True if i in reduced else False for i in processing.global_sim_num]
   
   # Plot UMAP embedding of atomic descriptor space of the reduced database with
   #   each atom colored byfilename and rename the plot to "reduced_database.html"
   processing.plot(mapper, color_data=processing.filenames,
                   hover_info={'Global Simulation Number': processing.global_sim_num,
                               'Local Simulation Number': processing.local_sim_num,
                               'Local ID': processing.local_id},
                   subset_points=subset_points,
                   key_str='reduced_database')
   
   
   # Get unique filenames in the order that they are read for feature calculation
   unique_filenames, unique_index = np.unique(processing.filenames,
                                              return_index=True)
   unique_index = np.argsort(unique_index)
   unique_filenames = unique_filenames[unique_index]
   
   # Load dump files to a list of series instances in the order they are read for
   #   feature calculation
   series = [rann.load(i, series=True) for i in unique_filenames]
   
   # Get new series instances for reduced database
   global_count = 0
   reduced_series = []
   for i in range(len(unique_filenames)):
       file_series = []
       for j in range(len(series[i].systems)):
           if global_count in reduced:
               file_series.append(series[i].systems[j])
           global_count += 1
       reduced_series.append(rann.series(file_series))
   
   # Export reduced database to new dump files
   for i in range(len(reduced_series)):
       reduced_series[i].export(f'{unique_filenames[i]}_reduced.dump',
                                directory='reduced_DUMPS')
   
   # Getting the energy per atom for each atom for each simulation
   epa = [j.energy / j.natoms for i in series for j in i.systems for _ in range(j.natoms)]
   
   # Plotting the full database with atoms colored by energy per atom (eV/atom)
   processing.plot(mapper, color_data=epa,
                   color_type='linear',
                   hover_info={'Global Simulation Number': processing.global_sim_num,
                               'Local Simulation Number': processing.local_sim_num,
                               'Local ID': processing.local_id,
                               'eV/atom': epa},
                   key_str='full_database_epa')
   
   # Plotting the reduced database with atoms colored by energy per atom (eV/atom)
   processing.plot(mapper, color_data=epa,
                   color_type='linear',
                   hover_info={'Global Simulation Number': processing.global_sim_num,
                               'Local Simulation Number': processing.local_sim_num,
                               'Local ID': processing.local_id,
                               'eV/atom': epa},
                   subset_points=subset_points,
                   key_str='reduced_database_epa')


full_database.html

This plot is interactive. Hover over a datapoint to show the global simulation number, local simulation number, and local id for that particular atom. Click a legend entry to toggle between hide/show for that particular dump file.

.. raw:: html

   <iframe src="_static/full_database.html" height="1000px" width="100%" style="border:none;"></iframe>




reduced_database.html

This plot is interactive. Hover over a datapoint to show the global simulation number, local simulation number, and local id for that particular atom. Click a legend entry to toggle between hide/show for that particular dump file.

.. raw:: html

   <iframe src="_static/reduced_database.html" height="1000px" width="100%" style="border:none;"></iframe>




full_database_epa.html

This plot is interactive. Hover over a datapoint to show the global simulation number, local simulation number, local id, and energy/atom for that particular atom. There is a dual slider at the bottom left of the page. Adjusting this slider will remove atoms with energies outside of the slider range.

.. raw:: html

   <iframe src="_static/full_database_epa.html" height="1000px" width="100%" style="border:none;"></iframe>




reduced_database_epa.html

This plot is interactive. Hover over a datapoint to show the global simulation number, local simulation number, local id, and energy/atom for that particular atom. There is a dual slider at the bottom left of the page. Adjusting this slider will remove atoms with energies outside of the slider range.

.. raw:: html

   <iframe src="_static/reduced_database_epa.html" height="1000px" width="100%" style="border:none;"></iframe>
