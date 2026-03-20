1. SML 
    + run_sml.py
    + data/SML.py
2. SMAP
    + run_smap.py
    + data/SMAP.py
3. SMD - Split into 2 datasets
    + run_smd.py 
    + data/SMD.py   
4. Yahoo
    + run_yahoo.py
    + data/yahoo.py
5. SWaT - Use CPU
    + run_swat.py
    + data/SWAT.py
6. WADI - Use CPU
    + run_wadi.py
    + data/WADI.py
7. KPI - Split into 2 datasets
    + run_kpi.py
    + data/KPI.py

# Update general to use CPU
    data
        + augment.py
        + custom_dataset.py
    utils
        + common_config.py
        + utils.py
        + repository.py
    carla_pretext.py

# Update local
    run_swat.py
    data
        + SWAT.py
    configs
        + classification
            + carla_classification_swat.yml
        + pretext
            + carla_pretext_swat.yml