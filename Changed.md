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
5. KPI - Split into 2 datasets
    + run_kpi.py
    + data/KPI.py
6. SWaT - Use CPU
    + run_swat.py
    + data
        + SWAT.py
    + configs
        + classification
            + carla_classification_swat.yml
        + pretext
            + carla_pretext_swat.yml
7. WADI - Use CPU
    + run_wadi.py
    + data
        + WADI.py
    + configs
        + classification
            + carla_classification_wadi.yml
        + pretext
            + carla_pretext_wadi.yml

# Update general to use CPU
    data
        + augment.py
        + custom_dataset.py
    utils
        + mypath.py
        + common_config.py
        + utils.py
        + repository.py
    carla_pretext.py


    