1. SML 
    + run_sml.py
    + data/SML.py
2. SMAP
    + run_smap.py
    + data/SMAP.py
3. SMD
    + run_smd.py chia 2 stage
    + data/SMD.py   

# Update CPU
4. SWaT
    + run_swat.py
    + data/SWAT.py
5. WADI
    + run_wadi.py
    + data/WADI.py

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