Copying train file: /kaggle/input/datasets/giovannimonco/wadi-data/WADI_14days_new.csv → /kaggle/working/mamba_carla/datasets/wadi/WADI_14days_new.csv
Copying test file:  /kaggle/input/datasets/giovannimonco/wadi-data/WADI_attackdataLABLE.csv → /kaggle/working/mamba_carla/datasets/wadi/WADI_attackdataLABLE.csv
Set wadi_DATASET_PATH to /kaggle/working/mamba_carla/datasets/wadi

==============================
STARTING EXPERIMENTS
==============================
GPU available: Tesla T4

Running dataset: wadi
Error running pretext for wadi: Command '['/usr/bin/python3', 'carla_pretext.py', '--config_env', 'configs/env.yml', '--config_exp', 'configs/pretext/carla_pretext_wadi.yml', '--fname', 'wadi']' returned non-zero exit status 1.
/kaggle/working/mamba_carla/data/WADI.py:67: DtypeWarning: Columns (0,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130) have mixed types. Specify dtype option on import or set low_memory=False.
  raw = pd.read_csv(file_path, header=0)
Traceback (most recent call last):
  File "/kaggle/working/mamba_carla/carla_pretext.py", line 240, in <module>
    main()
  File "/kaggle/working/mamba_carla/carla_pretext.py", line 144, in main
    val_dataset = get_val_dataset(p, val_transforms, sanomaly, False, train_dataset.mean,
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/mamba_carla/utils/common_config.py", line 235, in get_val_dataset
    dataset = WADI(p['fname'], train=False, transform=transform, sanomaly=sanomaly,
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/mamba_carla/data/WADI.py", line 119, in __init__
    temp = (temp - self.mean) / self.std
            ~~~~~^~~~~~~~~~~
ValueError: operands could not be broadcast together with shapes (172803,0) (127,) 

Error running classification for wadi: Command '['/usr/bin/python3', 'carla_classification.py', '--config_env', 'configs/env.yml', '--config_exp', 'configs/classification/carla_classification_wadi.yml', '--fname', 'wadi']' returned non-zero exit status 1.
Traceback (most recent call last):
  File "/kaggle/working/mamba_carla/carla_classification.py", line 213, in <module>
    main()
  File "/kaggle/working/mamba_carla/carla_classification.py", line 51, in main
    train_dataset = get_aug_train_dataset(p, train_transformations, to_neighbors_dataset = True)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/kaggle/working/mamba_carla/utils/common_config.py", line 172, in get_aug_train_dataset
    data_dict = torch.load(p['contrastive_dataset'], weights_only=False)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 1065, in load
    with _open_file_like(f, 'rb') as opened_file:
         ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 468, in _open_file_like
    return _open_file(name_or_buffer, mode)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 449, in __init__
    super().__init__(open(name, mode))
                     ^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'results/wadi/wadi/pretext/con_train_dataset.pth'

Max GPU Memory after wadi: 0.00 MB

==============================
DONE ALL WADI DATASETS
Total time: 55.12 s
Avg / dataset: 55.12 s
==============================

Time results saved to results/wadi/time_results.json