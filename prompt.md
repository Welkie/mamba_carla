239.8s	1156	STARTING EXPERIMENTS
239.8s	1157	==============================
240.1s	1158	GPU available: Tesla T4
240.1s	1159	
240.1s	1160	Running dataset: wadi
7251.7s	1161	Error running pretext for wadi: Command '['/usr/bin/python3', 'carla_pretext.py', '--config_env', 'configs/env.yml', '--config_exp', 'configs/pretext/carla_pretext_wadi.yml', '--fname', 'wadi']' died with <Signals.SIGKILL: 9>.
7251.7s	1162	
7264.0s	1163	Error running classification for wadi: Command '['/usr/bin/python3', 'carla_classification.py', '--config_env', 'configs/env.yml', '--config_exp', 'configs/classification/carla_classification_wadi.yml', '--fname', 'wadi']' returned non-zero exit status 1.
7264.0s	1164	Traceback (most recent call last):
7264.0s	1165	  File "/kaggle/working/mamba_carla/carla_classification.py", line 213, in <module>
7264.0s	1166	    main()
7264.0s	1167	  File "/kaggle/working/mamba_carla/carla_classification.py", line 51, in main
7264.0s	1168	    train_dataset = get_aug_train_dataset(p, train_transformations, to_neighbors_dataset = True)
7264.0s	1169	                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
7264.0s	1170	  File "/kaggle/working/mamba_carla/utils/common_config.py", line 172, in get_aug_train_dataset
7264.0s	1171	    data_dict = torch.load(p['contrastive_dataset'], weights_only=False)
7264.0s	1172	                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
7264.0s	1173	  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 1065, in load
7264.0s	1174	    with _open_file_like(f, 'rb') as opened_file:
7264.0s	1175	         ^^^^^^^^^^^^^^^^^^^^^^^^
7264.0s	1176	  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 468, in _open_file_like
7264.0s	1177	    return _open_file(name_or_buffer, mode)
7264.0s	1178	           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
7264.0s	1179	  File "/usr/local/lib/python3.12/dist-packages/torch/serialization.py", line 449, in __init__
7264.0s	1180	    super().__init__(open(name, mode))
7264.0s	1181	                     ^^^^^^^^^^^^^^^^
7264.0s	1182	FileNotFoundError: [Errno 2] No such file or directory: 'results/wadi/wadi/pretext/con_train_dataset.pth'
7264.0s	1183	
7264.0s	1184	Max GPU Memory after wadi: 0.00 MB
7264.0s	1185	
7264.0s	1186	==============================
7264.0s	1187	DONE ALL WADI DATASETS
7264.0s	1188	Total time: 7024.23 s
7264.0s	1189	Avg / dataset: 7024.23 s
7264.0s	1190	==============================
7264.0s	1191	
7264.0s	1192	Time results saved to results/wadi/time_results.json
7264.0s	1193	
7264.0s	1194	==============================
7264.0s	1195	STARTING EVALUATION (PAPER STYLE)
7264.0s	1196	==============================
7264.0s	1197	Skip wadi (missing files)
7264.0s	1198	No results!
7264.8s	1199	[Errno 2] No such file or directory: 'mamba_carla'
7264.8s	1200	/kaggle/working/mamba_carla