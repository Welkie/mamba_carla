caption{Hyperparameter settings for CARLA-ResNet and CARLA-MAMBA. Shared parameters are the same for both variants. Window sizes are tuned independently for each backbone.}
\label{tab:hyperparams}
\resizebox{\linewidth}{!}{
\begin{tabular}{lcc}
\hline
\textbf{Parameter} & \textbf{CARLA-ResNet} & \textbf{CARLA-MAMBA} \\
\hline
Backbone (pretext) & ResNet (3 layers, kernels $[8,5,3]$) & Mamba (3 layers, $d\_state{=}16$, $d\_conv{=}4$, $expand{=}2$) \\
Backbone (classification) & ResNet & ResNet \\
Window size (MSL) & 200 & 512 \\
Window size (SMAP, SMD, SWaT) & 200 & 256 \\
Window size (WADI) & 400 & 256 \\
Window size (Yahoo-A1, KPI) & 250 & 512 \\
Epochs (pretext) & 30 (10 for SWaT/WADI) & 30 (10 for SWaT/WADI) \\
Epochs (classification) & 50 & 50 \\
Learning rate (pretext) & 0.001 & 0.001 \\
Batch size & 50 (64--256 for SWaT/WADI) & 50 (64--256 for SWaT/WADI) \\
Optimizer & Adam & Adam \\
sao lại ghi là Shared parameters are the same for both variants
