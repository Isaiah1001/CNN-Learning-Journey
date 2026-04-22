
# Stage 3.1 — Lightning Module
This subfolder refactors the EfficientNet‑B0 fine‑tuning pipeline from a hand‑written training loop (Stage 2) into a PyTorch Lightning setup. It keeps the same model architecture and dataset（only training classifier head）, 
but moves all training orchestration (epochs, device placement, checkpointing, LR logging, profiling) into `Trainer` and callbacks.

## File Structure
```
📁 01_Lightning_module/
├── 📁 preprocess/ 
├── 📁 logs/
├── 📁 profiler_output/
├── 📁 profiler_output/
├── lightning_flower.py
└── README.md 
```

## Results
**Code:** lightning_flower.py  
**Artifact:**  
| Metric | Value |
|--------|-------|
| Dataset | Oxford 102 Flowers |
| Top-1 Accuracy | 93.49% (best:93.73%) |
| Epochs | 40 |
| Optimizer | SGD, lr=0.1, weight_decay=1e-4 |

![Loss and Accuracy](./Lightning.png)  

-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                          ProfilerStep*         1.15%       4.187ms       100.00%     365.015ms      52.145ms       2.610ms         0.72%     365.000ms      52.143ms           0 b           0 b           0 b    -518.01 Mb             7  
[pl][profile][Strategy]SingleDeviceStrategy.validati...         2.59%       9.436ms        97.11%     354.473ms      50.639ms       8.633ms         2.37%     336.463ms      48.066ms           0 b           0 b           0 b    -252.00 Kb             7  
[pl][module]torchvision.models.efficientnet.Efficien...         0.17%     612.000us        92.91%     339.129ms      48.447ms     395.000us         0.11%     321.656ms      45.951ms           0 b           0 b     178.50 Kb    -114.19 Mb             7  
[pl][module]torch.nn.modules.container.Sequential: m...         0.41%       1.490ms        91.98%     335.755ms      47.965ms     693.000us         0.19%     318.443ms      45.492ms           0 b           0 b     112.00 Mb      -1.29 Gb             7  
                                           aten::conv2d         5.04%      18.400ms        52.16%     190.380ms     167.883us      10.025ms         2.75%     232.974ms     205.444us           0 b           0 b      22.63 Gb    -305.36 Mb          1134  
                                       aten::batch_norm         1.00%       3.649ms        23.89%      87.188ms     254.192us       1.468ms         0.40%      98.047ms     285.851us           0 b           0 b      11.27 Gb      -1.25 Mb           343  
                           aten::_batch_norm_impl_index         1.80%       6.556ms        22.89%      83.538ms     243.552us       2.814ms         0.77%      96.579ms     281.571us           0 b           0 b      11.27 Gb           0 b           343  
                                      aten::convolution         1.52%       5.534ms        16.91%      61.725ms     108.863us       4.047ms         1.11%      87.293ms     153.956us           0 b           0 b      11.31 Gb           0 b           567  
                                aten::native_batch_norm        10.29%      37.561ms        19.91%      72.670ms     211.865us      49.183ms        13.47%      83.872ms     244.525us           0 b           0 b      11.27 Gb           0 b           343  
                                     aten::_convolution         3.13%      11.440ms        15.39%      56.191ms      99.103us       6.835ms         1.87%      83.246ms     146.818us           0 b           0 b      11.31 Gb           0 b           567  
                                               aten::to         2.14%       7.819ms        12.69%      46.308ms      48.695us       8.375ms         2.29%      71.241ms      74.912us           0 b           0 b     830.47 Mb           0 b           951  
[pl][module]torch.nn.modules.container.Sequential: m...         0.16%     598.100us        18.08%      66.001ms       9.429ms     560.000us         0.15%      66.063ms       9.438ms           0 b           0 b      16.08 Mb     -48.23 Mb             7  
                                         aten::_to_copy         4.35%      15.861ms        10.54%      38.490ms      45.070us      10.633ms         2.91%      62.866ms      73.614us           0 b           0 b     830.47 Mb           0 b           854  
[pl][module]torch.nn.modules.container.Sequential: m...         0.14%     525.300us        30.51%     111.351ms      15.907ms     305.000us         0.08%      62.335ms       8.905ms           0 b           0 b      26.80 Mb     -53.59 Mb             7  
[pl][module]torch.nn.modules.container.Sequential: m...         0.55%       2.016ms        10.06%      36.703ms       5.243ms      61.000us         0.02%      58.681ms       8.383ms           0 b           0 b     128.62 Mb    -128.62 Mb             7  
[pl][module]torch.nn.modules.container.Sequential: m...         0.13%     490.800us        13.27%      48.452ms       6.922ms     478.000us         0.13%      48.511ms       6.930ms           0 b           0 b      37.52 Mb     -75.03 Mb             7  
                                aten::_conv_depthwise2d         5.96%      21.764ms         7.83%      28.581ms     255.191us      36.299ms         9.94%      46.374ms     414.054us           0 b           0 b       3.87 Gb           0 b           112  
                                            aten::fill_         3.90%      14.228ms         3.90%      14.228ms       5.508us      43.262ms        11.85%      43.262ms      16.749us           0 b           0 b           0 b           0 b          2583  
                                            aten::copy_         5.03%      18.345ms         5.03%      18.345ms      15.236us      34.554ms         9.47%      34.554ms      28.699us           0 b           0 b           0 b           0 b          1204  
[pl][module]torch.nn.modules.container.Sequential: m...         0.11%     415.300us         9.39%      34.281ms       4.897ms      44.000us         0.01%      33.802ms       4.829ms           0 b           0 b      53.59 Mb     -53.59 Mb             7  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 365.015ms
Self CUDA time total: 365.000ms


## Key Finding
