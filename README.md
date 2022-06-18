# Audio-Visual Generalized Few-Shot Learning with Prototype-Based Co-Adaptation

The code repository for "Audio-Visual Generalized Few-Shot Learning with Prototype-Based Co-Adaptation" [[paper, to appear]]() [[slides, to appear]]() [[poster, to appear]]() in PyTorch. If you use any content of this repo for your work, please cite the following bib entry:

```
@misc{Proto-CAT,
  author = {Yi-Kai Zhang},
  title = {Audio-Visual Generalized Few-Shot Learning with Prototype-Based Co-Adaptation},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ZhangYikaii/Proto-CAT}},
  commit = {main}
}
```

## Prototype-based Co-Adaptation with Transformer

<p align="center">
    <img src="assets/model.png" width = "66%">
</p>

**Illustration of Proto-CAT.** The model transforms the classification space using ![[公式]](assets/T.svg) based on two kinds of audio-visual prototypes (class centers): (1) the base training categories (color with blue, green, and pink); and (2) the additional novel test categories (color with burning transition). Proto-CAT learns and generalizes on novel test categories from *limited labeled* examples, maintaining performance on the base training ones. ![Tsvg](assets/T.svg) includes *audio-visual level* and *category level* prototype-based co-adaptation. From left to right, more coverage and more bright colors represent a more reliable classification space.

&nbsp;

## Results

<table>
    <tr>
        <td><b>Dataset</b></td>
        <td colspan="5" align="center"><b><a href="https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html">LRW</a></b></td>
        <td colspan="3" align="center"><b><a href="https://vipl.ict.ac.cn/en/view_database.php?id=13">LRW-1000</a></b></td>
    </tr>
    <tr>
        <td><b>Data Source</b></td>
        <td align="center">Audio (<img src = "assets/A.svg"/>)</td>
        <td align="center">Video (<img src = "assets/V.svg"/>)</td>
        <td colspan="3" align="center">Audio-Video (<img src = "assets/AandV.svg"/>)</td>
        <td colspan="3" align="center">Audio-Video (<img src = "assets/AandV.svg"/>)</td>
    </tr>
    <tr>
        <td><b>Perf. Measures on</b></td>
        <td align="center">H-mean</td>
        <td align="center">H-mean</td>
        <td align="center">Base</td>
        <td align="center">Novel</td>
        <td align="center">H-mean</td>
        <td align="center">Base</td>
        <td align="center">Novel</td>
        <td align="center">H-mean</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/1601.08188">LSTM-based</a></td>
        <td align="center">32.20</td>
        <td align="center">8.00</td>
        <td align="center">97.09</td>
        <td align="center">23.76</td>
        <td align="center">37.22</td>
        <td align="center">71.34</td>
        <td align="center">0.03</td>
        <td align="center">0.07</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/2011.07557">GRU-based</a></td>
        <td align="center">37.01</td>
        <td align="center">10.58</td>
        <td align="center"><b>97.44</b></td>
        <td align="center">27.35</td>
        <td align="center">41.71</td>
        <td align="center">71.34</td>
        <td align="center">0.05</td>
        <td align="center">0.09</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/2001.08702">MS-TCN-based</a></td>
        <td align="center">62.29</td>
        <td align="center">19.06</td>
        <td align="center">80.96</td>
        <td align="center">51.28</td>
        <td align="center">61.76</td>
        <td align="center">71.55</td>
        <td align="center">0.33</td>
        <td align="center">0.63</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/1703.03400">MAML</a></td>
        <td align="center">35.49</td>
        <td align="center">10.25</td>
        <td align="center">40.09</td>
        <td align="center">66.70</td>
        <td align="center">49.20</td>
        <td align="center">29.40</td>
        <td align="center">23.21</td>
        <td align="center">25.83</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/2109.04504">BootstrappedMAML</a></td>
        <td align="center">33.75</td>
        <td align="center">6.52</td>
        <td align="center">35.29</td>
        <td align="center">64.20</td>
        <td align="center">45.17</td>
        <td align="center">28.15</td>
        <td align="center">27.98</td>
        <td align="center">28.09</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/1703.05175">ProtoNet</a></td>
        <td align="center">39.95</td>
        <td align="center">14.40</td>
        <td align="center">96.33</td>
        <td align="center">39.23</td>
        <td align="center">54.79</td>
        <td align="center">69.33</td>
        <td align="center">0.76</td>
        <td align="center">1.47</td>
    </tr>
        <tr>
        <td><a href="https://arxiv.org/abs/1606.04080">MatchingNet</a></td>
        <td align="center">36.76</td>
        <td align="center">12.09</td>
        <td align="center">94.54</td>
        <td align="center">36.57</td>
        <td align="center">52.31</td>
        <td align="center">68.42</td>
        <td align="center">0.95</td>
        <td align="center">1.89</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/1904.03758">MetaOptNet</a></td>
        <td align="center">43.81</td>
        <td align="center">19.59</td>
        <td align="center">88.20</td>
        <td align="center">47.06</td>
        <td align="center">60.73</td>
        <td align="center">69.01</td>
        <td align="center">1.79</td>
        <td align="center">3.44</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/2003.06777">DeepEMD</a></td>
        <td align="center">--</td>
        <td align="center">27.02</td>
        <td align="center">82.53</td>
        <td align="center">16.43</td>
        <td align="center">27.02</td>
        <td align="center">64.54</td>
        <td align="center">0.80</td>
        <td align="center">1.56</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/1812.03664">FEAT</a></td>
        <td align="center">49.90</td>
        <td align="center">25.75</td>
        <td align="center">96.26</td>
        <td align="center">54.52</td>
        <td align="center">68.83</td>
        <td align="center"><b>71.69</b></td>
        <td align="center">2.62</td>
        <td align="center">4.89</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/1804.09458">DFSL</a></td>
        <td align="center">72.13</td>
        <td align="center">42.56</td>
        <td align="center">66.10</td>
        <td align="center">84.62</td>
        <td align="center">73.81</td>
        <td align="center">31.68</td>
        <td align="center"><b>68.72</b></td>
        <td align="center">42.56</td>
    </tr>
    <tr>
        <td><a href="https://arxiv.org/abs/1906.02944">CASTLE</a></td>
        <td align="center">75.48</td>
        <td align="center">34.68</td>
        <td align="center">73.50</td>
        <td align="center">90.20</td>
        <td align="center">80.74</td>
        <td align="center">11.13</td>
        <td align="center">54.07</td>
        <td align="center">17.84</td>
    </tr>
    <tr>
        <td>Proto-CAT (Ours)</td>
        <td style="vertical-align:middle;" align="center" rowspan="2"><b>84.18</b></td>
        <td style="vertical-align:middle;" align="center" rowspan="2"><b>74.55</b></td>
        <td align="center">93.37</td>
        <td align="center"><b>91.20</b></td>
        <td align="center"><b>92.13</b></td>
        <td align="center">49.70</td>
        <td align="center">38.27</td>
        <td align="center">42.25</td>
    </tr>
    <tr>
        <td>Proto-CAT+ (Ours)</td>
        <td align="center">93.18</td>
        <td align="center">90.16</td>
        <td align="center">91.49</td>
        <td align="center">54.55</td>
        <td align="center">38.16</td>
        <td align="center"><b>43.88</b></td>
    </tr>
</table>

Audio-visual generalized few-shot learning classification performance (in %; measured over 10,000 rounds; higher is better) of 5-way 1-shot training tasks on LRW and LRW-1000 datasets. The best result of each scenario is in bold font. The performance measure on both base and novel classes (Base, Novel in the table) is mean accuracy. Harmonic mean (i.e., H-mean) of the above two is a better generalized few-shot learning performance measure.

&nbsp;

## Prerequisites

### Environment

Please refer to [`requirements.txt`](requirements.txt) and run:

```bash
pip install -r requirement.txt
```

### Dataset

+ ~~**Use preprocessed data (suggested):**~~

  LRW and LRW-1000 forbid directly share the preprocessed data.

+ **Use raw data and do preprocess:**

  Download [LRW](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html) Dataset and unzip, like,

    ```
    /your data_path set in .sh file
    ├── lipread_mp4
    │   ├── [ALL CLASS FOLDER]
    │   ├── ...
    ```

  Run [`prepare_lrw_audio.py`](run/prepare_lrw_audio.py) and [`prepare_lrw_video.py`](run/prepare_lrw_video.py) to preprocess data on video and audio modality, respectively. Please modify the data path in the above preprocessing file in advance.

  Similarly, Download [LRW-1000](https://vipl.ict.ac.cn/en/view_database.php?id=13) dataset and unzip. Run [`prepare_lrw1000_audio.py`](run/prepare_lrw1000_audio.py) and [`prepare_lrw1000_video.py`](run/prepare_lrw1000_video.py) to preprocess it.

### Pretrained Weights

We provide pretrained weights on LRW and LRW-1000 dataset. Download from [Google Drive](https://drive.google.com/drive/folders/1wEaMxIg6XZFwvMGusUbIQhUPRwTxtAsk?usp=sharing) or [Baidu Yun(password: 3ad2)](https://pan.baidu.com/s/1Fa5aSf2iJV26zFITX1lnIA) and put them as:

  ```
  /your init_weights set in .sh file
  ├── Conv1dResNetGRU_LRW-pre.pth
  ├── Conv3dResNetLSTM_LRW-pre.pth
  ├── Conv1dResNetGRU_LRW1000-pre.pth
  ├── Conv3dResNetLSTM_LRW1000-pre.pth
  ```

&nbsp;

## How to Train Proto-CAT

For LRW dataset, fine-tune the parameters in `run/protocat_lrw.sh`, and run:

```bash
cd ./Proto-CAT/run
bash protocat_lrw.sh
```

Similarly, run `bash protocat_lrw1000.sh` for dataset LRW-1000.

Run `bash protocat_plus_lrw.sh` / `bash protocat_plus_lrw1000.sh` to train Proto-CAT+.

## How to Reproduce the Result of Proto-CAT

Download the trained models from [Google Drive](https://drive.google.com/drive/folders/1KWhRJ7VPr1gazn4rqz7qu9TmhL9mBbfX?usp=sharing) or [Baidu Yun(password: swzd)](https://pan.baidu.com/s/1ZXWWE1U4Y3gRxd6Wnpndkw) and run:

```bash
bash test_protocat_lrw.sh
```

Run `bash test_protocat_lrw1000.sh`, `bash test_protocat_plus_lrw.sh`, or `bash test_protocat_plus_lrw1000.sh` to evaluate other models.

&nbsp;

## Code Structures

Proto-CAT's entry function is in `main.py`. It calls the manager `Trainer` in `models/train.py` that contains the main training logic. In `Trainer`, `prepare_handle.prepare_dataloader` combined with `train_prepare_batch` inputs and preprocesses generalized few-shot style data. `fit_handle` controls forward and backward propagation. `callbacks` deals with the behaviors at each stage.

### Arguments

All parameters are defined in `models/utils.py`. We list the main ones below:

+ `do_train`, `do_test`: Store-true switch for whether to train or test.
+ `data_path`: Data directory to be set.
+ `model_save_path`: Optimal model save directory to be set.
+ `init_weights`: Pretrained weights to be set.
+ `dataset`: Option for the dataset.
+ `model_class`: Option for the top model.
+ `backend_type`: Option list for the backend type.
+ `train_way`, `val_way`, `test_way`, `train_shot`, `val_shot`, `test_shot`, `train_query`, `val_query`, `test_query`: Tasks setting of generalized few-shot learning.
+ `gfsl_train`, `gfsl_test`: Switch for whether train or test in generalized few-shot learning way, i.e., whether additional base class data is included.
+ `mm_list`: Participating modalities.
+ `lr_scheduler`: List of learning rate scheduler.
+ `loss_fn`: Option for the loss function.
+ `max_epoch`: Maximum training epoch.
+ `episodes_per_train_epoch`, `episodes_per_val_epoch`, `episodes_per_test_epoch`: Number of sampled episodes per epoch.
+ `num_tasks`: Number of tasks per episode.
+ `meta_batch_size`: Batch size of each task.
+ `test_model_filepath`: Trained weights `.pth` file path when testing a model.
+ `gpu`: Multi-GPU option like `--gpu 0,1,2,3`.
+ `logger_filename`: Logger file save directory.
+ `time_str`: Token for each run, and will generate by itself if empty.
+ `acc_per_class`: Switch for whether to measure the accuracy of each class with base, novel, and harmonic mean.
+ `verbose`, `epoch_verbose`: Switch for whether to output message or output progress bar.
+ `torch_seed`, `cuda_seed`, `np_seed`, `random_seed`: Seeds of random number generation.

&nbsp;

## Acknowledgment

We thank the following repos providing helpful components/functions in our work.

+ [Few-shot-Framework](https://github.com/ZhangYikaii/Few-shot-Framework)
+ [FEAT](https://github.com/Sha-Lab/FEAT)
+ [aCASTLE](https://github.com/Sha-Lab/aCASTLE)
+ [learn-an-effective-lip-reading-model-without-pains](https://github.com/Fengdalu/learn-an-effective-lip-reading-model-without-pains)
+ [FewShotWithoutForgetting](https://github.com/gidariss/FewShotWithoutForgetting)
+ [few-shot](https://github.com/oscarknagg/few-shot)
+ [Lipreading_using_Temporal_Convolutional_Networks](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks)
