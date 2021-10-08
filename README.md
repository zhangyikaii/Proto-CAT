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

&nbsp;

## Prototype-based Co-Adaptation with Transformer

<p align="center">
    <img src="assets/model.png" width = "60%">
</p>
**Illustration of Proto-CAT.** The model transforms the classification space using ![[公式]](assets/T.svg) based on two kinds of audio-visual prototypes (class centers): (1) the base training categories (color with ![[公式]](https://via.placeholder.com/15/4099d8/000000?text=+) blue, ![[公式]](https://via.placeholder.com/15/3fa864/000000?text=+) green, and ![[公式]](https://via.placeholder.com/15/f256ad/000000?text=+) pink); and (2) the additional novel test categories (color with ![[公式]](https://via.placeholder.com/15/b53600/000000?text=+) burning transition). Proto-CAT learns and generalizes on novel test categories from *limited labeled* examples, maintaining performance on the base training ones. ![[公式]](assets/T.svg) includes *audio-visual level* and *category level* prototype-based co-adaptation. From left to right, more coverage and more bright colors represent a more reliable classification space.

&nbsp;

## Results

<table>
    <tr>
        <td><b>Dataset</b></td>
        <td colspan="5" align="center"><b>LRW</b></td>
        <td colspan="3" align="center"><b>LRW-1000</b></td>
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
        <td>LSTM-based</td>
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
        <td>GRU-based</td>
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
        <td>MS-TCN-based</td>
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
        <td>ProtoNet-GFSL</td>
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
        <td>FEAT-GFSL</td>
        <td align="center">39.95</td>
        <td align="center">14.40</td>
        <td align="center">96.33</td>
        <td align="center">39.23</td>
        <td align="center">54.79</td>
        <td align="center">69.33</td>
        <td align="center">0.76</td>
        <td align="center">1.47</td>
    </tr>
</table>

