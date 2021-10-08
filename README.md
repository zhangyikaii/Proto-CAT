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
**Illustration of Proto-CAT.** The model transforms the classification space using ![T](assets/T.svg) based on two kinds of audio-visual prototypes (class centers): (1) the base training categories (color with ![#4099d8](https://via.placeholder.com/15/4099d8/000000?text=+)blue, ![#3fa864](https://via.placeholder.com/15/3fa864/000000?text=+)green, and ![#f256ad](https://via.placeholder.com/15/f256ad/000000?text=+)pink); and (2) the additional novel test categories (color with ![#b53600](https://via.placeholder.com/15/b53600/000000?text=+)burning transition). Proto-CAT learns and generalizes on novel test categories from *limited labeled* examples, maintaining performance on the base training ones. ![T](assets/T.svg) includes *audio-visual level* and *category level* prototype-based co-adaptation. From left to right, more coverage and more bright colors represent a more reliable classification space.

&nbsp;

## Results

<table>
    <tr>
        <td><b>Data Source</b></td>
        <td>Audio (<img src = "assets/A.svg"/>)</td>
        <td>Video (<img src = "assets/V.svg"/>)</td>
        <td colspan="3">Audio-Video (<img src = "assets/AandV.svg"/>)</td>
        <td colspan="3">Audio-Video (<img src = "assets/AandV.svg"/>)</td>
    </tr>
    <tr>
        <td><b>Perf. Measures on</b></td>
        <td>H-mean</td>
        <td>H-mean</td>
        <td>Base</td>
        <td>Novel</td>
        <td>H-mean</td>
        <td>Base</td>
        <td>Novel</td>
        <td>H-mean</td>
    </tr>
    <tr>
        <td>LSTM-based</td>
        <td>32.20</td>
        <td>8.00</td>
        <td>97.09</td>
        <td>23.76</td>
        <td>32.77</td>
        <td>71.34</td>
        <td>0.03</td>
        <td>0.07</td>
    </tr>
</table>

