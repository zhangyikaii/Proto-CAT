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

<center>
    <img src="assets/model.png" width = "60%">
</center>
**Illustration of Proto-CAT.** The model transforms the classification space using $\mathbf{T}$ based on two kinds of audio-visual prototypes (class centers): (1) the base training categories (color with ![#4099d8](https://via.placeholder.com/15/4099d8/000000?text=+)blue, ![#3fa864](https://via.placeholder.com/15/3fa864/000000?text=+)green, and ![#f256ad](https://via.placeholder.com/15/f256ad/000000?text=+)pink); and (2) the additional novel test categories (color with ![#b53600](https://via.placeholder.com/15/b53600/000000?text=+)burning transition). Proto-CAT learns and generalizes on novel test categories from *limited labeled* examples, maintaining performance on the base training ones. $\mathbf{T}$ includes *audio-visual level* and *category level* prototype-based co-adaptation. From left to right, more coverage and more bright colors represent a more reliable classification space.

&nbsp;

The code is coming soon.
