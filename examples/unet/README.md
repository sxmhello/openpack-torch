# U-Net

## Notebooks

### [Attention_U-Net_plus.ipynb](./notebooks/Attention_U-Net_plus.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lly1117/openpack-torch/blob/main/examples/unet/notebooks/Attention_U-Net_plus.ipynb)

In this notebook, you can train and test the U-Net with `openpack_torch` package.
Also, you can learn the basic usage of (1) pytorch-lightning's `LightningDataModule`, and (2) `LightinigModule` supported by `openpack_torch`.

## Script

```bash
# Training
$ python main.py mode=train debug=false

# Test
$ python main.py mode=test debug=false

# Make submission zip file
$ python main.py mode=submission debug=false
```

## Reference

TBA
