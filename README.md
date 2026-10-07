# RestoreLab

**Motion deblurring and denoising: a comparison of variational methods, Plug-and-Play, and deep learning.**

RestoreLab is an experimental computational imaging project for reconstructing RGB images degraded by motion blur and Gaussian noise. Through Jupyter notebooks, it compares Total Variation regularization, neural and generative priors, and networks trained for image restoration, using a shared degradation model and image quality metrics.

## The problem

The goal is to estimate a clean image `x` from a degraded observation:

```text
y = Kx + noise
```

`K` represents motion blur. In the implementation, observations are also clipped to the range `[0, 1]`.

| Parameter | Configuration |
| --- | --- |
| Images | RGB, 256 × 256 pixels |
| Blur | Motion blur, 9 × 9 kernel, 45° angle |
| Noise levels | 0.005, 0.01, 0.05, 0.1 |
| Metrics | PSNR and SSIM; relative error (RE) in some modules |

The degradation model is defined in [utilities/degradation.py](utilities/degradation.py).

## Implemented methods

| Method | Description | Notebook |
| --- | --- | --- |
| Total Variation | Variational regularization with a Chambolle–Pock solver | [tv_regularisation.ipynb](tv_regularisation.ipynb) |
| Plug-and-Play HQS | Half-Quadratic Splitting with a pretrained, frozen DRUNet denoiser | [hqs_pnp_heuristic.ipynb](hqs_pnp_heuristic.ipynb) |
| HQSNet | Unrolled HQS algorithm with learned parameters and a frozen DRUNet prior | [hqs_pnp_net.ipynb](hqs_pnp_net.ipynb) |
| NAFNet | Neural network trained for deblurring and denoising | [naf_net.ipynb](naf_net.ipynb) |
| Deep Generative Prior | BigGAN latent optimization, with the class estimated using ResNet50 | [dgp_regularisation.ipynb](dgp_regularisation.ipynb) |
| GAN | Generative adversarial network training experiment | [gan.ipynb](gan.ipynb) |

The [final_comparison.ipynb](final_comparison.ipynb) notebook collects visual comparisons and metrics for TV, PnP-HQS, NAFNet, and DGP.

## Dataset

The experiments load the `benjamin-paine/imagenet-1k-256x256` dataset through Hugging Face Datasets. Images are converted to RGB, resized to 256 × 256, and converted to tensors.

Subset sizes are defined in [utilities/config.py](utilities/config.py):

- **Training:** 10,000 images.
- **Validation:** 500 images.
- **Test:** 200 images.

The same file contains `TO_RECONSTRUCT_INDEXES`, the indices used for qualitative comparisons. The initial load requires access to the dataset and storage space for the local cache.

## Project structure

```text
Project/
├── IPPy/                      # Operators, solvers, and metrics for inverse problems
├── models/
│   ├── heuristic_tv_regularizer.py
│   ├── heuristic_hqs_pnp.py
│   ├── heuristic_dgp.py
│   ├── network_hqs_net.py
│   ├── network_unet.py         # Architecture used for DRUNet
│   ├── network_gan.py
│   └── basicblock.py
├── utilities/
│   ├── config.py              # Dataset sizes and test indices
│   ├── degradation.py         # Blur and noise
│   ├── image_dataset.py       # Image preprocessing
│   └── plotter.py             # Displaying and saving comparisons
├── weights/                   # Pretrained weights and checkpoints
├── results/                   # Experiment results
├── SPECIFICA.pdf
├── tv_regularisation.ipynb
├── hqs_pnp_heuristic.ipynb
├── hqs_pnp_net.ipynb
├── naf_net.ipynb
├── dgp_regularisation.ipynb
├── gan.ipynb
└── final_comparison.ipynb
```

## Environment setup

The project uses Python and PyTorch. A CUDA-compatible GPU is recommended for training and reconstruction with generative priors.

Example starting environment in **PowerShell**, with commands run from the `Project/` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision numpy matplotlib pillow scikit-image datasets tqdm jupyterlab ipykernel
.\.venv\Scripts\python.exe -m ipykernel install --user --name restorelab --display-name "Python (RestoreLab)"
```

Additional dependencies for DGP and NAFNet, also imported by the final comparison notebook:

```powershell
.\.venv\Scripts\python.exe -m pip install pytorch-pretrained-biggan focal-frequency-loss
```

These commands list the dependencies required by the main imports; the folder does not contain an environment with pinned versions verified across all notebooks. To use CUDA, choose a PyTorch build compatible with your hardware.

### NAFNet module outside the folder

`naf_net.ipynb` and `final_comparison.ipynb` import `nafnet_heightmap.network_naf_net`. In the current structure, this module is located in the parent directory, alongside `Project/`.

To run these notebooks while keeping the kernel in `Project/`, add the following before the imports:

```python
import sys
from pathlib import Path

sys.path.append(str(Path.cwd().parent))
```

If you distribute only the `Project/` folder, you must also include the `nafnet_heightmap` package or adjust the imports: both notebooks currently depend on code outside this folder.

## Running the notebooks

Start JupyterLab from the `Project/` folder:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab
```

1. Open the notebook for the method you want to run and select the `Python (RestoreLab)` kernel.
2. Keep `Project/` as the working directory: the `models` and `utilities` imports and the `./weights/...` paths depend on this location.
3. Check the dataset configuration, device, training or reconstruction parameters, and checkpoint paths.
4. Run the cells in order. For learned models, train first or prepare a compatible checkpoint.
5. Run the evaluation cells to display the original image, degraded observation, and reconstruction with their respective metrics.

To start with the variational method, open `tv_regularisation.ipynb`. Then move on to PnP-HQS and the learned models; run the final comparison after preparing the dependencies and weights for the methods involved.

### Weights and checkpoints

| Method | Expected resource |
| --- | --- |
| PnP-HQS and HQSNet | `weights/DRUNet/drunet_color.pth` |
| HQSNet | Training configured to use `weights/HQSNet/HQS_checkpoint.pth`; evaluation uses the corresponding `_best.pth` file |
| NAFNet | `weights/NAFNet/NAFImgDeblur&Denoise.pth` |
| GAN | `weights/GAN/gan.pth` |
| DGP | BigGAN `biggan-deep-256` and ResNet50 with ImageNet weights, loaded through their respective libraries |

The presence of these directories does not guarantee that all weights are available. Check the required files before evaluation; the first use of the pretrained DGP models may require a download.

## Evaluation and reproducibility

PSNR and SSIM compare the reconstruction with the clean image. The notebooks produce visual comparisons and, depending on the experiment, save images and checkpoints to the configured paths.

In the TV, PnP-HQS, and DGP notebooks, the best regularization parameter or schedule is selected using PSNR against the ground truth. This evaluation therefore assumes that the clean image is available; real data without a reference requires a different selection criterion.

For reproducible comparisons, record the degradation configuration, random seeds, selected images, checkpoints, and reconstruction parameters.

**DGP note:** the notebook introduction mentions StyleGAN-XL, but the current implementation in `models/heuristic_dgp.py` uses BigGAN. This README describes the implementation in the code.
