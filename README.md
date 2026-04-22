# Facial Composite - Quick Setup

This project generates a facial composite from CelebA using a Tkinter UI and a VAE + genetic algorithm pipeline.

## 1) Requirements

- Python 3.10 or 3.11 or 3.12 recommended
- Git
- (Optional) Kaggle account + API key to download CelebA automatically

## 2) Installation (new machine)

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) CelebA dataset (not versioned)

The dataset is intentionally not tracked in Git because it is too large.

### Option A - Kaggle (recommended)

1. Configure Kaggle API (`~/.kaggle/kaggle.json`).
2. Download and unzip:

```bash
mkdir -p data
kaggle datasets download -d jessicali9530/celeba-dataset -p data --unzip
```

3. Normalize the dataset structure expected by the app:

```bash
python scripts/prepare_celeba.py --source data/celeba-dataset --target celeba
```

By default, the script creates a symbolic link to images (fast, no duplication). If your OS blocks symlinks:

```bash
python scripts/prepare_celeba.py --source data/celeba-dataset --target celeba --copy-images
```

### Option B - Dataset already available locally

If you want to download manually (without Kaggle CLI), use:

- Dataset page: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
- Download from your browser, unzip it (for example to `data/celeba-dataset`), then run:

```bash
python scripts/prepare_celeba.py --source data/celeba-dataset --target celeba
```

If you already have a local CelebA folder, run:

```bash
python scripts/prepare_celeba.py --source /path/to/celeba --target celeba
```

## 4) Run

```bash
python UI_v5.py
```

## 5) Useful environment variables (optional)

If you want custom paths:

- `CELEBA_DIR`: dataset root folder
- `CELEBA_FACES_DIR`: direct path to the image folder
- `CELEBA_ATTRS_PATH`: direct path to `list_attr_celeba.csv`
- `VAE_WEIGHTS_PATH`: direct path to `vae_128_epoch-10.pth`

Example:

```bash
export CELEBA_DIR=/data/celeba
export VAE_WEIGHTS_PATH=/models/vae_128_epoch-10.pth
python UI_v5.py
```

## 6) Expected project layout

```text
facial-composite-final/
  UI_v5.py
  celeba_ga.py
  vae_128_epoch-10.pth
  celeba/
    list_attr_celeba.csv
    img_align_celeba/
      img_align_celeba/
        000001.jpg
        ...
```

## 7) Important notes

- On first run, `torchvision` may download a segmentation model (DeepLab) used for background removal.
- Do not version `celeba/` in Git (very large volume).
- Never commit a `.env.local` file containing API keys.
