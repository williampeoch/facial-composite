# Facial Composite - Installation rapide

Ce projet genere un portrait robot a partir de CelebA avec une UI Tkinter et un pipeline VAE + algo genetique.

## 1) Prerequis

- Python 3.10 ou 3.11 recommande
- Git
- (Optionnel) Compte Kaggle + API key pour telecharger CelebA automatiquement

## 2) Installation (nouveau PC)

Depuis la racine du projet:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Dataset CelebA (non versionne)

Le dataset est volontairement hors Git car trop lourd.

### Option A - Kaggle (recommande)

1. Configure l'API Kaggle (fichier `~/.kaggle/kaggle.json`).
2. Telecharge puis dezippe:

```bash
mkdir -p data
kaggle datasets download -d jessicali9530/celeba-dataset -p data --unzip
```

3. Normalise la structure attendue par l'application:

```bash
python scripts/prepare_celeba.py --source data/celeba-dataset --target celeba
```

Par defaut, le script cree un lien symbolique vers les images (rapide, sans duplication). Si ton OS bloque les symlinks:

```bash
python scripts/prepare_celeba.py --source data/celeba-dataset --target celeba --copy-images
```

### Option B - Dataset deja disponible

Si tu as deja le dossier CelebA localement, lance:

```bash
python scripts/prepare_celeba.py --source /chemin/vers/celeba --target celeba
```

## 4) Lancement

```bash
python UI_v5.py
```

## 5) Variables d'environnement utiles (optionnel)

Si tu veux des chemins personnalises:

- `CELEBA_DIR`: dossier racine du dataset
- `CELEBA_FACES_DIR`: chemin direct vers le dossier d'images
- `CELEBA_ATTRS_PATH`: chemin direct vers `list_attr_celeba.csv`
- `VAE_WEIGHTS_PATH`: chemin direct vers `vae_128_epoch-10.pth`

Exemple:

```bash
export CELEBA_DIR=/data/celeba
export VAE_WEIGHTS_PATH=/models/vae_128_epoch-10.pth
python UI_v5.py
```

## 6) Arborescence attendue

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

## 7) Notes importantes

- Au premier lancement, `torchvision` peut telecharger un modele de segmentation (DeepLab) utilise pour le retrait de fond.
- Ne versionne pas `celeba/` dans Git (volume tres important).
- Ne versionne jamais un fichier `.env.local` contenant des cles API.
