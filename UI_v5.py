######################design plan################################################
# page 1 : welcome page & instructions
# page 2 : criteria checkboxes
# page 3 : image selection connected to VAE + Genetic Algorithm
#######################import libraries##########################################
import os
import tkinter as tk
from tkinter import messagebox

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageTk
from torch import nn
from torchvision import models

from celeba_ga import (
    Individual,
    create_next_generation,
    initialize_population_from_attributes,
)

########################configure################################################
title = "Portrait Robot"
size = "720x720"
IMG_SIZE = 128
Z_DIM = 256
BASE = 64

# ================== PARAMETRES GA / VAE ==================
GA_ELITE_SIZE = 1
GA_MUTATION_STD = 0.40
GA_RANDOM_INJECTION_COUNT = 0
GA_LATENT_CLAMP = 2.8
GA_PROJECTION_BLEND = 0.25
GA_PROJECTION_STEPS = 1
GA_REMOVE_PARENT_BACKGROUND = True
GA_SEGMENTATION_THRESHOLD = 0.55
GA_SEGMENTATION_BG_VALUE = 0.5
GA_PERSON_CLASS_INDEX = 15
SEGMENTATION_MEAN = (0.485, 0.456, 0.406)
SEGMENTATION_STD = (0.229, 0.224, 0.225)
# ===========================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_CANDIDATE_DIRS = [
    os.path.join(BASE_DIR, "faces"),
    os.path.join(BASE_DIR, "celeba", "img_align_celeba", "img_align_celeba"),
    os.path.join(BASE_DIR, "celeba", "img_align_celeba"),
]
ATTR_CANDIDATE_PATHS = [
    os.path.join(BASE_DIR, "celeba", "list_attr_celeba.csv"),
]
MODEL_CANDIDATE_PATHS = [
    os.path.join(BASE_DIR, "vae_128_epoch-10.pth"),
]


def resolve_dir(candidates):
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return None


def resolve_file(candidates):
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


faces_path = resolve_dir(FACE_CANDIDATE_DIRS)
attrs_path = resolve_file(ATTR_CANDIDATE_PATHS)
model_path = resolve_file(MODEL_CANDIDATE_PATHS)

# aesthetic
BG = "#ebe8e8"  # lightgrey
CARD = "#ffffff"
PRIMARY = "#96ebcb"  # lightblue
SECONDARY = "#7d85f5"  # light purple
ACCENT = "#ff5252"  # pastel reload
TEXT = "#333"  # black

TITLE_FONT = ("Helvetica", 20, "bold")
SUBTITLE_FONT = ("Helvetica", 16, "bold")
TEXT_FONT = ("Helvetica", 11)


###################### VAE + GA BRIDGE #########################################
class ConvVAE128(nn.Module):
    def __init__(self, in_channels=3, z_dim=256, base=64):
        super().__init__()

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, base * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.enc_out = base * 8 * 4 * 4
        self.fc_mu = nn.Linear(self.enc_out, z_dim)
        self.fc_logvar = nn.Linear(self.enc_out, z_dim)
        self.fc_dec = nn.Linear(z_dim, self.enc_out)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 8, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 8, base * 4, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, in_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(h.size(0), 512, 4, 4)
        return self.dec(h)


class VAEGABridge:
    def __init__(self, faces_dir, attrs_csv, vae_weights):
        if not os.path.isdir(faces_dir):
            raise FileNotFoundError(f"Dossier images introuvable: {faces_dir}")
        if not os.path.isfile(attrs_csv):
            raise FileNotFoundError(f"Fichier attributs introuvable: {attrs_csv}")
        if not os.path.isfile(vae_weights):
            raise FileNotFoundError(f"Checkpoint VAE introuvable: {vae_weights}")

        self.faces_dir = faces_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvVAE128(in_channels=3, z_dim=Z_DIM, base=BASE).to(self.device)

        state_dict = torch.load(vae_weights, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        self.seg_model = None
        if GA_REMOVE_PARENT_BACKGROUND:
            try:
                self.seg_model = self._load_person_segmentation_model()
            except Exception:
                self.seg_model = None

        self.attr_df = pd.read_csv(attrs_csv)
        available_images = {
            name
            for name in os.listdir(self.faces_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        }
        self.attr_df = self.attr_df[self.attr_df["image_id"].isin(available_images)].reset_index(
            drop=True
        )

        if self.attr_df.empty:
            raise RuntimeError("Aucune correspondance entre CSV attributs et images disponibles.")

        self._latent_cache = {}

    def _image_id_to_path(self, image_id):
        return os.path.join(self.faces_dir, image_id)

    def load_original_image(self, image_id):
        path = self._image_id_to_path(image_id)
        if not os.path.isfile(path):
            return None
        with Image.open(path) as img:
            return img.convert("RGB").copy()

    def _image_to_tensor(self, image_id):
        path = self._image_id_to_path(image_id)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image introuvable pour encodage: {path}")

        with Image.open(path) as img:
            img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            arr = arr * 2.0 - 1.0

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def encode_image_id_to_latent(self, image_id):
        if image_id in self._latent_cache:
            return self._latent_cache[image_id].clone()

        with torch.no_grad():
            x = self._image_to_tensor(image_id)
            mu, _ = self.model.encode(x)
            z = mu.squeeze(0).detach().cpu()

        self._latent_cache[image_id] = z
        return z.clone()

    def encode_tensor_to_latent(self, image_tensor):
        with torch.no_grad():
            mu, _ = self.model.encode(image_tensor.unsqueeze(0).to(self.device))
        return mu.squeeze(0).detach().cpu()

    def decode_latent_to_pil(self, z):
        with torch.no_grad():
            z_batch = z.unsqueeze(0) if z.ndim == 1 else z
            recon = self.model.decode(z_batch.to(self.device)).squeeze(0).detach().cpu()
            recon = (recon * 0.5 + 0.5).clamp(0.0, 1.0)

        arr = (recon.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(arr)

    def project_latent_to_manifold(self, z, blend=GA_PROJECTION_BLEND, steps=GA_PROJECTION_STEPS):
        if blend <= 0 or steps <= 0:
            return z

        z_batch = z.unsqueeze(0).to(self.device)
        with torch.no_grad():
            for _ in range(max(0, steps)):
                recon = self.model.decode(z_batch)
                mu, _ = self.model.encode(recon)
                z_batch = (1.0 - blend) * z_batch + blend * mu
        return z_batch.squeeze(0).detach().cpu()

    def project_latent(self, z):
        projected = self.project_latent_to_manifold(z)
        return torch.clamp(projected, -GA_LATENT_CLAMP, GA_LATENT_CLAMP)

    def _load_person_segmentation_model(self):
        try:
            from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights

            seg_model = models.segmentation.deeplabv3_resnet50(
                weights=DeepLabV3_ResNet50_Weights.DEFAULT
            ).to(self.device)
        except Exception:
            seg_model = models.segmentation.deeplabv3_resnet50(pretrained=True).to(self.device)
        seg_model.eval()
        return seg_model

    @staticmethod
    def _denorm(img_tensor):
        return (img_tensor * 0.5 + 0.5).clamp(0, 1)

    def remove_background_from_image_tensor(
        self,
        img_tensor,
        threshold=GA_SEGMENTATION_THRESHOLD,
        bg_value=GA_SEGMENTATION_BG_VALUE,
    ):
        if self.seg_model is None:
            return img_tensor

        image_01 = self._denorm(img_tensor).to(self.device)
        seg_input = image_01.clone()
        for c in range(3):
            seg_input[c] = (seg_input[c] - SEGMENTATION_MEAN[c]) / SEGMENTATION_STD[c]
        seg_input = seg_input.unsqueeze(0)

        with torch.no_grad():
            seg_out = self.seg_model(seg_input)["out"][0]
            probs = torch.softmax(seg_out, dim=0)
            person_mask = (probs[GA_PERSON_CLASS_INDEX] > threshold).float().unsqueeze(0)

        background = torch.full_like(image_01, bg_value)
        fg_img = image_01 * person_mask + background * (1.0 - person_mask)
        return (fg_img * 2.0 - 1.0).detach().cpu()

    def remove_background_from_latent(self, z):
        decoded = self.model.decode(z.unsqueeze(0).to(self.device)).squeeze(0).detach().cpu()
        bg_removed = self.remove_background_from_image_tensor(decoded)
        return self.encode_tensor_to_latent(bg_removed)

    def initialize_population(self, selected_attrs, pop_size):
        used_attrs = {k: int(v) for k, v in selected_attrs.items() if k in self.attr_df.columns}
        return initialize_population_from_attributes(
            attr_df=self.attr_df,
            selected_attrs=used_attrs,
            pop_size=pop_size,
            encode_image_id_to_latent=self.encode_image_id_to_latent,
        )

    def evolve_population(
        self,
        population,
        selected_indices,
        pop_size,
        elite_size=GA_ELITE_SIZE,
        mutation_std=GA_MUTATION_STD,
        random_injection_count=GA_RANDOM_INJECTION_COUNT,
    ):
        working_population = population
        if GA_REMOVE_PARENT_BACKGROUND and self.seg_model is not None and selected_indices:
            selected_idx = selected_indices[0]
            selected_parent = population[selected_idx]
            cleaned_latent = self.remove_background_from_latent(selected_parent.z)
            working_population = list(population)
            working_population[selected_idx] = Individual(
                image_id=selected_parent.image_id,
                z=cleaned_latent,
                source=f"{selected_parent.source}+bg_removed",
                score=selected_parent.score,
            )

        return create_next_generation(
            population=working_population,
            selected_indices=selected_indices,
            attr_df=self.attr_df,
            pop_size=pop_size,
            elite_size=elite_size,
            mutation_std=mutation_std,
            encode_image_id_to_latent=self.encode_image_id_to_latent,
            random_injection_count=random_injection_count,
            latent_clamp=GA_LATENT_CLAMP,
            project_latent=self.project_latent,
        )


######################app class configuration####################################
class PortraitApp:
    def __init__(self, root):
        self.root = root
        self.root.title(title)
        self.root.geometry(size)
        self.root.configure(bg=BG)  # light grey

        self.backend = None

        # set up pages
        self.welcome_page = tk.Frame(root, bg=BG)  # page 1 - button "commencer" >>
        self.entries_page = tk.Frame(root, bg=BG)  # page 2 + pop up window
        self.images_page = tk.Frame(root, bg=BG)  # page 3 + ga loop

        # setup first page
        self.welcome_page.pack(fill="both", expand=True)

        # declare functions to create pages
        self.create_welcome_page()
        self.create_entries_page()
        self.create_images_page()

    ######################## WELCOME PAGE #######################################
    def create_welcome_page(self):
        title_label = tk.Label(
            self.welcome_page,
            text="BIENVENUE",
            font=("Helvetica", 20, "bold"),
            bg="#fafafa",
        )
        title_label.pack(pady=40)

        instructions = tk.Label(
            self.welcome_page,
            text=(
                "Objectif : retrouver le visage cible\n"
                "Etape 1 : cochez les caractéristiques qui correspondent à votre cible\n"
                "Etape 2 : cliquez sur 'Suivant' sous l'image la plus proche\n"
                "Etape 3 : cliquez sur 'Choisir' quand vous pensez avoir trouvé"
            ),
            font=("Helvetica", 12),
            bg="#fafafa",
            justify="center",
        )
        instructions.pack(pady=20)

        start_btn = tk.Button(
            self.welcome_page,
            text="Commencer",
            bg=PRIMARY,
            fg="white",
            font=("Helvetica", 12, "bold"),
            padx=10,
            pady=5,
            relief="flat",
            command=self.change_to_entries,
        )
        start_btn.pack(pady=30)

    def change_to_entries(self):
        self.welcome_page.pack_forget()
        self.entries_page.pack(fill="both", expand=True)

    ######################## ENTRIES PAGE #######################################
    def create_entries_page(self):
        titre = tk.Label(
            self.entries_page,
            text="Veuillez cocher les caractéristiques qui correspondent à la personne que vous cherchez",
            fg="black",
            font=("Helvetica", 12, "bold"),
            justify="center",
        )
        titre.pack(pady=40)

        # declare array & dictionary
        self.attributs = {
            "5_o_Clock_Shadow": 0,
            "Arched_Eyebrows": 0,
            "Attractive": 0,
            "Bags_Under_Eyes": -1,
            "Bald": -1,
            "Bangs": -1,
            "Big_Lips": -1,
            "Big_Nose": -1,
            "Black_Hair": -1,
            "Blond_Hair": -1,
            "Blurry": 0,
            "Brown_Hair": -1,
            "Bushy_Eyebrows": -1,
            "Chubby": -1,
            "Double_Chin": -1,
            "Eyeglasses": -1,
            "Goatee": -1,
            "Gray_Hair": -1,
            "Heavy_Makeup": -1,
            "High_Cheekbones": 0,
            "Male": -1,
            "Mouth_Slightly_Open": 0,
            "Mustache": -1,
            "Narrow_Eyes": -1,
            "No_Beard": -1,
            "Oval_Face": -1,
            "Pale_Skin": -1,
            "Pointy_Nose": -1,
            "Receding_Hairline": -1,
            "Rosy_Cheeks": -1,
            "Sideburns": 0,
            "Smiling": 0,
            "Straight_Hair": -1,
            "Wavy_Hair": -1,
            "Wearing_Earrings": 0,
            "Wearing_Hat": 0,
            "Wearing_Lipstick": -1,
            "Wearing_Necklace": 0,
            "Wearing_Necktie": 0,
            "Young": -1,
        }

        self.cles = [
            "Sacs sous les yeux",
            "Chauve",
            "Frange",
            "Grosses lèvres",
            "Grand nez",
            "Cheveux noirs",
            "Cheveux blonds",
            "Cheveux bruns",
            "Sourcils épais",
            "Potelé",
            "Double menton",
            "Lunettes",
            "Bouc",
            "Cheveux gris",
            "Maquillage lourd",
            "Homme",
            "Moustache",
            "Yeux étroits",
            "Pas de barde",
            "Visage ovale",
            "Peau pâle",
            "Nez pointu",
            "Calvitie naissante",
            "Joues roses",
            "Cheveux lisses",
            "Cheveux bouclés",
            "Porte du rouge à lèvre",
            "Jeune",
        ]

        self.add_checkboxes()

        self.resultat = {}
        for k in range(len(self.cles)):
            self.resultat[self.cles[k]] = -1

        bouton = tk.Button(
            self.entries_page,
            text="Valider",
            command=self.button_clicked,
            bg=PRIMARY,
            fg="white",
            font=("Helvetica", 12, "bold"),
            padx=10,
            pady=5,
            relief="flat",
        )
        bouton.pack(pady=10)

        # back page button
        bouton_retour_welcome = tk.Button(
            self.entries_page,
            text="⬅Retour",
            bg=PRIMARY,
            fg="white",
            font=("Helvetica", 12, "bold"),
            padx=10,
            pady=5,
            relief="flat",
            command=self.retour_welcome,
        )
        bouton_retour_welcome.pack(pady=5)

    def retour_welcome(self):
        self.entries_page.pack_forget()
        self.welcome_page.pack(fill="both", expand=True)

    def add_checkboxes(self):
        l_values = [
            "Sacs sous les yeux",
            "Chauve",
            "Frange",
            "Grosses lèvres",
            "Grand nez",
            "Cheveux noirs",
            "Cheveux blonds",
            "Cheveux bruns",
            "Sourcils épais",
            "Potelé",
            "Double menton",
            "Lunettes",
            "Bouc",
            "Cheveux gris",
            "Maquillage lourd",
            "Homme",
            "Moustache",
            "Yeux étroits",
            "Pas de barde",
            "Visage ovale",
            "Peau pâle",
            "Nez pointu",
            "Calvitie naissante",
            "Joues roses",
            "Cheveux lisses",
            "Cheveux bouclés",
            "Porte du rouge à lèvre",
            "Jeune",
        ]
        max_per_line = 5  # nombre de boutons par ligne
        line_frame = None

        for i, checkbox_value in enumerate(l_values):
            if i % max_per_line == 0:
                line_frame = tk.Frame(self.entries_page)
                line_frame.pack(anchor="w", pady=8)

            checkbox_var = tk.BooleanVar(value=False)
            tk.Checkbutton(
                line_frame,
                text=checkbox_value,
                variable=checkbox_var,
                command=lambda v=checkbox_value, var=checkbox_var: self.on_checkbox_change(v, var),
            ).pack(side="left", padx=10)

    def on_checkbox_change(self, checkbox_value, variable):
        cocher = variable.get()
        if cocher:
            self.resultat[checkbox_value] = 1
        else:
            self.resultat[checkbox_value] = -1

    def button_clicked(self):
        i = 0
        for k in self.attributs.keys():
            if self.attributs[k] != 0:
                self.attributs[k] = self.resultat[self.cles[i]]
                i += 1

        # create new window
        self.nouvelle = tk.Toplevel(self.root)
        self.nouvelle.title("Caractéristiques choisies")
        self.nouvelle.geometry(size)

        label = tk.Label(
            self.nouvelle,
            text="Voici les caractéristiques que vous avez sélectionnées",
            fg=TEXT,
            font=("Helvetica", 16, "bold"),
        )
        label.pack(pady=5)

        selected = []
        display_text = ""
        for k, v in self.resultat.items():
            if v == 1:
                selected.append(k)
                display_text += f"\n{k}"

        if display_text == "":
            display_text = "Aucune caractéristique sélectionnée"

        label2 = tk.Label(self.nouvelle, text=display_text, fg="black", font=("Helvetica", 10))
        label2.pack(pady=5)

        bouton2 = tk.Button(
            self.nouvelle,
            text="Suivant",
            bg=PRIMARY,
            fg="white",
            font=("Helvetica", 12, "bold"),
            padx=10,
            pady=5,
            relief="flat",
            command=self.change_to_images,
        )
        bouton2.pack()

    def _ensure_backend_ready(self):
        if self.backend is not None:
            return True

        if faces_path is None:
            messagebox.showerror(
                "Erreur",
                "Aucune image trouvée. Vérifie les dossiers:\n- " + "\n- ".join(FACE_CANDIDATE_DIRS),
            )
            return False
        if attrs_path is None:
            messagebox.showerror("Erreur", "Fichier d'attributs CelebA introuvable.")
            return False
        if model_path is None:
            messagebox.showerror("Erreur", "Checkpoint VAE introuvable.")
            return False

        try:
            self.root.config(cursor="watch")
            self.root.update_idletasks()
            self.backend = VAEGABridge(
                faces_dir=faces_path,
                attrs_csv=attrs_path,
                vae_weights=model_path,
            )
            return True
        except Exception as exc:
            messagebox.showerror("Erreur chargement modèle", str(exc))
            self.backend = None
            return False
        finally:
            self.root.config(cursor="")

    def change_to_images(self):
        if not self._ensure_backend_ready():
            return

        self.entries_page.pack_forget()
        self.images_page.pack(fill="both", expand=True)

        if hasattr(self, "nouvelle"):
            self.nouvelle.destroy()

        self.reset()

    def _selected_attributes_for_ga(self):
        return {k: v for k, v in self.attributs.items() if v != 0}

    @staticmethod
    def _clone_population(population):
        cloned = []
        for ind in population:
            cloned.append(
                Individual(
                    image_id=ind.image_id,
                    z=ind.z.clone(),
                    source=ind.source,
                    score=ind.score,
                )
            )
        return cloned

    ############# IMAGES PAGE ###################################################
    def create_images_page(self):
        self.images = []  # store rendered images
        self.current_population = []
        self.history = []  # list of tuple(round, population)
        self.round = 0

        # title
        title_label = tk.Label(
            self.images_page,
            text="Sélectionnez l'image la plus ressemblante",
            font=("Helvetica", 16, "bold"),
            bg="#fafafa",
        )
        title_label.pack(pady=10)

        subtitle = tk.Label(
            self.images_page,
            text="Cliquez sur 'Suivant' sous l'image la plus proche pour générer la suite.",
            font=("Helvetica", 10),
            bg="#fafafa",
        )
        subtitle.pack(pady=2)

        # counter
        self.counter_label = tk.Label(self.images_page, text="Round: 0", font=("Helvetica", 12), bg="#fafafa")
        self.counter_label.pack()

        # images container
        self.images_frame = tk.Frame(self.images_page, bg="#fafafa")
        self.images_frame.pack(expand=True, fill="both", padx=10, pady=10)

        self.labels = []

        # back images button
        back_button_entries = tk.Button(
            self.images_page,
            text="Retour images",
            bg=PRIMARY,
            fg="white",
            font=("Helvetica", 12, "bold"),
            padx=10,
            pady=5,
            relief="flat",
            command=self.go_back,
        )
        back_button_entries.pack(pady=5)

        # setup layout 5 images & 2 buttons each
        for i in range(5):
            frame = tk.Frame(self.images_frame, bg="#fafafa", bd=1, relief="solid")
            frame.grid(row=0, column=i)
            self.images_frame.columnconfigure(i, weight=1)

            lbl = tk.Label(frame, bg="#fafafa")
            lbl.pack(pady=5)

            tk.Button(
                frame,
                text="Choisir",
                command=lambda i=i: self.select_image(i),
                bg=PRIMARY,
                fg="white",
                font=("Helvetica", 12, "bold"),
                padx=10,
                pady=5,
                relief="flat",
            ).pack(fill="x", pady=2)

            tk.Button(
                frame,
                text="Suivant",
                command=lambda i=i: self.next_images(i),
                bg=PRIMARY,
                fg="white",
                font=("Helvetica", 12, "bold"),
                padx=10,
                pady=5,
                relief="flat",
            ).pack(fill="x")

            self.labels.append(lbl)

        # bottom buttons
        bottom = tk.Frame(self.images_page, bg="#fafafa")
        bottom.pack(fill="x", pady=10)

        tk.Button(
            bottom,
            text="⬅ Retour",
            command=self.back_entries,
            bg=PRIMARY,
            fg="white",
            font=("Helvetica", 12, "bold"),
            padx=10,
            pady=5,
            relief="flat",
        ).pack(side="left", padx=10)

        tk.Button(
            bottom,
            text="🔄 Reset",
            command=self.reset,
            bg=PRIMARY,
            fg="white",
            font=("Helvetica", 12, "bold"),
            padx=10,
            pady=5,
            relief="flat",
        ).pack(side="right", padx=10)

    def back_entries(self):  # back to entries page
        self.images_page.pack_forget()
        self.entries_page.pack(fill="both", expand=True)

    def _render_population(self):
        if not self.current_population:
            return

        self.images.clear()
        for lbl, ind in zip(self.labels, self.current_population):
            # Show real dataset images when available (less blurry).
            # Decode only synthetic children that do not map to a real file.
            pil_img = self.backend.load_original_image(ind.image_id)
            if pil_img is None:
                pil_img = self.backend.decode_latent_to_pil(ind.z)

            pil_img = pil_img.resize((128, 128))
            tk_img = ImageTk.PhotoImage(pil_img)
            self.images.append(tk_img)
            lbl.config(image=tk_img)
            lbl.image = tk_img

        self.counter_label.config(text=f"Round: {self.round}")

    def _initialize_population_from_selection(self):
        selected_attrs = self._selected_attributes_for_ga()
        pop_size = len(self.labels)

        population = self.backend.initialize_population(selected_attrs=selected_attrs, pop_size=pop_size)
        if not population:
            raise RuntimeError("Impossible d'initialiser la population.")

        self.current_population = self._clone_population(population[:pop_size])
        self.round = 1
        self._render_population()

    def next_images(self, index):
        if not self.current_population:
            return

        self.history.append((self.round, self._clone_population(self.current_population)))
        pop_size = len(self.labels)
        try:
            self.current_population = self.backend.evolve_population(
                population=self.current_population,
                selected_indices=[index],
                pop_size=pop_size,
                elite_size=GA_ELITE_SIZE,
                mutation_std=GA_MUTATION_STD,
                random_injection_count=GA_RANDOM_INJECTION_COUNT,
            )
            self.round += 1
            self._render_population()
        except Exception as exc:
            messagebox.showerror("Erreur génération", str(exc))
            if self.history:
                prev_round, prev_population = self.history.pop()
                self.round = prev_round
                self.current_population = prev_population
                self._render_population()

    def select_image(self, index):  # final image selected - new window with image shown
        if index < 0 or index >= len(self.images):
            return

        new_win = tk.Toplevel(self.root)
        new_win.title("Image choisie")
        new_win.geometry(size)

        img = self.images[index]
        lbl = tk.Label(new_win, image=img)
        lbl.image = img
        lbl.pack(padx=20, pady=20)

        tk.Label(new_win, text=f"Trouvé en {self.round} tours", font=("Helvetica", 12)).pack(pady=5)

    def go_back(self):  # back button selected - show previous generation
        if not self.history:
            return

        prev_round, prev_population = self.history.pop()
        self.round = prev_round
        self.current_population = prev_population
        self._render_population()

    def reset(self):  # reset button selected - restart from selected attributes
        if not self._ensure_backend_ready():
            return

        self.round = 0
        self.history.clear()
        try:
            self._initialize_population_from_selection()
        except Exception as exc:
            messagebox.showerror("Erreur reset", str(exc))


########################### MAIN PROGRAM ######################################
if __name__ == "__main__":
    root = tk.Tk()
    app = PortraitApp(root)
    root.mainloop()
