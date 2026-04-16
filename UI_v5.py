######################design plan################################################
#page 1 : welcome page & instructions
#page 2 : criterea checkboxes
#page 3 : image selection - show chosen image
#######################import libraries##########################################
import tkinter as tk
from PIL import Image, ImageTk
import random
import os
########################configure################################################
title = "Portrait Robot"
size = "720x720"
faces_path = "./faces" #A CHANGER POUR LE DOSSIER AVEC DES PHOTOS

#aesthetic
BG = "#ebe8e8" #lightgrey
CARD = "#ffffff"
PRIMARY =  "#96ebcb" #lightblue
SECONDARY = "#7d85f5" #light purple
ACCENT = "#ff5252" #pastel reload
TEXT = "#333" #black

TITLE_FONT = ("Helvetica", 20, "bold")
SUBTITLE_FONT = ("Helvetica", 16, "bold")
TEXT_FONT = ("Helvetica", 11)

######################app class configuration####################################
class PortraitApp:
    def __init__(self, root):
        self.root = root
        self.root.title(title)
        self.root.geometry(size)
        self.root.configure(bg=BG) #light grey

        #set up pages
        self.welcome_page = tk.Frame(root, bg= BG) #page 1 - button "commencer" >>
        self.entries_page = tk.Frame(root, bg = BG) #page 2 + pop up window - button "suivant" >>
        self.images_page = tk.Frame(root, bg = BG) #page 3 + pop up window

        #setup first page
        self.welcome_page.pack(fill = "both", expand=True)

        #declare functions to create pages
        self.create_welcome_page()
        self.create_entries_page()
        self.create_images_page()

    ######################## WELCOME PAGE #######################################
    def create_welcome_page(self):
        title = tk.Label(self.welcome_page, text = "BIENVENUE", font = ("Helvetica", 20, "bold"), bg = "#fafafa")
        title.pack(pady=40)
        instructions = tk.Label(self.welcome_page, text=("Objectif : retrouver le visage cible \n Etape 1 : \n  Cochez les caractéristiques qui correspondent à votre cible \n Etape 2 \n  Cliquez sur 'Suivant' pour sélectionnnez l'image qui ressemble le plus \n Cliquez sur 'Choisir' quand vous pensez avoir trouvé \n"), font = ("Helvetica", 12), bg = "#fafafa", justify = "center")
        instructions.pack(pady=20)

        start_btn = tk.Button(self.welcome_page, text = "Commencer", bg = PRIMARY, fg = "white", font = ("Helvetica", 12, "bold"), padx = 10, pady = 5, relief = "flat", command=self.change_to_entries )
        start_btn.pack(pady=30)

    def change_to_entries(self):
        self.welcome_page.pack_forget()
        self.entries_page.pack(fill="both", expand = True)

    ######################## ENTRIES PAGE #######################################
    def create_entries_page(self):
        titre = tk.Label(self.entries_page, text="Veuillez cocher les caractéristiques qui correspondent à la personne que vous cherchez",fg="black", font=("Helvetica", 12, "bold"), justify = "center")
        titre.pack(pady=40)

        #declare array & dictionnary
        self.attributs= {"5_o_Clock_Shadow" : 0, "Arched_Eyebrows" : 0, "Attractive" : 0, "Bags_Under_Eyes" : -1, "Bald" : -1, "Bangs" : -1, "Big_Lips" : -1, "Big_Nose" : -1, "Black_Hair" : -1, "Blond_Hair" : -1,
                    "Blurry" : 0, "Brown_Hair" : -1, "Bushy_Eyebrows" : -1, "Chubby" : -1, "Double_Chin" : -1, "Eyeglasses" : -1, "Goatee" : -1, "Gray_Hair" : -1, "Heavy_Makeup" : -1, "High_Cheekbones" : 0,
                    "Male" : -1, "Mouth_Slightly_Open" : 0, "Mustache" : -1, "Narrow_Eyes" : -1, "No_Beard" : -1, "Oval_Face" : -1, "Pale_Skin" : -1, "Pointy_Nose" : -1, "Receding_Hairline" : -1, "Rosy_Cheeks" : -1,
                    "Sideburns" : 0, "Smiling" : 0, "Straight_Hair" : -1, "Wavy_Hair" : -1, "Wearing_Earrings" : 0, "Wearing_Hat" : 0, "Wearing_Lipstick" : -1, "Wearing_Necklace" : 0, "Wearing_Necktie" : 0, "Young" : -1}

        self.cles = ["Sacs sous les yeux","Chauve", "Frange","Grosses lèvres", "Grand nez", "Cheveux noirs", "Cheveux blonds",
                "Cheveux bruns", "Sourcils épais", "Potelé", "Double menton", "Lunettes", "Bouc", "Cheveux gris", "Maquillage lourd",
                "Homme", "Moustache", "Yeux étroits", "Pas de barde", "Visage ovale", "Peau pâle", "Nez pointu", "Calvitie naissante", "Joues roses",
                "Cheveux lisses", "Cheveux bouclés", "Porte du rouge à lèvre", "Jeune"]

        self.add_checkboxes()

        self.resultat=dict()
        for k in range (len(self.cles)):
            self.resultat[self.cles[k]]=-1

        bouton = tk.Button(self.entries_page, text ="Valider", command= self.button_clicked, bg = PRIMARY, fg = "white", font = ("Helvetica", 12, "bold"), padx = 10, pady = 5, relief = "flat")
        bouton.pack(pady=10)

        #back page button
        bouton_retour_welcome = tk.Button(self.entries_page, text = "⬅Retour", bg = PRIMARY, fg = "white", font = ("Helvetica", 12, "bold"), padx = 10, pady = 5, relief = "flat", command = self.retour_welcome)
        bouton_retour_welcome.pack(pady=5)

    def retour_welcome(self):
        self.entries_page.pack_forget()
        self.welcome_page.pack(fill="both", expand=True)

    def add_checkboxes(self):
        L=["Sacs sous les yeux","Chauve", "Frange","Grosses lèvres", "Grand nez", "Cheveux noirs", "Cheveux blonds",
                "Cheveux bruns", "Sourcils épais", "Potelé", "Double menton", "Lunettes", "Bouc", "Cheveux gris", "Maquillage lourd",
                "Homme", "Moustache", "Yeux étroits", "Pas de barde", "Visage ovale", "Peau pâle", "Nez pointu", "Calvitie naissante", "Joues roses",
                "Cheveux lisses", "Cheveux bouclés", "Porte du rouge à lèvre", "Jeune"]
        max_per_line = 5  # nombre de boutons par ligne
        line_frame = None

        for i in range(len(L)):
            if i % max_per_line == 0:
                line_frame = tk.Frame(self.entries_page)
                line_frame.pack(anchor="w", pady=8)
            checkbox_value = L[i]
            checkbox_var = tk.BooleanVar(value = False)  # Set the initial state (unchecked)

            checkbox = tk.Checkbutton(line_frame, text=checkbox_value, variable=checkbox_var, command=lambda v=checkbox_value, var=checkbox_var: self.on_checkbox_change(v, var)).pack(side="left", padx=10)


    def on_checkbox_change(self, checkbox_value, variable):
        cocher = variable.get()
        if cocher == True :
            self.resultat[checkbox_value]=1
        else :
            self.resultat[checkbox_value]=-1


    def button_clicked(self):
        i = 0
        for k in self.attributs.keys():
            if self.attributs[k] != 0 :
                self.attributs[k] = self.resultat[self.cles[i]]
                i+=1
        print(self.attributs)
        #create new window
        self.nouvelle = tk.Toplevel(self.root)
        self.nouvelle.title("Caractéristiques choisies")
        self.nouvelle.geometry(size)

        label = tk.Label(self.nouvelle, text="Voici les caractéristiques que vous avez sélectionnées",fg= TEXT, font=("Helvetica", 16, "bold"))
        label.pack(pady=5)
        t = tk.Text(self.nouvelle, fg="blue", padx=50)
        t.config(font=("broadway", 12))

        #show selected characteristics in new window
        selected = [] #create empty array
        display_text = "" #create empty string
        for k, v in self.resultat.items():
            if v == 1: #if checkbox ticked
                selected.append(k) #add characteristics to array
                display_text += f"\n {k}" #add characteristics to list

        if display_text == "": #if no selection made
            display_text = "Aucune caractéristique selectionnée"



        label2 = tk.Label(self.nouvelle, text= display_text,fg="black", font=("Helvetica", 10))
        label2.pack(pady=5)

        bouton2 = tk.Button(self.nouvelle, text ="Suivant", bg = PRIMARY, fg = "white", font = ("Helvetica", 12, "bold"), padx = 10, pady = 5, relief = "flat", command= self.change_to_images)
        bouton2.pack()

    def change_to_images(self):
        self.entries_page.pack_forget()
        self.images_page.pack(fill="both", expand=True)

        #close window safely
        if hasattr(self, "nouvelle"):
            self.nouvelle.destroy()


    ############# IMAGES PAGE ###################################################
    def create_images_page(self):
        # declare and initialise arrays
        self.images = [] #store images to load
        self.current = [] #store current images
        self.history = [] #store previous images - back function
        self.round = 0 #count image selection rounds

        self.files = os.listdir(faces_path) #convert image files in path to list

        # title
        title = tk.Label(self.images_page, text="Sélectionnez l'image la plus ressemblante", font=("Helvetica", 16, "bold"), bg="#fafafa")
        title.pack(pady=10)

        # counter
        self.counter_label = tk.Label(self.images_page, text="Round: 0", font=("Helvetica", 12), bg="#fafafa")
        self.counter_label.pack()

        # images container
        self.images_frame = tk.Frame(self.images_page, bg="#fafafa")
        self.images_frame.pack(expand=True, fill="both", padx=10, pady=10)

        self.labels = [] #declare lables array

        #back images button
        back_button_entries = tk.Button(self.images_page, text = "Retour images", bg = PRIMARY, fg = "white", font = ("Helvetica", 12, "bold"), padx = 10, pady = 5, relief = "flat", command = self.go_back)
        back_button_entries.pack(pady=5)

        #setup layout 5 images & 2 buttons each
        for i in range(5):
            frame = tk.Frame(self.images_frame, bg="#fafafa", bd=1, relief="solid")
            frame.grid(row=0, column=i) #window geometry

            self.images_frame.columnconfigure(i, weight=1)


            lbl = tk.Label(frame, bg="#fafafa")
            lbl.pack(pady=5)

            #final choice button
            tk.Button(frame, text="Choisir", command=lambda i=i: self.select_image(i), bg = PRIMARY, fg = "white", font = ("Helvetica", 12, "bold"), padx = 10, pady = 5, relief = "flat").pack(fill="x", pady=2)

            #select image preference
            tk.Button(frame, text="Suivant", command=lambda i=i: self.next_images(), bg = PRIMARY, fg = "white", font = ("Helvetica", 12, "bold"), padx = 10, pady = 5, relief = "flat").pack(fill="x")

            self.labels.append(lbl)

        # bottom buttons
        bottom = tk.Frame(self.images_page, bg="#fafafa")
        bottom.pack(fill="x", pady=10)

        #back page button
        tk.Button(bottom, text="⬅ Retour", command=self.back_entries, bg = PRIMARY, fg = "white", font = ("Helvetica", 12, "bold"), padx = 10, pady = 5, relief = "flat").pack(side="left", padx=10)

        #reset button
        tk.Button(bottom, text="🔄 Reset", command=self.reset, bg = PRIMARY, fg = "white", font = ("Helvetica", 12, "bold"), padx = 10, pady = 5, relief = "flat").pack(side="right", padx=10)

        self.load_images()

    def back_entries(self): #back to entries page
        self.images_page.pack_forget()
        self.entries_page.pack(fill="both", expand=True)

    def load_images(self):
        self.round += 1
        self.counter_label.config(text=f"Round: {self.round}") #update label

        if self.current:
            self.history.append(self.current.copy()) #store current images in history

        self.images.clear()
        self.current.clear()

        for lbl in self.labels:
            file_path = os.path.join(faces_path, random.choice(self.files)) #randomly select images from file image list
            self.current.append(file_path)

            img = Image.open(file_path).resize((128, 128))
            tk_img = ImageTk.PhotoImage(img)

            self.images.append(tk_img)
            lbl.config(image=tk_img)
            lbl.image = tk_img

    def next_images(self):
        self.load_images()

    def select_image(self, index): # final image selected - new window with image shown
        new_win = tk.Toplevel(self.root)
        new_win.title("Image choisie")
        new_win.geometry(size)

        img = self.images[index] #store image selected
        lbl = tk.Label(new_win, image=img)
        lbl.image = img
        lbl.pack(padx=20, pady=20)

        tk.Label(new_win, text=f"Trouvé en {self.round} tours", font=("Helvetica", 12)).pack(pady=5)

    def go_back(self): #back button selected - show previous images
        if not self.history: #if history empty
            return

        self.round -= 1
        self.counter_label.config(text=f"Round: {self.round}") #reload count label

        previous = self.history.pop() #reselect previous images
        self.current = previous #current images now previous images
        self.images.clear()

        for lbl, path in zip(self.labels, previous): #show images
            img = Image.open(path).resize((120, 120))
            tk_img = ImageTk.PhotoImage(img)

            self.images.append(tk_img)
            lbl.config(image=tk_img)
            lbl.image = tk_img

    def reset(self): #reset button selected - reload new images
        self.round = 0
        self.history.clear()
        self.load_images()


########################### MAIN PROGRAM ######################################
root = tk.Tk()
app = PortraitApp(root)
root.mainloop()
