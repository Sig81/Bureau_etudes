import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk 

from scipy.signal import butter, lfilter
from tkinter import *
from tkinter.ttk import Combobox
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib 
matplotlib.use("TkAgg")
# Fonction permettant de définir les variables en global
def var_glob():
    global amplitude, frequence, phase, frequence_ref, phase_ref
    # Obtention des paramètres
    amplitude = float(amplitude_var.get())
    frequence = float(freq_var.get())
    phase = np.radians(float(phase_var.get()))  # Conversion degrés en radians
    frequence_ref = frequence
    phase_ref = 0

# Fonction pour concevoir un filtre passe-bas RC
def rc(signal, fc, fs):
    dt = 1 / fs
    alpha = dt / (1/(2*np.pi*fc) + dt)
    signal_filtre = np.zeros_like(signal)
    for i in range(1, len(signal)):
        signal_filtre[i] = alpha * signal[i] + (1 - alpha) * signal_filtre[i - 1]
    return signal_filtre

# Fonction pour créer le signal d'entrée
def signal_entree():
    # Création du signal d'entrée
    signal_entree = amplitude * np.sin(2 * np.pi * frequence * temps + phase)
    return signal_entree

#Fonction pour la création du signal de référence pour le Lock-In non déphasé
def signal_ref(frequence_ref, phase_ref):
    signal_ref = np.sin(2 * np.pi * frequence_ref * temps + phase_ref)
    return signal_ref

#Fonction pour la création du signal de référence pour le Lock-In déphasé
def signal_ref_perp(frequence_ref, phase_ref):
     signal_ref_perp = np.sin(2 * np.pi * frequence_ref * temps + phase_ref + np.pi/2)
     return signal_ref_perp

# Mixage du signal d'entrée avec le signal de référence
def signal_mixe(signal_ref, signal_entree):
    signal_mixe = signal_ref * signal_entree
    return signal_mixe

#Mixage du signal d'entrée avec le signal de référence perpendiculaire
def signal_mixe_perp(signal_ref_perp, signal_entree):
    signal_mixe_perp = signal_ref_perp * signal_entree
    return signal_mixe_perp

# Fonction utilisant le filtre passe-bas RC pour extraire la composante à la fréquence de référence
def composante(rc, signal_mixe, signal_mixe_perp):
        composante_verrouillee = rc(signal_mixe, 100, 10000)
        composante_verrouillee_perp = rc(signal_mixe_perp, 100, 10000)
        return composante_verrouillee, composante_verrouillee_perp

# Fonction permettant le tracé du plot et le calcul du signal d'entrée
def fplot_signal(event=None):
    var_glob()
    signal = signal_entree()

    # Nécessaire, sinon créé un graphique pour chaque valeur modifiée
    for widget in cadre_plot.winfo_children():
        widget.destroy()
    
    # Créer le plot
    fig, ax = plt.subplots()
    ax.plot(temps, signal)
    ax.set_xlabel("Temps")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal d'entrée")
    
    # Ajoute le graphe au second cadre
    canvas = FigureCanvasTkAgg(fig, master=cadre_plot)
    canvas.get_tk_widget().grid(row=0, column=0)
    canvas.draw()
    
    # Mixage du signal d'entrée avec le signal de référence
    signal_ref_non_dephase = signal_ref(frequence_ref, phase_ref)
    signal_ref_dephase = signal_ref_perp(frequence_ref, phase_ref)

    signal_melange = signal_mixe(signal_ref_non_dephase, signal)
    signal_melange_perp = signal_mixe_perp(signal_ref_dephase, signal)

    # Utilisation du filtre passe-bas RC pour extraire la composante à la fréquence de référence
    global composante_verrouillee, composante_verrouillee_perp
    composante_verrouillee, composante_verrouillee_perp = composante(rc, signal_melange, signal_melange_perp)

    # Calcul de R et Theta
    R = 2 * np.sqrt(composante_verrouillee**2 + composante_verrouillee_perp**2)
    theta = np.arctan2(composante_verrouillee_perp, composante_verrouillee)
    
    theta_abs = abs(theta)

    # Conversion des radians en degrés
    theta_deg = np.degrees(theta_abs)

    moyenne_R = np.mean(R)
    moyenne_theta = np.mean(theta_deg)
    
    # Mettre à jour les valeurs sous le graphique
    maj_valeurs(moyenne_R, moyenne_theta)

    plt.close()
# Fonction pour mettre à jour les valeurs sous le graphique
def maj_valeurs(R, theta_deg):
    moyenne_R.set("{:.6f}".format(R))
    moyenne_theta.set("{:.6f}".format(theta_deg))

# Création fenetre graphique
fenetre = Tk()
fenetre.title("Lock-IN")

# Définir une variable globale de type DoubleVar pour l'amplitude
amplitude_var = DoubleVar()
freq_var = DoubleVar()
phase_var = DoubleVar()

# Variables globales pour les valeurs sous le graphique
moyenne_R = DoubleVar()
moyenne_theta = DoubleVar()

temps = np.linspace(0, 1, 1000)

# Création d'un premier cadre permettant de saisir les valeurs d'entrées du signal
cadre_valeurs_entree = LabelFrame(fenetre, text="Simulation Signal", bg="lightgrey", borderwidth=2, relief=GROOVE)
cadre_valeurs_entree.grid(row=0, column=0, rowspan=2, sticky="nsew")

# Création liste déroulante pour le choix du signal
label_signal = Label(cadre_valeurs_entree, text="Type de signal", bg="lightgrey")
label_signal.grid(row=0, column=0, pady=(2, 0))
signaux = ["sin"]
selection_signal = StringVar()
selection_signal.set(signaux[0])
combo = Combobox(cadre_valeurs_entree, textvariable=selection_signal, values=signaux, width=8)
combo.grid(row=0, column=1, padx=20, pady=(10, 20))

# Définition amplitude
label_amplitude = Label(cadre_valeurs_entree, text="Amp (V)", bg="lightgrey")
label_amplitude.grid(row=1, column=0, pady=(2, 0))
amplitude = Entry(cadre_valeurs_entree, textvariable=amplitude_var, width=5)
amplitude.grid(row=2, column=0, pady=(0, 10))

# Entrée incertitude de l'amplitude
label_incertitude_amp = Label(cadre_valeurs_entree, text="Incert (%)", bg="lightgrey")
label_incertitude_amp.grid(row=1, column=1, pady=(2, 0))
incertitude_amp = Entry(cadre_valeurs_entree, width=5)
incertitude_amp.grid(row=2, column=1, pady=(0, 10))

# Choix de la distribution
label_distribution = Label(cadre_valeurs_entree, text="Distribution", bg="lightgrey")
label_distribution.grid(row=1, column=2, pady=(2, 0))
distributions = ["Normale", "Uniforme"]
selection_distrib = StringVar()
selection_distrib.set(distributions[0])
combo = Combobox(cadre_valeurs_entree, textvariable=selection_distrib, values=distributions, width=8)
combo.grid(row=2, column=2, pady=(0, 10))

# Définition phase
label_phase = Label(cadre_valeurs_entree, text="Phase (deg)", bg="lightgrey")
label_phase.grid(row=3, column=0, pady=(2, 0))
phase = Entry(cadre_valeurs_entree, textvariable=phase_var, width=5)
phase.grid(row=4, column=0, pady=(0, 10))

# Entrée incertitude de la phase
label_incertitude_phase = Label(cadre_valeurs_entree, text="Incert (%)", bg="lightgrey")
label_incertitude_phase.grid(row=3, column=1, pady=(2, 0))
incertitude_phase = Entry(cadre_valeurs_entree, width=5)
incertitude_phase.grid(row=4, column=1, pady=(0, 10))

# Choix de la distribution
label_distribution = Label(cadre_valeurs_entree, text="Distribution", bg="lightgrey")
label_distribution.grid(row=3, column=2, pady=(2, 0))
distributions = ["Normale", "Uniforme"]
selection_distrib = StringVar()
selection_distrib.set(distributions[0])
combo = Combobox(cadre_valeurs_entree, textvariable=selection_distrib, values=distributions, width=8)
combo.grid(row=4, column=2, pady=(0, 10))

# Définition fréquence
label_freq = Label(cadre_valeurs_entree, text="Freq (HZ)", bg="lightgrey")
label_freq.grid(row=5, column=0, pady=(2, 0))
freq = Entry(cadre_valeurs_entree, textvariable=freq_var, width=5)
freq.grid(row=6, column=0, pady=(0, 10))

# Entrée incertitude de la fréquence
label_incertitude_freq = Label(cadre_valeurs_entree, text="Incert (%)", bg="lightgrey")
label_incertitude_freq.grid(row=5, column=1, pady=(2, 0))
incertitude_freq = Entry(cadre_valeurs_entree, width=5)
incertitude_freq.grid(row=6, column=1, pady=(0, 10))

# Choix de la distribution
label_distribution = Label(cadre_valeurs_entree, text="Distribution", bg="lightgrey")
label_distribution.grid(row=5, column=2, pady=(2, 0))
distributions = ["Normale", "Uniforme"]
selection_distrib = StringVar()
selection_distrib.set(distributions[0])
combo = Combobox(cadre_valeurs_entree, textvariable=selection_distrib, values=distributions, width=8)
combo.grid(row=6, column=2, pady=(0, 10))

# Définition bruit
label_amp_bruit = Label(cadre_valeurs_entree, text="Amp Bruit (V)", bg="lightgrey")
label_amp_bruit.grid(row=7, column=0, pady=(20, 10))

amplitude_bruit = Entry(cadre_valeurs_entree, width=5)
amplitude_bruit.grid(row=7, column=1, pady=(20, 10))

distributions = ["Normale", "Uniforme"]
selection_distrib = StringVar()
selection_distrib.set(distributions[0])
combo = Combobox(cadre_valeurs_entree, textvariable=selection_distrib, values=distributions, width=8)
combo.grid(row=7, column=2, pady=(20, 10))

label_phase_bruit = Label(cadre_valeurs_entree, text="Gigue", bg="lightgrey")
label_phase_bruit.grid(row=8, column=0, pady=(0, 10))

phase_bruit = Entry(cadre_valeurs_entree, width=5)
phase_bruit.grid(row=8, column=1, pady=(0, 10))

distributions = ["Normale", "Uniforme"]
selection_distrib = StringVar()
selection_distrib.set(distributions[0])
combo = Combobox(cadre_valeurs_entree, textvariable=selection_distrib, values=distributions, width=8)
combo.grid(row=8, column=2, pady=(0, 10))


# Création d'un deuxième cadre permettant d'afficher le signal d'entrée et les résultats
cadre_plot = LabelFrame(fenetre, text="Signal et résultats", bg="lightgrey", borderwidth=2, relief=GROOVE)
cadre_plot.grid(row=0, column=1, rowspan=4, sticky="nsew")

# Associer la mise à jour du signal à l'événement de focus sur une entrée
for entry in [amplitude, freq, phase]:
    entry.bind("<FocusOut>", fplot_signal)
    
fplot_signal()

cadre_resultat = LabelFrame(fenetre, text="Résultats", bg="lightgrey", borderwidth=2, relief=GROOVE)
cadre_resultat.grid(row=2, column=1, rowspan=4, sticky="nsew")

# Ajout de Labels pour afficher des valeurs sous le graphique
label_valeur1 = Label(cadre_resultat, text="R:", bg="lightgrey")
label_valeur1.grid(row=1, column=0, pady=(10, 5), padx=(10, 10))

label_moyenne_R = Label(cadre_resultat, textvariable=moyenne_R, width=10)
label_moyenne_R.grid(row=1, column=1, pady=(10, 5))

label_valeur2 = Label(cadre_resultat, text="Theta:", bg="lightgrey")
label_valeur2.grid(row=2, column=0, pady=(5, 10), padx=(10, 10))

label_moyenne_theta = Label(cadre_resultat, textvariable=moyenne_theta, width=10)
label_moyenne_theta.grid(row=2, column=1, pady=(5, 10))





#########################################
# Création d'un troisième cadre permettant de saisir les valeurs du filtre RC
cadre_RC = LabelFrame(fenetre, text="Filtre RC", bg="lightgrey", borderwidth=2, relief=GROOVE)
cadre_RC.grid(row=0, column=2, rowspan=1, sticky="nsew")

Label(cadre_RC, text="Frame 3", bg="white").grid(row=0, column=2, rowspan=1, sticky="nsew")
cadre_sin_ref = LabelFrame(fenetre, text="Signal de ref", bg="lightgrey", borderwidth=2, relief=GROOVE)
cadre_sin_ref.grid(row=2, column=0, rowspan=2, sticky="nsew")

Label(cadre_sin_ref, text="Frame 4", bg="white").pack(padx=10, pady=10)


# bouton de sortie
bouton=Button(fenetre, text="Fermer", command=fenetre.quit)
bouton.grid()

fenetre.mainloop()
# bouton de sortie
# bouton=Button(fenetre, text="Fermer", command=fenetre.quit)
# bouton.pack()