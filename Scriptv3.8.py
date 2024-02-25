"""
Logiciel de simulation d'un LOCK-IN Amplifier avec évaluation des incertitudes utilisant un algorithme de Monte Carlo

Projet realise par : 
    - LE RUYET Guillaume : Mise ne place incertitude Monte Carlo
    - LORVELLEC Evan : Generation signaux aléatoires & Lock-IN
    - SIGUIER Adrien : Chef de projet & Interface graphique & Exportation en logiciel
    
Date : Le 25 fevrier 2024

Entrees : Parametres de signaux et de Lock-IN
Sortie : Graphiques & Amplitude (R) du signal de sortie avec sa phase (theta)

Lieu : Universite Toulouse III Paul Sabatier

Client : Mr Cafarelli
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk 
import os
import json
import time
import sys
import fitz

from tkinter import Label, messagebox, Entry, END, StringVar, Button, LabelFrame, GROOVE, Frame
from tkinter import ttk
from tkinter import Menu, filedialog
from tkinter.ttk import Combobox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk

#Fonction permettant de definir les parametres globaux
def var_glob():
    global amplitude, frequence, phase, amplitude_incertitude, frequence_incertitude, phase_incertitude, amplitude_bruit, gigue, amplitude_ref, frequence_ref, phase_ref, amplitude_incertitude_ref, frequence_incertitude_ref, phase_incertitude_ref, Orth, ortho_incertitude, phase_bruit_ref, fe, Ta, tau, tau_incertitude, Nbr_simulation, fc, temps
    #Obtention des parametres
    amplitude = float(amplitude_val_entree.get())
    frequence = float(frequence_val_entree.get())
    phase = float(phase_val_entree.get())  # Conversion degres en radians
    #Incertitude signal entree
    amplitude_incertitude = float(incertitude_amp_entree.get())
    frequence_incertitude = float(incertitude_freq_entree.get())
    phase_incertitude = float(incertitude_phase_entree.get())
    #Bruit
    amplitude_bruit = float(amplitude_bruit_entree.get())
    gigue = float(phase_bruit_entree.get())
    #Signal de reference
    amplitude_ref = float(amplitude_val_entree_ref.get())
    frequence_ref = float(frequence_val_entree_ref.get())
    phase_ref = float(phase_val_entree_ref.get())
    #Incertitude signal de reference
    amplitude_incertitude_ref = float(incertitude_amp_entree_ref.get())
    frequence_incertitude_ref = float(incertitude_freq_entree_ref.get())
    phase_incertitude_ref = float(incertitude_phase_entree_ref.get())
    #Orthogonalite
    Orth = float(ortho_val_entree_ref.get())
    ortho_incertitude = float(incertitude_ortho_entree_ref.get())
    phase_bruit_ref = float(phase_bruit_ref_entree.get())
    #ADC
    fe = float(freq_ech_entree.get())
    Ta = float(Tps_acq_entree.get())
    #Filtre RC
    tau = float(cste_tps_entree.get())
    tau_incertitude= float(incertitude_cste_tps.get())
    #Monte Carlo
    Nbr_simulation = int(Nbre_simul_entree.get())
    #Frequence de coupure
    fc = 1 / (2 * np.pi * tau)
    temps = np.linspace(0, Ta, int(fe*Ta))

#Fonction pour generer des valeurs aleatoires en fonction de la distribution choisie
def generer_valeurs_aleatoires(distribution):
    if distribution == "Normale":
        return np.random.randn(len(temps))
    elif distribution == "Uniforme":
        return np.random.rand(len(temps))
    
#Fonction permettant de definir des parametres avec les incertitudes 
def var_glob_2():
    global amplitude_var, amplitude_ref_var,frequence_var,frequence_ref_var,phase_var,phase_ref_var,ortho_var,tau_var,fc_var, gigue_var, amplitude_bruit_var, phase_bruit_ref_var
    
    #Amplitude
    np.random.seed(int(time.time()))
    var_glob()
    amplitude_var= amplitude + amplitude * amplitude_incertitude / 100 * generer_valeurs_aleatoires(combo_amp.get())
    amplitude_ref_var= amplitude_ref + amplitude_ref * amplitude_incertitude_ref / 100 * generer_valeurs_aleatoires(combo_amp_ref.get())
    #Frequence
    frequence_var = frequence + frequence * frequence_incertitude / 100 * generer_valeurs_aleatoires(combo_freq.get())
    frequence_ref_var = frequence_ref + frequence_ref * frequence_incertitude_ref / 100 * generer_valeurs_aleatoires(combo_freq_ref.get())
    #Phase
    phase_var = phase + phase * phase_incertitude / 100 * generer_valeurs_aleatoires(combo_phase.get())
    phase_ref_var = phase_ref + phase_ref * phase_incertitude_ref / 100 * generer_valeurs_aleatoires(combo_phase_ref.get())   
    #Ortogonalite 
    ortho_var = Orth + Orth * ortho_incertitude / 100 * generer_valeurs_aleatoires(combo_ortho.get())
    #Tau
    tau_var = tau + tau * tau_incertitude / 100 * generer_valeurs_aleatoires(combo_tau.get())
    fc_var = 1 / (2* np.pi * tau_var)
    gigue_var = gigue * generer_valeurs_aleatoires(combo_gigue.get())
    amplitude_bruit_var = amplitude_bruit * generer_valeurs_aleatoires(combo_amp_bruit.get())
    phase_bruit_ref_var = phase_bruit_ref * generer_valeurs_aleatoires(combo_phase_bruit.get())

#Fonction permettant de selectionner la forme du signal
def forme_signal(signal_shape):
    var_glob_2()
    if signal_shape == "sin":
        #Generation signal sinusoidal
        signal_entree = amplitude_var * np.sin(2 * np.pi * frequence_var * (temps + gigue_var) + np.radians(phase_var))  + amplitude_bruit_var
        signal_ref = amplitude_ref_var * np.sin(2 * np.pi * frequence_ref_var * temps + np.radians(phase_ref_var) + np.radians(phase_bruit_ref_var))
        signal_ref_perp = amplitude_ref_var*np.sin(2 * np.pi * frequence_ref_var * temps + np.radians(phase_ref_var) + np.radians(phase_bruit_ref_var) + np.radians(ortho_var))
        
    elif signal_shape == "carre":
        #Generation signal carre
        signal_entree = amplitude_var * np.sign(np.sin(2 * np.pi * frequence_var * (temps+gigue_var) + np.radians(phase_var)))  + amplitude_bruit_var
        signal_ref = amplitude_ref_var * np.sign(np.sin(2 * np.pi * frequence_ref_var * temps + np.radians(phase_ref_var) + np.radians(phase_bruit_ref_var)))
        signal_ref_perp = amplitude_ref_var * np.sign(np.sin(2 * np.pi * frequence_ref_var * temps + np.radians(phase_ref_var) + np.radians(phase_bruit_ref_var) + np.radians(ortho_var)))
        
    return signal_entree, signal_ref, signal_ref_perp
    
#Fonction pour concevoir un filtre passe-bas RC
def rc(signal, fc, fs):
    dt = 1 / fs
    alpha = dt / (1/(2*np.pi*fc) + dt)
    signal_filtre = np.zeros_like(signal)
    for i in range(1, len(signal)):
        signal_filtre[i] = alpha * signal[i] + (1 - alpha) * signal_filtre[i - 1]
    return signal_filtre

#Fonction pour creer le signal d'entree
def signal_entree():
    signal_entree = forme_signal(combo_signal.get())[0]
    return signal_entree

#Fonction pour la creation du signal de reference pour le Lock-In non dephase
def signal_ref():
    signal_ref = forme_signal(combo_signal.get())[1]
    return signal_ref

#Fonction pour la creation du signal de reference pour le Lock-In dephase
def signal_ref_perp():
    signal_ref_perp = forme_signal(combo_signal.get())[2]
    return signal_ref_perp

#Mixage du signal d'entree avec le signal de reference
def signal_mixe(signal_ref, signal_entree):
    signal_mixe = signal_ref * signal_entree
    return signal_mixe

#Mixage du signal d'entree avec le signal de reference perpendiculaire
def signal_mixe_perp(signal_ref_perp, signal_entree):
    signal_mixe_perp = signal_ref_perp * signal_entree
    return signal_mixe_perp

#Fonction utilisant le filtre passe-bas RC pour extraire la composante a la frequence de reference
def composante(rc, signal_mixe, signal_mixe_perp):
        var_glob()
        composante_verrouillee = rc(signal_mixe, fc, fe)
        composante_verrouillee_perp = rc(signal_mixe_perp, fc, fe)
        return composante_verrouillee, composante_verrouillee_perp

#Fonctions pour mettre a jour les valeurs sous le graphique
def maj_valeurs(R, theta_deg):
    global label_moyenne_R, label_moyenne_theta
    label_R_sortie.config(text="{:.6f}".format(R))
    label_theta_sortie.config(text="{:.6f}".format(theta_deg))

def maj_valeurs2(moyenne_R, moyenne_theta):
    global label_R_moyenne, label_theta_moyenne
    label_R_moyenne_sortie.config(text="{:.6f}".format(moyenne_R))
    label_theta_moyenne_sortie.config(text="{:.6f}".format(moyenne_theta))

def maj_valeurs3(ecart_type_R, ecart_type_theta):
    global label_R_ecart_type, label_theta_ecart_type
    label_R_ecart_type_sortie.config(text="{:.2e}".format(ecart_type_R))
    label_theta_ecart_type_sortie.config(text="{:.2e}".format(ecart_type_theta))

#Fonction calculant R et theta et plot le signal d'entree
def mettre_a_jour():
    #Verification que tout les parametres soient entrees
    if verification_valeurs() !=0:
        var_glob()
        var_glob_2()
        
        #Verification condition tau
        if verification_cst_tps() !=0:
            #Permet d'ouvrir l'onglet des resultats
            notebook.select(0)
        
            signal_ent = signal_entree()
        
            #Mettre a jour le graphique
            ax.clear()
            ax.plot(temps, signal_ent)
            ax.set_xlabel("Temps")
            ax.set_ylabel("Amplitude")
            ax.set_title("Signal d'entree")
            canvas.draw()
        
            i=0
            for i in range(0,int(Ta*fe)):
                if(temps[i]>=np.mean(tau_var)):
                    t0=int(2*i)
                    break
                else:
                    i=i+1
                
            #Mixage du signal d'entree avec le signal de reference
            signal_ref_non_dephase = signal_ref()
            signal_ref_dephase = signal_ref_perp()
            signal_melange = signal_mixe(signal_ref_non_dephase, signal_ent)
            signal_melange_perp = signal_mixe_perp(signal_ref_dephase, signal_ent)
            
            #Utilisation du filtre passe-bas RC pour extraire la composante a la frequence de reference
            composante_verrouillee, composante_verrouillee_perp = composante(rc, signal_melange, signal_melange_perp)
        
            #Calcul de R et Theta
            R = (2 / amplitude_ref_var[t0:]) * np.sqrt(composante_verrouillee[t0:]**2 + composante_verrouillee_perp[t0:]**2)
            theta = np.arctan2(composante_verrouillee_perp[t0:], composante_verrouillee[t0:])
        
            #Conversion des radians en degres
            theta_deg = np.degrees(abs(theta))
            
            #Signal "normal"
            moyenne_R = np.mean(R)
            moyenne_theta = np.mean(theta_deg)
            
            #Signal simule N fois
            MC_R_moyenne, MC_theta_moyenne, MC_R_ecart_type, MC_theta_ecart_type = Methode_Monte_Carlo(Nbr_simulation)
            
            #Mettre a jour les valeurs sous le graphique
            maj_valeurs(moyenne_R, moyenne_theta)
            maj_valeurs2(MC_R_moyenne, MC_theta_moyenne)
            maj_valeurs3(MC_R_ecart_type, MC_theta_ecart_type)
            
            label_fc_sortie.config(text="{:2f}".format(fc))
        
#Fonction de Monte Carlos
def Methode_Monte_Carlo(Nbr_simulation):
    np.random.seed(int(time.time()))
    var_glob()
    var_glob_2()
    temps = np.linspace(0, Ta, int(fe*Ta));
    
    TAB_R_mean=np.zeros(Nbr_simulation)
    TAB_R_std=np.zeros(Nbr_simulation)
    TAB_theta_mean =np.zeros(Nbr_simulation)
    TAB_theta_std =np.zeros(Nbr_simulation)
    
    #Reinitialise les listes
    Amp_list.clear()
    phase_list.clear()
    frequence_list.clear()
    Amp_Bruit_list.clear()
    Gigue_list.clear()
    Amp_ref_list.clear()
    Phase_ref_list.clear()
    Freq_ref_list.clear()
    Ortho_list.clear()
    Phase_Bruit_list.clear()
    Tau_list.clear()
    R_list.clear()
    Theta_list.clear()
    
    tau_var = tau + tau * tau_incertitude / 100 * generer_valeurs_aleatoires(combo_tau.get())
    
    i=0
    for i in range(0,int(Ta*fe)):
        if(temps[i]>=np.mean(tau_var)):
            t0=int(2*i)
            break
        else:
            i=i+1
              
    n=0
    while(n<Nbr_simulation):
        
        var_glob()
        #Amplitude
        amplitude_var= amplitude + amplitude * amplitude_incertitude / 100 * generer_valeurs_aleatoires(combo_amp.get())
        amplitude_ref_var= amplitude_ref + amplitude_ref * amplitude_incertitude_ref / 100 * generer_valeurs_aleatoires(combo_amp_ref.get())
        #Frequence
        frequence_var = frequence + frequence * frequence_incertitude / 100 * generer_valeurs_aleatoires(combo_freq.get())
        frequence_ref_var = frequence_ref + frequence_ref * frequence_incertitude_ref / 100 * generer_valeurs_aleatoires(combo_freq_ref.get())
        #Phase
        phase_var = phase + phase * phase_incertitude / 100 * generer_valeurs_aleatoires(combo_phase.get())
        phase_ref_var = phase_ref + phase_ref * phase_incertitude_ref / 100 * generer_valeurs_aleatoires(combo_phase_ref.get())    
        #Ortogonalite 
        ortho_var = Orth + Orth * ortho_incertitude / 100 * generer_valeurs_aleatoires(combo_ortho.get())
        #Taux
        gigue_var = gigue * generer_valeurs_aleatoires(combo_gigue.get())
        amplitude_bruit_var = amplitude_bruit * generer_valeurs_aleatoires(combo_amp_bruit.get())
        phase_bruit_ref_var = phase_bruit_ref * generer_valeurs_aleatoires(combo_phase_bruit.get())
        
        #1er phase du processus
        signal_ent_mmc  = signal_entree()
        
        #Mixage du signal d'entree avec le signal de reference
        signal_ref_mmc       = signal_ref()
        signal_ref_perp_mmc  = signal_ref_perp()
        signal_mixe_mmc      = signal_mixe(signal_ref_mmc, signal_ent_mmc)
        signal_mixe_perp_mmc = signal_mixe_perp(signal_ref_perp_mmc, signal_ent_mmc)
        
        #3 eme phase du processus 
        composante_verrouillee_mmc, composante_verrouillee_perp_mmc = composante(rc, signal_mixe_mmc, signal_mixe_perp_mmc)
        
        #Calcul de R et Theta
        R = (2 / amplitude_ref_var[t0:]) * np.sqrt(composante_verrouillee_mmc[t0:]**2 + composante_verrouillee_perp_mmc[t0:]**2)
        theta = np.arctan2(composante_verrouillee_perp_mmc[t0:], composante_verrouillee_mmc[t0:])

        theta_abs = abs(theta)

        #Conversion des radians en degres
        theta_deg = np.degrees(theta_abs)

        TAB_R_mean[n] = np.mean(R)
        TAB_R_std[n]  = np.std(R)
        TAB_theta_mean[n] = np.mean(theta_deg)
        TAB_theta_std[n] = np.std(theta_deg)
        n=n+1
        
        #Ajout dans les listes
        Amp_list.append(amplitude_var)
        phase_list.append(phase_var)
        frequence_list.append(frequence_var)
        Amp_Bruit_list.append(amplitude_bruit_var)
        Gigue_list.append(gigue_var)
        Amp_ref_list.append(amplitude_ref_var)
        Phase_ref_list.append(phase_ref_var)
        Freq_ref_list.append(frequence_ref_var)
        Ortho_list.append(ortho_var)
        Phase_Bruit_list.append(phase_bruit_ref_var)
        Tau_list.append(tau_var)
        R_list.append(R)
        Theta_list.append(theta_deg)
    
    moyenne_R=np.mean(TAB_R_mean)
    moyenne_theta=np.mean(TAB_theta_mean)
    
    ecart_type_R=np.mean(TAB_R_std)
    ecart_type_theta=np.mean(TAB_theta_std)
    
    return moyenne_R, moyenne_theta, ecart_type_R, ecart_type_theta  

#Fonction permettant de tracer les histogrammes des distributions
def tracer_histogramme(Valeurs, titre):
    ax_3.clear()
    ax_3.hist(Valeurs, bins=50, color=plt.cm.viridis(np.linspace(0, 1, len(Valeurs))), edgecolor='lightsteelblue')
    ax_3.set_title(titre)
    ax_3.set_xlabel('Valeurs')
    ax_3.set_ylabel('Densite de probabilite')
    
    # Ajouter la moyenne et l'ecart-type
    Valeurs_moy = np.mean(Valeurs)
    Valeurs_ecart_type = np.std(Valeurs)

    ax_3.axvline(Valeurs_moy, color='crimson', linestyle='dashed', linewidth=2, label=f'Moyenne: {Valeurs_moy:.2f}')
    ax_3.axvline(Valeurs_moy + Valeurs_ecart_type, color='forestgreen', linestyle='dashed', linewidth=2, label=f'ecart-type: {Valeurs_ecart_type:.2f}')
    ax_3.axvline(Valeurs_moy - Valeurs_ecart_type, color='forestgreen', linestyle='dashed', linewidth=2)
    
    ax_3.legend()
    canvas_3.draw()
    
    #Permet d'ouvrir l'onglet des histogrammes
    notebook.select(2)
    
#Fonction permettant de tracer les signaux intermediaire
def tracer_signal_intermediaire(signal, titre):
    ax_2.clear()
    ax_2.plot(temps, signal)
    ax_2.set_title(titre)
    ax_2.set_xlabel('Temps')
    ax_2.set_ylabel('Amplitude')
    canvas_2.draw()
    
    #Permet d'ouvrir l'onglet des signaux intermediaire
    notebook.select(1)
    
#Fonction permettant de stocker les variables dans le fichier JSON
def sauvegarder_parametres(parametres, nom_fichier):
    with open(nom_fichier, 'w') as fichier:
        json.dump(parametres, fichier)

#Fonction permettant d'attribuer la variable enregistre dans le fichier JSON
def sauvegarder():
    if verification_valeurs() !=0:
        var_glob()
        parametres_a_sauvegarder = {
            'amplitude': amplitude,
            'frequence': frequence,
            'phase': phase,
            'amplitude_incertitude': amplitude_incertitude,
            'frequence_incertitude': frequence_incertitude,
            'phase_incertitude': phase_incertitude,
            'amplitude_bruit': amplitude_bruit,
            'gigue': gigue,
            'amplitude_ref': amplitude_ref,
            'frequence_ref': frequence_ref,
            'phase_ref': phase_ref,
            'amplitude_incertitude_ref': amplitude_incertitude_ref,
            'frequence_incertitude_ref': frequence_incertitude_ref,
            'phase_incertitude_ref': phase_incertitude_ref,
            'Orth': Orth,
            'ortho_incertitude': ortho_incertitude,
            'phase_bruit_ref':phase_bruit_ref,
            'fe': fe,
            'Ta': Ta,
            'tau': tau,
            'tau_incertitude': tau_incertitude,
            'Nbr_simulation': Nbr_simulation
        }
    
        nom_fichier = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Fichiers JSON", "*.json")])
        if nom_fichier:
            sauvegarder_parametres(parametres_a_sauvegarder, nom_fichier)

#Fonction permettant de charger les variables du fichier JSON
def charger_parametres(nom_fichier):
    with open(nom_fichier, 'r') as fichier:
        parametres = json.load(fichier)
    return parametres

#Fonction permettant de restituer les variables enregistre du fichier JSON lu
def charger():
    nom_fichier = filedialog.askopenfilename(defaultextension=".json", filetypes=[("Fichiers JSON", "*.json")])

    if nom_fichier:
        parametres_charge = charger_parametres(nom_fichier)

        #Mise a jour avec les parametres charges
        amplitude_val_entree.delete(0, END)
        amplitude_val_entree.insert(0, str(parametres_charge['amplitude']))

        frequence_val_entree.delete(0, END)
        frequence_val_entree.insert(0, str(parametres_charge['frequence']))

        phase_val_entree.delete(0, END)
        phase_val_entree.insert(0, str(parametres_charge['phase']))
        
        incertitude_amp_entree.delete(0, END)
        incertitude_amp_entree.insert(0, str(parametres_charge['amplitude_incertitude']))

        incertitude_freq_entree.delete(0,END)
        incertitude_freq_entree.insert(0, str(parametres_charge['frequence_incertitude']))
        
        incertitude_phase_entree.delete(0,END)
        incertitude_phase_entree.insert(0, str(parametres_charge['phase_incertitude']))

        amplitude_bruit_entree.delete(0,END)
        amplitude_bruit_entree.insert(0, str(parametres_charge['amplitude_bruit']))
        
        phase_bruit_entree.delete(0,END)
        phase_bruit_entree.insert(0, str(parametres_charge['gigue']))
        
        amplitude_val_entree_ref.delete(0,END)
        amplitude_val_entree_ref.insert(0, str(parametres_charge['amplitude_ref']))
        
        frequence_val_entree_ref.delete(0,END)
        frequence_val_entree_ref.insert(0, str(parametres_charge['frequence_ref']))
        
        phase_val_entree_ref.delete(0,END)
        phase_val_entree_ref.insert(0, str(parametres_charge['phase_ref']))
        
        incertitude_amp_entree_ref.delete(0,END)
        incertitude_amp_entree_ref.insert(0, str(parametres_charge['amplitude_incertitude_ref']))
        
        incertitude_freq_entree_ref.delete(0,END)
        incertitude_freq_entree_ref.insert(0, str(parametres_charge['frequence_incertitude_ref']))
        
        incertitude_phase_entree_ref.delete(0,END)
        incertitude_phase_entree_ref.insert(0, str(parametres_charge['phase_incertitude_ref']))
        
        ortho_val_entree_ref.delete(0,END)
        ortho_val_entree_ref.insert(0, str(parametres_charge['Orth']))
        
        incertitude_ortho_entree_ref.delete(0,END)
        incertitude_ortho_entree_ref.insert(0, str(parametres_charge['ortho_incertitude']))

        phase_bruit_ref_entree.delete(0,END)
        phase_bruit_ref_entree.insert(0, str(parametres_charge['phase_bruit_ref']))
        
        freq_ech_entree.delete(0,END)
        freq_ech_entree.insert(0, str(parametres_charge['fe']))
        
        Tps_acq_entree.delete(0,END)
        Tps_acq_entree.insert(0, str(parametres_charge['Ta']))
        
        cste_tps_entree.delete(0,END)
        cste_tps_entree.insert(0, str(parametres_charge['tau']))
        
        incertitude_cste_tps.delete(0,END)
        incertitude_cste_tps.insert(0, str(parametres_charge['tau_incertitude']))
        
        Nbre_simul_entree.delete(0,END)
        Nbre_simul_entree.insert(0, str(parametres_charge['Nbr_simulation']))
        
#Fonction pour reinitialiser les valeurs a zero
def reinitialiser():
    amplitude_val_entree.delete(0, END)
    frequence_val_entree.delete(0, END)
    phase_val_entree.delete(0, END)
    amplitude_val_entree.delete(0, END)
    frequence_val_entree.delete(0, END)
    phase_val_entree.delete(0, END)
    incertitude_amp_entree.delete(0, END)
    incertitude_freq_entree.delete(0,END)
    incertitude_phase_entree.delete(0,END)
    amplitude_bruit_entree.delete(0,END)
    phase_bruit_entree.delete(0,END)
    amplitude_val_entree_ref.delete(0,END)
    frequence_val_entree_ref.delete(0,END)
    phase_val_entree_ref.delete(0,END)
    incertitude_amp_entree_ref.delete(0,END)    
    incertitude_freq_entree_ref.delete(0,END)
    incertitude_phase_entree_ref.delete(0,END)
    ortho_val_entree_ref.delete(0,END)
    incertitude_ortho_entree_ref.delete(0,END)
    phase_bruit_ref_entree.delete(0,END)
    freq_ech_entree.delete(0,END)
    Tps_acq_entree.delete(0,END)
    cste_tps_entree.delete(0,END)
    incertitude_cste_tps.delete(0,END)
    Nbre_simul_entree.delete(0,END)
    
    #Clear les graphes
    ax.clear()
    ax_2.clear()
    ax_3.clear()
    
    #Actualise les graphes
    canvas.draw()
    canvas_2.draw()
    canvas_3.draw()
    
    label_R_sortie.config(text="0.000000")
    label_theta_sortie.config(text="0.000000")
    label_R_moyenne_sortie.config(text="0.000000")
    label_theta_moyenne_sortie.config(text="0.000000")
    label_R_ecart_type_sortie.config(text="0.000000")
    label_theta_ecart_type_sortie.config(text="0.000000")

#Fonction permettant de charger un fichier par defaut
def charger_fichier_par_defaut():
   
    repertoire_fichier = os.path.dirname(os.path.abspath(sys.argv[0]))
    fichier_par_defaut = os.path.join(repertoire_fichier, 'Utility\Preset_1.json')

    try:
        with open(fichier_par_defaut, 'r') as fichier:
            
            parametres_charge = charger_parametres(fichier_par_defaut)

            #Mise a jour parametre avec fichier par defaut
            amplitude_val_entree.delete(0, END)
            amplitude_val_entree.insert(0, str(parametres_charge['amplitude']))

            frequence_val_entree.delete(0, END)
            frequence_val_entree.insert(0, str(parametres_charge['frequence']))

            phase_val_entree.delete(0, END)
            phase_val_entree.insert(0, str(parametres_charge['phase']))
            
            incertitude_amp_entree.delete(0, END)
            incertitude_amp_entree.insert(0, str(parametres_charge['amplitude_incertitude']))

            incertitude_freq_entree.delete(0,END)
            incertitude_freq_entree.insert(0, str(parametres_charge['frequence_incertitude']))
            
            incertitude_phase_entree.delete(0,END)
            incertitude_phase_entree.insert(0, str(parametres_charge['phase_incertitude']))

            amplitude_bruit_entree.delete(0,END)
            amplitude_bruit_entree.insert(0, str(parametres_charge['amplitude_bruit']))
            
            phase_bruit_entree.delete(0,END)
            phase_bruit_entree.insert(0, str(parametres_charge['gigue']))
            
            amplitude_val_entree_ref.delete(0,END)
            amplitude_val_entree_ref.insert(0, str(parametres_charge['amplitude_ref']))
            
            frequence_val_entree_ref.delete(0,END)
            frequence_val_entree_ref.insert(0, str(parametres_charge['frequence_ref']))
            
            phase_val_entree_ref.delete(0,END)
            phase_val_entree_ref.insert(0, str(parametres_charge['phase_ref']))
            
            incertitude_amp_entree_ref.delete(0,END)
            incertitude_amp_entree_ref.insert(0, str(parametres_charge['amplitude_incertitude_ref']))
            
            incertitude_freq_entree_ref.delete(0,END)
            incertitude_freq_entree_ref.insert(0, str(parametres_charge['frequence_incertitude_ref']))
            
            incertitude_phase_entree_ref.delete(0,END)
            incertitude_phase_entree_ref.insert(0, str(parametres_charge['phase_incertitude_ref']))
            
            ortho_val_entree_ref.delete(0,END)
            ortho_val_entree_ref.insert(0, str(parametres_charge['Orth']))
            
            incertitude_ortho_entree_ref.delete(0,END)
            incertitude_ortho_entree_ref.insert(0, str(parametres_charge['ortho_incertitude']))

            phase_bruit_ref_entree.delete(0,END)
            phase_bruit_ref_entree.insert(0, str(parametres_charge['phase_bruit_ref']))
            
            freq_ech_entree.delete(0,END)
            freq_ech_entree.insert(0, str(parametres_charge['fe']))
            
            Tps_acq_entree.delete(0,END)
            Tps_acq_entree.insert(0, str(parametres_charge['Ta']))
            
            cste_tps_entree.delete(0,END)
            cste_tps_entree.insert(0, str(parametres_charge['tau']))
            
            incertitude_cste_tps.delete(0,END)
            incertitude_cste_tps.insert(0, str(parametres_charge['tau_incertitude']))
            
            Nbre_simul_entree.delete(0,END)
            Nbre_simul_entree.insert(0, str(parametres_charge['Nbr_simulation']))
    
    except FileNotFoundError:
        pass

#Fonction permettant d'enregistrer dans un chemin d'acces specifie
def enregistrer_avec_champ():
    if verification_valeurs() != 0:
        var_glob()
        chemin_complet = chemin_entree.get()

        if chemin_complet:
            # Séparation du chemin d'accès et nom du fichier
            chemin, nom_fichier = os.path.split(chemin_complet)

            if nom_fichier:
                # Créer le répertoire si nécessaire
                if not os.path.exists(chemin):
                    os.makedirs(chemin)

                chemin_complet = os.path.join(chemin, nom_fichier)
                parametres_a_sauvegarder = {
                    'amplitude': amplitude,
                    'frequence': frequence,
                    'phase': phase,
                    'amplitude_incertitude': amplitude_incertitude,
                    'frequence_incertitude': frequence_incertitude,
                    'phase_incertitude': phase_incertitude,
                    'amplitude_bruit': amplitude_bruit,
                    'gigue': gigue,
                    'amplitude_ref': amplitude_ref,
                    'frequence_ref': frequence_ref,
                    'phase_ref': phase_ref,
                    'amplitude_incertitude_ref': amplitude_incertitude_ref,
                    'frequence_incertitude_ref': frequence_incertitude_ref,
                    'phase_incertitude_ref': phase_incertitude_ref,
                    'Orth': Orth,
                    'ortho_incertitude': ortho_incertitude,
                    'phase_bruit_ref':phase_bruit_ref,
                    'fe': fe,
                    'Ta': Ta,
                    'tau': tau,
                    'tau_incertitude': tau_incertitude,
                    'Nbr_simulation': Nbr_simulation
                }
                
                sauvegarder_parametres(parametres_a_sauvegarder, chemin_complet)
                label_message.config(text="Enregistrement réussi.")
            else:
                # Affichage manquant si nom de fichier manquant
                label_message.config(text="Veuillez entrer un nom de fichier.")
        else:
            # Affichage message si rien n'est écrit
            label_message.config(text="Veuillez spécifier le chemin et le nom du fichier.")

#Fonction permettant de verifier si une valeur n'est pas saisie
def verification_valeurs():
    if amplitude_val_entree.get() == "" or frequence_val_entree.get() == "" or phase_val_entree.get() == "" or incertitude_amp_entree.get() == "" or incertitude_freq_entree.get() == "" or incertitude_phase_entree.get() == "" or amplitude_bruit_entree.get() == "" or phase_bruit_entree.get() == "" or amplitude_val_entree_ref.get()  == "" or frequence_val_entree_ref.get() == "" or phase_val_entree_ref.get() == "" or incertitude_amp_entree_ref.get() == "" or incertitude_freq_entree_ref.get() == "" or incertitude_phase_entree_ref.get() == "" or ortho_val_entree_ref.get() == "" or incertitude_ortho_entree_ref.get() == "" or freq_ech_entree.get() == "" or Tps_acq_entree.get() == "" or cste_tps_entree.get() == "" or incertitude_cste_tps.get() == "" or Nbre_simul_entree.get() == "":
        messagebox.showwarning("Echec", "Echec, veuillez remplir tous les paramètres.")
        return 0

#Fonction permettant d'avertir l'utilisateur si la constante de temps n'est pas correctement paramettre
def verification_cst_tps():
    if 2*tau >= Ta :
        messagebox.showwarning("Attention", "Veuillez saisir un temps d'acquisition 2 fois plus grand que la constante de temps.")
        return 0
    
#Fonction permettant d'afficher la notice d'utilisation
def affichage_notice():

    pdf_chemin_1 = os.path.dirname(os.path.abspath(sys.argv[0]))
    pdf_chemin_2 = os.path.join(pdf_chemin_1, 'Utility/Notice_Lock-IN2000_1.pdf')    
    
    pdf_document = fitz.open(pdf_chemin_2)
    
    PDF_Viewer = tk.Toplevel()
    PDF_Viewer.title("Notice d'utilisation")

    pdf_canvas = tk.Canvas(PDF_Viewer)
    pdf_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar_vertical = tk.Scrollbar(PDF_Viewer, command=pdf_canvas.yview)
    scrollbar_vertical.pack(side=tk.RIGHT, fill=tk.Y)

    pdf_canvas.config(yscrollcommand=scrollbar_vertical.set)

    images_list = []

    y_position = 0

    for nombre_page in range(pdf_document.page_count):
        page = pdf_document[nombre_page]
        image = page.get_pixmap()
        width, height = image.width, image.height
        mode = "RGBA" if image.alpha else "RGB"
        img_data = image.samples
        img = Image.frombytes(mode, (width, height), img_data)

        tk_img = ImageTk.PhotoImage(img)

        images_list.append(tk_img)

        pdf_canvas.create_image(0, y_position, anchor=tk.NW, image=tk_img)

        y_position += height

    pdf_canvas.config(scrollregion=(0, 0, width, y_position))
    pdf_canvas.images_list = images_list

    pdf_document.close()

    PDF_Viewer.geometry(f"{width}x600")
    
#=============================================================================================================================#
#=============================================================================================================================#
#=============================================================================================================================#

#Creation de la fenêtre principale
fenetre = tk.Tk()
fenetre.title("Lock-IN2000")

#Creation du menu
menu_principal = Menu(fenetre)
fenetre.config(menu=menu_principal)

#Creation d'un menu Fichier avec differentes options
menu_fichier = Menu(menu_principal, tearoff=0)

menu_fichier.add_command(label="Nouveau", command=reinitialiser)
menu_fichier.add_command(label="Ouvrir", command=charger)
menu_fichier.add_command(label="Enregistrer", command=sauvegarder)
menu_fichier.add_separator()
menu_fichier.add_command(label="Quitter", command=lambda: [fenetre.quit(), fenetre.destroy()])

#Ajout des sous menus au menu principal
menu_principal.add_cascade(label="Fichier", menu=menu_fichier)

menu_principal.add_command(label="Aide", command=affichage_notice)

#=============================================================================================================================#
#===================== Creation d'un cadre permettant de parametrer les signaux d'entree et de reference =====================#
#=============================================================================================================================#

cadre_signaux = Frame(fenetre, bg="lightgrey", borderwidth=2, relief=GROOVE)
cadre_signaux.grid(row=0, column=0, rowspan=1, sticky="nsew")

#=============================================================================================================================#
#====================== Creation d'un sous cadre permettant de saisir les valeurs d'entrees du signal ========================#
#=============================================================================================================================#

cadre_valeurs_entree = LabelFrame(cadre_signaux, text="Simulation Signal", bg="lightgrey", borderwidth=2, relief=GROOVE)
cadre_valeurs_entree.grid(row=0, column=0, rowspan=2, sticky="nsew")

#Creation liste deroulante pour le choix du signal
label_signal = Label(cadre_valeurs_entree, text="Type de signal", bg="lightgrey")
label_signal.grid(row=0, column=0, pady=(2, 0))
signal_type = ["sin", "carre"]
selection_signal = StringVar()
selection_signal.set(signal_type[0])
combo_signal = Combobox(cadre_valeurs_entree, textvariable=selection_signal, values=signal_type, width=8)
combo_signal.grid(row=0, column=1, padx=20, pady=(20, 20))
combo_signal.set(signal_type[0])

#Definition amplitude
label_amplitude = Label(cadre_valeurs_entree, text="Amp (V)", bg="lightgrey")
label_amplitude.grid(row=1, column=0, pady=(2, 0))
amplitude_val_entree = Entry(cadre_valeurs_entree, width=5, justify="center")
amplitude_val_entree.grid(row=2, column=0, pady=(0, 10))

#Entree incertitude de l'amplitude
label_incertitude_amp = Label(cadre_valeurs_entree, text="Incert (%)", bg="lightgrey")
label_incertitude_amp.grid(row=1, column=1, pady=(2, 0))
incertitude_amp_entree = Entry(cadre_valeurs_entree, width=5, justify="center")
incertitude_amp_entree.grid(row=2, column=1, pady=(0, 10))

#Choix de la distribution
label_distribution = Label(cadre_valeurs_entree, text="Distribution", bg="lightgrey")
label_distribution.grid(row=1, column=2, pady=(2, 0))
distribution_amp = ["Normale", "Uniforme"]
selection_distrib_amp = StringVar()
selection_distrib_amp.set(distribution_amp[0])
combo_amp = Combobox(cadre_valeurs_entree, textvariable=selection_distrib_amp, values=distribution_amp, width=8)
combo_amp.grid(row=2, column=2, pady=(0, 10))
combo_amp.set(distribution_amp[0])

#Definition frequence
label_freq = Label(cadre_valeurs_entree, text="Freq (HZ)", bg="lightgrey")
label_freq.grid(row=5, column=0, pady=(2, 0))
frequence_val_entree = Entry(cadre_valeurs_entree, width=5, justify="center")
frequence_val_entree.grid(row=6, column=0, pady=(0, 10))

#Entree incertitude de la frequence
label_incertitude_freq = Label(cadre_valeurs_entree, text="Incert (%)", bg="lightgrey")
label_incertitude_freq.grid(row=5, column=1, pady=(2, 0))
incertitude_freq_entree = Entry(cadre_valeurs_entree, width=5, justify="center")
incertitude_freq_entree.grid(row=6, column=1, pady=(0, 10))

#Choix de la distribution
label_distribution = Label(cadre_valeurs_entree, text="Distribution", bg="lightgrey")
label_distribution.grid(row=5, column=2, pady=(2, 0))
distribution_freq = ["Normale", "Uniforme"]
selection_distrib_freq = StringVar()
selection_distrib_freq.set(distribution_freq[0])
combo_freq = Combobox(cadre_valeurs_entree, textvariable=selection_distrib_freq, values=distribution_freq, width=8)
combo_freq.grid(row=6, column=2, pady=(0, 10))
combo_freq.set(distribution_freq[0])

#Definition phase
label_phase = Label(cadre_valeurs_entree, text="Phase (deg)", bg="lightgrey")
label_phase.grid(row=3, column=0, pady=(2, 0))
phase_val_entree = Entry(cadre_valeurs_entree, width=5, justify="center")
phase_val_entree.grid(row=4, column=0, pady=(0, 10))

#Entree incertitude de la phase
label_incertitude_phase = Label(cadre_valeurs_entree, text="Incert (%)", bg="lightgrey")
label_incertitude_phase.grid(row=3, column=1, pady=(2, 0))
incertitude_phase_entree = Entry(cadre_valeurs_entree, width=5, justify="center")
incertitude_phase_entree.grid(row=4, column=1, pady=(0, 10))

#Choix de la distribution
label_distribution = Label(cadre_valeurs_entree, text="Distribution", bg="lightgrey")
label_distribution.grid(row=3, column=2, pady=(2, 0))
distribution_phase = ["Normale", "Uniforme"]
selection_distrib_phase = StringVar()
selection_distrib_phase.set(distribution_phase[0])
combo_phase = Combobox(cadre_valeurs_entree, textvariable=selection_distrib_phase, values=distribution_phase, width=8)
combo_phase.grid(row=4, column=2, pady=(0, 10))
combo_phase.set(distribution_phase[0])

#Definition bruit
label_amp_bruit = Label(cadre_valeurs_entree, text="Amp Bruit (V)", bg="lightgrey")
label_amp_bruit.grid(row=7, column=0, pady=(20, 10))

amplitude_bruit_entree = Entry(cadre_valeurs_entree, width=5, justify="center")
amplitude_bruit_entree.grid(row=7, column=1, pady=(20, 10))

distribution_amp_bruit = ["Normale", "Uniforme"]
selection_distrib_amp_bruit = StringVar()
selection_distrib_amp_bruit.set(distribution_amp_bruit[0])
combo_amp_bruit = Combobox(cadre_valeurs_entree, textvariable=selection_distrib_amp_bruit, values=distribution_amp_bruit, width=8)
combo_amp_bruit.grid(row=7, column=2, pady=(20, 10))
combo_amp_bruit.set(distribution_amp_bruit[0])

label_phase_bruit = Label(cadre_valeurs_entree, text="Gigue", bg="lightgrey")
label_phase_bruit.grid(row=8, column=0, pady=(0, 10))

phase_bruit_entree = Entry(cadre_valeurs_entree, width=5, justify="center")
phase_bruit_entree.grid(row=8, column=1, pady=(0, 10))

distribution_gigue = ["Normale", "Uniforme"]
selection_distrib_gigue = StringVar()
selection_distrib_gigue.set(distribution_gigue[0])
combo_gigue = Combobox(cadre_valeurs_entree, textvariable=selection_distrib_gigue, values=distribution_gigue, width=8)
combo_gigue.grid(row=8, column=2, pady=(0, 10))
combo_gigue.set(distribution_gigue[0])

#=============================================================================================================================#
#========================== Creation d'un sous cadre permettant de definir le signal de reference ============================#
#=============================================================================================================================#

cadre_sin_ref = LabelFrame(cadre_signaux, text="Signal de ref", bg="lightgrey", borderwidth=2, relief=GROOVE)
cadre_sin_ref.grid(row=2, column=0, rowspan=2, sticky="nsew")

#Definition amplitude reference
label_amplitude_ref = Label(cadre_sin_ref, text="Amp (V)", bg="lightgrey")
label_amplitude_ref.grid(row=1, column=0, pady=(2, 0))
amplitude_val_entree_ref = Entry(cadre_sin_ref, width=5, justify="center")
amplitude_val_entree_ref.grid(row=2, column=0, pady=(0, 10))

#Entree incertitude de l'amplitude reference
label_incertitude_amp_ref = Label(cadre_sin_ref, text="Incert (%)", bg="lightgrey")
label_incertitude_amp_ref.grid(row=1, column=1, pady=(2, 0), padx=(12, 0))
incertitude_amp_entree_ref = Entry(cadre_sin_ref, width=5, justify="center")
incertitude_amp_entree_ref.grid(row=2, column=1, pady=(0, 10), padx=(10, 0))

#Choix de la distribution
label_distribution_amp_ref = Label(cadre_sin_ref, text="Distribution", bg="lightgrey")
label_distribution_amp_ref.grid(row=1, column=2, pady=(2, 0), padx=(20, 0))
distribution_amp_ref = ["Normale", "Uniforme"]
selection_distrib_amp_ref = StringVar()
selection_distrib_amp_ref.set(distribution_amp_ref[0])
combo_amp_ref = Combobox(cadre_sin_ref, textvariable=selection_distrib_amp_ref, values=distribution_amp_ref, width=8)
combo_amp_ref.grid(row=2, column=2, pady=(0, 10), padx=(27, 5))
combo_amp_ref.set(distribution_amp_ref[0])

#Definition phase reference
label_phase_ref = Label(cadre_sin_ref, text="Phase (deg)", bg="lightgrey")
label_phase_ref.grid(row=3, column=0, pady=(2, 0))
phase_val_entree_ref = Entry(cadre_sin_ref, width=5, justify="center")
phase_val_entree_ref.grid(row=4, column=0, pady=(0, 10))

#Entree incertitude de la phase reference
label_incertitude_phase_ref = Label(cadre_sin_ref, text="Incert (%)", bg="lightgrey")
label_incertitude_phase_ref.grid(row=3, column=1, pady=(2, 0), padx=(12, 0))
incertitude_phase_entree_ref = Entry(cadre_sin_ref, width=5, justify="center")
incertitude_phase_entree_ref.grid(row=4, column=1, pady=(0, 10), padx=(10, 0))

#Choix de la distribution
label_distribution_phase_ref = Label(cadre_sin_ref, text="Distribution", bg="lightgrey")
label_distribution_phase_ref.grid(row=3, column=2, pady=(2, 0), padx=(20, 0))
distribution_phase_ref = ["Normale", "Uniforme"]
selection_distrib_phase_ref = StringVar()
selection_distrib_phase_ref.set(distribution_phase_ref[0])
combo_phase_ref = Combobox(cadre_sin_ref, textvariable=selection_distrib_phase_ref, values=distribution_phase_ref, width=8)
combo_phase_ref.grid(row=4, column=2, pady=(0, 10), padx=(27, 5))
combo_phase_ref.set(distribution_phase_ref[0])

#Definition frequence de reference
label_freq_ref = Label(cadre_sin_ref, text="Freq (HZ)", bg="lightgrey")
label_freq_ref.grid(row=5, column=0, pady=(2, 0))
frequence_val_entree_ref = Entry(cadre_sin_ref, width=5, justify="center")
frequence_val_entree_ref.grid(row=6, column=0, pady=(0, 10))

#Entree incertitude de la frequence de reference
label_incertitude_freq_ref = Label(cadre_sin_ref, text="Incert (%)", bg="lightgrey")
label_incertitude_freq_ref.grid(row=5, column=1, pady=(2, 0), padx=(12, 0))
incertitude_freq_entree_ref = Entry(cadre_sin_ref, width=5, justify="center")
incertitude_freq_entree_ref.grid(row=6, column=1, pady=(0, 10), padx=(10, 0))

#Choix de la distribution
label_distribution_freq_ref = Label(cadre_sin_ref, text="Distribution", bg="lightgrey")
label_distribution_freq_ref.grid(row=5, column=2, pady=(2, 0), padx=(20, 0))
distribution_freq_ref = ["Normale", "Uniforme"]
selection_distrib_freq_ref = StringVar()
selection_distrib_freq_ref.set(distribution_freq_ref[0])
combo_freq_ref = Combobox(cadre_sin_ref, textvariable=selection_distrib_freq_ref, values=distribution_freq_ref, width=8)
combo_freq_ref.grid(row=6, column=2, pady=(0, 10), padx=(27, 5))
combo_freq_ref.set(distribution_freq_ref[0])

#Definition orthogonalite
label_orth_ref = Label(cadre_sin_ref, text="Orth (deg)", bg="lightgrey")
label_orth_ref.grid(row=7, column=0, pady=(2, 0))
ortho_val_entree_ref = Entry(cadre_sin_ref, width=5, justify="center")
ortho_val_entree_ref.grid(row=8, column=0, pady=(0, 10))

#Entree incertitude de l'hortogonalite
label_incertitude_ortho_ref = Label(cadre_sin_ref, text="Incert (%)", bg="lightgrey")
label_incertitude_ortho_ref.grid(row=7, column=1, pady=(2, 0), padx=(12, 0))
incertitude_ortho_entree_ref = Entry(cadre_sin_ref, width=5, justify="center")
incertitude_ortho_entree_ref.grid(row=8, column=1, pady=(0, 10), padx=(10, 0))

#Choix de la distribution
label_distribution_ortho_ref = Label(cadre_sin_ref, text="Distribution", bg="lightgrey")
label_distribution_ortho_ref.grid(row=7, column=2, pady=(2, 0), padx=(20, 0))
distribution_ortho_ref = ["Normale", "Uniforme"]
selection_distrib_ortho_ref = StringVar()
selection_distrib_ortho_ref.set(distribution_freq_ref[0])
combo_ortho = Combobox(cadre_sin_ref, textvariable=selection_distrib_ortho_ref, values=distribution_ortho_ref, width=8)
combo_ortho.grid(row=8, column=2, pady=(0, 10), padx=(27, 5))
combo_ortho.set(distribution_ortho_ref[0])

#Definition phase bruit
label_phase_bruit_ref = Label(cadre_sin_ref, text="Bruit phase (deg)", bg="lightgrey")
label_phase_bruit_ref.grid(row=9, column=0, pady=(20, 10))

phase_bruit_ref_entree = Entry(cadre_sin_ref, width=5, justify="center")
phase_bruit_ref_entree.grid(row=9, column=1, pady=(20, 10), padx=(10, 0))

distribution_bruit_ref = ["Normale", "Uniforme"]
selection_distrib_ref_bruit = StringVar()
selection_distrib_ref_bruit.set(distribution_bruit_ref[0])
combo_phase_bruit = Combobox(cadre_sin_ref, textvariable=selection_distrib_ref_bruit, values=distribution_bruit_ref, width=8)
combo_phase_bruit.grid(row=9, column=2, pady=(20, 10), padx=(27, 5))
combo_phase_bruit.set(distribution_bruit_ref[0])

#=============================================================================================================================#
#================================== Creation d'un cadre permettant d'afficher les resultats ==================================#
#=============================================================================================================================#

cadre_plot = LabelFrame(fenetre, text="Resultats", bg="lightgrey", borderwidth=2, relief=tk.GROOVE)
cadre_plot.grid(row=0, column=1, rowspan=9, sticky="nsew")

#Creation d'un menu d'onglets
notebook = ttk.Notebook(cadre_plot)

#Creation d'onglets
onglet1 = ttk.Frame(notebook)
onglet2 = ttk.Frame(notebook)
onglet3 = ttk.Frame(notebook)

#Ajout des onglets au menu d'onglets
notebook.add(onglet1, text="Signal d'entrée et résultats")
notebook.add(onglet2, text="Graphes Intermédiaires")
notebook.add(onglet3, text="Histogrammes")

notebook.grid(row=0, column=1, rowspan=4, sticky="nsew")

#Cadre pour l'onglet 1
cadre_entree_plot = Frame(onglet1, bg="lightgrey", borderwidth=2, relief=tk.GROOVE)
cadre_entree_plot.grid(row=0, column=0, sticky="nsew")

#Creer le trace du signal d'entree
fig, ax = plt.subplots(figsize=(8, 4.6))
# fig.set_size_inches(cadre_entree_plot.winfo_width()*10, cadre_entree_plot.winfo_height()*6.40, forward=False)
canvas = FigureCanvasTkAgg(fig, master=cadre_entree_plot)
widget_canvas = canvas.get_tk_widget()
widget_canvas.grid(row=0, column=0, columnspan=4, sticky="nsew")

#Cadre pour l'onglet 2
cadre_plot_signal_intermediaire = Frame(onglet2, bg="lightgrey", borderwidth=2, relief=tk.GROOVE)
cadre_plot_signal_intermediaire.grid(row=0, column=0, sticky="nsew")

#Creer le trace des signaux intermediaire
fig_2, ax_2 = plt.subplots(figsize=(8, 5.65))
# fig_2.set_size_inches(cadre_entree_plot.winfo_width()*10, cadre_entree_plot.winfo_height()*7.85, forward=False)
canvas_2 = FigureCanvasTkAgg(fig_2, master=cadre_plot_signal_intermediaire)
widget_canvas_2 = canvas_2.get_tk_widget()
widget_canvas_2.grid(row=0, column=0, sticky="nsew")

#Cadre pour l'onglet 3
cadre_hist_plot = Frame(onglet3, bg="lightgrey", borderwidth=2, relief=tk.GROOVE)
cadre_hist_plot.grid(row=0, column=0, sticky="nsew")

#Creer le trace des histogrammes
fig_3, ax_3 = plt.subplots(figsize=(8, 5.65))
# fig_3.set_size_inches(cadre_entree_plot.winfo_width()*10, cadre_entree_plot.winfo_height()*7.85, forward=False)
canvas_3 = FigureCanvasTkAgg(fig_3, master=cadre_hist_plot)
widget_canvas_3 = canvas_3.get_tk_widget()
widget_canvas_3.grid(row=0, column=0, sticky="nsew")

#Onglet 1
#Affichage des valeurs de R et theta
label_R = Label(cadre_entree_plot, text="Valeur amplitude (V)", bg="lightgrey")
label_R.grid(row=1, column=0, pady=(10, 5), padx=(10, 5), sticky="e")

label_R_sortie = Label(cadre_entree_plot, text="0.000000", width=10)
label_R_sortie.grid(row=1, column=1, pady=(10, 5), padx=(5, 10), sticky="w")

label_theta = Label(cadre_entree_plot, text="Différence de phase (deg)", bg="lightgrey")
label_theta.grid(row=1, column=2, pady=(10, 5), padx=(10, 5), sticky="e")

label_theta_sortie = Label(cadre_entree_plot, text="0.000000", width=10)
label_theta_sortie.grid(row=1, column=3, pady=(10, 5), padx=(5, 10), sticky="w")

#Affichage des valeurs de R et theta moyennes
label_R_moyenne = Label(cadre_entree_plot, text="Moyenne valeur amplitude (V) ", bg="lightgrey")
label_R_moyenne.grid(row=2, column=0, pady=(5, 5), padx=(10, 5), sticky="e")

label_R_moyenne_sortie = Label(cadre_entree_plot, text="0.000000", width=10)
label_R_moyenne_sortie.grid(row=2, column=1, pady=(5, 5), padx=(5, 10), sticky="w")

label_theta_moyenne = Label(cadre_entree_plot, text="Moyenne différence de phase (deg)", bg="lightgrey")
label_theta_moyenne.grid(row=2, column=2, pady=(5, 5), padx=(10, 5), sticky="e")

label_theta_moyenne_sortie = Label(cadre_entree_plot, text="0.000000", width=10)
label_theta_moyenne_sortie.grid(row=2, column=3, pady=(5, 5), padx=(5, 10), sticky="w")

#Affichage des valeurs de l'ecart-type de R et theta
label_R_ecart_type = Label(cadre_entree_plot, text="Ecart-type valeur amplitude (V)", bg="lightgrey")
label_R_ecart_type.grid(row=3, column=0, pady=(5, 13), padx=(10, 5), sticky="e")

label_R_ecart_type_sortie = Label(cadre_entree_plot, text="0.000000", width=10)
label_R_ecart_type_sortie.grid(row=3, column=1, pady=(5, 13), padx=(5, 10), sticky="w")

label_theta_ecart_type = Label(cadre_entree_plot, text="Ecart-type différence de phase (deg)", bg="lightgrey")
label_theta_ecart_type.grid(row=3, column=2, pady=(5, 13), padx=(10, 5), sticky="e")

label_theta_ecart_type_sortie = Label(cadre_entree_plot, text="0.000000", width=10)
label_theta_ecart_type_sortie.grid(row=3, column=3, pady=(5, 13), padx=(5, 10), sticky="w")

#=============================================================================================================================#
#=================== Creation d'un cadre permettant de definir les caracteristiques du LOCK-IN et autres =====================#
#=============================================================================================================================#

cadre_reglages = Frame(fenetre, bg="lightgrey", borderwidth=2, relief=GROOVE)
cadre_reglages.grid(row=0, column=2, rowspan=1, sticky="nsew")

#=============================================================================================================================#
#======================== Creation d'un sous cadre permettant de definir les constantes du LOCK-IN ===========================#
#=============================================================================================================================#

cadre_ADC = LabelFrame(cadre_reglages, text="ADC", bg="lightgrey", borderwidth=2, relief=GROOVE)
cadre_ADC.grid(row=0, column=2, rowspan=1, sticky="nsew")

#Definition frequence d'echantillonnage
label_freq_ech = Label(cadre_ADC, text="Frequence", bg="lightgrey")
label_freq_ech.grid(row=1, column=0, pady=(0, 0), padx=(40, 20))
label_freq_ech = Label(cadre_ADC, text="d'échantillonage (Hz)", bg="lightgrey")
label_freq_ech.grid(row=2, column=0, pady=(0, 0), padx=(40, 20))
freq_ech_entree = Entry(cadre_ADC, width=5, justify="center")
freq_ech_entree.grid(row=3, column=0, pady=(5, 7), padx=(40, 20))

#Definition temps d'acquisition
label_tps_acq = Label(cadre_ADC, text="Temps", bg="lightgrey")
label_tps_acq.grid(row=1, column=1, pady=(0, 0),  padx=(20, 40))
label_tps_acq = Label(cadre_ADC, text="d'acquisition (s)", bg="lightgrey")
label_tps_acq.grid(row=2, column=1, pady=(0, 0), padx=(20, 40))
Tps_acq_entree = Entry(cadre_ADC, width=5, justify="center")
Tps_acq_entree.grid(row=3, column=1, pady=(5, 7), padx=(20, 40))

#=============================================================================================================================#
#======================= Creation d'un sous cadre permettant de definir les constantes du filtre RC ==========================#
#=============================================================================================================================#

cadre_RC = LabelFrame(cadre_reglages, text="Filtre RC", bg="lightgrey", borderwidth=2, relief=GROOVE)
cadre_RC.grid(row=1, column=2, rowspan=1, sticky="nsew")

#Definition constante de temps
label_cste_tps_entree = Label(cadre_RC, text="Constante de temps (s)", bg="lightgrey")
label_cste_tps_entree.grid(row=1, column=0, pady=(2, 0), padx=(3, 0))
cste_tps_entree = Entry(cadre_RC, width=5, justify="center")
cste_tps_entree.grid(row=2, column=0, pady=(0, 5))

#Entree incertitude de la constante
label_incertitude_cste_tps = Label(cadre_RC, text="Incert (%)", bg="lightgrey")
label_incertitude_cste_tps.grid(row=1, column=1, pady=(2, 0), padx=(0, 0))
incertitude_cste_tps = Entry(cadre_RC, width=5, justify="center")
incertitude_cste_tps.grid(row=2, column=1, pady=(0, 5), padx=(5, 0))

#Choix de la distribution
label_distribution_cste_tps = Label(cadre_RC, text="Distribution", bg="lightgrey")
label_distribution_cste_tps.grid(row=1, column=2, pady=(2, 0), padx=(0, 0))
distribution_cste_tps = ["Normale", "Uniforme"]
selection_distrib_cste_tps = StringVar()
selection_distrib_cste_tps.set(distribution_cste_tps[0])
combo_tau = Combobox(cadre_RC, textvariable=selection_distrib_cste_tps, values=distribution_cste_tps, width=8)
combo_tau.grid(row=2, column=2, pady=(0, 5), padx=(3, 5))
combo_tau.set(distribution_cste_tps[0])

#Affichage de la frequence de coupure
label_fc = Label(cadre_RC, text="Frequence de coupure (Hz)", bg="lightgrey")
label_fc.grid(row=3, column=1, pady=(0, 0), padx=(0, 0))

label_fc_sortie = Label(cadre_RC, text="0", width=10, bg="lightgrey")
label_fc_sortie.grid(row=3, column=2, pady=(0, 0), padx=(0, 10), sticky="w")

label_fc.config(fg="dimgrey")
label_fc_sortie.config(fg="dimgrey")

#=============================================================================================================================#
#================== Creation d'un sous cadre permettant de definir le nombre d'iterations de Monte Carlo =====================#
#=============================================================================================================================#

cadre_MC = LabelFrame(cadre_reglages, text="Monte Carlo", bg="lightgrey", borderwidth=2, relief=GROOVE)
cadre_MC.grid(row=2, column=2, rowspan=1, sticky="nsew")

#Definition du nombre d'iterations
label_simul = Label(cadre_MC, text="Nombre de simulations", bg="lightgrey")
label_simul.grid(row=1, column=0, pady=(5, 10), padx=(35,0))
Nbre_simul_entree = Entry(cadre_MC, width=15, justify="center")
Nbre_simul_entree.grid(row=1, column=1, pady=(5, 10), padx=(70,30))

#=============================================================================================================================#
#=================== Creation d'un sous cadre permettant d'enregistrer les donnees dans un chemin donne ======================#
#=============================================================================================================================#

cadre_save = LabelFrame(cadre_reglages, text="Enregistrement", bg="lightgrey", borderwidth=2, relief=GROOVE)
cadre_save.grid(row=3, column=2, rowspan=1, sticky="nsew")

#Chemin d'acces
chemin_entree = tk.Entry(cadre_save, width=40)
chemin_entree.grid(row=0, column=0, padx=5, pady=5)

#Bouton enregistrement
bouton_enregistrer = tk.Button(cadre_save, text="Enregistrer", command=enregistrer_avec_champ)
bouton_enregistrer.grid(row=0, column=1, padx=(25,5), pady=5)

#Affichage des messages
label_message = tk.Label(cadre_save, text="")
label_message.grid(row=1, column=0, columnspan=2, pady=(5, 10))

#=============================================================================================================================#
#======================= Creation d'un sous cadre permettant d'afficher les graphes intermediaires ===========================#
#=============================================================================================================================#

cadre_courbe = LabelFrame(cadre_reglages, text="Affichage des courbes intermediaires", bg="lightgrey", borderwidth=2, relief=GROOVE)
cadre_courbe.grid(row=4, column=2, rowspan=1, sticky="nsew")

#Bouton affichage signal de reference
bouton_afficher_signal_ref = Button(cadre_courbe, text="Signal de ref", command=lambda:tracer_signal_intermediaire(signal_ref(), "Signal de reference"), width=13)
bouton_afficher_signal_ref.grid(row=0, column=0, padx=(15, 8), pady=(8, 8))
#Bouton affichage signal X
bouton_afficher_frequence = Button(cadre_courbe, text="X", command=lambda:tracer_signal_intermediaire(composante(rc, signal_mixe(signal_ref(), signal_entree()), signal_mixe_perp(signal_ref_perp(), signal_entree()))[0], "Signal issu du filtre passe-bas (X)"), width=13)
bouton_afficher_frequence.grid(row=0, column=1, padx=(8, 8), pady=(8, 8))
#Bouton affichage signal Y
bouton_afficher_phase = Button(cadre_courbe, text="Y", command=lambda:tracer_signal_intermediaire(composante(rc, signal_mixe(signal_ref(), signal_entree()), signal_mixe_perp(signal_ref_perp(), signal_entree()))[1], "Signal issu du filtre passe-bas (Y)"), width=13)
bouton_afficher_phase.grid(row=0, column=2, padx=(8, 15), pady=(8, 8))

#=============================================================================================================================#
#============================ Creation d'un sous cadre permettant d'afficher les histogrammes ================================#
#=============================================================================================================================#

cadre_hist = LabelFrame(cadre_reglages, text="Affichage des histogrammes", bg="lightgrey", borderwidth=2, relief=GROOVE)
cadre_hist.grid(row=5, column=2, rowspan=1, sticky="nsew")

#Definition des listes pour stocker les valeurs
Amp_list = []
phase_list = []
frequence_list = []
Amp_Bruit_list = []
Gigue_list = []
Amp_ref_list = []
Phase_ref_list = []
Freq_ref_list = []
Ortho_list = []
Phase_Bruit_list = []
Tau_list = []
R_list = []
Theta_list = []

#Bouton affichage amplitude
bouton_afficher_A = Button(cadre_hist, text="Amp", command=lambda:tracer_histogramme(Amp_list, "Distribution de l'amplitude"), width=7)
bouton_afficher_A.grid(row=0, column=0, padx=(8, 5), pady=(8, 8))
#Bouton affichage frequence
bouton_afficher_frequence = Button(cadre_hist, text="Freq", command=lambda:tracer_histogramme(frequence_list, "Distribution de la frequence"), width=7)
bouton_afficher_frequence.grid(row=0, column=1, padx=(5, 5), pady=(8, 8))
#Bouton affichage phase
bouton_afficher_phase = Button(cadre_hist, text="Phase", command=lambda:tracer_histogramme(phase_list, "Distribution de la phase"), width=7)
bouton_afficher_phase.grid(row=0, column=2, padx=(5, 5), pady=(8, 8))
#Bouton affichage amplitude bruit
bouton_afficher_amplitude_bruit = Button(cadre_hist, text="Amp Bruit", command=lambda:tracer_histogramme(Amp_Bruit_list, "Distribution du bruit de l'amplitude"), width=7)
bouton_afficher_amplitude_bruit.grid(row=0, column=3, padx=(5, 5), pady=(8, 8))
#Bouton affichage Gigue
bouton_afficher_Gigue = Button(cadre_hist, text='Gigue', command=lambda:tracer_histogramme(Gigue_list, "Distribution de la gigue"), width=7)
bouton_afficher_Gigue.grid(row=0, column=4, padx=(5, 8), pady=(8, 8))

#Bouton affichage amplitude ref
bouton_afficher_A_ref = Button(cadre_hist, text="Amp ref", command=lambda:tracer_histogramme(Amp_ref_list, "Distribution de l'amplitude de référence"), width=7)
bouton_afficher_A_ref.grid(row=1, column=0, padx=(8, 5), pady=(8, 8))
#Bouton affichage frequence ref
bouton_afficher_frequence_ref = Button(cadre_hist, text="Freq ref", command=lambda:tracer_histogramme(Freq_ref_list, "Distribution de la frequence de référence"), width=7)
bouton_afficher_frequence_ref.grid(row=1, column=1, padx=(5, 5), pady=(8, 8))
#Bouton affichage phase ref
bouton_afficher_phase_ref = Button(cadre_hist, text="Phase ref", command=lambda:tracer_histogramme(Phase_ref_list, "Distribution de la phase de référence"), width=7)
bouton_afficher_phase_ref.grid(row=1, column=2, padx=(5, 5), pady=(8, 8))
#Bouton affichage Bruit Phase
bouton_afficher_Phase_Bruit = Button(cadre_hist, text='Phase Bruit', command=lambda:tracer_histogramme(Phase_Bruit_list, "Distribution du bruit de phase"), width=7)
bouton_afficher_Phase_Bruit.grid(row=1, column=3, padx=(5, 5), pady=(8, 8))
#Bouton affichage Orthogonalite
bouton_afficher_ortho = Button(cadre_hist, text='Orth', command=lambda:tracer_histogramme(Ortho_list, "Distribution de l'orthogonalité"), width=7)
bouton_afficher_ortho.grid(row=1, column=4, padx=(5, 8), pady=(8, 8))

#Bouton affichage Tau
bouton_afficher_tau = Button(cadre_hist, text='Tau', command=lambda:tracer_histogramme(Tau_list, "Distribution de la constante de temps (Tau)"), width=7)
bouton_afficher_tau.grid(row=2, column=1, padx=(8,5), pady=(8, 8))
#Bouton affichage R
bouton_afficher_R = Button(cadre_hist, text="R", command=lambda:tracer_histogramme(R_list, "Distribution de l'amplitude de sortie (R)"), width=7)
bouton_afficher_R.grid(row=2, column=2, padx=(5, 5), pady=(8, 8))
#Bouton affichage Theta
bouton_afficher_theta = Button(cadre_hist, text='Theta', command=lambda:tracer_histogramme(Theta_list, "Distribution de la différence de phase (theta)"), width=7)
bouton_afficher_theta.grid(row=2, column=3, padx=(5, 8), pady=(8, 8))

#=============================================================================================================================#
#============================== Creation d'un sous cadre permettant d'executer le programme ==================================#
#=============================================================================================================================#

cadre_exe = LabelFrame(cadre_reglages, text="Processus", bg="lightgrey", borderwidth=2, relief=GROOVE)
cadre_exe.grid(row=6, column=2, rowspan=1, sticky="nsew")

#Bouton affichage
bouton_afficher_signal = Button(cadre_exe, text="Afficher Signal", command=mettre_a_jour, width=12, height=2)
bouton_afficher_signal.grid(row=10, column=1, padx=(25, 5), pady=(10, 8))

#Bouton stop
bouton_stop = Button(cadre_exe, text="Stop", command=fenetre.quit, width=12, height=2)
bouton_stop.grid(row=10, column=2, padx=(5, 5), pady=(10, 8))

#Bouton fermer
bouton_fermer = Button(cadre_exe, text="Fermer", command=lambda: [fenetre.quit(), fenetre.destroy()], width=12, height=2)
bouton_fermer.grid(row=10, column=3, padx=(5, 25), pady=(10, 8))

#=============================================================================================================================#
#=============================================================================================================================#
#=============================================================================================================================#

#Lancement de la boucle principale
charger_fichier_par_defaut()
fenetre.mainloop()