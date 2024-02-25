"""
Cree le 19/12/2023 par LE RUYET Guillaume, LORVELLEC Evan, SIGUIER Adrien

Version 3

But : 1. Modeliser la generation de signal sinusoidale ou SWEEP.
      2. Modeliser fonctionnement d'un Lock-In Amplifier. 
      3. Redemontrer Kramers-Kronig en utilisant la fonction SWEEP et un circuit RC externe.

Entrees  :  Amplitude, frequence et dephasage du signal. Constante de temps du Lock-IN Amplifier.

Sorties : Graphes de R, theta, X et Y ainsi que leurs moyennes.
"""
import numpy as np
import matplotlib.pyplot as plt

# Generation d'un signal SWEEP
def generate_sweep(amplitude, freq_deb, freq_fin, duree, echantillonnage, dephasage):
    t      = np.arange(taux, duree, 1/echantillonnage)
    freq   = np.linspace(freq_deb, freq_fin, len(t))
    signal = amplitude * np.sin(2 * np.pi * freq * t + dephasage)
    return signal

# Modelisation du circuit RC
def rc_circuit(v, t, R, C, Vin):
    tau  = R * C
    dvdt = (Vin - v) / tau
    return dvdt

# Fonction pour concevoir un filtre passe-bas RC
def rc(signal, fc, fs):
    dt = 1 / fs
    alpha = dt / (1/(2*np.pi*fc) + dt)
    signal_filtre = np.zeros_like(signal)
    for i in range(1, len(signal)):
        signal_filtre[i] = alpha * signal[i] + (1 - alpha) * signal_filtre[i - 1]
    return signal_filtre

# Parametres du signal d'entree
amplitude = float(input("Entrer l'amplitude du signal A0 : "))
taux      = float(input("Entrer la constante de temps : "))
# frequence = float(input("Entrer la frequence du signal f0 : "))
phase     = float(input("Entrer la phase du signal phi : "))

# Parametres du Lock-In Amplifier
R2            = 0.01
fc            = 1/R2*taux
# frequence_ref = frequence
phase_ref     = phase
R = 1.0  # Resistance en ohms
C = 1e-3  # Capacite en farads

# Variables pour le SWEEP
frequence_depart = 0        # Frequence de depart en Hz
frequence_fin    = 16000    # Frequence de fin en Hz
duree_sweep      = 60       # Duree de la sweep en secondes
echantillonnage  = 1000     # Taux d'echantillonnage en Hz
t                = np.arange(taux, duree_sweep, 1/echantillonnage)  # Axe des temps
f                = 1/t                                              # Axe des frequences

# Creation du signal d'entree
signal_entree = generate_sweep(amplitude, frequence_depart, frequence_fin, duree_sweep, echantillonnage, 0)

# Simulation du circuit RC
signal_sortie = np.zeros_like(signal_entree)
amplitude_sortie_filtre = 0.0

for i in range(len(signal_entree) - 1):
    dt = 1/echantillonnage
    amplitude_entree = signal_entree[i]
    amplitude_sortie_filtre = amplitude_sortie_filtre + rc_circuit(amplitude_sortie_filtre, i*dt, R, C, amplitude_entree) * dt
    signal_sortie[i + 1] = amplitude_sortie_filtre

# Creation du signal de reference pour le Lock-In Amplifier non dephase et dephase
signal_ref      = generate_sweep(1, frequence_depart, frequence_fin, duree_sweep, echantillonnage, 0)
signal_ref_perp = generate_sweep(1, frequence_depart, frequence_fin, duree_sweep, echantillonnage, 1.571)

# Mixage du signal d'entree avec le signal de reference
signal_mixe      = signal_ref * signal_sortie
signal_mixe_perp = signal_ref_perp * signal_sortie

# Filtre passe-bas RC pour extraire la composante a la frequence de reference
composante_verrouillee  = rc(signal_mixe, fc, 10000)
composante_verrouillee2 = rc(signal_mixe_perp, fc, 10000)

# Creation de R et Theta
R     = 2*np.sqrt(composante_verrouillee**2 + composante_verrouillee2**2)
theta = np.arctan2(composante_verrouillee2, composante_verrouillee)

# Conversion des radians en degres
theta_deg = np.degrees(theta)

# Affichage des signaux
plt.figure(figsize=(10, 6))

plt.figure(1)
plt.subplot(1, 2, 1)
plt.plot(f, signal_entree, label='Signal d\'entree') 
plt.title('Signal d\'entree')
plt.xlabel('Temps (s)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(f, signal_ref, label='Signal de reference')
plt.title('Signal de reference')
plt.xlabel('Temps (s)')
plt.legend()

plt.figure(2)
plt.subplot(1, 2, 1)
plt.plot(f, R, label='Composante verrouillee')
plt.title('R')
plt.xlabel('Temps (s)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(f, theta, label='Composante verrouillee 2')
plt.title('Theta')
plt.xlabel('Temps (s)')
plt.legend()

plt.figure(3)
plt.subplot(1, 2, 1)
plt.plot(f, composante_verrouillee, label='Composante verrouillee')
plt.title('X')
plt.xlabel('Temps (s)')

plt.subplot(1, 2, 2)
plt.plot(f, composante_verrouillee2, label='Composante verrouillee 2')
plt.title('Y')
plt.xlabel('Temps (s)')

plt.tight_layout()
plt.show()

moyenne_R     = np.mean(R)
moyenne_theta = np.mean(theta)

print("Moyenne de R:", moyenne_R)
print("Moyenne de Theta:", moyenne_theta)

