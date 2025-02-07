import wfdb
import matplotlib.pyplot as plt
import numpy as np
import random

#Adquirir datos
ruta = 'C:\\Users\\sachi\\Desktop\\examples-of-electromyograms-1.0.0\\emg_neuropathy'
record = wfdb.rdrecord(ruta)
#print(record.__dict__)
signal = record.p_signal
fs = record.fs
muestreo = int(2*fs)
print("Frecuencia de muestreo = ", fs)

time = [i / fs for i in range(len(signal))]
signal = signal[:muestreo]
time = time[:muestreo]

#Cálculos de estadísticos manuales
n = len(signal)
suma = 0
for i in range (0,n):
    suma = suma + signal[i]
    i = i+1
promma = suma/n
E = 0
for i in range (0,n):
    E = E + (signal[i]-promma)**2
varianzama = E/(n-1)
desviama = varianzama**0.5
cvariama = desviama/promma

#Cálculos de estadísticos con numpy
promnum = np.mean(signal)
varianum = np.var(signal)
desvinum = np.std(signal)
cvarianum = desvinum/promnum

#Imprimir estadísticos
print("\nEstadísticos descriptivos manuales")
print("Promedio: ",promma)
print("Varianza: ",varianzama)
print("Desviación estándar: ",desviama)
print("Coeficiente de variación: ",cvariama)

print("\nEstadísticos descriptivos programados")
print("Promedio: ",promnum)
print("Varianza: ",varianum)
print("Desviación estándar: ",desvinum)
print("Coeficiente de variación: ",cvarianum)

#Graficar la señal
plt.figure(figsize=(10,4))
plt.plot(time, signal, label="Señal EMG")
plt.title('Señal EMG con Neuropatía')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")
plt.legend()
plt.grid()
plt.show()

#Graficar histograma y funcion de probabilidad
hist, bins = np.histogram(signal, bins=30, density=True)
pdf = hist 
bin_centers = (bins[:-1] + bins[1:]) / 2

plt.figure(figsize=(10,4))
plt.hist(signal, bins=30, alpha=0.75, color='b', edgecolor='black', density=True)
plt.plot(bin_centers, pdf, marker='o', linestyle='-', color='r', label="Función de Probabilidad")
plt.xlabel("Voltaje [mV]")
plt.ylabel("Frecuencia")
plt.title("Histograma de la Señal EMG con Neuropatía")
plt.grid()
plt.show()

#Generar ruido Gaussiano
noise = np.random.normal(loc=0,scale=np.std(signal)*0.01, size=signal.shape)
signal_gauss = signal + noise
potencia_signal = np.mean(signal**2)
potencia_gauss = np.mean(noise**2)
SNRg = 10*np.log10(potencia_signal/potencia_gauss)
print("\nSNR Gaussiano: ",SNRg,"dB")

plt.figure(figsize=(10, 4))
#plt.plot(time, signal, label="Señal Original", alpha=0.8)
plt.plot(time, signal_gauss, label="Señal con Ruido")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [mV]")
plt.title("Señal neuropática con Ruido Gaussiano")
plt.legend()
plt.grid()
plt.show()

# Generar ruido de impulso
imp_noise = [random.uniform(-1,1) if random.random()<0.05 else 0 for _ in range(len(signal))]
signal_imp = [signal[i] + imp_noise[i] for i in range(len(signal))]

potenciai = np.mean(signal**2)
impulso = signal_imp-signal
ruidoi = np.mean(impulso**2)
SNRi = 10 * np.log10(potenciai / ruidoi)
print("\nSNR Impulso: ",SNRi,"dB")

plt.figure(figsize=(10, 4))
plt.plot(signal_imp, label="Señal con Ruido de Impulso")
plt.title("Señal con Ruido de Impulso")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [mV]")
plt.legend()
plt.grid(True)
plt.show()

#Generar ruido de aparato
art_noise = signal[:]
for _ in range(30):
    idx = random.randint(0, len(signal)-1)
    art_noise[idx] += random.uniform(-2,2)
signal_art = art_noise 

potenciase = np.mean(signal**2)
potenciart = np.mean(art_noise[idx]**2)
SNRa = 10*np.log10(potenciase/potenciart)
print("\nSNR Artefacto: ",SNRa,"dB")

plt.figure(figsize=(10,4))
plt.plot(time, signal_art)
plt.title("Señal con ruido de artefacto")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [mV]")
#plt.legend()
plt.grid(True)
plt.show()

#Comparación entre gráficas
plt.figure(figsize=(12,8))
plt.subplot(4,1,1)
plt.plot(signal, label="Señal original", color="royalblue")
plt.title("Señal original")
plt.legend()

plt.subplot(4,1,2)
plt.plot(signal_gauss, label="Señal Gauss", color="darkblue")
plt.title("Señal Gauss")
plt.legend()

plt.subplot(4,1,3)
plt.plot(signal_imp, label="Señal impulso", color="violet")
plt.title("Señal impulso")
plt.legend()

plt.subplot(4,1,4)
plt.plot(signal_art, label="Señal artefacto", color="purple")
plt.title("Señal artefacto")
plt.legend()

plt.tight_layout()
plt.show()
