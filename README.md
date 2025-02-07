# Laboratorio-1-PDS
## OBJETIVO
El presente laboratorio tiene como objetivo desarrollar habilidades para obtener, procesar y visualizar señales biomédicas utilizando herramientas de programación como Python mediante librerías específicas como Matplotlib y calcular e interpretar estadísticos descriptivos clave (media, desviación estándar, coeficiente de variación, histogramas y función de probabilidad) que permitan describir las características de las señales biomédicas.
## Procedimiento
La señal biomédica a estudiar se obtuvo a partir de bases de datos de señales fisiológicas, en este caso physionet, en este caso es una señal EMG con una neuropatía,seguido a esto se importó la señal a python. Para adquirir la señal se importaron tres librerias especificas: WFDB (Waveform DataBase): Esta librería es crucial para trabajar con datos de señales fisiológicas, como electrocardiogramas (ECG), electromiogramas (EMG) esta permite leer y esrcibir señales digitalizadas, Matplotlib: se utiliza para crear gráficos y visualizaciones de datos, NumPy: es fundamental para trabajar con arrays y matrices numéricas. Al importar estas librerias empezamos a realizar nuestro codigo para adquirir los datos de la señal descargada de la siguiente manera:
```ruby
import wfdb
import matplotlib.pyplot as plt
import numpy as np
import random

#Adquirir datos
ruta = 'C:\\Users\\sachi\\Desktop\\examples-of-electromyograms-1.0.0\\emg_neuropathy'
record = wfdb.rdrecord(ruta)
#print(record._dict_)
signal = record.p_signal
fs = record.fs
muestreo = int(2*fs)
print("Frecuencia de muestreo = ", fs)

time = [i / fs for i in range(len(signal))]
signal = signal[:muestreo]
time = time[:muestreo]
```
### Adquirir datos
En esta sección se explicara detalladamente el funcionamiento del codigo para compilarlo de manera correcta con cualquier compilador de python.\
Como se puede observar se utiliza la variable "ruta" que es la dirección de la carpeta del archivo de datos EMG obtenido de Physioent.\ 
**signal = record.p_signal**:  Extrae la señal propiamente dicha del objeto record y la guarda en la variable signal. p_signal suele contener los valores de la señal en sí.  Es un array de NumPy.\
**fs = record.fs**: Extrae la frecuencia de muestreo (sampling rate) del registro y la guarda en la variable fs. Esta indica cuántas muestras de la señal se tomaron por segundo.\
**muestreo = int(2*fs)**: Calcula el número de muestras que se van a utilizar.  En este caso, se toman dos segundos de la señal, ya que se multiplica la frecuencia de muestreo (fs) por 2.  El resultado se convierte a entero con int().\
**print("Frecuencia de muestreo = ", fs)**: Imprime la frecuencia de muestreo en la consola.\
**time = [i / fs for i in range(len(signal))]**: Crea una lista llamada "time" que representa el eje de tiempo para la señal.  Para cada muestra i en la señal, calcula el tiempo correspondiente dividiendo i por la frecuencia de muestreo fs.  Esto genera una lista de tiempos en segundos.\
**signal = signal[:muestreo]**: Permite cortar la señal signal para que solo contenga las primeras muestreo muestras.  Esto asegura que solo se utilicen los dos segundos de datos calculados anteriormente.\
**time = time[:muestreo]**:  De manera similar a la angterior, esta permite cortar la lista de tiempos time para que coincida con la longitud de la señal que se ha cortado. Esto asegura que los tiempos correspondan a las muestras de señal que se están utilizando.\
Este fragmento de código nos ayudará a cargar los datos del EMG desde un archivo.A extraer la señal y la frecuencia de muestreo a calcular un vector de un tiempo correspondiente y luego a seleccionar los primeros dos segundos de la señal para su posterior análisis o visualización.
### Cálculos de estadísticos manuales
Este código calcula manualmente varias estadísticas descriptivas para un conjunto de datos numéricos representados por la variable signal, utilizando bucles for y operaciones aritméticas básicas.
```ruby
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
```
**n = len(signal)**: Se calcula el número total de elementos en la lista "signal" y se almacena en la variable n.\
**suma = 0**: Se inicializa una variable suma a 0. Esta variable se utilizará para calcular la suma de todos los elementos en "signal".\
**for i in range(0, n)**: Se inicia un bucle for que repite sobre cada elemento de la lista signal. La variable i representa el índice del elemento actual.\
**suma = suma + signal[i]**: En cada repetición, el valor del elemento actual signal[i] se añade a la variable suma.\
**varianzama = E / (n - 1)**: Una vez calculada la suma de los cuadrados de las diferencias, se divide por n - 1 (en lugar de n) para obtener la varianza muestral, que se almacena en la variable "varianzama".\
**promma = suma / n**: Una vez calculada la suma de todos los elementos, se divide por el número total de elementos n para obtener la media aritmética, que se almacena en la variable promma.\
**desviama = varianzama ** 0.5**: Se calcula la raíz cuadrada de la varianza varianzama para obtener la desviación estándar, que se almacena en la variable desviama. La desviación estándar mide la dispersión de los datos alrededor de la media.\
**cvariama = desviama / promma**: Se divide la desviación estándar por la media para obtener el coeficiente de variación, que se almacena en la variable cvariama. El coeficiente de variación es una medida de dispersión relativa que permite comparar la variabilidad de conjuntos de datos con diferentes unidades o escalas.

### Cálculos de estadísticos con numpy
En el siguiente código se evidenciará como se obtuvieron cuatro estadísticos descriptivos: Media, Varianza, Desviación estándar, Coeficiente de variación utilizando la libreria numpy de la siguiente manera:
```ruby
#Cálculos de estadísticos con numpy
promnum = np.mean(signal)
varianum = np.var(signal)
desvinum = np.std(signal)
cvarianum = desvinum/promnum
```
**promnum = np.mean(signal)**: Calcula la media aritmética de los elementos de la señal y la almacena. La función np.mean() de NumPy calcula el promedio de los valores en un array.\
**varianum = np.var(signal)**: Calcula la varianza de los elementos de la señal y la almacena. La varianza es una medida de dispersión que indica cuánto se alejan los valores individuales de la media.\
**desvinum = np.std(signal)**: Calcula la desviación estándar de los elementos de la señal y la almacena. La desviación estándar es la raíz cuadrada de la varianza y proporciona una medida de dispersión más interpretable en las unidades originales de los datos.\
**cvarianum = desvinum / promnum**: Calcula el coeficiente de variación de los elementos de la señal y lo almacena. El coeficiente de variación es una medida de dispersión relativa que compara la desviación estándar con la media.\
Estos estadísticos proporcionan información sobre la tendencia central y la dispersión de los datos en la señal EMG.
### Imprimir estadísticos
Este código muestra una comparación de estadísticas descriptivas (promedio, varianza, desviación estándar y coeficiente de variación) calculadas de dos formas distintas: manualmente y mediante programación. Esto permite verificar si los resultados manuales concuerdan con los obtenidos a través de métodos programación.
```ruby
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
```
![image](https://github.com/user-attachments/assets/162efda8-1d59-461f-8c49-8c6c064c969a)

### Graficar la señal
Este código utiliza la biblioteca matplotlib.pyplot para crear el gráfico de la señal de EMG con neuropatía de la siguiente forma:
```ruby
#Graficar la señal
plt.figure(figsize=(10,4))
plt.plot(time, signal, label="Señal EMG")
plt.title('Señal EMG con Neuropatía')
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud [mV]")
plt.legend()
plt.grid()
plt.show()
```
![señal_original](https://github.com/user-attachments/assets/7b74ad29-4f1d-4a63-964f-6462f469e16e)

### Histograma y función de probabilidad 
El histograma es la representación gráfica de cuántas veces se repite cierto dato o cierto rango de datos en una señal y muestra como estos están distrubuidos, lo cual indica las tendencias que presenta el conjunto de datos y la función de probabilidad representa qué tan probable es que un dato al azar sea exactamente igual a algún valor de la muestra.\
En el código se implementaron comandos de la librería numpy para realizar el histograma con una cantidad de 30 intervalos o "cajas" y pyplot para graficar.

```ruby
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
```
La funciónn **np.histogram()** realiza el histograma de un arreglo de datos asignado, que en este caso es la variable **signal** 
![histo_funpro](https://github.com/user-attachments/assets/2a1fd296-4f56-4edd-8484-4ce0f2f618ae)



### Generación de ruido y el SNR
Para este laboratorio se contaminó la señal con tres tipos de ruido: Gaussiano, de impulso y de artefacto.\
El ruido Gaussiano está asociado a la radiación electromagnética y es aquel que tiene una distribución normal (tiene una tendencia similar a la campana de Gauss), el ruido de impulso es el que tiene picos de alta amplitud pero de corta duración y el ruido de artefactos es el que producen elementos externos como corrientes eléctricas o, en el caso de la adquisición de señales biológicas, piel o corriente de otro tipo de señales. Para discernir entre cuánto hay de señal y cuánto hay de ruido en la toma de una señal, existe el SNR, el cual por sus siglas en inglés (Signal to Noise Ratio) establece la cantidad de información útil hay en una medición respecto al ruido. Esta relación está dada por la ecuación $SNR = 10 \times \log_{10}(\frac{potencia de la señal}{potencia del ruido})$\
**Ruido Gaussiano**\
Se refiere a un tipo específico de ruido que se caracteriza por tener una distribución de probabilidad normal o gaussiana. En otras palabras, los valores de este ruido se distribuyen de forma simétrica alrededor de un valor promedio, y la probabilidad de encontrar valores alejados de este promedio disminuye exponencialmente.
```ruby
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
```
El ruido genera valores aleatorios con una distribución normal, por lo que su media está localizada en 0 y se tiene en cuenta la desviación de la señal original para generar el ruido a partir de esta en el comando **scale=np.std(signal)0.01**. Después de generar el ruido se le añade a la señal original y se grafíca, y a su vez se calcula el respectivo SNR.\
![gauss_noise](https://github.com/user-attachments/assets/cb3e6d23-6c94-49b9-adae-abe9787ed489)


**Ruido de impulso**\
Se refiere a un tipo de interferencia que se caracteriza por ser de corta duración y alta amplitud. Estos impulsos de ruido pueden aparecer de forma repentina y aislada o pueden ocurrir de manera repetitiva a intervalos regulares o irregulares.
```ruby
##Ruido de impulso
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
```
En este código se agrega aleatoriamente un número entre -1 y 1 con una probabilidad del 5%. Luego de esto se añade a la señal original y se calcula la potencia de ambas para el SNR y se genera la gráfia correspondiente.\
![imp_noise](https://github.com/user-attachments/assets/3fb4f5f9-9e84-4812-9b62-efb9d209c7bb)


**Ruido de aparato**\
Se refiere a las interferencias o señales no deseadas que pueden ser generadas por el propio equipo de medición o sus componentes. Estas señales pueden contaminar la señal EMG real, dificultando su interpretación y análisis.
```ruby
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
```
La línea **art_noise = signal[:]** genera una copia de la señal en el arreglo de la señal orginal, en el **for** se añaden 30 picos aleatorios donde se selecciona un índice aleatorio dentro del tamaño del arreglo **signal** en la línea **idx = random.randint(0, len(signal)-1)** y luego de esto se le añade el ruido por un número de forma aleatoria en un rango entre -2 y 2. La gráfica de esta señal será la que tenga más ruido ya que por los picos que pueden alcanzar un valor significativo la señal se verá más alterada y menos parecida a la original.\
![art_noise](https://github.com/user-attachments/assets/3bd63371-efb5-4a68-a926-cca48f29a2c5)

![compare_signals](https://github.com/user-attachments/assets/a488e154-b873-4b87-9c93-b82facf7c245)

