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
En esta sección se explicara detalladamente el funcnionamiento del codigo para compilarlo de manera correcta con cualquier compilador de python.\
Como se puede observar se utiliza la variable "ruta" que almacena la ruta al archivo de datos EMG.  Esta ruta específica indica que el archivo se encuentra en el escritorio del usuario "sachi" dentro de una carpeta llamada **"examples-of-electromyograms-1.0.0"** y el archivo se llama "emg_neuropathy".\
Posterior a esto se utiliza la función "wfdb.rdrecord(ruta)" para leer los datos del registro. Esta función de la librería WFDB mencionada anteriormente carga los datos de la señal y otra información asociada (como la frecuencia de muestreo) desde el archivo especificado en ruta. El objeto "record" contiene toda esta información.\
**signal = record.p_signal**:  Extrae la señal propiamente dicha del objeto record y la guarda en la variable signal. p_signal suele contener los valores de la señal en sí.  Es un array de NumPy.\
**fs = record.fs**: Extrae la frecuencia de muestreo (sampling rate) del registro y la guarda en la variable fs. Esta indica cuántas muestras de la señal se tomaron por segundo.\
**muestreo = int(2*fs)**: Calcula el número de muestras que se van a utilizar.  En este caso, se toman dos segundos de la señal, ya que se multiplica la frecuencia de muestreo (fs) por 2.  El resultado se convierte a entero con int().\
**print("Frecuencia de muestreo = ", fs)**: Imprime la frecuencia de muestreo en la consola.\
**time = [i / fs for i in range(len(signal))]**: Crea una lista llamada "time" que representa el eje de tiempo para la señal.  Para cada muestra i en la señal, calcula el tiempo correspondiente dividiendo i por la frecuencia de muestreo fs.  Esto genera una lista de tiempos en segundos.\
**signal = signal[:muestreo]**: Permite cortar la señal signal para que solo contenga las primeras muestreo muestras.  Esto asegura que solo se utilicen los dos segundos de datos calculados anteriormente.\
**time = time[:muestreo]**:  De manera similar a la angterior, esta permite cortar la lista de tiempos time para que coincida con la longitud de la señal que se ha cortado. Esto asegura que los tiempos correspondan a las muestras de señal que se están utilizando.\
