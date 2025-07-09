import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

def leer_audio(nombre_archivo):
    fs, data = wavfile.read(nombre_archivo)
    if len(data.shape) > 1:
        data = data[:, 0]
    return fs, data

def mu_law_encode(audio_signal, MU, xmax, ymax):
    """ Aplica la cuantización no uniforme Ley Mu """
    if MU == 0:
        return audio_signal
    return ymax * np.log(1 + MU * np.abs(audio_signal) / xmax) * np.sign(audio_signal) / np.log(1 + MU)

def mu_law_decode(audio_signal, MU, xmax, ymax):
    if MU == 0:
        return audio_signal
    return xmax * (np.power(1 + MU, np.abs(audio_signal) / ymax) - 1) * np.sign(audio_signal) / MU

def cuantificar_senal(audio_muestreado, niveles):
    max_valor_abs = np.max(abs(audio_muestreado))
    paso = (2 * max_valor_abs) / (niveles - 1)
    senal_cuantificada = np.round((audio_muestreado + max_valor_abs) / paso) * paso - max_valor_abs
    return senal_cuantificada, paso, niveles

def codificar_senal(senal_cuantificada, paso, niveles):
    bits = int(np.log2(niveles))
    codigos = ((senal_cuantificada + np.max(np.abs(senal_cuantificada))) / paso).astype(int)
    codigos_binarios = [format(codigo, f'0{bits}b') for codigo in codigos]
    return codigos_binarios, bits

# Ruta del archivo de audio de entrada (modifica esta línea según la ubicación de tu archivo)
audio_file_path = r"C:/Users/elosc/Desktop/Universidad/STransmision/0-audio.wav"

# Leer el archivo de audio
fs, audio_signal = leer_audio(audio_file_path)

# Parámetros
MU = float(input("Ingrese el parámetro mu de compresión (valor positivo): "))
bits = int(input("Ingrese la cantidad de bits de cuantizacion: "))
niveles = int(2 ** bits)

xmax = np.max(np.abs(audio_signal))
ymax = xmax

# Aplicar la cuantización no uniforme Ley Mu
senal_comprimida = mu_law_encode(audio_signal, MU, xmax, ymax)
senal_cuantificada, paso, niveles = cuantificar_senal(senal_comprimida, niveles)
senal_expandida = mu_law_decode(senal_cuantificada, MU, xmax, ymax)

np.savetxt('1-senal_cuantizada.txt', senal_cuantificada)

# Aplicar codificacion
codigos_binarios, bits_por_caracter = codificar_senal(senal_cuantificada, paso, niveles)

# Unir todos los códigos binarios en una sola cadena de bits
bitstream = ''.join(codigos_binarios)

# Contar la cantidad de '0' y '1'
bit_zeros = bitstream.count('0')
bit_ones = bitstream.count('1')

# Calcular la equiprobabilidad
total_bits = len(bitstream)

if total_bits > 0:
    prob_zeros = bit_zeros / total_bits
    prob_ones = bit_ones / total_bits

    print(f"Bits '0': {bit_zeros}, Bits '1': {bit_ones}")
    print(f"Equiprobabilidad: {prob_zeros:.2%} de '0', {prob_ones:.2%} de '1'")
else:
    print("Error: No hay bits en la secuencia.")

# Guardar los codigos binarios
with open(r"C:/Users/elosc/Desktop/Universidad/STransmision/1-senal_codificada.txt", "w") as f:
    f.writelines("\n".join(codigos_binarios) + "\n")
    f.write(f"Bits por carácter: {bits_por_caracter}\n")  # Agrega la cantidad de bits al final

print("Bits generados y guardados en '1-senal_codificada.txt'")

# Gráficos
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.hist(audio_signal, bins=200, color='blue', alpha=0.7)
plt.title("Histograma - Señal muestreada original")
plt.xlabel("Amplitud")
plt.ylabel("Frecuencia")

plt.subplot(2, 1, 2)
plt.hist(senal_comprimida, bins=200, color='red', alpha=0.7)
plt.title("Histograma - Señal comprimida")
plt.xlabel("Amplitud")
plt.ylabel("Frecuencia")

plt.tight_layout()
plt.show()
