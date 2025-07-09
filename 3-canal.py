import numpy as np
import json
import os
import matplotlib.pyplot as plt

def load_signal_and_metadata(signal_path, metadata_path):
    signal = np.loadtxt(signal_path, dtype=int)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return signal, metadata

def add_awgn(signal, N0):
    sigma = np.sqrt(N0 / 2)
    noise = np.random.normal(0, sigma, size=signal.shape)
    noisy_signal = signal + noise
    return noisy_signal, noise

def calculate_snr_metrics(metadata, N0):
    M = metadata['constellation_size']
    bits_per_symbol = int(np.log2(M))
    Eb = 1 / bits_per_symbol
    Es = Eb * bits_per_symbol
    Eb_N0 = Eb / N0
    Es_N0 = Es / N0
    return {
        'Eb': Eb,
        'Es': Es,
        'Eb_N0_dB': 10 * np.log10(Eb_N0),
        'Es_N0_dB': 10 * np.log10(Es_N0)
    }

def save_outputs(output_dir, noisy_signal, noise, metadata, snr_metrics, N0):
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(f"{output_dir}/3_channel_output.txt", noisy_signal)
    np.savetxt(f"{output_dir}/3_channel_noise.txt", noise)
    channel_info = {
        'modulation_type': metadata['modulation_type'],
        'constellation_size': metadata['constellation_size'],
        'total_symbols': metadata['total_symbols'],
        'padding': metadata['padding'],
        'N0': N0,
        'snr_metrics': snr_metrics
    }
    with open(f"{output_dir}/3_channel_info.json", 'w') as f:
        json.dump(channel_info, f, indent=2)

def plot_comparison(signal, noisy_signal):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.title("Señal Original (One-Hot)")
    plt.imshow(signal[:20].T, cmap='Blues', aspect='auto')
    plt.xlabel("Símbolos")
    plt.ylabel("Componentes")

    plt.subplot(1, 2, 2)
    plt.title("Señal con Ruido")
    plt.imshow(noisy_signal[:20].T, cmap='Reds', aspect='auto')
    plt.xlabel("Símbolos")

    plt.tight_layout()
    plt.savefig("3_channel_analysis.png", dpi=300)
    plt.show()

def main():
    print("="*60)
    print("CANAL AWGN UNIVERSAL COMPATIBLE CON MODULADOR EXISTENTE")
    print("="*60)

    # Ruta esperada por tu sistema
    base_path = r"C:/Users/elosc/Desktop/Universidad/STransmision/STransmision"
    signal_path = os.path.join(base_path, "2_modulated_signal.txt")
    metadata_path = os.path.join(base_path, "2_modulation_metadata.json")

    if not os.path.exists(signal_path) or not os.path.exists(metadata_path):
        print("Error: faltan archivos necesarios del modulador.")
        return

    # Entrada de N0
    try:
        N0 = float(input("Ingrese el valor de N0: "))
    except ValueError:
        print("Valor inválido.")
        return

    # Procesamiento
    signal, metadata = load_signal_and_metadata(signal_path, metadata_path)
    noisy_signal, noise = add_awgn(signal, N0)
    snr_metrics = calculate_snr_metrics(metadata, N0)
    save_outputs(base_path, noisy_signal, noise, metadata, snr_metrics, N0)

    # Resultados
    print(f"\nResultado del canal con N0 = {N0}:")
    print(f"Eb/N0 = {snr_metrics['Eb_N0_dB']:.2f} dB")
    print(f"Es/N0 = {snr_metrics['Es_N0_dB']:.2f} dB")
    print("\nArchivos guardados en:", base_path)

    # Gráfica
    plot_comparison(signal, noisy_signal)

if __name__ == "__main__":
    main()
