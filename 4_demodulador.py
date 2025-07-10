import numpy as np
import json
import os
import time
from abc import ABC, abstractmethod

class BaseDemodulator(ABC):
    """Clase base abstracta para todos los demoduladores"""
    
    def __init__(self, name, bits_per_symbol, constellation_size):
        self.name = name
        self.bits_per_symbol = bits_per_symbol
        self.constellation_size = constellation_size
    
    @abstractmethod
    def demodulate_bayes(self, received_matrix, A=1.0):
        """Método abstracto para demodulación bayesiana"""
        pass
    
    @abstractmethod
    def symbol_to_bits(self, symbol):
        """Convierte símbolo a bits con codificación específica"""
        pass
    
    def gray_to_binary(self, n):
        """Convierte un número en código Gray a su equivalente en binario"""
        b = 0
        while n:
            b ^= n
            n >>= 1
        return b

class FSK8Demodulator(BaseDemodulator):
    """Demodulador 8FSK con codificación Gray"""
    
    def __init__(self):
        super().__init__("8FSK", 3, 8)
        self.frequencies = np.linspace(1e3, 8e3, 8)  # 1kHz a 8kHz
    
    def demodulate_bayes(self, received_matrix, A=1.0):
        """Demodulación Bayesiana para 8FSK"""
        num_symbols = received_matrix.shape[0]
        detected_gray_symbols = np.zeros(num_symbols, dtype=int)
        
        for idx in range(num_symbols):
            r = received_matrix[idx, :]
            distances = []
            
            # Calcular distancia euclidiana para cada símbolo posible
            for i in range(self.constellation_size):
                s_i = np.zeros(self.constellation_size)
                s_i[i] = A
                d = np.sum((r - s_i) ** 2)
                distances.append(d)
            
            # Detectar símbolo con menor distancia
            detected_gray_symbols[idx] = np.argmin(distances)
        
        return detected_gray_symbols
    
    def symbol_to_bits(self, symbol):
        """Convierte símbolo Gray a bits binarios"""
        binary_symbol = self.gray_to_binary(symbol)
        return list(bin(binary_symbol)[2:].zfill(self.bits_per_symbol))
    
    def get_symbol_info(self, detected_symbols):
        """Información específica para 8FSK"""
        frequencies = self.frequencies[detected_symbols]
        binary_symbols = np.array([self.gray_to_binary(sym) for sym in detected_symbols])
        return {
            'frequencies': frequencies,
            'binary_symbols': binary_symbols,
            'gray_symbols': detected_symbols
        }

class QAM16Demodulator(BaseDemodulator):
    """Demodulador 16QAM con codificación Gray"""
    
    def __init__(self):
        super().__init__("16QAM", 4, 16)
        # Mapeo Gray para 16-QAM (mismo que el modulador)
        self.gray_symbol_map = {
            '0100': (-3, +3), '0110': (-1, +3), '1110': (+1, +3), '1100': (+3, +3),
            '0101': (-3, +1), '0111': (-1, +1), '1111': (+1, +1), '1101': (+3, +1),
            '0001': (-3, -1), '0011': (-1, -1), '1011': (+1, -1), '1001': (+3, -1),
            '0000': (-3, -3), '0010': (-1, -3), '1010': (+1, -3), '1000': (+3, -3),
        }
        # Crear mapeo inverso
        self.index_to_bits = {i: bits for i, bits in enumerate(self.gray_symbol_map.keys())}
        self.iq_constellation = np.array(list(self.gray_symbol_map.values()))
    
    def demodulate_bayes(self, received_matrix, A=1.0):
        """Demodulación Bayesiana para 16QAM"""
        num_symbols = received_matrix.shape[0]
        detected_symbols = np.zeros(num_symbols, dtype=int)
        
        for idx in range(num_symbols):
            r = received_matrix[idx, :]
            distances = []
            
            # Calcular distancia euclidiana para cada símbolo posible
            for i in range(self.constellation_size):
                s_i = np.zeros(self.constellation_size)
                s_i[i] = A
                d = np.sum((r - s_i) ** 2)
                distances.append(d)
            
            # Detectar símbolo con menor distancia
            detected_symbols[idx] = np.argmin(distances)
        
        return detected_symbols
    
    def symbol_to_bits(self, symbol):
        """Convierte índice de símbolo a bits Gray"""
        bit_string = self.index_to_bits[symbol]
        return list(bit_string)
    
    def get_symbol_info(self, detected_symbols):
        """Información específica para 16QAM"""
        iq_values = []
        bit_strings = []
        
        for symbol in detected_symbols:
            bit_string = self.index_to_bits[symbol]
            iq_values.append(self.gray_symbol_map[bit_string])
            bit_strings.append(bit_string)
        
        return {
            'iq_values': iq_values,
            'bit_strings': bit_strings,
            'detected_symbols': detected_symbols
        }

class PSK8Demodulator(BaseDemodulator):
    """Demodulador 8PSK con codificación Gray"""
    
    def __init__(self):
        super().__init__("8PSK", 3, 8)
        # Fases para 8PSK (en radianes)
        self.phases = np.array([i * 2 * np.pi / 8 for i in range(8)])
        # Mapeo Gray para 8PSK (mismo que el modulador)
        self.gray_map = [0, 1, 3, 2, 6, 7, 5, 4]
        # Mapeo inverso
        self.inverse_gray_map = {v: k for k, v in enumerate(self.gray_map)}
    
    def demodulate_bayes(self, received_matrix, A=1.0):
        """Demodulación Bayesiana para 8PSK"""
        num_symbols = received_matrix.shape[0]
        detected_gray_symbols = np.zeros(num_symbols, dtype=int)
        
        for idx in range(num_symbols):
            r = received_matrix[idx, :]
            distances = []
            
            # Calcular distancia euclidiana para cada símbolo posible
            for i in range(self.constellation_size):
                s_i = np.zeros(self.constellation_size)
                s_i[i] = A
                d = np.sum((r - s_i) ** 2)
                distances.append(d)
            
            # Detectar símbolo con menor distancia
            detected_gray_symbols[idx] = np.argmin(distances)
        
        return detected_gray_symbols
    
    def symbol_to_bits(self, symbol):
        """Convierte símbolo Gray a bits binarios"""
        # Convertir de Gray a binario original
        binary_symbol = self.inverse_gray_map[symbol]
        return list(bin(binary_symbol)[2:].zfill(self.bits_per_symbol))
    
    def get_symbol_info(self, detected_symbols):
        """Información específica para 8PSK"""
        phases = self.phases[detected_symbols]
        iq_values = []
        binary_symbols = []
        
        for symbol in detected_symbols:
            phase = self.phases[symbol]
            I = np.cos(phase)
            Q = np.sin(phase)
            iq_values.append((I, Q))
            binary_symbols.append(self.inverse_gray_map[symbol])
        
        return {
            'phases': phases,
            'iq_values': iq_values,
            'binary_symbols': binary_symbols,
            'gray_symbols': detected_symbols
        }

class DemodulationSystem:
    """Sistema principal que maneja la selección y ejecución de demoduladores"""
    
    def __init__(self):
        self.demodulators = {
            '8FSK': FSK8Demodulator(),
            '16QAM': QAM16Demodulator(),
            '8PSK': PSK8Demodulator()
        }
    
    def load_channel_output(self, base_path):
        """Carga la salida del canal y metadatos"""
        try:
            # Cargar señal con ruido
            signal_path = os.path.join(base_path, "3_channel_output.txt")
            received_matrix = np.loadtxt(signal_path)
            
            # Cargar información del canal
            info_path = os.path.join(base_path, "3_channel_info.json")
            with open(info_path, 'r') as f:
                channel_info = json.load(f)
            
            # Cargar metadatos originales de modulación
            metadata_path = os.path.join(base_path, "2_modulation_metadata.json")
            with open(metadata_path, 'r') as f:
                modulation_metadata = json.load(f)
            
            return received_matrix, channel_info, modulation_metadata
        
        except Exception as e:
            raise Exception(f"Error cargando archivos del canal: {str(e)}")
    
    def demodulate_signal(self, received_matrix, modulation_type, A=1.0):
        """Demodula la señal según el tipo de modulación"""
        if modulation_type not in self.demodulators:
            raise ValueError(f"Tipo de modulación no soportado: {modulation_type}")
        
        demodulator = self.demodulators[modulation_type]
        
        start_time = time.time()
        
        # Demodulación bayesiana
        detected_symbols = demodulator.demodulate_bayes(received_matrix, A)
        
        # Convertir símbolos a bits
        demodulated_bits = []
        for symbol in detected_symbols:
            bits = demodulator.symbol_to_bits(symbol)
            demodulated_bits.extend([int(b) for b in bits])
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Obtener información específica del demodulador
        symbol_info = demodulator.get_symbol_info(detected_symbols)
        
        return {
            'demodulated_bits': np.array(demodulated_bits, dtype=int),
            'detected_symbols': detected_symbols,
            'symbol_info': symbol_info,
            'elapsed_time': elapsed_time,
            'demodulator': demodulator
        }
    
    def remove_padding(self, bits, padding):
        """Elimina el padding agregado durante la modulación"""
        if padding > 0:
            return bits[:-padding]
        return bits
    
    def save_results(self, results, channel_info, modulation_metadata, base_path):
        """Guarda los resultados de la demodulación"""
        # Archivo principal de bits demodulados
        bits_file = os.path.join(base_path, "4_demodulated_bits.txt")
        np.savetxt(bits_file, results['demodulated_bits'], fmt='%d')
        
        # Archivo de información de demodulación
        demod_info = {
            'modulation_type': modulation_metadata['modulation_type'],
            'constellation_size': modulation_metadata['constellation_size'],
            'total_symbols_processed': len(results['detected_symbols']),
            'padding_removed': modulation_metadata['padding'],
            'bits_per_character': modulation_metadata.get('bits_per_character', None),
            'elapsed_time': results['elapsed_time'],
            'snr_info': channel_info['snr_metrics']
        }
        
        info_file = os.path.join(base_path, "4_demodulation_info.json")
        with open(info_file, 'w') as f:
            json.dump(demod_info, f, indent=2)
        
        # Archivo específico del tipo de modulación
        specific_file = os.path.join(base_path, f"4_demodulated_{modulation_metadata['modulation_type'].lower()}_specific.txt")
        self.save_specific_info(results['symbol_info'], specific_file, modulation_metadata['modulation_type'])
        
        return bits_file, info_file, specific_file
    
    def save_specific_info(self, symbol_info, file_path, modulation_type):
        """Guarda información específica según el tipo de modulación"""
        with open(file_path, 'w') as f:
            if modulation_type == '8FSK':
                f.write("Símbolo_Gray\tBinario\tFrecuencia_Hz\n")
                for i, (gray, binary, freq) in enumerate(zip(
                    symbol_info['gray_symbols'], 
                    symbol_info['binary_symbols'], 
                    symbol_info['frequencies']
                )):
                    f.write(f"{gray}\t{binary}\t{freq:.0f}\n")
            
            elif modulation_type == '16QAM':
                f.write("Índice\tBits_Gray\tI\tQ\n")
                for i, (bits, iq) in enumerate(zip(symbol_info['bit_strings'], symbol_info['iq_values'])):
                    f.write(f"{symbol_info['detected_symbols'][i]}\t{bits}\t{iq[0]}\t{iq[1]}\n")
            
            elif modulation_type == '8PSK':
                f.write("Símbolo_Gray\tBinario\tFase_rad\tI\tQ\n")
                for i, (gray, binary, phase, iq) in enumerate(zip(
                    symbol_info['gray_symbols'],
                    symbol_info['binary_symbols'],
                    symbol_info['phases'],
                    symbol_info['iq_values']
                )):
                    f.write(f"{gray}\t{binary}\t{phase:.4f}\t{iq[0]:.4f}\t{iq[1]:.4f}\n")
    
    def display_results(self, results, channel_info, modulation_metadata):
        """Muestra estadísticas de la demodulación"""
        modulation_type = modulation_metadata['modulation_type']
        
        print(f"\n{'='*60}")
        print(f"RESULTADOS DE DEMODULACIÓN {modulation_type}")
        print(f"{'='*60}")
        print(f"Símbolos procesados: {len(results['detected_symbols'])}")
        print(f"Bits demodulados: {len(results['demodulated_bits'])}")
        print(f"Padding eliminado: {modulation_metadata['padding']} bits")
        print(f"Tiempo de ejecución: {results['elapsed_time']:.4f} segundos")
        
        # Información del canal
        snr_info = channel_info['snr_metrics']
        print(f"\nINFORMACIÓN DEL CANAL:")
        print(f"N0: {channel_info['N0']}")
        print(f"Eb/N0: {snr_info['Eb_N0_dB']:.2f} dB")
        print(f"Es/N0: {snr_info['Es_N0_dB']:.2f} dB")
        
        # Información específica según modulación
        symbol_info = results['symbol_info']
        if modulation_type == '8FSK':
            print(f"\nRANGO DE FRECUENCIAS:")
            print(f"Mínima: {min(symbol_info['frequencies']):.0f} Hz")
            print(f"Máxima: {max(symbol_info['frequencies']):.0f} Hz")
        
        elif modulation_type in ['16QAM', '8PSK']:
            print(f"\nPRIMEROS 5 SÍMBOLOS I/Q:")
            for i, (I, Q) in enumerate(symbol_info['iq_values'][:5]):
                print(f"  Símbolo {i+1}: I={I:.3f}, Q={Q:.3f}")
    
    def run(self):
        """Ejecuta el sistema de demodulación"""
        print("="*60)
        print("SISTEMA DE DEMODULACIÓN UNIVERSAL")
        print("Compatible con modulador y canal existentes")
        print("="*60)
        
        # Ruta base del sistema
        base_path = r"C:/Users/elosc/Desktop/Universidad/STransmision"
        
        try:
            # Cargar datos del canal
            print("Cargando salida del canal...")
            received_matrix, channel_info, modulation_metadata = self.load_channel_output(base_path)
            
            modulation_type = modulation_metadata['modulation_type']
            print(f"Tipo de modulación detectado: {modulation_type}")
            
            # Obtener amplitud de referencia
            try:
                A = float(input("Ingrese la amplitud de referencia A (Enter para 1.0): ") or "1.0")
            except ValueError:
                A = 1.0
                print("Usando amplitud por defecto: 1.0")
            
            # Demodular
            print(f"Demodulando con {modulation_type}...")
            results = self.demodulate_signal(received_matrix, modulation_type, A)
            
            # Remover padding
            final_bits = self.remove_padding(results['demodulated_bits'], modulation_metadata['padding'])
            results['demodulated_bits'] = final_bits
            
            # Guardar resultados
            bits_file, info_file, specific_file = self.save_results(
                results, channel_info, modulation_metadata, base_path
            )
            
            # Mostrar resultados
            self.display_results(results, channel_info, modulation_metadata)
            
            print(f"\nARCHIVOS GENERADOS:")
            print(f"- Bits demodulados: {bits_file}")
            print(f"- Información: {info_file}")
            print(f"- Específico {modulation_type}: {specific_file}")
            
            print(f"\n✅ Demodulación completada exitosamente!")
            
        except Exception as e:
            print(f"Error durante la demodulación: {str(e)}")
            print("\nVerifique que existan los archivos generados por el canal:")
            print("- 3_channel_output.txt")
            print("- 3_channel_info.json")
            print("- 2_modulation_metadata.json")

if __name__ == "__main__":
    system = DemodulationSystem()
    system.run()