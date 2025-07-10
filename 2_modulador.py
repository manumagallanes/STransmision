import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import os
import json

class BaseModulator(ABC):
    """Clase base abstracta para todos los moduladores"""
    
    def __init__(self, name, bits_per_symbol):
        self.name = name
        self.bits_per_symbol = bits_per_symbol
    
    @abstractmethod
    def modulate(self, bits):
        """Método abstracto para modular bits"""
        pass
    
    @abstractmethod
    def get_constellation_size(self):
        """Retorna el tamaño de la constelación"""
        pass
    
    def read_bits_from_file(self, file_path):
        """Lee bits desde archivo manteniendo formato original"""
        bits = []
        bits_per_character = None
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Formato 1: Como 1-senal_codificada.txt (códigos binarios por línea)
        if any(line.strip().startswith("Bits por carácter:") for line in lines):
            for line in lines:
                line = line.strip()
                if line.startswith("Bits por carácter:"):
                    bits_per_character = int(line.split(":")[1].strip())
                    continue
                # Cada línea es un código binario completo
                if line and not line.startswith("Bits por carácter:"):
                    bits.extend(list(map(int, line)))
        
        # Formato 2: Como 1_bit_sequence.txt (un bit por línea)
        elif all(line.strip() in ['0', '1'] for line in lines if line.strip()):
            for line in lines:
                line = line.strip()
                if line:
                    bits.append(int(line))
        
        return bits, bits_per_character
    
    def apply_padding(self, bits):
        """Aplica padding para completar símbolos"""
        remainder = len(bits) % self.bits_per_symbol
        padding = 0
        if remainder != 0:
            padding = self.bits_per_symbol - remainder
            bits.extend([0] * padding)
            print(f"Se agregaron {padding} ceros para completar grupos de {self.bits_per_symbol} bits.")
        else:
            print("No fue necesario agregar ceros.")
        
        return bits, padding

class FSK8Modulator(BaseModulator):
    """Modulador 8FSK con codificación Gray"""
    
    def __init__(self):
        super().__init__("8FSK", 3)
        self.frequencies = np.linspace(1e3, 8e3, 8)  # 1kHz a 8kHz
        self.constellation_size = 8
    
    def binary_to_gray(self, n):
        """Convierte binario a Gray"""
        return n ^ (n >> 1)
    
    def modulate(self, bits):
        """Modula bits usando 8FSK con codificación Gray"""
        bits, padding = self.apply_padding(bits)
        
        # Convertir grupos de 3 bits en enteros
        symbols = []
        for i in range(0, len(bits), 3):
            bit_group = bits[i:i+3]
            decimal_value = int("".join(map(str, bit_group)), 2)
            symbols.append(decimal_value)
        
        # Aplicar codificación Gray
        gray_symbols = [self.binary_to_gray(sym) for sym in symbols]
        
        # Mapear a frecuencias
        modulated_values = [self.frequencies[gray_sym] for gray_sym in gray_symbols]
        
        # Crear matriz one-hot para compatibilidad con canal
        one_hot_matrix = np.zeros((len(gray_symbols), self.constellation_size))
        for i, symbol in enumerate(gray_symbols):
            one_hot_matrix[i, symbol] = 1
        
        return {
            'modulated_signal': one_hot_matrix,  # Formato estándar para el canal
            'symbol_values': modulated_values,   # Valores de frecuencia originales
            'gray_symbols': gray_symbols,
            'padding': padding,
            'modulation_type': '8FSK',
            'constellation_size': self.constellation_size
        }
    
    def get_constellation_size(self):
        return self.constellation_size

class QAM16Modulator(BaseModulator):
    """Modulador 16QAM con codificación Gray"""
    
    def __init__(self):
        super().__init__("16QAM", 4)
        self.constellation_size = 16
        # Mapeo Gray para 16-QAM
        self.gray_symbol_map = {
            '0100': (-3, +3), '0110': (-1, +3), '1110': (+1, +3), '1100': (+3, +3),
            '0101': (-3, +1), '0111': (-1, +1), '1111': (+1, +1), '1101': (+3, +1),
            '0001': (-3, -1), '0011': (-1, -1), '1011': (+1, -1), '1001': (+3, -1),
            '0000': (-3, -3), '0010': (-1, -3), '1010': (+1, -3), '1000': (+3, -3),
        }
        # Crear mapeo inverso para one-hot
        self.symbol_to_index = {bits: i for i, bits in enumerate(self.gray_symbol_map.keys())}
    
    def modulate(self, bits):
        """Modula bits usando 16QAM con codificación Gray"""
        bits, padding = self.apply_padding(bits)
        
        # Dividir bits en grupos de 4
        symbols = [bits[i:i+4] for i in range(0, len(bits), 4)]
        
        # Convertir a amplitudes I,Q y crear one-hot
        iq_values = []
        one_hot_matrix = np.zeros((len(symbols), self.constellation_size))
        
        for i, symbol in enumerate(symbols):
            bit_string = ''.join(map(str, symbol))
            iq_values.append(self.gray_symbol_map[bit_string])
            # Crear one-hot para compatibilidad con canal
            symbol_index = self.symbol_to_index[bit_string]
            one_hot_matrix[i, symbol_index] = 1
        
        return {
            'modulated_signal': one_hot_matrix,  # Formato estándar para el canal
            'iq_values': iq_values,              # Valores I,Q originales
            'padding': padding,
            'modulation_type': '16QAM',
            'constellation_size': self.constellation_size
        }
    
    def get_constellation_size(self):
        return self.constellation_size

class PSK8Modulator(BaseModulator):
    """Modulador 8PSK con codificación Gray"""
    
    def __init__(self):
        super().__init__("8PSK", 3)
        self.constellation_size = 8
        # Fases para 8PSK (en radianes)
        self.phases = np.array([i * 2 * np.pi / 8 for i in range(8)])
        # Mapeo Gray para 8PSK
        self.gray_map = [0, 1, 3, 2, 6, 7, 5, 4]  # Codificación Gray
    
    def modulate(self, bits):
        """Modula bits usando 8PSK con codificación Gray"""
        bits, padding = self.apply_padding(bits)
        
        # Convertir grupos de 3 bits en enteros
        symbols = []
        for i in range(0, len(bits), 3):
            bit_group = bits[i:i+3]
            decimal_value = int("".join(map(str, bit_group)), 2)
            symbols.append(decimal_value)
        
        # Aplicar codificación Gray
        gray_symbols = [self.gray_map[sym] for sym in symbols]
        
        # Convertir a coordenadas I,Q
        iq_values = []
        for symbol in gray_symbols:
            phase = self.phases[symbol]
            I = np.cos(phase)
            Q = np.sin(phase)
            iq_values.append((I, Q))
        
        # Crear matriz one-hot para compatibilidad con canal
        one_hot_matrix = np.zeros((len(gray_symbols), self.constellation_size))
        for i, symbol in enumerate(gray_symbols):
            one_hot_matrix[i, symbol] = 1
        
        return {
            'modulated_signal': one_hot_matrix,  # Formato estándar para el canal
            'iq_values': iq_values,              # Valores I,Q
            'gray_symbols': gray_symbols,
            'padding': padding,
            'modulation_type': '8PSK',
            'constellation_size': self.constellation_size
        }
    
    def get_constellation_size(self):
        return self.constellation_size

class ModulationSystem:
    """Sistema principal que maneja la selección y ejecución de moduladores"""
    
    def __init__(self):
        self.modulators = {
            '1': FSK8Modulator(),
            '2': QAM16Modulator(),
            '3': PSK8Modulator()
        }
    
    def show_menu(self):
        """Muestra el menú de opciones de modulación"""
        print("\n" + "="*50)
        print("SISTEMA DE MODULACIÓN DIGITAL")
        print("="*50)
        print("Seleccione el tipo de modulación:")
        print("1. 8FSK (Frequency Shift Keying)")
        print("2. 16QAM (Quadrature Amplitude Modulation)")
        print("3. 8PSK (Phase Shift Keying)")
        print("0. Salir")
        print("="*50)
    
    def get_user_choice(self):
        """Obtiene y valida la elección del usuario"""
        while True:
            choice = input("Ingrese su opción (0-3): ").strip()
            if choice == '0':
                return None
            elif choice in self.modulators:
                return choice
            else:
                print("Opción inválida. Por favor, ingrese un número entre 0 y 3.")
    
    def get_file_path(self):
        """Obtiene la ruta del archivo de entrada"""
        print("\nArchivos de entrada esperados:")
        print("- TPFinal/punto1/1-senal_codificada.txt (generado por 1-procesamientovoz.py)")
        print("- 1-senal_codificada.txt")
        print("- 1_bit_sequence.txt")
        
        while True:
            file_path = input("\nIngrese la ruta del archivo de bits (Enter para usar por defecto): ").strip()
            if file_path == "":
                file_path = r"1-senal_codificada.txt"
                print(f"Usando ruta por defecto: {file_path}")
            
            if os.path.exists(file_path):
                return file_path
            else:
                print("El archivo no existe. Por favor, verifique la ruta.")
                retry = input("¿Intentar otra ruta? (s/n): ").strip().lower()
                if retry not in ['s', 'si', 'sí', 'y', 'yes']:
                    return None
    
    def save_results(self, modulator, results, bits_per_character, output_dir=""):
        """Guarda los resultados en formato estándar para el canal"""
        
        # Archivo principal para el canal (siempre el mismo formato)
        channel_input_file = os.path.join(output_dir, "2_modulated_signal.txt")
        np.savetxt(channel_input_file, results['modulated_signal'], fmt='%d')
        
        # Archivo de metadatos para el sistema
        metadata_file = os.path.join(output_dir, "2_modulation_metadata.json")
        metadata = {
            'modulation_type': results['modulation_type'],
            'constellation_size': results['constellation_size'],
            'padding': results['padding'],
            'bits_per_character': bits_per_character,
            'total_symbols': len(results['modulated_signal'])
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Archivo específico de la modulación (para análisis)
        modulation_specific_file = os.path.join(output_dir, f"2_modulated_{modulator.name.lower()}_specific.txt")
        
        if modulator.name == "8FSK":
            # Guardar frecuencias específicas para FSK
            np.savetxt(modulation_specific_file, results['symbol_values'], fmt='%.0f')
        elif modulator.name in ["16QAM", "8PSK"]:
            # Guardar valores I,Q para QAM y PSK
            with open(modulation_specific_file, 'w') as f:
                for I, Q in results['iq_values']:
                    f.write(f"{I:.6f} {Q:.6f}\n")
        
        return channel_input_file, metadata_file, modulation_specific_file
    
    def display_results(self, modulator, results):
        """Muestra estadísticas de la modulación"""
        print(f"\n{'='*50}")
        print(f"RESULTADOS DE MODULACIÓN {modulator.name}")
        print(f"{'='*50}")
        print(f"Símbolos modulados: {len(results['modulated_signal'])}")
        print(f"Tamaño de constelación: {results['constellation_size']}")
        print(f"Padding aplicado: {results['padding']} bits")
        
        if modulator.name == "8FSK":
            print(f"Rango de frecuencias: {min(results['symbol_values']):.0f} - {max(results['symbol_values']):.0f} Hz")
        elif modulator.name in ["16QAM", "8PSK"]:
            print(f"Símbolos I/Q generados: {len(results['iq_values'])}")
            # Mostrar algunos ejemplos de símbolos
            print("Primeros 5 símbolos I/Q:")
            for i, (I, Q) in enumerate(results['iq_values'][:5]):
                print(f"  Símbolo {i+1}: I={I:.3f}, Q={Q:.3f}")
    
    def run(self):
        """Ejecuta el sistema principal"""
        print("="*60)
        print("SISTEMA DE MODULACIÓN DIGITAL INTEGRADO")
        print("Compatible con sistema de procesamiento de voz")
        print("="*60)
        
        while True:
            self.show_menu()
            choice = self.get_user_choice()
            
            if choice is None:
                print("¡Hasta luego!")
                break
            
            modulator = self.modulators[choice]
            print(f"\nHa seleccionado: {modulator.name}")
            
            # Obtener archivo de entrada
            file_path = self.get_file_path()
            if file_path is None:
                continue
            
            try:
                # Leer bits
                print("Leyendo bits desde archivo...")
                bits, bits_per_character = modulator.read_bits_from_file(file_path)
                print(f"Se leyeron {len(bits)} bits")
                
                # Modular
                print(f"Modulando con {modulator.name}...")
                results = modulator.modulate(bits)
                
                # Crear directorio de salida si no existe
                output_dir = os.getcwd()
                os.makedirs(output_dir, exist_ok=True)
                
                # Guardar resultados
                channel_file, metadata_file, specific_file = self.save_results(
                    modulator, results, bits_per_character
                )
                
                # Mostrar resultados
                self.display_results(modulator, results)
                
                print(f"\nARCHIVOS GENERADOS:")
                print(f"- Entrada al canal: {channel_file}")
                print(f"- Metadatos: {metadata_file}")
                print(f"- Específico {modulator.name}: {specific_file}")
                print(f"\nEl archivo '{channel_file}' es compatible con el canal universal.")
                
            except Exception as e:
                print(f"Error durante la modulación: {str(e)}")
                print("Verifique que el archivo contenga el formato correcto.")
            
            # Preguntar si continuar
            continue_choice = input("\n¿Desea realizar otra modulación? (s/n): ").strip().lower()
            if continue_choice not in ['s', 'si', 'sí', 'y', 'yes']:
                print("¡Hasta luego!")
                break

if __name__ == "__main__":
    system = ModulationSystem()
    system.run()