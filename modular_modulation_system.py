import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import os

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
    def get_output_format(self):
        """Retorna el formato de salida esperado"""
        pass
    
    def read_bits_from_file(self, file_path):
        """Lee bits desde archivo manteniendo formato original"""
        bits = []
        bits_per_character = None
        
        # Detectar formato de archivo
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Formato 1: Como tu 1-senal_codificada.txt (códigos binarios por línea)
        if any(line.strip().startswith("Bits por carácter:") for line in lines):
            for line in lines:
                line = line.strip()
                if line.startswith("Bits por carácter:"):
                    bits_per_character = int(line.split(":")[1].strip())
                    continue
                # Cada línea es un código binario completo
                if line and not line.startswith("Bits por carácter:"):
                    bits.extend(list(map(int, line)))
        
        # Formato 2: Como tu 1_bit_sequence.txt (un bit por línea)
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
        
        # Guardar también información adicional como en tu implementación original
        additional_info = {
            'symbols': symbols,
            'decimal_symbols': symbols,
            'gray_symbols': gray_symbols,
            'frequencies': modulated_values
        }
        
        return {
            'modulated_values': modulated_values,
            'padding': padding,
            'gray_symbols': gray_symbols,
            'additional_info': additional_info
        }
    
    def get_output_format(self):
        return "frequency_values"

class QAM16Modulator(BaseModulator):
    """Modulador 16QAM con codificación Gray"""
    
    def __init__(self):
        super().__init__("16QAM", 4)
        # Mapeo Gray para 16-QAM (manteniendo tu implementación exacta)
        self.gray_symbol_map = {
            '0100': (-3, +3), '0110': (-1, +3), '1110': (+1, +3), '1100': (+3, +3),
            '0101': (-3, +1), '0111': (-1, +1), '1111': (+1, +1), '1101': (+3, +1),
            '0001': (-3, -1), '0011': (-1, -1), '1011': (+1, -1), '1001': (+3, -1),
            '0000': (-3, -3), '0010': (-1, -3), '1010': (+1, -3), '1000': (+3, -3),
        }
    
    def modulate(self, bits):
        """Modula bits usando 16QAM con codificación Gray"""
        bits, padding = self.apply_padding(bits)
        
        # Dividir bits en grupos de 4
        symbols = [bits[i:i+4] for i in range(0, len(bits), 4)]
        
        # Convertir a amplitudes I,Q con mapeo Gray
        iq_values = []
        for symbol in symbols:
            bit_string = ''.join(map(str, symbol))
            iq_values.append(self.gray_symbol_map[bit_string])
        
        return {
            'iq_values': iq_values,
            'padding': padding
        }
    
    def get_output_format(self):
        return "iq_values"

class ModulationSystem:
    """Sistema principal que maneja la selección y ejecución de moduladores"""
    
    def __init__(self):
        self.modulators = {
            '1': FSK8Modulator(),
            '2': QAM16Modulator()
        }
    
    def show_menu(self):
        """Muestra el menú de opciones de modulación"""
        print("\n" + "="*50)
        print("SISTEMA DE MODULACIÓN DIGITAL")
        print("="*50)
        print("Seleccione el tipo de modulación:")
        print("1. 8FSK (Frequency Shift Keying)")
        print("2. 16QAM (Quadrature Amplitude Modulation)")
        print("0. Salir")
        print("="*50)
    
    def get_user_choice(self):
        """Obtiene y valida la elección del usuario"""
        while True:
            choice = input("Ingrese su opción (0-2): ").strip()
            if choice == '0':
                return None
            elif choice in self.modulators:
                return choice
            else:
                print("Opción inválida. Por favor, ingrese un número entre 0 y 2.")
    
    def get_file_path(self):
        """Obtiene la ruta del archivo de entrada"""
        print("\nArchivos comunes esperados:")
        print("- TPFinal/punto1/1-senal_codificada.txt (generado por 1-procesamientovoz.py)")
        print("- 1-senal_codificada.txt")
        print("- 1_bit_sequence.txt")
        
        while True:
            file_path = input("\nIngrese la ruta del archivo de bits: ").strip()
            if file_path == "":
                file_path = "TPFinal/punto1/1-senal_codificada.txt"  # Ruta por defecto
                print(f"Usando ruta por defecto: {file_path}")
            
            if os.path.exists(file_path):
                return file_path
            else:
                print("El archivo no existe. Por favor, verifique la ruta.")
                retry = input("¿Intentar otra ruta? (s/n): ").strip().lower()
                if retry not in ['s', 'si', 'sí', 'y', 'yes']:
                    return None
    
    def save_results(self, modulator, results, bits_per_character, output_file):
        """Guarda los resultados según el tipo de modulación"""
        with open(output_file, 'w') as file:
            if modulator.get_output_format() == "frequency_values":
                # Formato para 8FSK - compatible con tu 3_Correlacion.py
                for freq in results['modulated_values']:
                    file.write(f"{freq:.0f}\n")  # Sin decimales como en tu formato original
                
                # Archivo adicional con información extra
                info_file = output_file.replace('.txt', '_info.txt')
                with open(info_file, 'w') as info:
                    info.write(f"Padding: {results['padding']}\n")
                    info.write(f"Bits por carácter: {bits_per_character}\n")
                    info.write(f"Símbolos Gray: {results['gray_symbols']}\n")
                
            elif modulator.get_output_format() == "iq_values":
                # Formato para 16QAM
                for (I, Q) in results['iq_values']:
                    file.write(f"{I} {Q}\n")
                file.write(f"Padding: {results['padding']}\n")
                file.write(f"Bits por carácter: {bits_per_character}\n")
    
    def run(self):
        """Ejecuta el sistema principal"""
        print("Bienvenido al Sistema de Modulación Digital")
        print("Compatible con archivos de FuenteInformacion.py")
        
        while True:
            self.show_menu()
            choice = self.get_user_choice()
            
            if choice is None:
                print("¡Hasta luego!")
                break
            
            modulator = self.modulators[choice]
            print(f"\nHa seleccionado: {modulator.name}")
            
            # Sugerir archivos de entrada comunes
            if modulator.name == "8FSK":
                print("Archivos de entrada sugeridos:")
                print("- 1-senal_codificada.txt (código binario por línea)")
                print("- 1_bit_sequence.txt (un bit por línea)")
            
            # Obtener archivo de entrada
            file_path = self.get_file_path()
            
            try:
                # Leer bits
                print("Leyendo bits desde archivo...")
                bits, bits_per_character = modulator.read_bits_from_file(file_path)
                print(f"Se leyeron {len(bits)} bits")
                
                # Modular
                print(f"Modulando con {modulator.name}...")
                results = modulator.modulate(bits)
                
                # Guardar resultados
                if modulator.name == "8FSK":
                    output_file = "2_modulated_values.txt"  # Compatible con tu 3_Correlacion.py
                else:
                    output_file = f"2_senal_modulada_{modulator.name.lower()}.txt"
                
                self.save_results(modulator, results, bits_per_character, output_file)
                
                print(f"Modulación completada exitosamente!")
                print(f"Resultados guardados en: {output_file}")
                if modulator.name == "8FSK":
                    print(f"Información adicional en: {output_file.replace('.txt', '_info.txt')}")
                    print("Archivo compatible con 3_Correlacion.py")
                
                # Mostrar estadísticas
                if modulator.get_output_format() == "frequency_values":
                    print(f"Símbolos modulados: {len(results['modulated_values'])}")
                    print(f"Símbolos Gray: {len(results['gray_symbols'])}")
                    print(f"Rango de frecuencias: {min(results['modulated_values']):.0f} - {max(results['modulated_values']):.0f} Hz")
                else:
                    print(f"Símbolos I/Q generados: {len(results['iq_values'])}")
                
                print(f"Padding aplicado: {results['padding']} bits")
                
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