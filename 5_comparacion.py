import numpy as np
import os
import json
from typing import Tuple, List, Optional

class VoiceComparisonSystem:
    """Sistema para comparar procesamiento de voz original con señal demodulada"""
    
    def __init__(self, base_path: str = r"C:/Users/elosc/Desktop/Universidad/STransmision"):
        self.base_path = base_path
        self.system_path = base_path
    
    def read_voice_processing_bits(self, file_path: str) -> Tuple[List[str], Optional[int]]:
        """
        Lee los bits del procesamiento de voz original.
        Maneja el formato de 1-senal_codificada.txt
        """
        bits = []
        bits_per_character = None
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if line.startswith("Bits por carácter:"):
                    bits_per_character = int(line.split(":")[1].strip())
                    continue
                # Cada línea es un código binario completo
                if line and not line.startswith("Bits por carácter:"):
                    bits.append(line)
            
            print(f"✓ Archivo de procesamiento de voz cargado: {len(bits)} códigos binarios")
            if bits_per_character:
                print(f"✓ Bits por carácter: {bits_per_character}")
            
            return bits, bits_per_character
            
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
        except Exception as e:
            raise Exception(f"Error leyendo archivo de procesamiento de voz: {str(e)}")
    
    def read_demodulated_bits(self, file_path: str) -> List[str]:
        """
        Lee los bits demodulados.
        Maneja el formato de 4_demodulated_bits.txt (un bit por línea)
        """
        try:
            bits = []
            with open(file_path, 'r') as f:
                for line in f:
                    bit = line.strip()
                    if bit in ['0', '1']:
                        bits.append(bit)
            
            print(f"✓ Archivo demodulado cargado: {len(bits)} bits individuales")
            return bits
            
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
        except Exception as e:
            raise Exception(f"Error leyendo archivo demodulado: {str(e)}")
    
    def load_demodulation_info(self) -> dict:
        """Carga información del proceso de demodulación"""
        info_path = os.path.join(self.system_path, "4_demodulation_info.json")
        try:
            with open(info_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("⚠ No se encontró información de demodulación")
            return {}
        except Exception as e:
            print(f"⚠ Error cargando información de demodulación: {str(e)}")
            return {}
    
    def reconstruct_voice_format(self, demodulated_bits: List[str], bits_per_character: int) -> List[str]:
        """
        Reconstruye el formato original del procesamiento de voz 
        agrupando bits individuales en códigos binarios
        """
        if not bits_per_character:
            raise ValueError("No se puede reconstruir sin conocer bits por carácter")
        
        reconstructed = []
        for i in range(0, len(demodulated_bits), bits_per_character):
            group = demodulated_bits[i:i+bits_per_character]
            if len(group) == bits_per_character:
                reconstructed.append(''.join(group))
            else:
                # Grupo incompleto (posible padding)
                print(f"⚠ Grupo incompleto al final: {len(group)} bits (esperados: {bits_per_character})")
                break
        
        print(f"✓ Reconstruidos {len(reconstructed)} códigos binarios desde bits individuales")
        return reconstructed
    
    def compare_bit_sequences(self, original_bits: List[str], demodulated_bits: List[str]) -> dict:
        """
        Compara dos secuencias de bits y calcula estadísticas de error
        """
        # Verificar longitudes
        min_length = min(len(original_bits), len(demodulated_bits))
        if len(original_bits) != len(demodulated_bits):
            print(f"⚠ Longitudes diferentes: Original={len(original_bits)}, Demodulado={len(demodulated_bits)}")
            print(f"Se compararán los primeros {min_length} elementos")
        
        # Comparar código por código
        errores_por_codigo = 0
        errores_por_bit = 0
        total_bits = 0
        codigos_erroneos = []
        
        for i in range(min_length):
            codigo_original = original_bits[i]
            codigo_demodulado = demodulated_bits[i]
            
            # Verificar si el código completo es diferente
            if codigo_original != codigo_demodulado:
                errores_por_codigo += 1
                codigos_erroneos.append({
                    'posicion': i,
                    'original': codigo_original,
                    'demodulado': codigo_demodulado
                })
            
            # Contar errores bit por bit
            total_bits += len(codigo_original)
            errores_por_bit += sum(1 for b1, b2 in zip(codigo_original, codigo_demodulado) if b1 != b2)
        
        # Calcular estadísticas
        porcentaje_error_codigo = (errores_por_codigo / min_length) * 100 if min_length > 0 else 0
        porcentaje_error_bit = (errores_por_bit / total_bits) * 100 if total_bits > 0 else 0
        relacion_error_bit = errores_por_bit / total_bits if total_bits > 0 else 0
        
        # Probabilidad de error de símbolo (para modulación con 3 bits por símbolo)
        bits_por_simbolo = 3  # Para 8FSK, 8PSK
        if total_bits > 0:
            Ps = 1 - (1 - relacion_error_bit) ** bits_por_simbolo
        else:
            Ps = 0
        
        return {
            'total_codigos': min_length,
            'total_bits': total_bits,
            'errores_por_codigo': errores_por_codigo,
            'errores_por_bit': errores_por_bit,
            'porcentaje_error_codigo': porcentaje_error_codigo,
            'porcentaje_error_bit': porcentaje_error_bit,
            'relacion_error_bit': relacion_error_bit,
            'probabilidad_error_simbolo': Ps,
            'porcentaje_error_simbolo': Ps * 100,
            'codigos_erroneos': codigos_erroneos[:10]  # Mostrar solo los primeros 10
        }
    
    def display_results(self, results: dict, demod_info: dict):
        """Muestra los resultados de la comparación"""
        print("\n" + "="*70)
        print("RESULTADOS DE COMPARACIÓN - PROCESAMIENTO DE VOZ")
        print("="*70)
        
        # Información general
        print(f"Total de códigos binarios: {results['total_codigos']}")
        print(f"Total de bits: {results['total_bits']}")
        
        # Información del sistema (si disponible)
        if demod_info:
            print(f"Tipo de modulación: {demod_info.get('modulation_type', 'N/A')}")
            print(f"Constelación: {demod_info.get('constellation_size', 'N/A')}")
            if 'snr_info' in demod_info:
                snr = demod_info['snr_info']
                print(f"Eb/N0: {snr.get('Eb_N0_dB', 'N/A'):.2f} dB")
                print(f"Es/N0: {snr.get('Es_N0_dB', 'N/A'):.2f} dB")
        
        print("\n" + "-"*70)
        print("ANÁLISIS DE ERRORES")
        print("-"*70)
        
        # Errores por código
        print(f"Códigos binarios erróneos: {results['errores_por_codigo']}")
        print(f"Porcentaje de códigos erróneos: {results['porcentaje_error_codigo']:.2f}%")
        
        # Errores por bit
        print(f"Bits erróneos: {results['errores_por_bit']}")
        print(f"Relación error/total bits: {results['relacion_error_bit']:.6f}")
        print(f"Porcentaje de error por bit: {results['porcentaje_error_bit']:.2f}%")
        
        # Probabilidad de error de símbolo
        print(f"Probabilidad de error de símbolo: {results['probabilidad_error_simbolo']:.6f}")
        print(f"Porcentaje de error de símbolo: {results['porcentaje_error_simbolo']:.2f}%")
        
        # Mostrar algunos errores específicos
        if results['codigos_erroneos']:
            print("\n" + "-"*70)
            print("PRIMEROS CÓDIGOS ERRÓNEOS")
            print("-"*70)
            print("Pos.\tOriginal\tDemodulado")
            for error in results['codigos_erroneos']:
                print(f"{error['posicion']}\t{error['original']}\t{error['demodulado']}")
    
    def save_comparison_report(self, results: dict, demod_info: dict):
        """Guarda un reporte detallado de la comparación"""
        report_path = os.path.join(self.base_path, "5_comparison_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("REPORTE DE COMPARACIÓN - PROCESAMIENTO DE VOZ\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Total de códigos binarios: {results['total_codigos']}\n")
            f.write(f"Total de bits: {results['total_bits']}\n")
            
            if demod_info:
                f.write(f"Tipo de modulación: {demod_info.get('modulation_type', 'N/A')}\n")
                f.write(f"Constelación: {demod_info.get('constellation_size', 'N/A')}\n")
                if 'snr_info' in demod_info:
                    snr = demod_info['snr_info']
                    f.write(f"Eb/N0: {snr.get('Eb_N0_dB', 'N/A'):.2f} dB\n")
                    f.write(f"Es/N0: {snr.get('Es_N0_dB', 'N/A'):.2f} dB\n")
            
            f.write("\nANÁLISIS DE ERRORES\n")
            f.write("-"*70 + "\n")
            f.write(f"Códigos binarios erróneos: {results['errores_por_codigo']}\n")
            f.write(f"Porcentaje de códigos erróneos: {results['porcentaje_error_codigo']:.2f}%\n")
            f.write(f"Bits erróneos: {results['errores_por_bit']}\n")
            f.write(f"Relación error/total bits: {results['relacion_error_bit']:.6f}\n")
            f.write(f"Porcentaje de error por bit: {results['porcentaje_error_bit']:.2f}%\n")
            f.write(f"Probabilidad de error de símbolo: {results['probabilidad_error_simbolo']:.6f}\n")
            f.write(f"Porcentaje de error de símbolo: {results['porcentaje_error_simbolo']:.2f}%\n")
            
            if results['codigos_erroneos']:
                f.write("\nCÓDIGOS ERRÓNEOS DETALLADOS\n")
                f.write("-"*70 + "\n")
                f.write("Posición\tOriginal\tDemodulado\n")
                for error in results['codigos_erroneos']:
                    f.write(f"{error['posicion']}\t{error['original']}\t{error['demodulado']}\n")
        
        print(f"\n✓ Reporte guardado en: {report_path}")
    
    def run_comparison(self):
        """Ejecuta la comparación completa"""
        print("="*70)
        print("SISTEMA DE COMPARACIÓN DE PROCESAMIENTO DE VOZ")
        print("="*70)
        
        # Archivos esperados
        voice_file = os.path.join(self.base_path, "1-senal_codificada.txt")
        demod_file = os.path.join(self.system_path, "4_demodulated_bits.txt")
        
        # Archivo alternativo si no existe el principal
        alt_voice_file = os.path.join(self.base_path, "1_bit_sequence.txt")
        
        try:
            # Intentar cargar archivo de procesamiento de voz
            if os.path.exists(voice_file):
                print(f"📂 Cargando procesamiento de voz desde: {voice_file}")
                original_bits, bits_per_character = self.read_voice_processing_bits(voice_file)
            elif os.path.exists(alt_voice_file):
                print(f"📂 Cargando secuencia de bits desde: {alt_voice_file}")
                # Leer como bits individuales
                with open(alt_voice_file, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                original_bits = lines
                bits_per_character = None
                print(f"✓ Archivo alternativo cargado: {len(original_bits)} elementos")
            else:
                raise FileNotFoundError("No se encontró archivo de procesamiento de voz")
            
            # Cargar bits demodulados
            print(f"📂 Cargando bits demodulados desde: {demod_file}")
            demodulated_bits = self.read_demodulated_bits(demod_file)
            
            # Cargar información de demodulación
            demod_info = self.load_demodulation_info()
            
            # Reconstruir formato si es necesario
            if bits_per_character and len(demodulated_bits) > 0:
                print(f"🔄 Reconstruyendo formato original ({bits_per_character} bits por carácter)...")
                reconstructed_bits = self.reconstruct_voice_format(demodulated_bits, bits_per_character)
            else:
                reconstructed_bits = demodulated_bits
                print("🔄 Comparando directamente (formato compatible)")
            
            # Realizar comparación
            print("⚖️ Realizando comparación...")
            results = self.compare_bit_sequences(original_bits, reconstructed_bits)
            
            # Mostrar resultados
            self.display_results(results, demod_info)
            
            # Guardar reporte
            self.save_comparison_report(results, demod_info)
            
            print(f"\n✅ Comparación completada exitosamente!")
            
        except Exception as e:
            print(f"❌ Error durante la comparación: {str(e)}")
            print("\nVerifique que existan los archivos:")
            print(f"- {voice_file}")
            print(f"- {alt_voice_file} (alternativo)")
            print(f"- {demod_file}")

if __name__ == "__main__":
    # Permitir configuración de ruta personalizada
    base_path = input("Ingrese la ruta base (Enter para usar por defecto): ").strip()
    if not base_path:
        base_path = r"C:/Users/elosc/Desktop/Universidad/STransmision"
    
    system = VoiceComparisonSystem(base_path)
    system.run_comparison()