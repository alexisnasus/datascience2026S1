import pdfplumber
import os
from pathlib import Path

def extraer_texto_presentacion(pdf_path, output_path):
    """
    Extrae el texto de un archivo PDF y lo guarda en un archivo de texto plano.
    Ideal para pasarle contexto limpio a modelos de lenguaje o LLMs.
    """
    print(f"Procesando: {pdf_path}...")
    texto_completo = "# Extracción de Presentación PDF\n\n"
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, pagina in enumerate(pdf.pages):
                # Extraer texto de la página
                texto_pagina = pagina.extract_text()
                
                if texto_pagina:
                    texto_completo += f"## --- PÁGINA {i + 1} ---\n\n"
                    texto_completo += texto_pagina + "\n\n"
        
        # Guardar el resultado
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(texto_completo)
            
        print(f"✅ ¡Éxito! Texto extraído y guardado en: {output_path}")
        
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo en la ruta {pdf_path}")
    except Exception as e:
        print(f"❌ Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    # Ajusta estas rutas según la estructura de tu proyecto (con Paths relativos)
    # Suponiendo que ejecutas el script desde la raíz del proyecto
    ROOT_DIR = Path(__file__).resolve().parent.parent
    
    archivo_entrada = ROOT_DIR / "docs" / "presentacion data science.pdf"
    archivo_salida = ROOT_DIR / "docs" / "presentacion_extraida.md"
    
    extraer_texto_presentacion(archivo_entrada, archivo_salida)