import pdfplumber
import os
from pathlib import Path

def extraer_texto_presentacion(pdf_path, output_path):
    """Extrae el texto de cada slide del PDF y lo guarda como Markdown.

    Numera TODAS las slides (incluso las puramente visuales, sin texto
    extraible) para que la numeracion del .md coincida exactamente con la
    del PDF. Esto es clave para armar el guion y referenciar "slide 9-13"
    sin desajustes.
    """
    print(f"Procesando: {pdf_path}...")
    partes = ["# Extracción de Presentación PDF\n"]

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            for i, pagina in enumerate(pdf.pages, start=1):
                texto_pagina = (pagina.extract_text() or "").strip()
                partes.append(f"## --- SLIDE {i} / {total} ---\n")
                if texto_pagina:
                    partes.append(texto_pagina + "\n")
                else:
                    partes.append("_(sin texto extraíble — slide visual)_\n")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(partes) + "\n")

        print(f"✅ ¡Éxito! Texto extraído y guardado en: {output_path}")

    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo en la ruta {pdf_path}")
    except Exception as e:
        print(f"❌ Ocurrió un error inesperado: {e}")

if __name__ == "__main__":
    # Ajusta estas rutas según la estructura de tu proyecto (con Paths relativos)
    # Suponiendo que ejecutas el script desde la raíz del proyecto
    ROOT_DIR = Path(__file__).resolve().parent.parent
    
    archivo_entrada = ROOT_DIR / "docs" / "presentacion.pdf"
    archivo_salida = ROOT_DIR / "docs" / "presentacion.md"
    
    extraer_texto_presentacion(archivo_entrada, archivo_salida)