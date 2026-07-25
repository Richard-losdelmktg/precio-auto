"""Descarga el catalogo oficial de Tasacion Fiscal de Vehiculos del SII.
Fuente publica: https://www.sii.cl/servicios_online/1049-2612.html
Uso: python scripts/descargar_sii.py [anio]
"""
import sys, urllib.request
from pathlib import Path
ANIO = sys.argv[1] if len(sys.argv) > 1 else '2026'
URL = f'https://www.sii.cl/servicios_online/tasacion_fiscal_vehiculos/liv{ANIO}.xlsx'
OUT = Path(__file__).resolve().parent.parent / 'data' / f'sii_liv{ANIO}.xlsx'
print(f'Descargando {URL} ...')
req = urllib.request.Request(URL, headers={'User-Agent':'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=300) as r, open(OUT,'wb') as f:
    f.write(r.read())
print(f'Guardado en {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)')
