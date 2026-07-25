"""
Banco de pruebas contra avisos reales (Chileautos u otra fuente).

Uso:
  1. Llena app/validacion.csv con avisos reales que mires en el sitio.
  2. python app/validar.py

Reporta, por auto y en conjunto, si el precio publicado cae dentro del rango
estimado y cuanto se desvia del valor central. Sirve para validar el modelo con
datos que NO vienen del entrenamiento.

Nota: los avisos se copian a mano mirando el sitio como cualquier usuario.
No hay scraping automatizado de fuentes que no lo permitan.
"""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

APP = Path(__file__).resolve().parent
CSV = APP / 'validacion.csv'

with open(APP / 'modelo.pkl', 'rb') as f:
    M = pickle.load(f)

if not CSV.exists():
    print(f'Falta {CSV}. Crealo con las columnas:')
    print('  marca,modelo,ano,kilometraje,combustible,transmision,precio_publicado')
    sys.exit(1)

df = pd.read_csv(CSV)
req = ['marca','modelo','ano','kilometraje','combustible','transmision','precio_publicado']
faltan = [c for c in req if c not in df.columns]
if faltan:
    print('Faltan columnas en el CSV:', faltan); sys.exit(1)

X = pd.DataFrame({
    'antiguedad': 2026 - df['ano'].astype(int),
    'Kilometraje': df['kilometraje'].astype(float),
    'Marca': df['marca'].astype(str).str.strip().str.title(),
    'Modelo': df['modelo'].astype(str).str.strip().str.title(),
    'Combustible': df['combustible'].astype(str).str.strip().str.title(),
    'Transmision': df['transmision'].astype(str).str.strip().str.title(),
})[M['FEATS']]
for c in M['CATS']:
    X[c] = X[c].astype(M['cat_dtypes'][c])

# Avisa si algun valor no existe en el vocabulario del modelo (quedaria como NaN)
for c in M['CATS']:
    desconocidos = df.index[X[c].isna()].tolist()
    if desconocidos:
        vals = [str(df.loc[i, c.lower()]) for i in desconocidos]
        print(f'AVISO: valores de {c} no vistos en entrenamiento -> {set(vals)}')

p50 = np.expm1(M['mmid'].predict(X))
lo  = np.expm1(M['mlo'].predict(X) - M['qhat'])
hi  = np.expm1(M['mhi'].predict(X) + M['qhat'])
real = df['precio_publicado'].astype(float).values

err = (p50 - real) / real * 100
dentro = (real >= lo) & (real <= hi)

def M_(v): return f'${v/1e6:.1f}M'
print(f'\n{"AUTO":34s} {"PUBLICADO":>10s} {"ESTIMADO":>10s} {"RANGO":>18s} {"ERROR":>8s}  EN RANGO')
print('-'*98)
for i in range(len(df)):
    nombre = f"{df.loc[i,'marca']} {df.loc[i,'modelo']} {int(df.loc[i,'ano'])}"
    print(f'{nombre:34s} {M_(real[i]):>10s} {M_(p50[i]):>10s} '
          f'{M_(lo[i])+" - "+M_(hi[i]):>18s} {err[i]:+7.1f}%  {"si" if dentro[i] else "NO"}')

ape = np.abs(err)
print('-'*98)
print(f'\nRESUMEN sobre {len(df)} autos reales:')
print(f'  Error mediano (MdAPE)      : {np.median(ape):.1f}%')
print(f'  Dentro de ±10%             : {(ape<=10).mean()*100:.0f}%')
print(f'  Precio real dentro del rango: {dentro.mean()*100:.0f}%   (esperado ~80%)')
print(f'  Sesgo medio                : {err.mean():+.1f}%  '
      f'({"sobreestima" if err.mean()>0 else "subestima"} en promedio)')
if len(df) < 25:
    print(f'\n  OJO: con {len(df)} autos el resultado es indicativo. Apunta a 30-50 para concluir.')
