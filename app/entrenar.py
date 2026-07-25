"""
Entrena el modelo final de pricing (LightGBM quantile P10/P50/P90 + calibracion
conformal) sobre el dataset combinado y lo guarda en app/modelo.pkl para que la
app local (app/servidor.py) lo sirva. Ejecutar una vez: python app/entrenar.py
"""
import warnings; warnings.filterwarnings('ignore')
import json, pickle
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgb

BASE = Path(__file__).resolve().parent.parent / 'data'
OUT  = Path(__file__).resolve().parent / 'modelo.pkl'

def limpiar_combustible(v):
    if pd.isna(v): return np.nan
    v = str(v).lower()
    if 'diesel' in v or 'petroleo' in v: return 'Diesel'
    if any(k in v for k in ['bencina','gasolina','gasoline','petrol']): return 'Bencina'
    if 'hibrido' in v or 'hybrid' in v: return 'Hibrido'
    if 'electrico' in v or 'electric' in v: return 'Electrico'
    if 'gas' in v: return 'Gas'
    return 'Otro'

def limpiar_transmision(v):
    if pd.isna(v): return np.nan
    v = str(v).lower().strip()
    if v in ['m','manual','mecanica','mechanical']: return 'Manual'
    if 'auto' in v or 'cvt' in v or 'tiptronic' in v: return 'Automatica'
    return np.nan

def clean(df):
    df = df.copy()
    for c in ['Ano','Kilometraje','price']: df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Combustible'] = df['Combustible'].apply(limpiar_combustible)
    df['Transmision'] = df['Transmision'].apply(limpiar_transmision)
    df = df.dropna(subset=['Marca','Modelo','Ano','Kilometraje','price','Transmision','Combustible'])
    for c in ['price','Kilometraje']:
        q1,q3 = df[c].quantile(.25), df[c].quantile(.75); iqr=q3-q1
        df = df[(df[c]>=q1-1.5*iqr)&(df[c]<=q3+1.5*iqr)]
    df = df[(df['price']>500000)&(df['Ano']>=1990)&(df['Ano']<=2026)&(df['Kilometraje']>0)]
    df['antiguedad'] = 2026 - df['Ano'].astype(int)
    df['Marca'] = df['Marca'].str.strip().str.title()
    df['Modelo'] = df['Modelo'].str.strip().str.title()
    return df[['Marca','Modelo','antiguedad','Kilometraje','Combustible','Transmision','price']]

print('Cargando datos...')
orig = clean(pd.read_csv(BASE/'datos_combinados_entrega2.csv'))
yapo = clean(pd.DataFrame(json.loads((BASE/'datos_scraped_yapo.json').read_text(encoding='utf-8'))))
full = pd.concat([orig, yapo], ignore_index=True).drop_duplicates(
    subset=['Marca','Modelo','antiguedad','Kilometraje','price']).reset_index(drop=True)
print(f'Original: {len(orig):,} | Yapo: {len(yapo):,} | Combinado: {len(full):,}')

CATS = ['Marca','Modelo','Combustible','Transmision']
FEATS = ['antiguedad','Kilometraje'] + CATS
cat_dtypes = {c: pd.api.types.CategoricalDtype(full[c].astype('category').cat.categories) for c in CATS}
def prep(X):
    X = X[FEATS].copy()
    for c in CATS: X[c] = X[c].astype(cat_dtypes[c])
    return X

y = np.log1p(full['price'])
bins = pd.qcut(full['price'], 5, labels=False, duplicates='drop')
Xtr, Xte, ytr, yte = train_test_split(full, y, test_size=.2, random_state=42, stratify=bins)
yte_o = np.expm1(yte)

def make(alpha):
    m = lgb.LGBMRegressor(objective='quantile', alpha=alpha, n_estimators=800,
                          learning_rate=.05, num_leaves=63, min_child_samples=20,
                          random_state=42, verbosity=-1)
    m.fit(prep(Xtr), ytr, categorical_feature=CATS)
    return m

print('Entrenando P10 / P50 / P90...')
mlo, mmid, mhi = make(.1), make(.5), make(.9)

# Calibracion conformal para que el rango P10-P90 cubra ~80% real
Xc_tr, Xc_cal, yc_tr, yc_cal = train_test_split(Xtr, ytr, test_size=.2, random_state=42)
clo = lgb.LGBMRegressor(objective='quantile', alpha=.1, n_estimators=800, learning_rate=.05,
                        num_leaves=63, min_child_samples=20, random_state=42, verbosity=-1).fit(prep(Xc_tr), yc_tr, categorical_feature=CATS)
chi = lgb.LGBMRegressor(objective='quantile', alpha=.9, n_estimators=800, learning_rate=.05,
                        num_leaves=63, min_child_samples=20, random_state=42, verbosity=-1).fit(prep(Xc_tr), yc_tr, categorical_feature=CATS)
lo_c, hi_c = clo.predict(prep(Xc_cal)), chi.predict(prep(Xc_cal))
scores = np.maximum(lo_c - yc_cal.values, yc_cal.values - hi_c)
qhat = float(np.quantile(scores, .8 * (1 + 1/len(scores))))

# Metricas en test
p50 = np.expm1(mmid.predict(prep(Xte)))
lo  = np.expm1(mlo.predict(prep(Xte)) - qhat)
hi  = np.expm1(mhi.predict(prep(Xte)) + qhat)
mdape = float(np.median(np.abs(yte_o - p50)/yte_o*100))
cov   = float(((yte_o.values>=lo)&(yte_o.values<=hi)).mean()*100)
print(f'  MdAPE (P50): {mdape:.1f}% | R2: {r2_score(yte,mmid.predict(prep(Xte))):.3f} | Cobertura rango: {cov:.1f}%')

# Opciones para los desplegables de la interfaz (marca -> modelos, valores validos)
marcas = sorted(full['Marca'].dropna().unique().tolist())
modelos_por_marca = {m: sorted(full[full['Marca']==m]['Modelo'].dropna().unique().tolist()) for m in marcas}
opciones = {
    'marcas': marcas,
    'modelos_por_marca': modelos_por_marca,
    'combustibles': sorted(full['Combustible'].dropna().unique().tolist()),
    'transmisiones': sorted(full['Transmision'].dropna().unique().tolist()),
    'km_mediano': int(full['Kilometraje'].median()),
}

with open(OUT, 'wb') as f:
    pickle.dump({'mlo':mlo,'mmid':mmid,'mhi':mhi,'qhat':qhat,'cat_dtypes':cat_dtypes,
                 'FEATS':FEATS,'CATS':CATS,'opciones':opciones,
                 'metricas':{'mdape':round(mdape,1),'cobertura':round(cov,1),'n':len(full)}}, f)
print(f'\nModelo guardado en {OUT}')
