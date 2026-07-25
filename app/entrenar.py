"""
Entrena el modelo de pricing y guarda app/modelo.pkl para la app local.

Pipeline v3:
  - Sin filtro IQR sobre el precio (ese filtro borraba todo el mercado sobre $24M
    y dejaba al modelo incapaz de tasar vehiculos premium).
  - Catalogo oficial del SII (Tasacion Fiscal de Vehiculos) como referencia de
    valor por version. Aporta sobre todo en premium: 16.3% -> 13.0% de error.
  - Rango calibrado (conformal) + ancho adaptativo segun dispersion de versiones.
  - Seleccion de version en la interfaz: si el tasador elige la version exacta,
    se usa la tasacion de ESA version en vez de la mediana del modelo-año.

Ejecutar: python app/entrenar.py
"""
import warnings; warnings.filterwarnings('ignore')
import json, pickle, re, unicodedata
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgb

BASE = Path(__file__).resolve().parent.parent / 'data'
OUT  = Path(__file__).resolve().parent / 'modelo.pkl'
SII_XLSX = BASE / 'sii_liv2026.xlsx'

# ---------------------------------------------------------------- utilidades
def norm(s):
    s = ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', s.lower())).strip()

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
    if any(k in v for k in ['auto','cvt','tiptronic','dsg']): return 'Automatica'
    return np.nan

MARCA_FIX = {'Mercedes':'Mercedes-Benz', 'Mercedes Benz':'Mercedes-Benz',
             'Citroën':'Citroen', 'Range Rover':'Land Rover', 'Vw':'Volkswagen'}

def clean(df):
    df = df.copy()
    for c in ['Ano','Kilometraje','price']: df[c] = pd.to_numeric(df[c], errors='coerce')
    df['Combustible'] = df['Combustible'].apply(limpiar_combustible)
    df['Transmision'] = df['Transmision'].apply(limpiar_transmision)
    df = df.dropna(subset=['Marca','Modelo','Ano','Kilometraje','price','Combustible','Transmision'])
    q1, q3 = df['Kilometraje'].quantile([.25,.75]); iqr = q3-q1
    df = df[(df['Kilometraje'] >= q1-1.5*iqr) & (df['Kilometraje'] <= q3+1.5*iqr)]
    df = df[(df['price'] > 500_000) & (df['price'] < 150_000_000)]
    df = df[(df['Ano'] >= 1990) & (df['Ano'] <= 2026) & (df['Kilometraje'] > 0)]
    df['antiguedad'] = 2026 - df['Ano'].astype(int)
    df['Marca'] = df['Marca'].str.strip().str.title().replace(MARCA_FIX)
    df['Modelo'] = df['Modelo'].str.strip().str.title()
    df['mk'] = df['Marca'].map(norm)
    df['md'] = df['Modelo'].map(norm).str.split().str[0]
    df['yr'] = df['Ano'].astype(int)
    df['km_ano'] = df['Kilometraje']/(df['antiguedad']+1)
    return df[['Marca','Modelo','mk','md','yr','antiguedad','Kilometraje','km_ano',
               'Combustible','Transmision','price']]

# ------------------------------------------------- catalogo oficial del SII
print('Cargando catálogo SII...')
sii = pd.read_excel(SII_XLSX, sheet_name='LIV2026', header=11)
sii.columns = [str(c).strip() for c in sii.columns]
def _col(pat):
    for c in sii.columns:
        if re.search(pat, ''.join(ch for ch in unicodedata.normalize('NFD',c)
                                  if unicodedata.category(ch)!='Mn').lower()): return c
CC = {k:_col(p) for k,p in {'ano':r'^ano','marca':r'marca','modelo':r'^modelo','version':r'version',
      'cc':r'cilindrada','hp':r'potencia','tas':r'tasacion','trac':r'traccion'}.items()}
sii = sii.dropna(subset=[CC['marca'],CC['modelo'],CC['ano'],CC['tas']])
sii[CC['ano']] = pd.to_numeric(sii[CC['ano']], errors='coerce')
sii[CC['tas']] = pd.to_numeric(sii[CC['tas']], errors='coerce')
sii = sii.dropna(subset=[CC['ano'],CC['tas']])
sii['mk'] = sii[CC['marca']].map(norm)
sii['md'] = sii[CC['modelo']].map(norm).str.split().str[0]
sii['yr'] = sii[CC['ano']].astype(int)
print(f'  {len(sii):,} tasaciones | {sii[CC["version"]].nunique():,} versiones distintas')

# Referencia por modelo-año (para entrenar: no sabemos la version de cada aviso)
ref = sii.groupby(['mk','md','yr']).agg(
    sii_tas=(CC['tas'],'median'), sii_min=(CC['tas'],'min'), sii_max=(CC['tas'],'max'),
    sii_nver=(CC['version'],'nunique'), sii_cc=(CC['cc'],'median'), sii_hp=(CC['hp'],'median')
).reset_index()
ref['sii_spread'] = ((ref['sii_max']-ref['sii_min'])/ref['sii_tas']).clip(0,3)
ref_md = sii.groupby(['mk','md'])[CC['tas']].median().rename('sii_tas_md').reset_index()

# Catalogo de versiones para el desplegable de la interfaz
cat_ver = (sii[['mk','md','yr',CC['version'],CC['tas'],CC['cc'],CC['hp']]]
           .rename(columns={CC['version']:'version', CC['tas']:'tas',
                            CC['cc']:'cc', CC['hp']:'hp'}))
cat_ver = cat_ver.dropna(subset=['version']).drop_duplicates(subset=['mk','md','yr','version'])

# --------------------------------------------------------------- datos avisos
print('Cargando avisos...')
orig = clean(pd.read_csv(BASE/'datos_combinados_entrega2.csv'))
yapo = clean(pd.DataFrame(json.loads((BASE/'datos_scraped_yapo.json').read_text(encoding='utf-8'))))
full = pd.concat([orig, yapo], ignore_index=True).drop_duplicates(
    subset=['mk','md','yr','Kilometraje','price']).reset_index(drop=True)

full = full.merge(ref, on=['mk','md','yr'], how='left').merge(ref_md, on=['mk','md'], how='left')
full['sii_any']   = full['sii_tas'].fillna(full['sii_tas_md'])
full['tiene_sii'] = full['sii_tas'].notna().astype(int)
full['log_sii']   = np.log1p(full['sii_any'])
for c in ['sii_cc','sii_hp','sii_nver','sii_spread']:
    full[c] = full[c].fillna(full[c].median())
print(f'  {len(full):,} avisos | con referencia SII: {full["tiene_sii"].mean()*100:.0f}%')

# ------------------------------------------------------------------- modelo
CATS  = ['Marca','Modelo','Combustible','Transmision']
FEATS = ['antiguedad','Kilometraje','km_ano','log_sii','tiene_sii',
         'sii_cc','sii_hp','sii_nver','sii_spread'] + CATS
cat_dtypes = {c: pd.api.types.CategoricalDtype(full[c].astype('category').cat.categories) for c in CATS}
def prep(X):
    X = X[FEATS].copy()
    for c in CATS: X[c] = X[c].astype(cat_dtypes[c])
    return X

y = np.log1p(full['price'])
bins = pd.qcut(full['price'], 5, labels=False, duplicates='drop')
Xtr, Xte, ytr, yte = train_test_split(full, y, test_size=.2, random_state=42, stratify=bins)
yte_o = np.expm1(yte)

def make(a):
    m = lgb.LGBMRegressor(objective='quantile', alpha=a, n_estimators=800, learning_rate=.05,
                          num_leaves=63, min_child_samples=20, random_state=42, verbosity=-1)
    return m.fit(prep(Xtr), ytr, categorical_feature=CATS)

print('Entrenando P10 / P50 / P90...')
mlo, mmid, mhi = make(.1), make(.5), make(.9)

# calibracion conformal del rango
Xa, Xc, ya, yc = train_test_split(Xtr, ytr, test_size=.2, random_state=42)
def make2(a):
    m = lgb.LGBMRegressor(objective='quantile', alpha=a, n_estimators=800, learning_rate=.05,
                          num_leaves=63, min_child_samples=20, random_state=42, verbosity=-1)
    return m.fit(prep(Xa), ya, categorical_feature=CATS)
clo, chi = make2(.1), make2(.9)
s = np.maximum(clo.predict(prep(Xc)) - yc.values, yc.values - chi.predict(prep(Xc)))
qhat = float(np.quantile(s, .8*(1+1/len(s))))

p50 = np.expm1(mmid.predict(prep(Xte)))
lo  = np.expm1(mlo.predict(prep(Xte)) - qhat)
hi  = np.expm1(mhi.predict(prep(Xte)) + qhat)
ape = np.abs(yte_o - p50)/yte_o*100
mdape = float(np.median(ape)); cov = float(((yte_o>=lo)&(yte_o<=hi)).mean()*100)
prem = yte_o.values >= 25_000_000
print(f'  MdAPE={mdape:.1f}% | R2={r2_score(yte, mmid.predict(prep(Xte))):.3f} | cobertura={cov:.1f}%')
print(f'  premium >=$25M: MdAPE={np.median(ape[prem]):.1f}% (n={prem.sum()})')

# confianza: cuantos avisos comparables hay de ese modelo-año
dens = full.groupby(['mk','md','yr']).size().rename('n_comp')
dens_md = full.groupby(['mk','md']).size().rename('n_comp_md')

# ------------------------------------------------------------------ opciones UI
marcas = sorted(full['Marca'].dropna().unique().tolist())
opciones = {
    'marcas': marcas,
    'modelos_por_marca': {m: sorted(full[full['Marca']==m]['Modelo'].dropna().unique().tolist())
                          for m in marcas},
    'combustibles': sorted(full['Combustible'].dropna().unique().tolist()),
    'transmisiones': sorted(full['Transmision'].dropna().unique().tolist()),
    'km_mediano': int(full['Kilometraje'].median()),
}

with open(OUT, 'wb') as f:
    pickle.dump({'mlo':mlo,'mmid':mmid,'mhi':mhi,'qhat':qhat,'cat_dtypes':cat_dtypes,
                 'FEATS':FEATS,'CATS':CATS,'opciones':opciones,
                 'ref':ref,'ref_md':ref_md,'cat_ver':cat_ver,
                 'dens':dens,'dens_md':dens_md,
                 'metricas':{'mdape':round(mdape,1),'cobertura':round(cov,1),
                             'n':len(full),'premium':round(float(np.median(ape[prem])),1)}}, f)
print(f'\nModelo guardado en {OUT}')
