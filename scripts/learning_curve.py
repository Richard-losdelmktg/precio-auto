# Curva de aprendizaje: MdAPE vs tamaño de train, para estimar cuantos datos hacen falta.
# Test set SIEMPRE el mismo (congelado), solo varia cuanto train se usa.
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import lightgbm as lgb

BASE = Path(__file__).resolve().parent.parent / 'data'

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
    df['Transmision'] = df['Transmision'].fillna(df['Transmision'].mode()[0])
    df['Combustible'] = df['Combustible'].fillna(df['Combustible'].mode()[0])
    df = df.dropna(subset=['Marca','Modelo','Ano','Kilometraje','price'])
    for c in ['price','Kilometraje']:
        q1,q3 = df[c].quantile(.25), df[c].quantile(.75); iqr=q3-q1
        df = df[(df[c]>=q1-1.5*iqr)&(df[c]<=q3+1.5*iqr)]
    df = df[(df['price']>500000)&(df['Ano']>=1990)&(df['Ano']<=2026)&(df['Kilometraje']>0)]
    df['antiguedad'] = 2026 - df['Ano']
    df['Marca'] = df['Marca'].str.strip().str.title()
    df['Modelo'] = df['Modelo'].str.strip().str.title()
    return df.reset_index(drop=True)

df0 = clean(pd.read_csv(BASE/'datos_combinados_entrega2.csv'))
df0 = df0.drop_duplicates(subset=['Marca','Modelo','Ano','Kilometraje','price']).reset_index(drop=True)

dfy = pd.DataFrame(pd.read_json(BASE/'datos_scraped_yapo.json'))
dfy = clean(dfy)
dfy = dfy.drop_duplicates(subset=['Marca','Modelo','Ano','Kilometraje','price']).reset_index(drop=True)

full = pd.concat([df0, dfy], ignore_index=True).drop_duplicates(
    subset=['Marca','Modelo','Ano','Kilometraje','price']).reset_index(drop=True)
print(f'Original limpio: {len(df0):,} | Yapo limpio: {len(dfy):,} | Combinado total: {len(full):,}')

CATS = ['Marca','Modelo','Combustible','Transmision']
FEATS = ['antiguedad','Kilometraje'] + CATS
y_all = np.log1p(full['price'])
bins = pd.qcut(full['price'], 5, labels=False, duplicates='drop')

# Test set congelado: 20% del dataset combinado, semilla 42 (misma metodologia que MEJORAS.md)
Xtr_pool, Xte, ytr_pool, yte = train_test_split(full, y_all, test_size=.2, random_state=42, stratify=bins)
yte_o = np.expm1(yte)
cat_maps = {c: pd.api.types.CategoricalDtype(Xtr_pool[c].astype('category').cat.categories) for c in CATS}
def prep(X):
    X = X[FEATS].copy()
    for c in CATS: X[c] = X[c].astype(cat_maps[c])
    return X
Pte = prep(Xte)

def mdape(y, p): return np.median(np.abs(y-p)/y*100)

print(f'\nTrain pool disponible: {len(Xtr_pool):,} | Test congelado: {len(Xte):,}')
print('='*70)
sizes = [5000, 10000, 15000, 20000, 25000, len(Xtr_pool)]
sizes = sorted(set(s for s in sizes if s <= len(Xtr_pool)))
for n in sizes:
    Xs, ys = Xtr_pool.sample(n=n, random_state=42), None
    ys = ytr_pool.loc[Xs.index]
    Ps = prep(Xs)
    m = lgb.LGBMRegressor(n_estimators=500, learning_rate=.05, num_leaves=63,
                          min_child_samples=15, random_state=42, verbosity=-1)
    m.fit(Ps, ys, categorical_feature=CATS)
    pred = np.expm1(m.predict(Pte))
    r2 = r2_score(yte_o, pred)
    md = mdape(yte_o, pred)
    pct10 = (np.abs(yte_o-pred)/yte_o*100 <= 10).mean()*100
    print(f'  n_train={n:>7,}  R2={r2:.3f}  MdAPE={md:5.1f}%  ±10%={pct10:4.0f}%')
print('='*70)
