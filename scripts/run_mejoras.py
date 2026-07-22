# Experimentos de mejora del modelo de pricing — ver MEJORAS.md
# Metodologia: un solo test set congelado (seed 42). Toda mejora se mide contra el.
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

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

def metrics(y, p, name):
    ape = np.abs(y-p)/y*100
    r = dict(nombre=name, R2=r2_score(y,p), MAE_M=mean_absolute_error(y,p)/1e6,
             MAPE=ape.mean(), MdAPE=np.median(ape),
             pct10=(ape<=10).mean()*100, pct15=(ape<=15).mean()*100)
    print(f"{name:28s} R2={r['R2']:.3f} MAE=${r['MAE_M']:.2f}M MAPE={r['MAPE']:.1f}% "
          f"MdAPE={r['MdAPE']:.1f}% ±10%={r['pct10']:.0f}% ±15%={r['pct15']:.0f}%")
    return r

def te_fit(s, t, sm=10):
    gm = t.mean(); st = t.groupby(s).agg(['mean','count'])
    lam = st['count']/(st['count']+sm)
    return (lam*st['mean']+(1-lam)*gm).to_dict(), gm

def run_xgb(df, tag):
    """Pipeline actual: XGB + target encoding + OHE, split aleatorio estratificado."""
    import xgboost as xgb
    y = np.log1p(df['price'])
    bins = pd.qcut(df['price'], 5, labels=False, duplicates='drop')
    Xtr, Xte, ytr, yte = train_test_split(df, y, test_size=.2, random_state=42, stratify=bins)
    maps, glo = {}, {}
    for c in ['Marca','Modelo']: maps[c], glo[c] = te_fit(Xtr[c], ytr)
    ohe_tr = pd.get_dummies(Xtr[['Combustible','Transmision']], drop_first=True)
    ohe_te = pd.get_dummies(Xte[['Combustible','Transmision']], drop_first=True).reindex(columns=ohe_tr.columns, fill_value=0)
    def enc(X, ohe):
        te = pd.DataFrame({c+'_te': X[c].map(maps[c]).fillna(glo[c]).values for c in ['Marca','Modelo']})
        return pd.concat([X[['antiguedad','Kilometraje']].reset_index(drop=True), te, ohe.reset_index(drop=True)], axis=1)
    m = xgb.XGBRegressor(n_estimators=300, max_depth=7, learning_rate=.1, random_state=42, verbosity=0)
    m.fit(enc(Xtr, ohe_tr), ytr)
    return metrics(np.expm1(yte), np.expm1(m.predict(enc(Xte, ohe_te))), tag)

print('='*100)
df0 = clean(pd.read_csv(BASE/'datos_combinados_entrega2.csv'))
print(f'Registros limpios: {len(df0):,}')

# ── Paso 0: replica del pipeline actual (punto de partida) ──
run_xgb(df0, '0. Actual (XGB+TE, con dups)')

# ── Paso 1: deduplicacion ──
KEY = ['Marca','Modelo','Ano','Kilometraje','price']
n0 = len(df0)
df1 = df0.drop_duplicates(subset=KEY, keep='first').reset_index(drop=True)
nd_specs = df0.duplicated(subset=KEY[:4]).sum()
print(f'\nDups exactos (5 cols): {n0-len(df1):,} ({(n0-len(df1))/n0*100:.1f}%) | '
      f'mismo auto specs (4 cols): {nd_specs:,}')
run_xgb(df1, '1. Dedup + XGB')

# ── Split congelado sobre datos dedup (para pasos 2-3) ──
y = np.log1p(df1['price'])
bins = pd.qcut(df1['price'], 5, labels=False, duplicates='drop')
Xtr, Xte, ytr, yte = train_test_split(df1, y, test_size=.2, random_state=42, stratify=bins)
yte_o = np.expm1(yte)
CATS = ['Marca','Modelo','Combustible','Transmision']
FEATS = ['antiguedad','Kilometraje'] + CATS
def prep(X):
    X = X[FEATS].copy()
    for c in CATS: X[c] = X[c].astype('category')
    return X
# alinear categorias train/test
cat_maps = {c: pd.api.types.CategoricalDtype(Xtr[c].astype('category').cat.categories) for c in CATS}
def prep2(X):
    X = X[FEATS].copy()
    for c in CATS: X[c] = X[c].astype(cat_maps[c])
    return X
Ptr, Pte = prep2(Xtr), prep2(Xte)

# ── Paso 2: LightGBM categoricas nativas ──
import lightgbm as lgb
m2 = lgb.LGBMRegressor(n_estimators=800, learning_rate=.05, num_leaves=63,
                       min_child_samples=20, random_state=42, verbosity=-1)
m2.fit(Ptr, ytr, categorical_feature=CATS)
metrics(yte_o, np.expm1(m2.predict(Pte)), '2. Dedup + LGBM nativo')

# ── Paso 3: rango calibrado (quantile P10/P50/P90) ──
qm = {}
for a in [.1,.5,.9]:
    q = lgb.LGBMRegressor(objective='quantile', alpha=a, n_estimators=800,
                          learning_rate=.05, num_leaves=63, min_child_samples=20,
                          random_state=42, verbosity=-1)
    q.fit(Ptr, ytr, categorical_feature=CATS)
    qm[a] = np.expm1(q.predict(Pte))
metrics(yte_o, qm[.5], '3. LGBM quantile P50')
cov = ((yte_o.values>=qm[.1])&(yte_o.values<=qm[.9])).mean()*100
width = np.median((qm[.9]-qm[.1])/qm[.5])*100
print(f'   Rango P10-P90: cobertura={cov:.1f}% (objetivo ~80%) | ancho mediano={width:.0f}% del precio')

# ── Paso 4: feature km/ano (intensidad de uso) ──
for X in (Ptr, Pte): pass
Ptr4 = Ptr.copy(); Pte4 = Pte.copy()
Ptr4['km_ano'] = Xtr['Kilometraje']/(Xtr['antiguedad']+1)
Pte4['km_ano'] = Xte['Kilometraje']/(Xte['antiguedad']+1)
m4 = lgb.LGBMRegressor(objective='quantile', alpha=.5, n_estimators=800, learning_rate=.05,
                       num_leaves=63, min_child_samples=20, random_state=42, verbosity=-1)
m4.fit(Ptr4, ytr, categorical_feature=CATS)
metrics(yte_o, np.expm1(m4.predict(Pte4)), '4. + km/ano quantile P50')

# ── Paso 5: calibracion conformal del rango (CQR) ──
# Se separa un set de calibracion del train; se ajusta el rango con el residuo conformal.
Xt2, Xcal, yt2, ycal = train_test_split(Ptr4, ytr, test_size=.2, random_state=42)
qmods = {}
for a in [.1,.5,.9]:
    q = lgb.LGBMRegressor(objective='quantile', alpha=a, n_estimators=800, learning_rate=.05,
                          num_leaves=63, min_child_samples=20, random_state=42, verbosity=-1)
    q.fit(Xt2, yt2, categorical_feature=CATS)
    qmods[a] = q
lo_c, hi_c = qmods[.1].predict(Xcal), qmods[.9].predict(Xcal)
s = np.maximum(lo_c-ycal, ycal-hi_c)                      # score conformal (en log)
qhat = np.quantile(s, .8*(1+1/len(s)))                    # correccion para 80% cobertura
lo_t = np.expm1(qmods[.1].predict(Pte4)-qhat)
hi_t = np.expm1(qmods[.9].predict(Pte4)+qhat)
p50_t = np.expm1(qmods[.5].predict(Pte4))
cov5 = ((yte_o.values>=lo_t)&(yte_o.values<=hi_t)).mean()*100
w5 = np.median((hi_t-lo_t)/p50_t)*100
print(f'5. Rango conformal (CQR):    cobertura={cov5:.1f}% | ancho mediano={w5:.0f}% del precio')
print('='*100)
