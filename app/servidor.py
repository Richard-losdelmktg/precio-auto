"""
Servidor local del estimador de precios.
  python app/entrenar.py    (una vez, genera modelo.pkl)
  python app/servidor.py    -> http://localhost:5000
"""
import pickle, unicodedata, re
from pathlib import Path
import numpy as np, pandas as pd
from flask import Flask, request, jsonify, render_template

APP = Path(__file__).resolve().parent
with open(APP/'modelo.pkl','rb') as f: M = pickle.load(f)
app = Flask(__name__, template_folder=str(APP/'templates'))

CAT_VER = M['cat_ver']; REF = M['ref'].set_index(['mk','md','yr'])
REF_MD  = M['ref_md'].set_index(['mk','md']); DENS = M['dens']; DENS_MD = M['dens_md']

def norm(s):
    s = ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c)!='Mn')
    return re.sub(r'\s+',' ', re.sub(r'[^a-z0-9 ]',' ', s.lower())).strip()

def fmt(v): return f"${v:,.0f}".replace(',','.')

@app.route('/')
def index():
    return render_template('index.html', op=M['opciones'], met=M['metricas'])

@app.route('/versiones')
def versiones():
    """Versiones oficiales del SII para ese marca+modelo+año (opción B)."""
    mk, md = norm(request.args.get('marca','')), norm(request.args.get('modelo','')).split(' ')[0]
    try: yr = int(request.args.get('ano'))
    except (TypeError, ValueError): return jsonify([])
    sub = CAT_VER[(CAT_VER.mk==mk)&(CAT_VER.md==md)&(CAT_VER.yr==yr)]
    if sub.empty:   # sin año exacto, ofrecer las del modelo
        sub = CAT_VER[(CAT_VER.mk==mk)&(CAT_VER.md==md)]
    sub = sub.sort_values('tas')
    return jsonify([{'v':r.version,'tas':float(r.tas)} for r in sub.itertuples()][:60])

@app.route('/estimar', methods=['POST'])
def estimar():
    d = request.get_json()
    try:
        ano = int(d['ano']); km = float(d['kilometraje'])
        marca, modelo = d['marca'], d['modelo']
    except (KeyError, TypeError, ValueError):
        return jsonify({'error':'Datos incompletos o inválidos'}), 400

    mk, md = norm(marca), norm(modelo).split(' ')[0]
    ant = 2026 - ano

    # Referencia SII. Si el usuario eligió una versión, se usa ESA tasación
    # (captura la diferencia entre versión básica y full); si no, la mediana.
    r = REF.loc[(mk,md,ano)] if (mk,md,ano) in REF.index else None
    tas_ver = d.get('tasacion_version')
    if tas_ver:
        try: sii_val = float(tas_ver)
        except ValueError: sii_val = None
    else:
        sii_val = None
    if sii_val is None:
        if r is not None: sii_val = float(r['sii_tas'])
        elif (mk,md) in REF_MD.index: sii_val = float(REF_MD.loc[(mk,md),'sii_tas_md'])
        else: sii_val = float(np.expm1(np.median(M['mmid'].predict(_dummy()))) ) if False else np.nan

    tiene = 1 if r is not None else 0
    med = lambda k, dv: float(r[k]) if r is not None and not pd.isna(r[k]) else dv
    fila = {
        'antiguedad': ant, 'Kilometraje': km, 'km_ano': km/(ant+1),
        'log_sii': np.log1p(sii_val) if sii_val==sii_val else np.log1p(8_000_000),
        'tiene_sii': tiene,
        'sii_cc': med('sii_cc',1600), 'sii_hp': med('sii_hp',110),
        'sii_nver': med('sii_nver',1), 'sii_spread': med('sii_spread',0.2),
        'Marca': marca, 'Modelo': modelo,
        'Combustible': d.get('combustible'), 'Transmision': d.get('transmision'),
    }
    X = pd.DataFrame([fila])[M['FEATS']]
    for c in M['CATS']: X[c] = X[c].astype(M['cat_dtypes'][c])

    p50 = float(np.expm1(M['mmid'].predict(X)[0]))
    lo  = float(np.expm1(M['mlo'].predict(X)[0] - M['qhat']))
    hi  = float(np.expm1(M['mhi'].predict(X)[0] + M['qhat']))

    # (C) Rango adaptativo: si el modelo-año tiene muchas versiones y el usuario
    # NO eligió una, la incertidumbre real es mayor -> se ensancha el rango.
    spread = fila['sii_spread']
    if not tas_ver and spread > 0.15:
        f = 1 + min(spread, 0.6)*0.5
        lo, hi = p50 - (p50-lo)*f, p50 + (hi-p50)*f
    lo, hi = min(lo,p50), max(hi,p50)

    # Confianza: cuántos avisos comparables existen
    n = int(DENS.get((mk,md,ano), 0)); n_md = int(DENS_MD.get((mk,md), 0))
    if n >= 10:   conf, conf_txt = 'alta',  f'{n} autos comparables'
    elif n >= 3:  conf, conf_txt = 'media', f'solo {n} comparables — verificar'
    elif n_md > 0:conf, conf_txt = 'baja',  f'sin datos de ese año ({n_md} del modelo) — requiere tasación manual'
    else:         conf, conf_txt = 'baja',  'vehículo poco frecuente — requiere tasación manual'

    res = {'p50':p50,'lo':lo,'hi':hi,'p50_txt':fmt(p50),'lo_txt':fmt(lo),'hi_txt':fmt(hi),
           'conf':conf,'conf_txt':conf_txt,
           'sii_txt': fmt(sii_val) if sii_val==sii_val else None}

    precio = d.get('precio')
    if precio not in (None,'',0):
        try:
            precio = float(precio)
            estado, clase = (('SUBVALORADO','bajo') if precio < lo else
                             ('SOBREVALORADO','alto') if precio > hi else ('PRECIO JUSTO','justo'))
            res.update({'precio':precio,'precio_txt':fmt(precio),'estado':estado,'clase':clase})
        except ValueError: pass
    return jsonify(res)

def _dummy(): return None

if __name__ == '__main__':
    print('Estimador corriendo en  http://localhost:5000')
    app.run(host='127.0.0.1', port=5000, debug=False)
