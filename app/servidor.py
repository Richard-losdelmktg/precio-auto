"""
Servidor local del estimador de precios. Carga app/modelo.pkl y expone una
interfaz web en http://localhost:5000 para testear el modelo nuevo.
Correr:  python app/servidor.py   (antes: python app/entrenar.py una vez)
"""
import pickle
from pathlib import Path
import numpy as np, pandas as pd
from flask import Flask, request, jsonify, render_template

APP_DIR = Path(__file__).resolve().parent
with open(APP_DIR / 'modelo.pkl', 'rb') as f:
    M = pickle.load(f)

app = Flask(__name__, template_folder=str(APP_DIR / 'templates'))

def fmt(v):
    return f"${v:,.0f}".replace(',', '.')

@app.route('/')
def index():
    return render_template('index.html', op=M['opciones'], met=M['metricas'])

@app.route('/estimar', methods=['POST'])
def estimar():
    d = request.get_json()
    try:
        fila = {
            'antiguedad': 2026 - int(d['ano']),
            'Kilometraje': float(d['kilometraje']),
            'Marca': d['marca'], 'Modelo': d['modelo'],
            'Combustible': d['combustible'], 'Transmision': d['transmision'],
        }
    except (KeyError, ValueError):
        return jsonify({'error': 'Datos incompletos o inválidos'}), 400

    X = pd.DataFrame([fila])[M['FEATS']]
    for c in M['CATS']:
        X[c] = X[c].astype(M['cat_dtypes'][c])

    p50 = float(np.expm1(M['mmid'].predict(X)[0]))
    lo  = float(np.expm1(M['mlo'].predict(X)[0] - M['qhat']))
    hi  = float(np.expm1(M['mhi'].predict(X)[0] + M['qhat']))
    lo, hi = min(lo, p50), max(hi, p50)

    res = {'p50': p50, 'lo': lo, 'hi': hi,
           'p50_txt': fmt(p50), 'lo_txt': fmt(lo), 'hi_txt': fmt(hi)}

    # Si el usuario ingresó un precio, lo clasifica contra el rango
    precio = d.get('precio')
    if precio not in (None, '', 0):
        try:
            precio = float(precio)
            if precio < lo:      estado, clase = 'SUBVALORADO', 'bajo'
            elif precio > hi:    estado, clase = 'SOBREVALORADO', 'alto'
            else:                estado, clase = 'PRECIO JUSTO', 'justo'
            res.update({'precio': precio, 'precio_txt': fmt(precio),
                        'estado': estado, 'clase': clase})
        except ValueError:
            pass
    return jsonify(res)

if __name__ == '__main__':
    print('Estimador corriendo en  http://localhost:5000')
    app.run(host='127.0.0.1', port=5000, debug=False)
