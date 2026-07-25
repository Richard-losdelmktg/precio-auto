"""Valida el modelo contra avisos reales cargados a mano en validacion.csv.
Consulta el servidor local (mismo camino que usa la interfaz).
  python app/servidor.py     (en otra consola)
  python app/validar.py
"""
import csv, json, sys, urllib.request
from pathlib import Path

API = 'http://127.0.0.1:5000/estimar'
CSV = Path(__file__).resolve().parent / 'validacion.csv'

def pedir(fila):
    body = json.dumps({
        'marca': fila['marca'], 'modelo': fila['modelo'], 'ano': fila['ano'],
        'kilometraje': fila['kilometraje'], 'combustible': fila['combustible'],
        'transmision': fila['transmision'], 'precio': fila['precio_publicado'],
    }).encode()
    req = urllib.request.Request(API, data=body, headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=30))

filas = list(csv.DictReader(open(CSV, encoding='utf-8')))
print(f'\n{"AUTO":30s} {"REAL":>11s} {"ESTIMADO":>11s} {"RANGO":>25s} {"ERROR":>8s} {"OK":>4s}  CONFIANZA')
print('-'*112)
errs, dentro = [], 0
for f in filas:
    try: d = pedir(f)
    except Exception as e:
        print(f'{f["marca"]} {f["modelo"]}: error {e}'); continue
    real = float(f['precio_publicado'])
    err = (d['p50'] - real)/real*100
    ok = d.get('estado') == 'PRECIO JUSTO'
    errs.append(err); dentro += ok
    nombre = f"{f['marca']} {f['modelo']} {f['ano']}"
    rango = f"{d['lo_txt']} - {d['hi_txt']}"
    print(f'{nombre:30s} {"$"+f"{real/1e6:.1f}M":>11s} {"$"+f"{d["p50"]/1e6:.1f}M":>11s} '
          f'{rango:>25s} {err:+7.1f}% {"si" if ok else "NO":>4s}  {d["conf"]}')
print('-'*112)
if errs:
    a = sorted(abs(e) for e in errs)
    med = a[len(a)//2] if len(a)%2 else (a[len(a)//2-1]+a[len(a)//2])/2
    print(f'\n{len(errs)} autos | error mediano {med:.1f}% | dentro del rango {dentro}/{len(errs)} '
          f'({dentro/len(errs)*100:.0f}%) | sesgo medio {sum(errs)/len(errs):+.1f}%')
