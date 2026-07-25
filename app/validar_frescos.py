"""
Validacion temporal: avisos NUEVOS de Yapo que el modelo nunca vio.

Trae los avisos mas recientes de Yapo (rutas permitidas por su robots.txt),
descarta los que ya estaban en el entrenamiento y evalua el modelo sobre ese
conjunto fresco. Mide dos cosas que el test interno no puede medir:
  - si el modelo se degrada con avisos posteriores al entrenamiento
  - como se comporta la señal de confianza en la practica

Uso:  python app/servidor.py     (en otra consola)
      python app/validar_frescos.py [n_objetivo]
"""
import json, re, sys, time, urllib.request
from pathlib import Path

APP  = Path(__file__).resolve().parent
DATA = APP.parent / 'data'
BASE = 'https://www.yapo.cl'
API  = 'http://127.0.0.1:5000/estimar'
H = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
     'Accept-Language':'es-CL,es;q=0.9'}
DELAY = 1.2
LD_RE = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.S)
AD_RE = re.compile(r'href="(/autos-usados/[^"]+/(\d+))"')

def get(url):
    try:
        req = urllib.request.Request(url, headers=H)
        return urllib.request.urlopen(req, timeout=25).read().decode('utf-8','ignore')
    except Exception:
        return None

def parse(html):
    for m in LD_RE.finditer(html):
        try: d = json.loads(m.group(1), strict=False)
        except Exception: continue
        if isinstance(d, dict) and d.get('@type') == 'Car':
            mod = d.get('model'); mod = mod.get('name') if isinstance(mod, dict) else mod
            try: km = int(str((d.get('mileageFromOdometer') or {}).get('value','')).replace('.',''))
            except Exception: km = None
            try: pr = int(float((d.get('offers') or {}).get('price') or 0))
            except Exception: pr = None
            return {'marca': (d.get('brand') or '').strip(), 'modelo': str(mod or '').strip(),
                    'ano': d.get('vehicleModelDate'), 'kilometraje': km, 'precio': pr,
                    'combustible': (d.get('fuelType') or '').strip(),
                    'transmision': (d.get('vehicleTransmission') or '').strip()}
    return None

def estimar(a):
    body = json.dumps({'marca':a['marca'],'modelo':a['modelo'],'ano':a['ano'],
                       'kilometraje':a['kilometraje'],'combustible':a['combustible'],
                       'transmision':a['transmision'],'precio':a['precio']}).encode()
    req = urllib.request.Request(API, data=body, headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=30))

def main(objetivo=50):
    ya = set()
    p = DATA/'datos_scraped_yapo.json'
    if p.exists():
        ya = {str(r['source_id']) for r in json.loads(p.read_text(encoding='utf-8'))}
    print(f'Avisos ya conocidos por el modelo: {len(ya):,}')

    frescos, pagina = [], 1
    while len(frescos) < objetivo and pagina <= 40:
        u = f'{BASE}/autos-usados' if pagina == 1 else f'{BASE}/autos-usados.{pagina}'
        h = get(u)
        if h:
            for m in AD_RE.finditer(h):
                aid = m.group(2)
                if aid in ya or any(f[0] == aid for f in frescos): continue
                frescos.append((aid, BASE + m.group(1)))
                if len(frescos) >= objetivo: break
        pagina += 1; time.sleep(DELAY)
    print(f'Avisos NUEVOS encontrados: {len(frescos)} (revisadas {pagina-1} páginas)\n')

    filas = []
    for i, (aid, url) in enumerate(frescos, 1):
        h = get(url)
        a = parse(h) if h else None
        if a and a['marca'] and a['precio'] and a['ano'] and a['kilometraje']:
            filas.append(a)
        if i % 20 == 0: print(f'  descargados {i}/{len(frescos)}...')
        time.sleep(DELAY)

    print(f'\n{"AUTO":32s} {"REAL":>9s} {"ESTIM":>9s} {"ERROR":>8s} {"RANGO":>5s}  CONF')
    print('-'*80)
    res = []
    for a in filas:
        try: d = estimar(a)
        except Exception: continue
        if 'p50' not in d: continue
        err = (d['p50'] - a['precio'])/a['precio']*100
        dentro = a['precio'] >= d['lo'] and a['precio'] <= d['hi']
        res.append((err, dentro, d['conf']))
        print(f'{a["marca"][:14]+" "+a["modelo"][:12]+" "+str(a["ano"]):32s} '
              f'{"$"+format(a["precio"]/1e6,".1f")+"M":>9s} {"$"+format(d["p50"]/1e6,".1f")+"M":>9s} '
              f'{err:+7.1f}% {"si" if dentro else "NO":>5s}  {d["conf"]}')

    if not res: print('sin resultados'); return
    print('-'*80)
    def resumen(sub, tag):
        if not sub: return
        a = sorted(abs(e) for e,_,_ in sub)
        med = a[len(a)//2] if len(a)%2 else (a[len(a)//2-1]+a[len(a)//2])/2
        p10 = sum(1 for e,_,_ in sub if abs(e)<=10)/len(sub)*100
        cob = sum(1 for _,d,_ in sub if d)/len(sub)*100
        ses = sum(e for e,_,_ in sub)/len(sub)
        print(f'  {tag:26s} n={len(sub):3d}  MdAPE={med:5.1f}%  ±10%={p10:3.0f}%  '
              f'en rango={cob:3.0f}%  sesgo={ses:+5.1f}%')
    print(f'\nRESULTADOS sobre {len(res)} avisos nuevos (nunca vistos por el modelo):')
    resumen(res, 'TODOS')
    resumen([r for r in res if r[2]=='alta'],  'confianza alta')
    resumen([r for r in res if r[2]=='media'], 'confianza media')
    resumen([r for r in res if r[2]=='baja'],  'confianza baja')

if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
