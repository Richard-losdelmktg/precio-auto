#!/usr/bin/env python3
"""
Scraper de Yapo.cl — avisos de autos usados.
Rutas permitidas por robots.txt de yapo.cl (verificado 2026-07-21):
  - /autos-usados.N  (listado paginado estatico)
  - /autos-usados/<slug>/<id>  (detalle del aviso, con JSON-LD schema.org/Car)
NO usa /ajax/ ni /chile-es/* (prohibidos por robots.txt).
Rate limit: >=1.2s entre requests. Incremental: reanuda sin repetir avisos.
"""
import json, os, re, sys, time
from datetime import datetime, timezone
from pathlib import Path
import requests

BASE = "https://www.yapo.cl"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "datos_scraped_yapo.json"
STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "yapo_scrape_state.json"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "es-CL,es;q=0.9",
}
DELAY = 1.2
LD_RE = re.compile(r'<script type="application/ld\+json">(.*?)</script>', re.S)
AD_RE = re.compile(r'href="(/autos-usados/[^"]+/(\d+))"')

def log(m): print(m, flush=True)

def get(url, s, retries=2):
    for i in range(retries + 1):
        try:
            r = s.get(url, headers=HEADERS, timeout=25)
            if r.status_code == 200: return r.text
            if r.status_code in (403, 429):
                log(f"  HTTP {r.status_code} en {url} — pausa 60s")
                time.sleep(60)
        except Exception as e:
            log(f"  error ({i+1}): {e}")
        time.sleep(3)
    return None

def parse_ad(html, ad_id, url):
    car = None
    for m in LD_RE.finditer(html):
        try: d = json.loads(m.group(1), strict=False)  # tolera \n, \t sin escapar
        except Exception: continue
        if isinstance(d, dict) and d.get("@type") == "Car": car = d; break
    if not car: return None
    desc = car.get("description") or ""
    km = None
    mo = car.get("mileageFromOdometer") or {}
    try: km = int(str(mo.get("value", "")).replace(".", ""))
    except Exception: pass
    offers = car.get("offers") or {}
    try: price = int(float(offers.get("price") or 0))
    except Exception: price = None
    model = car.get("model")
    if isinstance(model, dict): model = model.get("name")
    # region/comuna: a veces en seller.address o en la descripcion
    addr = (offers.get("seller") or {}).get("address") or {}
    region = (addr.get("addressRegion") or "").strip() or None
    comuna = (addr.get("addressLocality") or "").strip() or None
    return {
        "Marca": (car.get("brand") or "").strip().title() or None,
        "Modelo": (str(model or "")).strip().title() or None,
        "Titulo": (car.get("name") or "").strip() or None,
        "Descripcion": desc[:600],
        "Ano": car.get("vehicleModelDate"),
        "Kilometraje": km, "price": price,
        "Combustible": (car.get("fuelType") or "").strip() or None,
        "Transmision": (car.get("vehicleTransmission") or "").strip() or None,
        "Color": (car.get("color") or "").strip() or None,
        "location": region, "Comuna": comuna,
        "source_id": ad_id, "url": url, "source": "yapo",
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }

def scrape(target=12000, max_pages=1148):
    # Presupuesto de tiempo (min): al superarlo el scraper se detiene y guarda,
    # para que el paso de commit del workflow alcance a correr antes del limite
    # de 6h de GitHub. La corrida siguiente continua desde el checkpoint.
    budget_min = float(os.environ.get("MAX_MINUTES", "0"))
    deadline = time.time() + budget_min * 60 if budget_min > 0 else None
    def out_of_time(): return deadline is not None and time.time() >= deadline

    s = requests.Session()
    results, done = [], set()
    if OUT_PATH.exists():
        try:
            results = json.loads(OUT_PATH.read_text(encoding="utf-8"))
            done = {r["source_id"] for r in results}
            log(f"Reanudando: {len(done)} avisos ya guardados")
        except Exception: pass

    def save():
        OUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    # Fase 1: recolectar URLs de avisos desde el listado paginado.
    # El listado esta ordenado "mas reciente primero": si se reiniciara siempre
    # en la pagina 1, en corridas repetidas esas paginas quedarian saturadas de
    # avisos ya guardados y el corte por "paginas vacias" terminaria demasiado
    # pronto sin llegar a las paginas mas antiguas que si tienen avisos nuevos.
    # Por eso se guarda un checkpoint y cada corrida continua donde quedo la
    # anterior (con vuelta a la pagina 1 al llegar al final del listado).
    start_page = 1
    if STATE_PATH.exists():
        try: start_page = json.loads(STATE_PATH.read_text(encoding="utf-8")).get("last_page", 1)
        except Exception: pass
    log(f"Fase 1 desde pagina {start_page}")

    ad_urls, page, empty, scanned = {}, start_page, 0, 0
    while scanned <= max_pages and len(done) + len(ad_urls) < target * 1.15:
        if out_of_time(): log("Presupuesto de tiempo agotado en Fase 1"); break
        u = f"{BASE}/autos-usados.{page}" if page > 1 else f"{BASE}/autos-usados"
        h = get(u, s)
        if not h:
            empty += 1
        else:
            found = 0
            for m in AD_RE.finditer(h):
                aid = m.group(2)
                if aid not in done and aid not in ad_urls:
                    ad_urls[aid] = BASE + m.group(1); found += 1
            empty = 0 if found else empty + 1
            if empty >= 40:
                log(f"40 paginas sin avisos nuevos — fin real del listado en p.{page}")
                page = 1; scanned += 1; empty = 0
                STATE_PATH.write_text(json.dumps({"last_page": 1}), encoding="utf-8")
                continue
        scanned += 1
        if scanned % 25 == 0: log(f"  listado p.{page} | urls nuevas: {len(ad_urls)}")
        page = page + 1 if page < max_pages else 1
        STATE_PATH.write_text(json.dumps({"last_page": page}), encoding="utf-8")
        time.sleep(DELAY)
    log(f"Fase 1 lista: {len(ad_urls)} avisos nuevos por descargar (checkpoint: pagina {page})")

    # Fase 2: detalle de cada aviso
    for i, (aid, url) in enumerate(ad_urls.items(), 1):
        if len(results) >= target: break
        if out_of_time():
            save(); log(f"Presupuesto de tiempo agotado en Fase 2 — guardado parcial ({len(results)})")
            return
        h = get(url, s)
        if h:
            row = parse_ad(h, aid, url)
            if row and row["Marca"] and row["price"] and row["Ano"]:
                results.append(row); done.add(aid)
        if i % 100 == 0:
            save(); log(f"  [{i}/{len(ad_urls)}] guardados: {len(results)}")
        time.sleep(DELAY)

    save()
    log(f"COMPLETO: {len(results)} avisos en {OUT_PATH}")

if __name__ == "__main__":
    scrape(target=int(sys.argv[1]) if len(sys.argv) > 1 else 12000)
