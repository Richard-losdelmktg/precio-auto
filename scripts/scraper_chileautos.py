#!/usr/bin/env python3
"""
Scraper de Chileautos.cl — adaptado del repo Seba-vp/car-tracker
Usa los endpoints JSON internos de Next.js que NO tienen DataDome.
"""

import json, re, sys, time
from datetime import datetime, timezone
from pathlib import Path
import requests

BASE = "https://www.chileautos.cl"
SEARCH_API = f"{BASE}/_api/search-core/"
DETAILS_API = f"{BASE}/_api/details-core/"
QUERY_BASE = "(And.(C.Category.autos.)_.State.Usado.)"

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "datos_scraped_chileautos.json"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "es-CL,es;q=0.9,en;q=0.8",
    "Referer": f"{BASE}/",
}

FUEL_MAP = {
    "bencina": "Bencina", "gasolina": "Bencina",
    "diesel": "Diesel", "diésel": "Diesel",
    "hibrido": "Híbrido", "híbrido": "Híbrido",
    "electrico": "Eléctrico", "eléctrico": "Eléctrico",
}
TRANS_MAP = {
    "manual": "Manual", "mecánico": "Manual",
    "automatic": "Automática", "automático": "Automática",
    "automatica": "Automática", "automática": "Automática",
    "cvt": "Automática", "tiptronic": "Automática",
    "dsg": "Automática", "dct": "Automática",
}
SELLER_MAP = {"agencia": "Agencia", "particular": "Particular", "dealer": "Agencia", "private": "Particular"}

MAKES = [
    "toyota", "chevrolet", "nissan", "hyundai", "kia", "mazda", "suzuki",
    "ford", "mitsubishi", "peugeot", "volkswagen", "honda", "jeep",
    "subaru", "mercedes-benz", "bmw", "audi", "renault", "mg",
    "great-wall", "chery", "changan", "jac", "jetour", "dongfeng",
    "geely", "baic", "haval", "dfsk", "zotye", "foton", "jmc",
    "volvo", "land-rover", "lexus", "infiniti", "porsche", "mini",
    "fiat", "alfa-romeo", "citroen", "skoda", "seat", "opel",
    "ram", "dodge", "chrysler", "jeep", "cadillac", "gmc",
    "ssangyong", "daewoo", "dacia", "isuzu", "hummer",
]
REGIONS = [
    "metropolitana-de-santiago", "valparaiso", "biobio", "maule",
    "ohiggins", "araucania", "los-lagos", "coquimbo", "antofagasta",
    "los-rios", "atacama", "nuble", "tarapaca",
]
YEAR_BUCKETS = [(2000,2005),(2006,2009),(2010,2012),(2013,2015),(2016,2018),(2019,2021),(2022,2026)]
PRICE_BUCKETS = [(0,6_000_000),(6_000_000,10_000_000),(10_000_000,15_000_000),(15_000_000,22_000_000),(22_000_000,1_000_000_000)]


def log(msg):
    print(msg, flush=True)


def walk(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from walk(v)


def fetch_json(url, session, retries=2):
    for attempt in range(retries + 1):
        try:
            r = session.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200 and "json" in r.headers.get("content-type", ""):
                return r.json()
            if r.status_code == 403:
                log(f"  403 bloqueado en {url}")
                return None
        except Exception as e:
            log(f"  Error intento {attempt+1}: {e}")
        time.sleep(1.5)
    return None


def extract_ids(search_json):
    if not search_json:
        return {}
    seen = {}
    for d in walk(search_json):
        nid = d.get("networkId")
        if isinstance(nid, str) and nid.startswith("CP-AD-"):
            cur = seen.setdefault(nid, {})
            for k, v in d.items():
                if not isinstance(v, (dict, list)) and k not in cur:
                    cur[k] = v
    url_map = {}
    for d in walk(search_json):
        u = d.get("url")
        if isinstance(u, str):
            m = re.match(r"^(/vehiculos/detalles/[a-zA-Z0-9\-]+/(CP-AD-\d+)/)", u)
            if m:
                url_map.setdefault(m.group(2), BASE + m.group(1))
    return {nid: {"networkId": nid, "url": url_map.get(nid, f"{BASE}/vehiculos/{nid}"), "tracking": tr}
            for nid, tr in seen.items()}


def extract_details(details_json):
    if not details_json:
        return None
    best, best_score = None, -1
    for d in walk(details_json):
        if "networkId" in d and "make" in d and "model" in d:
            score = sum(1 for k in ("year","price","odometermin","fueltype","genericgeartype","bodystyle","sellertype","publishDate","state") if k in d)
            if score > best_score:
                best_score, best = score, d
    return best


def to_int(v):
    if v is None:
        return None
    try:
        return int(str(v).replace(".", "").replace(",", "").strip())
    except:
        return None


def normalise(nid, url, tracking, details):
    t = details or tracking or {}
    make  = (t.get("make") or "").strip().title() or None
    model = (t.get("model") or "").strip().title() or None
    year  = to_int(t.get("year"))
    price = to_int(t.get("price"))
    km    = to_int(t.get("odometermin"))
    fuel  = FUEL_MAP.get((t.get("fueltype") or "").strip().lower())
    trans = TRANS_MAP.get((t.get("genericgeartype") or "").strip().lower())
    seller = SELLER_MAP.get((t.get("sellertype") or "").strip().lower())
    region = (t.get("state") or "").strip().title() or None
    body   = (t.get("bodystyle") or "").strip().title() or None
    version = (t.get("badge") or "").strip().lower() or None      # version/trim del vehiculo
    pub    = (t.get("publishDate") or "").strip() or None         # fecha de publicacion del aviso
    color  = (t.get("colour") or "").strip().title() or None
    return {
        "Marca": make, "Modelo": model, "Version": version, "Ano": year,
        "Kilometraje": km, "price": price, "Combustible": fuel, "Transmision": trans,
        "Tipo_de_vendedor": seller, "location": region, "Category": body,
        "Color": color, "publish_date": pub, "seller_id": t.get("sellerId"),
        "source_id": nid, "url": url,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def scrape(target=3000):
    session = requests.Session()
    session.get(BASE + "/", timeout=20)

    seen_ids = {}

    def collect(q, max_offsets=4):
        for off in range(0, max_offsets * 31, 31):
            url = f"{SEARCH_API}?q={q}&offset={off}"
            j = fetch_json(url, session)
            if not j:
                return
            new_ids = extract_ids(j)
            added = {k: v for k, v in new_ids.items() if k not in seen_ids}
            seen_ids.update(added)
            log(f"  +{len(added)} nuevos | total: {len(seen_ids)} | query: {q[:70]}...")
            if len(added) == 0:
                break
            if len(seen_ids) >= target:
                return
            time.sleep(0.7)

    log("=== FASE 1: base + regiones ===")
    collect(QUERY_BASE, max_offsets=5)
    for region in REGIONS:
        if len(seen_ids) >= target: break
        collect(f"(And.(C.Category.autos.)_.State.Usado._.Region.{region}.)", max_offsets=3)

    log("=== FASE 2: por marca ===")
    for make in MAKES:
        if len(seen_ids) >= target: break
        collect(f"(And.(C.Category.autos.)_.State.Usado._.Make.{make}.)", max_offsets=3)

    log("=== FASE 3: marca × año ===")
    for make in MAKES:
        for ylo, yhi in YEAR_BUCKETS:
            if len(seen_ids) >= target: break
            collect(f"(And.(C.Category.autos.)_.State.Usado._.Make.{make}._.Year.range({ylo}..{yhi}).)", max_offsets=2)
        if len(seen_ids) >= target: break

    log("=== FASE 4: marca × precio ===")
    for make in MAKES:
        for plo, phi in PRICE_BUCKETS:
            if len(seen_ids) >= target: break
            collect(f"(And.(C.Category.autos.)_.State.Usado._.Make.{make}._.Price.range({plo}..{phi}).)", max_offsets=2)
        if len(seen_ids) >= target: break

    log("=== FASE 5 (4D): región × marca × año × precio ===")
    for region in REGIONS:
        for make in MAKES:
            for ylo, yhi in YEAR_BUCKETS:
                for plo, phi in PRICE_BUCKETS:
                    if len(seen_ids) >= target: break
                    q = (f"(And.(C.Category.autos.)_.State.Usado."
                         f"_.Region.{region}._.Make.{make}"
                         f"._.Year.range({ylo}..{yhi})"
                         f"._.Price.range({plo}..{phi}).)")
                    collect(q, max_offsets=1)
                if len(seen_ids) >= target: break
            if len(seen_ids) >= target: break
        if len(seen_ids) >= target: break

    log(f"\n=== IDs recolectados: {len(seen_ids)} ===")
    log("=== Obteniendo detalles de cada auto... ===")

    # Reanudacion incremental: cargar lo ya scrapeado y saltar esos ids
    results = []
    done_ids = set()
    if OUT_PATH.exists():
        try:
            results = json.loads(OUT_PATH.read_text(encoding="utf-8"))
            done_ids = {r["source_id"] for r in results if "Version" in r}  # re-scrapear los viejos sin Version
            log(f"  Reanudando: {len(done_ids)} ya scrapeados con formato nuevo")
        except Exception:
            pass
    results = [r for r in results if r["source_id"] in done_ids]
    refs = [r for r in seen_ids.values() if r["networkId"] not in done_ids]
    for i, ref in enumerate(refs, 1):
        nid = ref["networkId"]
        if i % 50 == 0:
            log(f"  [{i}/{len(refs)}] procesados... ({len(results)} con datos completos)")
            # Guardar progreso parcial
            with open(OUT_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        dj = fetch_json(f"{DETAILS_API}{nid}/", session)
        details = extract_details(dj)
        row = normalise(nid, ref["url"], ref.get("tracking", {}), details)
        # Solo guardar si tiene los campos mínimos
        if row["Marca"] and row["price"] and row["Kilometraje"]:
            results.append(row)
        time.sleep(0.7)

    log(f"\n=== SCRAPING COMPLETO: {len(results)} autos con datos completos ===")
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log(f"Guardado en: {OUT_PATH}")
    return results


if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    results = scrape(target=target)
    # Preview
    import pandas as pd
    df = pd.DataFrame(results)
    print(df[["Marca","Modelo","Ano","Kilometraje","price","Combustible","Transmision"]].head(10).to_string())
    print(f"\nShape final: {df.shape}")
    print(df["Marca"].value_counts().head(10))
