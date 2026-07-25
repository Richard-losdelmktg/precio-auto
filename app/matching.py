"""
Cruce entre nuestros avisos y el catalogo del SII.

El SII escribe los nombres distinto a los portales de avisos, y por eso un
cruce ingenuo (primer token del modelo) fallaba en el 15% de los casos:

    aviso              SII                  motivo
    KIA                KIA MOTORS           la marca se llama distinto
    Mitsubishi L 200   mitsubishi / l200    el espacio partia el modelo en "l"
    Mazda BT-50        mazda / bt50         el guion se convertia en espacio
    Mazda 3            mazda / mazda3       el SII antepone la marca
    Mercedes C180      mercedes benz / c    el numero va en la version

Se resuelve con alias de marca y una cascada de estrategias de modelo.
"""
import re, unicodedata

def norm(s):
    s = ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')
    return re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9 ]', ' ', s.lower())).strip()

def compact(s):
    """'l 200' -> 'l200'  ·  'bt 50' -> 'bt50'  ·  'cx 5' -> 'cx5'"""
    return norm(s).replace(' ', '')

# Marcas: como las escribimos -> como las escribe el SII
ALIAS_MARCA = {
    'kia': 'kia motors', 'mercedes': 'mercedes benz', 'mercedes-benz': 'mercedes benz',
    'vw': 'volkswagen', 'range rover': 'land rover', 'gac motors': 'gac', 'gac motor': 'gac',
    'gwm': 'great wall', 'zxauto': 'zx auto', 'dfm': 'dfsk', 'citroen': 'citroen',
}

def marca_sii(marca):
    m = norm(marca)
    return ALIAS_MARCA.get(m, m)

def candidatos_modelo(marca, modelo):
    """Formas posibles del modelo, de la mas especifica a la mas laxa."""
    mk, md = marca_sii(marca), norm(modelo)
    if not md: return []
    c = []
    c.append(compact(md))                    # 'l 200'  -> 'l200'
    c.append(compact(mk.split()[0] + md))    # mazda + '3' -> 'mazda3'
    c.append(md.split()[0])                  # 'hilux dc 4x2' -> 'hilux'
    c.append(compact(' '.join(md.split()[:2])))
    vistos, out = set(), []
    for x in c:
        if x and x not in vistos:
            vistos.add(x); out.append(x)
    return out

def resolver(marca, modelo, modelos_por_marca):
    """Devuelve (mk_sii, md_sii) o (mk_sii, None) si no hay match.

    modelos_por_marca: dict {mk_sii: set(modelos_sii)}
    """
    mk = marca_sii(marca)
    disponibles = modelos_por_marca.get(mk)
    if not disponibles:
        return mk, None
    for cand in candidatos_modelo(marca, modelo):
        if cand in disponibles:
            return mk, cand
    # Ultimo recurso: el modelo del aviso empieza con un modelo del SII y lo que
    # sobra es la cilindrada/serie. Mercedes 'c180' -> SII 'c' + resto '180';
    # BMW '116i' -> SII '116' + resto 'i'. Se exige que el resto empiece en
    # digito o sea un sufijo corto, para no aceptar coincidencias casuales.
    cc = compact(modelo)
    mejor = None
    for m in disponibles:
        if not cc.startswith(m) or len(m) > len(cc): continue
        resto = cc[len(m):]
        if resto and not (resto[0].isdigit() or len(resto) <= 2): continue
        if mejor is None or len(m) > len(mejor): mejor = m
    return mk, mejor
