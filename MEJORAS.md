# MEJORAS.md — Registro de cambios y mejoras del modelo de pricing

Trazabilidad de qué se cambió, por qué, y el efecto medido. Para el equipo.

## Metodología de validación (aplica a todo)

- **Un solo test set congelado** (20%, semilla 42, estratificado por quintil de precio). Ningún modelo lo ve durante entrenamiento ni ajuste. Toda mejora se mide contra ese mismo set.
- **Métricas nuevas** además de R²/MAPE: **MdAPE** (error mediano, no lo distorsionan los autos baratos) y **% de predicciones dentro de ±10% y ±15%** del precio real — lo que el negocio realmente siente.
- Script reproducible: `scripts/run_mejoras.py` (corre todos los pasos y imprime la tabla).

## Punto de partida (pipeline original)

XGBoost + target encoding manual, 6 features (marca, modelo, año, km, combustible, transmisión), 32.371 avisos limpios de Yapo/Chileautos.

| Métrica | Valor |
|---|---|
| R² test | 0.808 |
| MAPE | 18.0% |
| **MdAPE** | **11.4%** |
| Dentro de ±10% | 45% |
| Dentro de ±15% | 61% |

Lectura: la mitad de las predicciones tiene error ≤11.4%. El MAPE de 18% está inflado por autos baratos/raros donde el error porcentual explota.

## Cambio 1 — Deduplicación de avisos

**Qué se hizo:** eliminar avisos idénticos (misma marca/modelo/año/km/precio) antes de entrenar.
**Por qué:** avisos republicados cuentan doble y pueden caer uno en train y otro en test, inflando artificialmente la métrica.
**Resultado:** solo 138 duplicados exactos (0.4%) — el dataset estaba más limpio de lo esperado. Las métricas casi no cambian (MdAPE 11.4→11.8), pero ahora son *honestas*: el número anterior tenía una leve inflación por fuga de duplicados. Se mantiene el cambio por corrección metodológica, no por ganancia.

## Cambio 2 — LightGBM con categóricas nativas (reemplaza XGBoost + target encoding)

**Qué se hizo:** cambiar a LightGBM, que maneja marca/modelo como categorías nativas, eliminando el target encoding manual.
**Por qué:** el target encoding manual es una fuente típica de fuga de información y de errores al llevar el modelo a producción (hay que mantener los diccionarios de encoding sincronizados). Con categóricas nativas el pipeline es más simple y robusto.
**Efecto:** R² 0.804→0.815, MAE $1.41M→$1.36M. Mejora leve + pipeline más simple de mantener y exportar.

## Cambio 3 — Predicción por rango en vez de punto único (quantile regression)

**Qué se hizo:** entrenar el modelo con objetivo quantile (P10/P50/P90): en vez de "tu auto vale $8.4M", entrega "vale entre $7.1M y $9.6M, valor central $8.4M".
**Por qué:** es el cambio clave para el caso Auto360. Un punto único que falla un 11% pierde clientes; un rango honesto abre la conversación y le da al tasador un piso y techo de negociación. Además la mediana (P50) es más robusta que la media ante avisos con precios absurdos.
**Efecto:** el P50 es además el mejor modelo puntual: **MdAPE 10.9%**, dentro de ±10% sube 45%→47%, dentro de ±15% 61%→63%.

## Cambio 4 — Feature km/año (intensidad de uso)

**Qué se hizo:** agregar kilometraje promedio anual como variable.
**Por qué:** un auto de 5 años con 150.000 km vale menos que uno igual con 40.000 km; la relación km/edad aporta señal que el km absoluto no captura del todo.
**Efecto:** marginal (MdAPE 10.9→10.8). Se mantiene: costo cero y no perjudica.

## Cambio 5 — Calibración conformal del rango (CQR)

**Qué se hizo:** ajustar el ancho del rango P10–P90 con un set de calibración para garantizar cobertura real del 80%.
**Por qué:** el rango sin calibrar solo contenía el precio real el 71% de las veces (prometía 80%). Un rango mal calibrado es un rango mentiroso.
**Efecto:** cobertura 71.4% → **78.9%** (objetivo 80%). El ancho mediano del rango es ±23% en torno al valor central — refleja la incertidumbre *real* que existe con solo 6 variables.

## Resumen ejecutivo

| | Original | Final (P50 + rango calibrado) |
|---|---|---|
| MdAPE | 11.4% | **10.8%** |
| Dentro de ±10% | 45% | **47%** |
| Dentro de ±15% | 61% | **63%** |
| Rango calibrado | no existía | **sí, cobertura 79%** |
| Pipeline | XGB + encoding manual | LGBM nativo (más simple) |

**Conclusión honesta:** las mejoras de proceso dieron ganancias reales pero moderadas. El modelo llegó al **techo de lo que permiten 6 variables**: dos autos idénticos en papel pueden valer 20-30% distinto por versión/equipamiento y estado. Para bajar de ~10% MdAPE se necesita **más información por auto**, no más tuning.

## Cambio 6 — Scraper v2: versión, fecha, región y vendedor

**Qué se hizo:** se actualizó `scripts/scraper_chileautos.py` para capturar por cada aviso: **Versión/trim** (ej: "2.0t deluxe diesel 4x4 dob. cab. at"), **fecha de publicación**, color, región, tipo de vendedor y seller_id. Además ahora es incremental: reanuda donde quedó sin repetir avisos ya scrapeados.
**Por qué:** la versión es la variable ausente de mayor impacto (dos autos iguales en papel difieren 20-30% por trim); la fecha permite deflactar precios y validar temporalmente; región/vendedor aportan señal adicional. Todo esto ya existía en el endpoint de Chileautos — solo no se estaba guardando.
**Efecto:** validado con prueba de 60 autos (93% trae versión). Scraping masivo a 12.000 autos en curso. El efecto en el error se medirá al reentrenar con los datos nuevos.

## Cambio 7 — Cumplimiento robots.txt: se detuvo Chileautos, nuevo scraper de Yapo

**Qué se hizo:** se detectó que el `robots.txt` de Chileautos prohíbe explícitamente `/vehiculos/detalles/*` (las páginas que scrapeábamos). Se detuvo ese scraping de inmediato. Se verificaron alternativas: **Yapo.cl permite** sus rutas de listado (`/autos-usados.N`) y detalle en robots.txt; MercadoLibre exige API con OAuth (requiere crear app de desarrollador); Autocosmos permite pero con crawl-delay de 20s (inviable en volumen). Se construyó `scripts/scraper_yapo.py`: parsea el JSON-LD estructurado (schema.org/Car) de cada aviso, respeta ≥1.2s entre requests, guarda incremental y reanuda solo.
**Por qué:** riesgo legal/comercial — un producto para Auto360 no puede construirse sobre datos obtenidos violando restricciones explícitas del sitio fuente. Yapo entrega además **título y descripción completa** del aviso, de donde se puede extraer la versión/trim.
**Efecto:** scraping de Yapo validado (30/30 avisos en prueba) y corriendo a volumen (objetivo 12.000). Los ~200 registros de Chileautos quedan sin uso comercial. Pendiente: revisión de Términos de Servicio por alguien con conocimiento legal antes de comercializar.

## Cambio 8 — Curva de aprendizaje: cuánto dato hace falta realmente

**Qué se hizo:** `scripts/learning_curve.py` entrena el mismo modelo (LightGBM, 6 features) con subconjuntos crecientes de train (5k→26k) contra el mismo test congelado, para medir en qué punto más volumen deja de ayudar.
**Por qué:** antes de invertir horas/días de scraping había que confirmar con datos, no con estimación, cuánto volumen es realmente necesario.
**Resultado (corrige una estimación previa de "80-100k"):**

| n_train | R² | MdAPE | ±10% |
|---|---|---|---|
| 5.000 | 0.731 | 14.6% | 36% |
| 10.000 | 0.781 | 12.6% | 41% |
| 20.000 | 0.807 | 11.4% | 45% |
| 26.327 | 0.815 | 11.4% | 45% |

**Conclusión:** el modelo se aplana en ~20-25k con el set de variables actual. Entre 20k y 26k el error casi no se mueve. **Esto confirma que el cuello de botella no es volumen, es la variable de versión/trim** — meter más filas sin versión no va a bajar el error significativamente. Prioridad correcta: que los datos nuevos de Yapo traigan versión extraída de título/descripción, no solo acumular más avisos.

## Cambio 9 — Scraping externalizado (GitHub Actions)

**Qué se hizo:** `.github/workflows/scrape_yapo.yml` — corre el scraper de Yapo cada 6 horas en la infraestructura de GitHub (gratis en repos públicos), hace commit incremental de los datos nuevos al repo. Se investigó Autocosmos como fuente adicional pero su listado se carga por JavaScript (no accesible con requests simples) y además exige 20s entre requests — dado que Yapo solo ya cubre la meta de volumen (20-25k), se pausó esa vía por ahora.
**Por qué:** liberar al usuario de mantener su PC prendido días enteros. El scraper ya era incremental/reanudable, lo que encaja naturalmente con ejecuciones programadas cortas en vez de un proceso continuo.
**Efecto:** scraping ahora corre solo, sin depender de la máquina local. Pendiente: push del repo con este workflow para que se active.

## Pendientes (requieren re-scrapear — próxima etapa)

1. **Versión/trim** (la mejora de mayor impacto disponible): la URL de Chileautos ya la contiene (ej: `santa-fe-2-2-crdi-auto-plus`). El scraper actual ya guarda `url`, `Tipo_de_vendedor`, `Category` y `scraped_at` — pero solo hay ~200 registros con ese detalle vs 32.000 sin él. Hay que correr el scraper a volumen.
2. **Fecha del aviso**: para deflactar precios (UF/IPC) y validar con split temporal, que es la validación correcta para un sistema en producción.
3. **Región y tipo de vendedor** como features (ya vienen en el scraper nuevo).
4. **Datos de transacciones reales de Auto360** (cuando exista el acuerdo): permitiría modelar el precio de *toma* además del precio de *publicación* — el activo diferenciador del proyecto.
