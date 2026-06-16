# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

dataset = [
    ("spam", "oferta gratis urgente"),
    ("spam", "gana dinero rapido"),
    ("spam", "oferta exclusiva limitada"),
    ("spam", "urgente gana dinero ahora"),
    ("spam", "promocion especial gratis"),
    ("spam", "dinero facil rapido"),
    ("spam", "oferta limitada ahora"),
    ("spam", "gana dinero facil"),

    ("legit", "reunion mañana proyecto"),
    ("legit", "agenda reunion equipo"),
    ("legit", "avance del proyecto final"),
    ("legit", "reunion mañana oficina"),
    ("legit", "entrega final proyecto"),
    ("legit", "planificacion equipo proyecto"),
    ("legit", "reunion de seguimiento"),
    ("legit", "agenda semanal equipo")
]

# ------------------------------------------------------------
# Vocabulario permitido
# ------------------------------------------------------------

vocabulario = [
    "gratis",
    "oferta",
    "urgente",
    "dinero",
    "gana",
    "proyecto",
    "reunion"
]

# ============================================================
# ENTRENAMIENTO
# ============================================================

total_emails = len(dataset)

spam_emails = [texto for clase, texto in dataset if clase == "spam"]
legit_emails = [texto for clase, texto in dataset if clase == "legit"]

cantidad_spam = len(spam_emails)
cantidad_legit = len(legit_emails)

# ------------------------------------------------------------
# Probabilidades de clase
# ------------------------------------------------------------

p_spam = cantidad_spam / total_emails
p_legit = cantidad_legit / total_emails

# ------------------------------------------------------------
# Probabilidades condicionales
# ------------------------------------------------------------

prob_spam = {}
prob_legit = {}

for palabra in vocabulario:

    apariciones_spam = 0
    for email in spam_emails:
        palabras = email.split()

        if palabra in palabras:
            apariciones_spam += 1

    prob_spam[palabra] = apariciones_spam / cantidad_spam

    apariciones_legit = 0
    for email in legit_emails:
        palabras = email.split()

        if palabra in palabras:
            apariciones_legit += 1

    prob_legit[palabra] = apariciones_legit / cantidad_legit

# ============================================================
# MOSTRAR MODELO APRENDIDO
# ============================================================

print("=" * 60)
print("PARTE 1 - PROBABILIDADES DE CLASE")
print("=" * 60)

print(f"P(spam)  = {p_spam}")
print(f"P(legit) = {p_legit}")

print()

print("=" * 60)
print("PARTE 1 - PROBABILIDADES CONDICIONALES")
print("=" * 60)

for palabra in vocabulario:
    print(
        f"{palabra:10} | "
        f"P({palabra}|spam)={prob_spam[palabra]:.3f} | "
        f"P({palabra}|legit)={prob_legit[palabra]:.3f}"
    )

# ============================================================
# CLASIFICADOR
# ============================================================

def clasificar(email):

    palabras_email = email.split()

    palabras_relevantes = []

    for palabra in palabras_email:
        if palabra in vocabulario:
            palabras_relevantes.append(palabra)

    score_spam = p_spam
    score_legit = p_legit

    for palabra in palabras_relevantes:
        score_spam *= prob_spam[palabra]
        score_legit *= prob_legit[palabra]

    if score_spam > score_legit:
        clase = "spam"
    elif score_legit > score_spam:
        clase = "legit"
    else:
        clase = "empate"

    return score_spam, score_legit, clase

# ============================================================
# PARTE 2 - NUEVOS CASOS
# ============================================================

casos = [
    "gratis dinero urgente",
    "reunion proyecto agenda",
    "gratis proyecto urgente"
]

print()
print("=" * 60)
print("PARTE 2 - CLASIFICACION")
print("=" * 60)

for i, caso in enumerate(casos, start=1):

    score_spam, score_legit, clase = clasificar(caso)

    print()
    print(f"CASO {i}")
    print(f'Email: "{caso}"')
    print(f"Score Spam : {score_spam}")
    print(f"Score Legit: {score_legit}")
    print(f"Clasificacion final: {clase}")

# ============================================================
# PARTE 3 - RESPUESTAS CONCEPTUALES
# ============================================================

print()
print("=" * 60)
print("PARTE 3 - INTERPRETACION")
print("=" * 60)

print("""
1) Cada término de la multiplicación representa evidencia
   aportada por una palabra para una determinada clase.

2) Una clase puede recibir probabilidad 0 porque no usamos
   suavizado. Si una palabra nunca apareció en una clase,
   entonces P(palabra|clase)=0 y todo el producto vale 0.

3) Las palabras con mayor influencia son aquellas que tienen
   alta probabilidad en una clase y probabilidad 0 en la otra.
   Por ejemplo: dinero, proyecto y reunion.

4) Este proceso es Machine Learning porque el modelo aprende
   patrones observando ejemplos etiquetados en lugar de utilizar
   reglas escritas manualmente.

5) Si duplicamos todos los correos spam:
      - Cambian las probabilidades de clase.
      - No cambian las probabilidades condicionales.
   Las proporciones internas permanecen iguales.

6) Un ejemplo donde podría fallar sería:
      "Reunion urgente del proyecto"

   Contiene palabras típicas de correos legítimos y palabras
   asociadas a spam. Además, al no usar suavizado, pueden
   aparecer probabilidades 0 que afecten la decisión.
""")