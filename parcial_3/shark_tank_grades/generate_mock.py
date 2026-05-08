import pandas as pd
import numpy as np
import random

teams = ["Team 1", "Team 2", "Team 3", "Team 4"]
places = ["1st Place", "2nd Place", "3rd Place", np.nan]
justifications = [
    "Me pareció una arquitectura increíblemente robusta, sobre todo la integración con YOLO que resolvió el problema en tiempo real.",
    "El prompt de Gemma fue impecable, generó respuestas con un tono natural y sin alucinaciones notorias.",
    "Sin duda, este es el mejor caso de uso para el mundo real que hemos visto hoy, la viabilidad comercial es altísima.",
    "Me encantó la demostración en vivo, todo fluyó sin errores y la interfaz es súper intuitiva para el usuario final.",
    "Es un pipeline de datos muy creativo y eficiente, me impresiona cómo manejaron la latencia con la base de datos vectorial."
]

data = []
# Generar 20 alumnos aleatorios
for i in range(20):
    row = [f"2026-05-06 19:0{i:02d}:00", f"100{i:02d}", random.choice(teams)]
    # Votos aleatorios para las 16 columnas de métricas
    for _ in range(16):
        row.append(random.choice(places))
    row.append(random.choice(justifications))
    data.append(row)

# Agregar el voto del Shark (Profesor)
shark_row = ["2026-05-06 19:30:00", "3120523", "Team 1"] # Team no importa por el override
for _ in range(16): shark_row.append(random.choice(places[:3])) # Shark no deja nulos
shark_row.append("Excelente integración de los embeddings en el pipeline de latencia baja.")
data.append(shark_row)

cols = ["Timestamp", "Student ID", "Which team are you in?"]
for metric in ["Architecture", "Creativity", "Pitch", "Investment"]:
    for t in teams:
        cols.append(f"{metric} [{t}]")
cols.append("Shark Justification")

df = pd.DataFrame(data, columns=cols)
df.to_csv("mock_shark_tank.csv", index=False)
print("Mockup CSV generado con éxito.")
