from transformers import T5Tokenizer, T5ForConditionalGeneration

nombre_modelo = "t5-large"

# Importar los módulos necesarios del modelo T5
convertir_vectores = T5Tokenizer.from_pretrained(nombre_modelo)
modelo = T5ForConditionalGeneration.from_pretrained(nombre_modelo)

# Mostrar un menú para que el usuario elija la opción que desea realizar
print("¿Qué tarea quieres realizar?")
print("1. Resumir")
print("2. Traducir")
print("3. Hacer una pregunta")
print("4. Generar preguntas a partir de un texto")

# Leer la elección del usuario
eleccion = int(input("Elige una opción (1-4): "))

# Variable para evaluar qué orden se le dará al modelo
tipo_tarea = ""

# Opción 4: Generar preguntas
if eleccion == 4:
    texto = input("Ingresa el texto del cual generar preguntas: ")
    tipo_tarea = "generate questions: "
    texto = f"{tipo_tarea} {texto}"
    num_preguntas = 10  # Número de preguntas a generar
    max_length = 512

    vectores_entrada = convertir_vectores(texto, return_tensors="pt").input_ids
    vectores_salidad = modelo.generate(
        vectores_entrada,
        max_length=max_length,
        num_return_sequences=num_preguntas,
        do_sample=True,
        top_k=50,
        top_p=0.95
    )

    print("\n========= PREGUNTAS GENERADAS =========")
    for i, salida in enumerate(vectores_salidad, 1):
        pregunta = convertir_vectores.decode(salida, skip_special_tokens=True)
        print(f"{i}. {pregunta}")

else:
    texto = input("Ingresa el texto: ")

    if eleccion == 1:
        tipo_tarea = "summarize: "
        texto = f"{tipo_tarea} {texto}"

    elif eleccion == 2:
        tipo_tarea = "translate English to French: "
        texto = f"{tipo_tarea} {texto}"

    elif eleccion == 3:
        contexto = input("Ingresa el contexto: ")
        tipo_tarea = "question: "
        texto = f"{tipo_tarea} {texto}. context: {contexto}"

    vectores_entrada = convertir_vectores(texto, return_tensors="pt").input_ids
    vectores_salidad = modelo.generate(vectores_entrada, max_length=512)
    texto_salida = convertir_vectores.decode(vectores_salidad[0], skip_special_tokens=True)

    print("\n========= RESULTADO =========")
    print(texto_salida)
