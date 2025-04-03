from transformers import T5Tokenizer, T5ForConditionalGeneration

nombre_modelo = "google/mt5-base"

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
eleccion = int(input())

# Variable para evaluar qué orden se le dará al modelo
tipo_tarea = ""

# Si la tarea es generar preguntas
if eleccion == 4:
    texto = input("Ingresa el texto del cual quieres generar preguntas: ")
    tipo_tarea = "generate questions: "
    texto = f"{tipo_tarea} {texto}"
    
    # Convertir texto a vectores
    vectores_entrada = convertir_vectores(texto, return_tensors="pt").input_ids
    
    # Generar múltiples preguntas (hasta 10)
    vectores_salida = modelo.generate(
        vectores_entrada,
        max_length=256,
        num_return_sequences=10,
        num_beams=10,
        early_stopping=True
    )
    
    # Mostrar todas las preguntas generadas
    print("\n========= PREGUNTAS GENERADAS =========")
    for i, salida in enumerate(vectores_salida, 1):
        pregunta = convertir_vectores.decode(salida, skip_special_tokens=True)
        print(f"{i}. {pregunta}")

else:
    # Solicitar el texto que se quiere procesar
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

    # Calcular los vectores de entrada
    vectores_entrada = convertir_vectores(texto, return_tensors="pt").input_ids

    # Calcular los vectores de salida
    vectores_salida = modelo.generate(vectores_entrada, max_length=512)

    # Decodificar los vectores de salida para generar el resultado
    texto_salida = convertir_vectores.decode(vectores_salida[0], skip_special_tokens=True)

    # Mostrar el texto procesado
    print("\n========= RESULTADO =========")
    print(texto_salida)
