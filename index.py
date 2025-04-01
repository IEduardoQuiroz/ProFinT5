from transformers import T5Tokenizer, T5ForConditionalGeneration

nombre_modelo = "t5-base"

# Importar los modulos necesarios del modelo T5
convertir_vectores = T5Tokenizer.from_pretrained(nombre_modelo)
modelo = T5ForConditionalGeneration.from_pretrained(nombre_modelo)

# Mostrar un menu para que el usuario elija la opción que desea realizar
print("Que tarea quieres realizar")
print("1. Resumir")
print("2. Traducir")
print("3. Hacer una pregunta")

# Leer la elección del usuario
eleccion = int(input())

# Variable para evaluar que orden se le dará al modelo
tipo_tarea = ""

# Solicitar el texto que se quiere procesar
texto = input("Ingresa el texto: ")

# Si la tarea es 1, es decir, resumir texto, el comando que se agrega es "summarize: "
if eleccion == 1:
    tipo_tarea = "summarize: "
    texto = f"{tipo_tarea} {texto}"

# Si la tarea es 2, es decir, traducir texto, el comando que se agrega es "translate idioma a idioma: "
elif eleccion == 2:
    tipo_tarea = "translate English to French: "
    texto = f"{tipo_tarea} {texto}"

# Si la tarea es 3, es decir, responder una pregunta, el comando que se agrega es "question: "
# Pero además se pide el contexto dentro del cual se va responder la pregunta
elif eleccion == 3:
    contexto = input("Ingresa el contexto: ")
    tipo_tarea = "question: "
    texto = f"{tipo_tarea} {texto}. context: {contexto}"

# Calcular los vectores de entrada
vectores_entrada = convertir_vectores(texto, return_tensors="pt").input_ids

# Calcular los vectores de salida
vectores_salidad = modelo.generate(vectores_entrada, max_length=512)

# Decodificar los vectores de salida para generar el resultado
texto_salida = convertir_vectores.decode(vectores_salidad[0], skip_special_tokens=True)

# Mostrar el texto resumido
print("\n ========= RESULTADO ==========")
print(texto_salida)
