from transformers import T5Tokenizer, T5ForConditionalGeneration

# Modelos generales para resumir, traducir y preguntas
modelo_general = "t5-base"
convertir_vectores = T5Tokenizer.from_pretrained(modelo_general)
modelo = T5ForConditionalGeneration.from_pretrained(modelo_general)

# Modelo especializado para generación de preguntas
modelo_qg = "valhalla/t5-base-qg-hl"
tokenizer_qg = T5Tokenizer.from_pretrained(modelo_qg)
modelo_preguntas = T5ForConditionalGeneration.from_pretrained(modelo_qg)

# Menú de opciones
print("¿Qué tarea quieres realizar?")
print("1. Resumir")
print("2. Traducir")
print("3. Hacer una pregunta")
print("4. Generar preguntas automáticamente")

# Leer la elección del usuario
eleccion = int(input())

# Solicitar el texto que se quiere procesar
texto = input("Ingresa el texto: ")

if eleccion == 1:
    texto = f"summarize: {texto}"

elif eleccion == 2:
    texto = f"translate English to French: {texto}"

elif eleccion == 3:
    contexto = input("Ingresa el contexto: ")
    texto = f"question: {texto}. context: {contexto}"

# Procesar generación automática de preguntas con modelo especializado
if eleccion == 4:
    texto_preparado = f"generate questions: {texto}"
    vectores_entrada = tokenizer_qg(texto_preparado, return_tensors="pt").input_ids

    preguntas_generadas = modelo_preguntas.generate(
        vectores_entrada,
        max_length=64,
        num_return_sequences=10,
        num_beams=20,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    print("\n========= PREGUNTAS GENERADAS ==========")
    preguntas = [tokenizer_qg.decode(preg, skip_special_tokens=True) for preg in preguntas_generadas]
    for idx, pregunta in enumerate(preguntas, 1):
        print(f"{idx}. {pregunta}")

# Procesar otras opciones con modelo general
else:
    vectores_entrada = convertir_vectores(texto, return_tensors="pt").input_ids
    vectores_salida = modelo.generate(
        vectores_entrada,
        max_length=512,
        early_stopping=True
    )

    texto_salida = convertir_vectores.decode(vectores_salida[0], skip_special_tokens=True)
    print("\n========= RESULTADO ==========")
    print(texto_salida)
