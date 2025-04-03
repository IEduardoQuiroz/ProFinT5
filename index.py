from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import re
import torch
from diffusers import StableDiffusionPipeline

# Modelos generales para resumir, traducir y preguntas
modelo_general = "t5-base"
convertir_vectores = T5Tokenizer.from_pretrained(modelo_general)
modelo = T5ForConditionalGeneration.from_pretrained(modelo_general)

# Modelo especializado para generación de preguntas
modelo_qg = "mrm8488/t5-base-finetuned-question-generation-ap"
tokenizer_qg = T5Tokenizer.from_pretrained(modelo_qg)
modelo_preguntas = T5ForConditionalGeneration.from_pretrained(modelo_qg)

# Modelo para generar imágenes con Stable Diffusion
modelo_imagen = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"
generador_imagenes = StableDiffusionPipeline.from_pretrained(modelo_imagen, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device)

# Menú de opciones
print("¿Qué tarea quieres realizar?")
print("1. Resumir")
print("2. Traducir")
print("3. Hacer una pregunta")
print("4. Generar preguntas automáticamente")
print("5. Generar una imagen (Stable Diffusion)")

# Leer la elección del usuario
eleccion = int(input())

# Función para generar preguntas resaltando frases automáticamente
def preparar_texto_para_preguntas(texto):
    frases = re.split(r'(\.|,|;|:) ', texto)
    textos_resaltados = []
    for frase in frases:
        frase_limpia = frase.strip()
        if len(frase_limpia.split()) > 3:
            textos_resaltados.append(texto.replace(frase_limpia, f"<hl> {frase_limpia} </hl>"))
    return textos_resaltados[:10]

# Solicitar el texto que se quiere procesar (excepto para opción imagen que pide prompt)
if eleccion != 5:
    texto = input("Ingresa el texto: ")

if eleccion == 1:
    texto = f"summarize: {texto}"

elif eleccion == 2:
    texto = f"translate English to French: {texto}"

elif eleccion == 3:
    contexto = input("Ingresa el contexto: ")
    texto = f"question: {texto}. context: {contexto}"

elif eleccion == 4:
    textos_resaltados = preparar_texto_para_preguntas(texto)

    print("\n========= PREGUNTAS GENERADAS ==========")
    preguntas_generadas = []
    for texto_resaltado in textos_resaltados:
        vectores_entrada = tokenizer_qg(texto_resaltado, return_tensors="pt").input_ids
        pregunta_generada = modelo_preguntas.generate(
            vectores_entrada,
            max_length=64,
            num_beams=10,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        pregunta = tokenizer_qg.decode(pregunta_generada[0], skip_special_tokens=True)
        preguntas_generadas.append(pregunta)

    for idx, pregunta in enumerate(preguntas_generadas, 1):
        print(f"{idx}. {pregunta}")

elif eleccion == 5:
    prompt = input("Describe la imagen que quieres generar: ")
    print("Generando imagen, esto puede tardar un momento...")
    imagen = generador_imagenes(prompt).images[0]
    imagen.save("imagen_generada.png")
    print("Imagen generada guardada como 'imagen_generada.png'")

# Procesar otras opciones con modelo general (1, 2, 3)
if eleccion in [1, 2, 3]:
    vectores_entrada = convertir_vectores(texto, return_tensors="pt").input_ids
    vectores_salida = modelo.generate(
        vectores_entrada,
        max_length=512,
        early_stopping=True
    )

    texto_salida = convertir_vectores.decode(vectores_salida[0], skip_special_tokens=True)
    print("\n========= RESULTADO ==========")
    print(texto_salida)