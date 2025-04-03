from flask import Flask, render_template, request, jsonify, send_file
from transformers import T5Tokenizer, T5ForConditionalGeneration
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import re

app = Flask(__name__)

# Configurar modelos
modelo_general = "t5-base"
convertir_vectores = T5Tokenizer.from_pretrained(modelo_general)
modelo = T5ForConditionalGeneration.from_pretrained(modelo_general)

modelo_qg = "mrm8488/t5-base-finetuned-question-generation-ap"
tokenizer_qg = T5Tokenizer.from_pretrained(modelo_qg)
modelo_preguntas = T5ForConditionalGeneration.from_pretrained(modelo_qg)

modelo_imagen = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"
generador_imagenes = StableDiffusionPipeline.from_pretrained(
    modelo_imagen,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

def preparar_texto_para_preguntas(texto):
    frases = re.split(r'(\.|,|;|:) ', texto)
    textos_resaltados = []
    for frase in frases:
        frase_limpia = frase.strip()
        if len(frase_limpia.split()) > 3:
            textos_resaltados.append(texto.replace(frase_limpia, f"<hl> {frase_limpia} </hl>"))
    return textos_resaltados[:5]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/procesar', methods=['POST'])
def procesar():
    datos = request.json
    tipo = datos['tipo']
    texto = datos.get('texto', '')

    if tipo == 'resumir':
        entrada = f"summarize: {texto}"
    elif tipo == 'traducir':
        entrada = f"translate English to French: {texto}"
    elif tipo == 'preguntar':
        contexto = datos.get('contexto', '')
        entrada = f"question: {texto}. context: {contexto}"
    else:
        entrada = texto

    if tipo in ['resumir', 'traducir', 'preguntar']:
        tokens = convertir_vectores(entrada, return_tensors="pt").input_ids
        salida = modelo.generate(tokens, max_length=512)
        respuesta = convertir_vectores.decode(salida[0], skip_special_tokens=True)
        return jsonify({'resultado': respuesta})

    elif tipo == 'generar_preguntas':
        textos_resaltados = preparar_texto_para_preguntas(texto)
        preguntas = []
        for entrada in textos_resaltados:
            ids = tokenizer_qg(entrada, return_tensors="pt").input_ids
            salida = modelo_preguntas.generate(ids, max_length=64, num_beams=10, early_stopping=True)
            pregunta = tokenizer_qg.decode(salida[0], skip_special_tokens=True)
            preguntas.append(pregunta)
        return jsonify({'preguntas': preguntas})

    elif tipo == 'generar_imagen':
        prompt = texto
        imagen = generador_imagenes(prompt).images[0]
        ruta = "static/imagen_generada.png"
        imagen.save(ruta)
        return jsonify({'imagen': ruta})

@app.route('/imagen')
def mostrar_imagen():
    return send_file("static/imagen_generada.png", mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
