import os
from dotenv import load_dotenv
from fastapi import FastAPI,Body
from mistralai.client import MistralClient
from fastapi.middleware.cors import CORSMiddleware
app= FastAPI()
app.title="Clasificador de intenciones con llm"
load_dotenv()
api_key_mistral= os.getenv('API_KEY_MISTRAL')
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = MistralClient(api_key=api_key_mistral)

tramites = [
        "Imprimir cupon de pago",
        "Abonar a la factura",
        "Separar servicio energia, agua o gas",
        "Sacar certificado servicios prepago",
        "Tramites y turnos"
    ]

examples = [
        {"user": "sacar la factura", "tramite": "Imprimir cupon de pago"},
        {"user": "Imprimir factura", "tramite": "Imprimir cupon de pago"},
        {"user": "Sacar cupon de pago", "tramite": "Imprimir cupon de pago"},
        {"user": "Sacar recibo de servicios", "tramite": "Imprimir cupon de pago"},
        {"user": "Sacar copia de los servicios", "tramite": "Imprimir cupon de pago"},
        {"user": "Pagar la factura", "tramite": "Abonar a la factura"},
        {"user": "abonar a una deuda de los servicios", "tramite": "Abonar a la factura"},
        {"user": "Abonar a la factura de servicios", "tramite": "Abonar a la factura"},
        {"user": "Imprimir factura para abonar", "tramite": "Abonar a la factura"},
        {"user": "Separar servicio de energia", "tramite": "Separar servicio energia, agua o gas"},
        {"user": "Sacar solo factura de energia", "tramite": "Separar servicio energia, agua o gas"},
        {"user": "Sacar solo factura de luz", "tramite": "Separar servicio energia, agua o gas"},
        {"user": "Sacar solo factura de agua", "tramite": "Separar servicio energia, agua o gas"},
        {"user": "Sacar solo factura de gas", "tramite": "Separar servicio energia, agua o gas"},
        {"user": "imprimir solo factura de energia", "tramite": "Separar servicio energia, agua o gas"},
        {"user": "imprimir solo factura de luz", "tramite": "Separar servicio energia, agua o gas"},
        {"user": "imprimir solo factura de agua", "tramite": "Separar servicio energia, agua o gas"},
        {"user": "imprimir solo factura de gas", "tramite": "Separar servicio energia, agua o gas"},
        {"user": "Necesito un certificado de servicios prepago", "tramite": "Sacar certificado servicios prepago"},
        {"user": "certificado de servicios prepago", "tramite": "Sacar certificado servicios prepago"},
        {"user": "Necesito información sobre un tramite", "tramite": "Tramites y turnos"},
        {"user": "Como hacer un tramite", "tramite": "Tramites y turnos"}
    ]

def detectar_inicio_kiosco_mistral(text):
    examples_text = "\n".join(
        [f"Pregunta: {ex['user']} Trámite: {ex['tramite']}" for ex in examples])
    message_text = [
        {"role": "system", "content": f"{examples_text}\n\nDada la siguiente pregunta, responde únicamente con el nombre del trámite correspondiente:"},
        {"role": "user", "content": text}
    ]
    completionMistral=client.chat(
        model="mistral-small-latest",
        messages=message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,       
)
    return completionMistral.choices[0].message.content

tramite_epm = [
        "Solicitud habilitación energia electrica",
        "Solicitud conexion del servicio de energia electrica",
        "Cambio de energia prepago a pospago",
        "Solicitar servicio de energía prepago",
        "Solicitar servicio de energía prepago",
        "Información de solicitudes de instalaciones",
        "Agua pospago",
        "Solicitud de conexión al servicio de alcantarillado",
        "Solicitud de supervision a las acometidas para la conexión de los servicios de acueducto y alcantarillado",
        "Solicitud de conexión al servicio de acueducto unicamente",
        "Agua prepago",
        "Cambio del titular del contrato de agua prepago",
        "Retiro agua prepago",
        "Facturación",
        "Epm a tu puerta",
        "Otras solicitudes"
    ]


examples_epm = [
        {"user": "¿Cómo solicitar la conexión del servicio de energía eléctrica si no tengo acomedida de energía?", "tramite_epm": "Solicitud habilitación energia electrica"},
        {"user": "¿Como solicitar el servicio de energía eléctrica?", "tramite_epm": "Solicitud conexion del servicio de energia electrica"},
        {"user": "¿Como cambiar de energia prepago a pospago?", "tramite_epm": "Cambio de energia prepago a pospago"},
        {"user": "¿Como tener energia prepago?", "tramite_epm": "Solicitar servicio de energía prepago"},
        {"user": "¿Como solicitar que me instalen energia prepago?", "tramite_epm": "Solicitar servicio de energía prepago"},
        {"user": "¿Como quitar la luz de una casa?", "tramite_epm": "Solicitar servicio de energía prepago"},
        {"user": "¿Como solicitar que le quiten la energia a una casa?", "tramite_epm": "Solicitar servicio de energía prepago"},
        {"user": "Información sobre una solicitud radicada sobre la instalación de servicios", "tramite_epm": "Información de solicitudes de instalaciones"},
        {"user": "Información sobre la instalación de servicios", "tramite_epm": "Información de solicitudes de instalaciones"},
        {"user": "Solicitud de conexion a los servicios de acueducto y alcantarillado", "tramite_epm": "Agua pospago"},
        {"user": "¿Como solicitar que me instalen el acueducto y alcantarillado?", "tramite_epm": "Agua pospago"},
        {"user": "¿Como solicitar que me instalen el agua y el alcatarillado?", "tramite_epm": "Agua pospago"},
        {"user": "¿Que hacer para que instalen el agua?", "tramite_epm": "Agua pospago"},
        {"user": "¿Como instalar el contador con un particular?", "tramite_epm": "Agua pospago"},
        {"user": "¿como instalar el contador del agua con epm?", "tramite_epm": "Agua pospago"},
        {"user": "Tengo acueducto, como solicito que epm me instalen alcantarillado", "tramite_epm": "Solicitud de conexión al servicio de alcantarillado"},
        {"user": "Tengo acueducto, puedo instalar el alcantarillado con un particular?", "tramite_epm": "Solicitud de conexión al servicio de alcantarillado"},
        {"user": "Necesito que supervicen la construcción del alcantarillado", "tramite_epm": "Solicitud de supervision a las acometidas para la conexión de los servicios de acueducto y alcantarillado"},
        {"user": "Como me hacen la visita para instalar el alcantarillado", "tramite_epm": "Solicitud de supervision a las acometidas para la conexión de los servicios de acueducto y alcantarillado"},
        {"user": "Como me hacen la visita para poner el agua", "tramite_epm": "Solicitud de supervision a las acometidas para la conexión de los servicios de acueducto y alcantarillado"},
        {"user": "¿Como solicitar que me instalen el acueducto?", "tramite_epm": "Solicitud de conexión al servicio de acueducto unicamente"},
        {"user": "¿Como solicitar que me instalen el agua?", "tramite_epm": "Solicitud de conexión al servicio de acueducto unicamente"},
        {"user": "¿Como solicitar que me instalen el acueducto solamente?", "tramite_epm": "Solicitud de conexión al servicio de acueducto unicamente"},
        {"user": "Soliciar servicio de agua prepago", "tramite_epm": "Agua prepago"},
        {"user": "Como solicitar el servicio de agua prepago?", "tramite_epm": "Agua prepago"},
        {"user": "Como tener agua prepago?", "tramite_epm": "Agua prepago"},
        {"user": "Como cambio el nombre que aparece en la factura de agua prepago", "tramite_epm": "Cambio del titular del contrato de agua prepago"},
        {"user": "Como pongo a mi nombre el contrato de agua prepago", "tramite_epm": "Cambio del titular del contrato de agua prepago"},
        {"user": "Como hago la solicitud para que me retiren el agua prepago", "tramite_epm": "Retiro agua prepago"},
        {"user": "como quitar el agua prepago", "tramite_epm": "Retiro agua prepago"},
        {"user": "Cambio nombre del cliente en la factura", "tramite_epm": "Facturación"},
        {"user": "¿Cómo puedo cambiar el nombre del cliente que aparece en la factura de servicios públicos? ", "tramite_epm": "Facturación"},
        {"user": "¿Cómo puedo cambiar el nombre que se imprime en la factura de servicios públicos? ", "tramite_epm": "Facturación"},
        {"user": "¿Cómo puedo pedir que no se imprima el dato del nombre que aparece en la factura de servicios públicos?", "tramite_epm": "Facturación"},
        {"user": "Necesito que vaya un tecnico a mi casa", "tramite_epm": "Epm a tu puerta"},
        {"user": "Necesito mantenimiento de la red de gas", "tramite_epm": "Epm a tu puerta"},
        {"user": "Radicados", "tramite_epm": "Otras solicitudes"},
        {"user": "Necesito radicar un documento", "tramite_epm": "Otras solicitudes"}
        
    ]

def detectar_tramites_epm_mistral(text):
    examples_text = "\n".join(
        [f"Pregunta: {ex['user']} Trámite: {ex['tramite_epm']}" for ex in examples_epm])
    message_text = [
        {"role": "system", "content": f"{examples_text}\n\nDada la siguiente pregunta, responde únicamente con el nombre del trámite correspondiente:"},
        {"role": "user", "content": text}
    ]
    completionMistral=client.chat(
        model="mistral-small-latest",
        messages=message_text,
        temperature=0.7,
        max_tokens=800,
        top_p=0.95,       
)
    return completionMistral.choices[0].message.content

@app.get("/inicio", tags=["inicio"])
def traer_inicio():
    return tramites

@app.get("/inicio-examples", tags=["inicio"])
def traer_inicio_examples():
    return examples

@app.get("/tramites", tags=["tramites"])
def traer_tramites():
    return tramite_epm

@app.get("/tramites-examples", tags=["tramites"])
def traer_tramites_epm():
    return examples_epm



@app.post('/inicio', tags=['generar'])
def clasificar_intencion_inicio( input: str = Body()):
    #tramites=aqui va la peticion a mongo para traer los almacenados en un tenant
    #examples=aqui los ejemplos que se tengan almacenados de cada tramite 
    intencion=detectar_inicio_kiosco_mistral(input)
    print('intencion:',intencion)
    return intencion

@app.get('/ejemplos', tags=['ejemplo'])
def ejemplo(input:str=Body(),tenant:str=Body()):
    tramite_ejemplo={}
    return ''


@app.post('/tramites', tags=['generar'])
def clasificar_intencion_tramites( input: str = Body()):
    intencion = detectar_tramites_epm_mistral(input)
    return intencion







@app.put('/inicio', tags=['insertar'])
def insertar_inicio_epm_mistral( input: str = Body(), tramite:str=Body()):
    tramites.append(tramite)
    print(tramites)
    examples.append({"user":input, "tramite":tramite})
    return examples


@app.put('/tramites', tags=['insertar'])
def insertar_example_epm_mistral( input: str = Body(), tramite:str=Body()):
    tramite_epm.append(tramite)
    print(tramite_epm)
    examples_epm.append({"user":input, "tramite_epm":tramite})
    return examples_epm


@app.delete('/inicio', tags=['delete'])
def eliminar_inicio_epm_mistral( tramite:str=Body()):
    if tramite in tramites:
        tramites.remove(tramite)
        print('tramite eliminado', tramites) 
    for example in examples:
        if example['tramite'] == tramite:
            examples.remove(example)
            print('example eliminado', examples)
    return {"message":"tramite eliminado de examples y tramites"}        

@app.delete('/tramites', tags=['delete'])
def eliminar_tramite_epm_mistral( tramite:str=Body()):
    if tramite in tramite_epm:
        tramite_epm.remove(tramite)
        print('tramite eliminado', tramites) 
    for example in examples_epm:
        if example['tramite_epm'] == tramite:
            examples_epm.remove(example)
            print('example eliminado', examples)
    return {"message":"tramite eliminado de examples_epm y tramites_epm"}   

    
    
