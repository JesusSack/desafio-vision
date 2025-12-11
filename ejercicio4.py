import time
from datetime import datetime

class AgenteBase:
    def __init__(self, nombre, rol):
        self.nombre = nombre
        self.rol = rol
        self.memoria = []

    def reportar(self):
        return self.memoria[-1] if self.memoria else None

class AgenteVision(AgenteBase):
    def __init__(self):
        super().__init__("VisioBot", "Percepción")

    def analizar(self, contexto):
        timestamp = datetime.now().isoformat()
        datos = {
            "ts": timestamp,
            "zona": "maquinaria",
            "objetos": ["taladro"]
        }
        
        if contexto == "SEGURO":
            datos["objetos"].extend(["casco", "guantes"])
            
        self.memoria.append(datos)
        return datos

class AgenteSeguridad(AgenteBase):
    def __init__(self):
        super().__init__("InspectorBot", "Compliance")
        self.normas = {"maquinaria": ["casco", "guantes"]}

    def evaluar(self, percepcion):
        zona = percepcion["zona"]
        detectados = percepcion["objetos"]
        requeridos = self.normas.get(zona, [])
        
        faltantes = [i for i in requeridos if i not in detectados]
        
        resultado = {
            "agente": self.nombre,
            "estado": "ALERTA" if faltantes else "OK",
            "detalle": f"Falta {faltantes}" if faltantes else "Cumplimiento verificado"
        }
        
        self.memoria.append(resultado)
        return resultado

def main():
    ojo = AgenteVision()
    inspector = AgenteSeguridad()
    
    escenarios_prueba = ["PELIGRO", "SEGURO"]
    
    print("--- SISTEMA MULTI-AGENTE INICIADO ---")
    
    for caso in escenarios_prueba:
        print(f"\n>>> Simulando Escenario: {caso}")
        
        datos_visuaes = ojo.analizar(caso)
        print(f"[{ojo.nombre}] Detectado: {datos_visuaes['objetos']}")
        
        informe = inspector.evaluar(datos_visuaes)
        print(f"[{inspector.nombre}] Dictamen: {informe['estado']} - {informe['detalle']}")

if __name__ == "__main__":
    main()