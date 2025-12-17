#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 22:27:25 2025

@author: isaias-gl
"""

# smart_coder.py - Versión mejorada
import subprocess
import json
import tempfile
import os
from datetime import datetime

class SpyderAIAssistant:
    def __init__(self, model="deepseek-coder:6.7b"):
        self.model = model
        self.history = []
        
    def code_completion(self, partial_code, context=""):
        """Completa código parcial"""
        prompt = f"""Completa este código Python:
        
        Contexto: {context}
        
        Código incompleto:
        ```python
        {partial_code}
        ```
        
        Solo devuelve el código completado.
        """
        return self._ollama_query(prompt)
    
    def debug_code(self, error_message, code_snippet):
        """Ayuda a depurar errores"""
        prompt = f"""Debug este error en Python:
        
        Error: {error_message}
        
        Código relacionado:
        ```python
        {code_snippet}
        ```
        
        Explica la causa y sugiere solución.
        """
        return self._ollama_query(prompt)
    
    def optimize_code(self, code, metric="performance"):
        """Optimiza código (performance/memoria/legibilidad)"""
        prompt = f"""Optimiza este código Python para {metric}:
        
        ```python
        {code}
        ```
        
        Muestra:
        1. Código optimizado
        2. Explicación de cambios
        3. Posible mejora en %
        """
        return self._ollama_query(prompt)
    
    def create_test(self, code):
        """Genera tests para código"""
        prompt = f"""Crea tests unitarios para:
        
        ```python
        {code}
        ```
        
        Usa pytest. Incluye casos edge.
        """
        return self._ollama_query(prompt)
    
    def _ollama_query(self, prompt, temperature=0.2):
        """Consulta Ollama con mejor manejo de errores"""
        try:
            # Guardar en archivo temporal
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                data = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature}
                }
                json.dump(data, f)
                temp_file = f.name
            
            # Ejecutar consulta
            cmd = f"curl -s -X POST http://localhost:11434/api/generate -H 'Content-Type: application/json' -d @{temp_file}"
            
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=True, 
                text=True, 
                timeout=45
            )
            
            # Limpiar archivo temporal
            os.unlink(temp_file)
            
            if result.returncode == 0:
                response_data = json.loads(result.stdout)
                response = response_data.get("response", "")
                
                # Guardar en historial
                self.history.append({
                    "timestamp": datetime.now().isoformat(),
                    "prompt": prompt[:100] + "...",
                    "response": response[:200] + "..."
                })
                
                return response
            else:
                # Intentar diagnóstico
                if "Connection refused" in result.stderr:
                    return "Error: Ollama no está corriendo. Ejecuta 'ollama serve &' en terminal."
                return f"Error: {result.stderr[:200]}"
                
        except subprocess.TimeoutExpired:
            return "Error: Timeout. El modelo está tardando demasiado."
        except Exception as e:
            return f"Error inesperado: {str(e)}"
    
    def get_available_models(self):
        """Lista modelos disponibles localmente"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )
            return result.stdout
        except:
            return "No se pudo obtener lista de modelos"

# ===== FUNCIONES DE CONVENIENCIA =====
def ai_generate(description, model="deepseek-coder:6.7b"):
    """Función rápida para generar código"""
    assistant = SpyderAIAssistant(model)
    return assistant.code_completion("", description)

def ai_review(code, model="deepseek-coder:6.7b"):
    """Función rápida para revisar código"""
    assistant = SpyderAIAssistant(model)
    return assistant.optimize_code(code, "legibilidad")

def ai_debug(error, code, model="deepseek-coder:6.7b"):
    """Función rápida para debug"""
    assistant = SpyderAIAssistant(model)
    return assistant.debug_code(error, code)