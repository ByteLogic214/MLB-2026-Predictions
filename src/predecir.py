"""
predecir.py - Script legacy redirigido.
Este script ahora delega a main.py para evitar duplicación de código
y problemas de deduplicación.

Si ejecutas 'python predecir.py', ejecutará main.py.
"""
import sys
import os

if __name__ == '__main__':
    print('⚠️ redirigiendo a main.py para evitar duplicación...')
    print('    Ejecuta: python main.py')
    print()

    from main import run
    run()
