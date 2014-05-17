# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 18:59:04 2013

@author: urielsandoval
@email: uriel_sandoval@ieee.org
"""

from numpy import array, zeros, ones
import sys


# Función para cargar los sistemas a analizar
# Se regresa un diccionario con los datos del sistema

def sistemas(nombre):
    """Función para cargar los sistemas a analizar
    Se regresa un diccionario con los datos del sistema"""

    if nombre == '3_nodos':
            nodos = {'Nodo': array([0, 1, 2]),
                     'Voltaje': array([1.05, 1.03, 1]),
                     'Angulo': zeros(3),
                     'Pgen': array([0, 20., 0]),
                     'Pcar': array([0, 50., 60]),
                     'Qgen': zeros(3),
                     'Qcar': array([0, 20., 25])}
            nodos['ns'] = 0 # Nodo slack
            nodos['npv'] = array([1])
            nodos['MVAbase'] = 100.
            nodos['num'] = len(nodos['Nodo'])
            
            lineas ={'R': array([0.08, 0.02, 0.06]),
                     'X': array([0.24, 0.06, 0.18]),
                     'B/2': zeros(3),
                     'NodoEnvio': array([0, 0 ,1]),
                     'NodoLlegada': array([1, 2, 2]),
                     'Tap': ones(3),
                     'Angulo': zeros(3)}
            lineas['num'] = len(lineas['NodoEnvio'])
            
    elif nombre == '5_nodos':
            nodos = {'Nodo': array([0, 1, 2, 3, 4]),
                     'Voltaje': array([1.06, 1, 1, 1, 1]),
                     'Angulo': zeros(5),
                     'Pgen': array([0, 40., 0, 0, 0]),
                     'Pcar': array([0, 20., 45., 40., 60.]),
                     'Qgen': array([0, 30., 0, 0, 0]),
                     'Qcar': array([0, 10., 15., 5., 10.]),
                     'ns': 0,
                     'npv': array([1]),
                     'MVAbase': 100.}
            nodos['num'] = len(nodos['Nodo'])
            
            lineas ={'R': array([0.02, 0.08, 0.06, 0.06, 0.04, 0.01, 0.08]),
                     'X': array([0.06, 0.24, 0.18, 0.18, 0.12, 0.03, 0.24]),
                     'B/2': array([0.03, 0.025, 0.02, 0.02, 0.015, 0.01, 0.025]),
                     'NodoEnvio': array([0, 0 , 1, 1, 1, 2, 3]),
                     'NodoLlegada': array([1, 2, 2, 3, 4, 3, 4]),
                     'Tap': ones(7),
                     'Angulo': zeros(7)}
            lineas['num'] = len(lineas['NodoEnvio'])
    else:
        sys.exit('No existe tal sistema')
    
    return nodos, lineas