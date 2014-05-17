# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 18:45:03 2013

@author: urielsandoval
"""

# Se importan las librerías necesarias

import sys



from numpy import array, zeros, ones, pi, exp, r_, dot, diagflat
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from scipy.sparse.linalg import spsolve



from sistemas import sistemas



def cambiar_a_pu(nodos, MVA_a_pu =  True):
    # Si se desea cambiarde MVAs a pu 
    if MVA_a_pu:
        for llave in ('Pgen', 'Pcar', 'Qgen', 'Qcar'):
            nodos[llave] /= nodos['MVAbase']
        nodos['Angulo'] *= pi/180
    # Si no se regresa de pu a MVAs
    else:
        for llave in ('Pgen', 'Pcar', 'Qgen', 'Qcar'):
            nodos[llave] *= nodos['MVAbase']
        nodos['Angulo'] /= pi/180
    
    return nodos
    
    
def crea_ybus(nodos, lineas):
    """Crea una matriz dispersa Ybus. También regresa las matrices Yp y Yq y 
    las matrices de conectividad Cp y Cq.
    Estas últimas son usadas para el cálculo de flujo de potencia en cada 
    elemento
    """
    # Número de nodos
    m = len(nodos['Nodo'])
    # Número de líneas
    nl = len(lineas['NodoEnvio'])
    # Se hacen alias a los datos del sistema para manejarlos de forma
    # compacta
    # Resistencia
    R = lineas['R']
    # Reactancia
    X = lineas['X']
    # Susceptancia
    B = lineas['B/2']
    # Protección al usuario
    if not 'Tap' in lineas:
        lineas['Tap'] = ones(nl)
    
    if not 'Angulo' in lineas:
        lineas['Angulo'] = zeros(nl)
        
    if not 'Activo' in lineas:
        lineas['Activo'] = ones(nl)
    
    # Angulo de defase    
    Angulo = lineas['Angulo']
    # Tap de transformadores
    Tap = array(lineas['Tap'], dtype=complex)
    # Estatus de las lineas
    Activo = lineas['Activo']
    
    # Nodos de envio
    NP = lineas['NodoEnvio']
    # Nodos de llegada
    NQ = lineas['NodoLlegada']
    
    # Admitancia serie
    Ys = Activo / (R + 1j*X)
    # Susceptancia de carga
    Bc = Activo * B
    # Modificando al tap complejo
    Tap *= exp(1j*Angulo)
    # Admitancias de linea
    Yqq = Ys + 1j*Bc
    Ypp = Yqq/(Tap*Tap.conj())
    Ypq = -Ys/Tap.conj()
    Yqp = -Ys/Tap

    # Matriz de conectividad
    Cp = csr_matrix((ones(nl), (range(nl), NP)), (nl, m))  
    Cq = csr_matrix((ones(nl), (range(nl), NQ)), (nl, m))
    
    indices = range(nl) * 2
    # Matriz de admitancias de los nodos de envío
    Yp = csr_matrix((r_[Ypp, Ypq], (indices, r_[NP, NQ])), (nl, m))
    # Matriz de admitancias de los nodos de lllegada
    Yq = csr_matrix((r_[Yqp, Yqq], (indices, r_[NP, NQ])), (nl, m))
    
    # Formacion final de Ybus
    Ybus = Cp.T * Yp + Cq.T * Yq
    
    # Las matrices de nodos de envío y de llegada se ocupan
    # para calcular el flujo a través de cada elemento de
    # transmisión
    return Ybus, Cp, Cq, Yp, Yq
    
def actualizar(nodos, Ybus):
    nodos['Vbus'] = nodos['Voltaje'] * exp(1j * nodos['Angulo'])
    nodos['Ibus'] = Ybus * nodos['Vbus']
    nodos['Sbus'] = dot(diagflat(nodos['Vbus'].conj()), nodos['Ibus'])
    
    return nodos
    
def Fx(nodos):
    """ Función para obtener las ecuaciones de balance de potencia
    nodal
    """
    m = nodos['num']
    ns = nodos['ns']
    npv = nodos['npv']    
    
    delta = zeros((m, 2))
    
    # Desbalance de potencia activa
    delta[:, 0] = nodos['Pgen'] - nodos['Pcar'] - nodos['Sbus'].real
    # Desbalance de potencia reactiva    
    delta[:, 1] = nodos['Qgen'] - nodos['Qcar'] + nodos['Sbus'].imag
    
    # Se elimina el desbalance de potencia para el nodo slack
    # tanto en potencia activa como en reactiva
    delta[ns, :] = 0
    # Si se tienen nodos PV se elimina el desbalance de potencia reactiva
    # en estos
    if npv.shape[0] >= 1:
        delta[npv, 1] = 0
    
    return delta
    
def Jx(nodos, Ybus):
    """ Función para obtener la matriz jacobiana utilizanco transformaciones
    singulares.
    Para más información de este procedimiento consultar el apéndice de:


    """
    
    Ibus = nodos['Ibus']
    Vbus = nodos['Vbus']
    # Nodo slack    
    ns = nodos['ns']
    # Número de nodos
    m = nodos['num']
    # Nodos PV
    npv = nodos['npv']
    # Un vector auxiliar de indices
    indices = range(m)
    # Crear una matriz diagonal 
    diagV = csr_matrix((Vbus, (indices, indices)), (m, m))
    diagI = csr_matrix((Ibus, (indices, indices)), (m, m))
        
    E = Vbus / abs (Vbus)
    diagE = csr_matrix((E, (indices, indices)), (m, m))
    # Se calculan las derivadas parciales
    Sbus_ang = 1j * diagV.conj() * (Ybus * diagV - diagI)    
    Sbus_mag = diagV.conj() * Ybus * diagE + diagI*diagE.conj()
     
    jac = vstack([hstack([-Sbus_ang.real, -Sbus_mag.real]),
                 hstack([Sbus_ang.imag, Sbus_mag.imag])], format='lil')
    
    jac[:, ns] = 0
    jac[ns, :] = 0
    jac[:, ns+m] = 0
    jac[ns+m, :] = 0
    jac[ns, ns] = 1
    jac[ns+m, ns+m]= 1
    
    # Si se tienen nodos PV se elimina el desbalance de potencia reactiva
    if npv.shape[0] >= 1:
        jac[npv+m, :] = 0
        jac[:, npv+m] = 0
        jac[npv+m, npv+m] = 1
        
    return jac
    
def solucion_final(nodos, lineas, Ybus, Cp, Cq, Yp, Yq):
    """ Calcula la solución final, la potencia generada en nodos PV
    y los flujos a través de líneas
    """
    
    ns = nodos['ns']
    npv = nodos['npv']
    nl = lineas['num']
    indices = range(nl)
    Vbus = nodos['Vbus']
    Sbus = nodos['Sbus']
    MVAbase = nodos['MVAbase']
    
    # Voltaje en los nodos de envio
    Vp = Cp * Vbus
    # Voltaje en los nodos de llegada
    Vq = Cq * Vbus
    if npv.shape[0] >= 1:
        # Se actuliza generación de potencia reactiva en los nodos PV
        nodos['Qgen'][npv] = -Sbus.imag[npv] + nodos['Qcar'][npv]
    # Se actualizan potencias en el nodo Slack
    nodos['Pgen'][ns] = Sbus.real[ns] + nodos['Pcar'][ns]
    nodos['Qgen'][ns] = -Sbus.imag[ns] + nodos['Qcar'][ns]
    # Se encuentra la potencia de pérdidas en las líneas
    # Potencia que sale de los nodos de envío
    Sp = csr_matrix((Vp.conj(), (indices, indices))) * Yp * Vbus * MVAbase
    # Potencia que sale de los nodos de llegada
    Sq = csr_matrix((Vq.conj(), (indices, indices))) * Yq * Vbus * MVAbase
    # Se calcula la potencia disipada por cada elemento (perdidas)
    Sper = Sp + Sq
    lineas['P_envio'] = Sp.real
    lineas['Q_envio'] = Sp.imag
    lineas['P_llegada'] = Sq.real
    lineas['Q_llegada'] = Sq.imag
    lineas['P_perdidas'] = Sper.real
    lineas['Q_perdidas'] = Sper.imag

    
    
    return nodos, lineas
    
    

def imprime(nodos, lineas, dec = 3):
    """ Función para imprimir los resultados
    """
    # Se despliegan resultados por salida standard    
    desp = sys.stdout.write
    desp('\n===================================================================')
    desp('\n|                               NODOS                             |')
    desp('\n===================================================================')
    desp('\nNodo| Voltaje  |  Angulo  |    Generacion   |       Carga     ') 
    desp('\n    |          | (grados) |   W    |   VAR  |   W    |  VAR   ')
    fmt1 = '\n{:<4} {:<10} {:<10}'
    for p in range(len(nodos['Nodo'])):

            
        desp(fmt1.format(nodos['Nodo'][p], round(nodos['Voltaje'][p], 4), round(nodos['Angulo'][p], 4) ))
        
        if nodos['Pgen'][p] == 0:
            desp(' {:^8}'.format('--'))
        else:
            desp(' {: ^8}'.format(round(nodos['Pgen'][p], 3)))
        if nodos['Qgen'][p] == 0:
            desp(' {:^8}'.format('--'))
        else:
            desp(' {: ^8}'.format(round(nodos['Qgen'][p], 3)))
        if nodos['Pcar'][p] == 0:
            desp(' {:^8}'.format('--'))
        else:
            desp(' {: ^8}'.format(round(nodos['Pcar'][p], 3)))
        if nodos['Qcar'][p] == 0:
            desp(' {:^8}'.format('--'))
        else:
            desp(' {:^8}'.format(round(nodos['Qcar'][p], 3)))

    fmt2 = '\n\nTotales                     {:^8} {:^8} {:^8} {:^8}'
    desp(fmt2.format(round(sum(nodos['Pgen']), 3), round(sum(nodos['Qgen']), 3), round(sum(nodos['Pcar'])  , 3), round(sum(nodos['Qcar']), 3)))
    desp('\n\n')

    desp('\n================================================================================')
    desp('\n|                          LINEAS Y TRANSFORMADORES                            |')
    desp('\n================================================================================')
    desp('\nLinea|  Nodo   |  Nodo   | Inyección envío |Inyección llegada|Pérdidas (I^2 * Z)') 
    desp('\n  #  |  envío  | llegada |   W    |   VAR  |   W    |   VAR  |    W   |  VAR   |')
    fmt3 = '\n{:^5} {:^9} {:^9} {: ^8} {: ^8} {: ^8} {: ^8} {: ^8} {: ^8}'
    fmt4 = '\n\nTotales                                                       {: ^8} {: ^8} '
    
    for t in range(len(lineas['NodoEnvio'])):
        p = lineas['NodoEnvio'][t]
        q = lineas['NodoLlegada'][t]
        l = t+1

        Ppq = lineas['P_envio'][t]
        Pqp = lineas['P_llegada'][t]
        Qpq = lineas['Q_envio'][t]
        Qqp = lineas['Q_llegada'][t]
        Pper = lineas['P_perdidas'][t]
        Qper = lineas['Q_perdidas'][t]

        
        desp(fmt3.format(l, p, q, round(Ppq, 3), round(-Qpq, 3), round(Pqp, 3), round(-Qqp, 3), round(Pper, 3), round(Qper, 3)))                

    
    desp(fmt4.format(round(sum(lineas['P_perdidas']), 3), round(-sum(lineas['Q_perdidas']), 3)))
    desp('\n')
    
def newton(nodos, lineas):
    """ Método de Newton-Raphson para el cálculo de estado periódico en una 
    red eléctrica de potencia.
    """
    
    # Calcular la matriz Ybus
    Ybus, Cp, Cq, Yp, Yq = crea_ybus(nodos, lineas)
    # Se cambian los valores a p.u.    
    nodos = cambiar_a_pu(nodos, True)
    # Toolerancia a la convergencia
    tol = 1e-12
    # Número máximo de iteraciones
    maxIte = 6
    # Se inicializa el contador de iteraciones
    ite = 0 
    # Número total de nodos
    m = nodos['num']
    # Inicia el ciclo iteratvo
    while ite <= maxIte:
        # Actualian voltajes a forma compleja, y se calcula
        # Ibus y Sbus
        nodos = actualizar(nodos, Ybus)
        # Se calcula el desbalance de potencia
        delta = Fx(nodos)
        # Máximo error
        error = np.max(np.abs(delta))
        if error <= tol:
            print 'Solución encontrada'
            break
        ite += 1
        # Se construye la matriz jacobiana
        jac = Jx(nodos, Ybus)
        # Se concatenan en forma de renglón la columna \Delta P y
        # \Delta Q
        B = r_[delta[:, 0], delta[:, 1]]
        # Se resuelve el sistema de ecuaciones lineales.
        sol = -spsolve(jac.tocsr(), B)
        # Actualizamos angulos y magnitudes
        #nodos['Angulo'] = nodos['Angulo'] + sol[:m]
        nodos['Angulo'] += sol[:m]
        nodos['Voltaje'] += sol[m:]

        
        
    # Calculamos las pérdidas y los flujos a través de las líneas
    nodos, lineas= solucion_final(nodos, lineas,Ybus, Cp, Cq, Yp, Yq)
    # Regresamos los valores a MVA y grados
    nodos = cambiar_a_pu(nodos, False)
    
    # Se despliegan los datos
    nodos['Nodo'] +=1
    lineas['NodoEnvio'] +=1
    lineas['NodoLlegada'] +=1
    imprime(nodos, lineas)

    nodos['Nodo'] -=1
    lineas['NodoEnvio'] -=1
    lineas['NodoLlegada'] -=1
    return nodos, lineas
    
        
def main():
    """ Función main de prueba
    """
    
    nodos, lineas = sistemas('5_nodos')
    nodos, lineas = newton(nodos, lineas)
    nodos['Pcar'] *= 1.1
    nodos['Qcar'] *= 1.1
    nodos, lineas = newton(nodos, lineas)
    

if __name__ == '__main__':
    main()

