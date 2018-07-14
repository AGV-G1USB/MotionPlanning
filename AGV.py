import serial
import requests
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import math
import pygame as pg

vec = pg.math.Vector2

"""
COMMUNICATIONS FUNCTION
"""
def communication(pwm1, pwm2, direction1, direction2):
    pwm1 = int(pwm1)
    pwm2 = int(pwm2)
    com = "F " + str(pwm1) + " " + str(direction1) + " " + str(pwm2) + " " + str(direction2) + chr(13)
    print("COMUNICACIÓN: ", com)
    return com

"""
CLASS OF ELEMENTS AND CLASSIFICATION FUNCTION
"""
class cordinates:

    def __init__(self):
        self.x = []
        self.y = []
        self.cordinates = []
        self.color = []

    def addValue (self, x, y, cordinates, color):
        self.x.append(x)
        self.y.append(y)
        self.cordinates.append(cordinates)
        self.color.append(color)

    def addValueT (self, i, response):
        self.addValue(response[i][0][0], response[i][0][1], response[i][1], response[i][2])

def colorClassification(response):

    balls, obstacles, car = cordinates(), cordinates(), cordinates()

    for i in range(len(response)):
        
        if response[i][2] == "RED":
            balls.addValueT(i, response)

        if response[i][2] == "CYAN" or response[i][2] == "GREEN":
            obstacles.addValueT(i, response)

        if response[i][2] == "YELLOW" or response[i][2] == "BLUE":
            car.addValueT(i, response)

    return balls, obstacles, car

"""
CONTROL FUNCTIONS
"""

def remoteController(key, bluetoothS):

    if key == "w":

        pwm1 = 550
        pwm2 = 507
        direction1 = 0
        direction2 = 0        
        com = communication(pwm1, pwm2, direction1, direction2)
        print("Adelante")
        bluetoothS.write(com.encode())

    if key == "d":
        pwm1 = 500
        pwm2 = 357
        direction1 = 1
        direction2 = 0        
        com = communication(pwm1, pwm2, direction1, direction2)
        print("Derecha")
        bluetoothS.write(com.encode())

    if key == "s":
        pwm1 = 350
        pwm2 = 450
        direction1 = 1
        direction2 = 1        
        com = communication(pwm1, pwm2, direction1, direction2)
        print("Atras")
        bluetoothS.write(com.encode())

    if key == "a":
        pwm1 = 400
        pwm2 = 600
        direction1 = 0
        direction2 = 1        
        com = communication(pwm1, pwm2, direction1, direction2)
        print("Izquierda")
        bluetoothS.write(com.encode())

    if key == " ":
        pwm1 = 0
        pwm2 = 0
        direction1 = 0
        direction2 = 0        
        com = communication(pwm1, pwm2, direction1, direction2)
        print("Stop")
        bluetoothS.write(com.encode())

def carMedia(x, y):

    return ((x[1]-x[0])/2+x[0]), ((y[1]-y[0])/2+y[0]) 

def vector(x1, x2, y1, y2):

    return (x2 - x1) , (y2 - y1)

def angleError(x, y, xVec, yVec):

    return ((math.acos((xVec*x + yVec*y)/(math.sqrt((pow(xVec,2) + pow(yVec, 2))*(pow(x, 2) + pow(y, 2))))))*180)/math.pi

def angleControl(error, errorAnt):
    kp = 0.35
    kd = 0.7
    
    return (error*kp + abs(error-errorAnt)*kd)

def distControl(coordX, coordY, mediaX, mediaY):

    return math.sqrt(pow((coordX-mediaX), 2) + pow((coordY-mediaY), 2))

def direction(xOrientation, yOrientation, xVec, yVec):
    
    if xOrientation != 0 and yOrientation != 0:
        #1er cuadrante
        if xOrientation > 0 and yOrientation > 0:
            sentido = "derecha"
        #2do cuadrante
        if xOrientation < 0 and yOrientation > 0:
            sentido = "izquierda"
        #3er cuadrante
        if xOrientation < 0 and yOrientation < 0:
            sentido = "izquierda"
        #4to cuadrante
        if xOrientation > 0 and yOrientation < 0:
            sentido = "derecha"

    if xOrientation == 0:
        if yOrientation < 0:
            #1er cuadrante
            if xVec > 0 and yVec >= 0:
                sentido = "derecha"
            #2do cuadrante
            if xVec <= 0 and yVec > 0:
                sentido = "izquierda"
            #3er cuadrante
            if xVec < 0 and yVec <= 0:
                sentido = "izquierda"
            #4to cuadrante
            if xVec >= 0 and yVec < 0:
                sentido = "derecha"

        if yOrientation > 0:
            #1er cuadrante
            if xVec > 0 and yVec >= 0:
                sentido = "izquierda"
            #2do cuadrante
            if xVec <= 0 and yVec > 0:
                sentido = "derecha"
            #3er cuadrante
            if xVec < 0 and yVec <= 0:
                sentido = "derecha"
            #4to cuadrante
            if xVec >= 0 and yVec < 0:
                sentido = "izquierda"

    if yOrientation == 0:
        if xOrientation < 0:
            #1er cuadrante
            if xVec > 0 and yVec >= 0:
                sentido = "izquierda"
            #2do cuadrante
            if xVec <= 0 and yVec > 0:
                sentido = "izquierda"
            #3er cuadrante
            if xVec < 0 and yVec <= 0:
                sentido = "derecha"
            #4to cuadrante
            if xVec >= 0 and yVec < 0:
                sentido = "derecha"

        if xOrientation > 0:
            #1er cuadrante
            if xVec > 0 and yVec >= 0:
                sentido = "derecha"
            #2do cuadrante
            if xVec <= 0 and yVec > 0:
                sentido = "derecha"
            #3er cuadrante
            if xVec < 0 and yVec >= 0:
                sentido = "izquierda"
            #4to cuadrante
            if xVec >= 0 and yVec < 0:
                sentido = "izquierda"

    """
    if xOrientation == 0:

        if yOrientation > 0:
           
            if xVec > 0 and yVec > 0:
                sentido = "derecha"

            elif xVec < 0 and yVec > 0:
                sentido = "izquierda"

            elif xVec < 0 and yVec < 0:
                sentido = "izquierda"

            elif xVec > 0 and yVec < 0:
                sentido = "derecha"
        
        else:

            if xVec > 0 and yVec > 0:
                sentido = "izquierda"

            elif xVec < 0 and yVec > 0:
                sentido = "derecha"

            elif xVec < 0 and yVec < 0:
                sentido = "derecha"

            elif xVec > 0 and yVec < 0:
                sentido = "izquierda"

    elif yOrientation == 0:

        if xOrientation > 0:
           
            if xVec > 0 and yVec > 0:
                sentido = "derecha"

            elif xVec < 0 and yVec > 0:
                sentido = "derecha"

            elif xVec < 0 and yVec < 0:
                sentido = "izquierda"

            elif xVec > 0 and yVec < 0:
                sentido = "izquierda"
        
        else:

            if xVec > 0 and yVec > 0:
                sentido = "izquierda"

            elif xVec < 0 and yVec > 0:
                sentido = "izquierda"

            elif xVec < 0 and yVec < 0:
                sentido = "derecha"

            elif xVec > 0 and yVec < 0:
                sentido = "derecha"

    elif xOrientation > 0 and yOrientation > 0:
        sentido = "derecha"

    elif xOrientation < 0 and yOrientation > 0:
        sentido = "izquierda"

    elif xOrientation < 0 and yOrientation < 0:
        sentido = "derecha"
        
    elif xOrientation > 0 and yOrientation < 0:
        sentido = "izquierda"
    """
    return sentido

def crossProduct(xVec, yVec, mediaXVec, mediaYVec, errorTetha):

    return abs(math.sqrt(pow(xVec, 2) + pow(yVec, 2)))*abs(math.sqrt(pow(mediaXVec, 2) + pow(mediaYVec, 2)))*math.sin((errorTetha*math.pi)/180)

def control(coordX, coordY, car, mediaX, mediaY, errorAnt, balls, obstacles, bluetoothS):

    # B, xb, yb = comparationColors(balls , obstacles , car)

    xVec, yVec = vector(car.x[0], car.x[1], car.y[0], car.y[1])
    vCar = vec(xVec, yVec)

    mediaXVec, mediaYVec = vector(car.x[0], coordX, car.y[0], coordY)
    vPoint = vec(mediaXVec, mediaYVec) 

    errorTetha = vCar.angle_to(vPoint)
    print("ANGULO VIEJO: ", errorTetha)
    if errorTetha > 100:
        errorTetha = 100
    elif errorTetha < -100:
        errorTetha = -100
    print("ANGULO NUEVO: ", errorTetha)
    pwmDif = angleControl(errorTetha, errorAnt)
    print("pwmDif: ", pwmDif)

    if errorTetha < 7 and errorTetha > -7:

        pwm1 = 520
        pwm2 = 500
        direction1 = 0
        direction2 = 0
        distance = distControl(coordX, coordY, mediaX, mediaY)
        print("DISTANCIA: ", distance)

        bit = 1
        com = communication(pwm1, pwm2, direction1, direction2)
        print("Adelante")

        bluetoothS.write(com.encode())

    else:
        
        bit = 0

        if errorTetha >= 10:
                
            # pwm1 = 400
            # pwm2 = 600
            pwm1 = 220
            pwm2 = 700
            pwm1 = pwm1 + pwmDif
            pwm2 = pwm2 - pwmDif
            direction1 = 0
            direction2 = 1        
            com = communication(pwm1, pwm2, direction1, direction2)
            print("Izquierda")
            bluetoothS.write(com.encode())
            
        elif errorTetha <= -10:

            # pwm1 = 500
            # pwm2 = 357
            pwm1 = 720
            pwm2 = 230
            pwm1 = pwm1 - pwmDif
            pwm2 = pwm2 + pwmDif
            direction1 = 1
            direction2 = 0        
            com = communication(pwm1, pwm2, direction1, direction2)
            print("Derecha")
            bluetoothS.write(com.encode())

    return bit

"""
A_STAR FUNCTIONS
"""

show_animation = False

class Node:

    def __init__(self, x, y, cost, pind):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind
        " Agregar atributo del potencial "

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.pind)

def calc_fianl_path(ngoal, closedset, reso):
    # generate final course

    # La construcción del camino inicia en el nodo final
    rx, ry = [ngoal.x * reso], [ngoal.y * reso]

    # Indicador que denota los nodos de inicio y llegada
    pind = ngoal.pind

    # Hasta que no se anexe el nodo de origen, se extraen elementos del conjunto
    # El indicador del nodo de origen es siempre -1.
    while pind != -1:

        # Extrae un nodo del conjunto de nodos evaluados
        n = closedset[pind]

        # Anexa el nuevo nodo a la lista de nodos del camino
        rx.append(n.x * reso)
        ry.append(n.y * reso)
        pind = n.pind

    return rx, ry

def a_star_planning(sx, sy, gx, gy, ox, oy, reso, rr):

    # Inicialización del nodo de partida
    nstart = Node(round(sx / reso), round(sy / reso), 0.0, -1)

    # Inicialización del nodo de llegada
    ngoal = Node(round(gx / reso), round(gy / reso), 0.0, -1)

    # Se ajustan las posiciones de los obstáculos a la resolución del mapa
    ox = [iox / reso for iox in ox]
    oy = [ioy / reso for ioy in oy]

    # Se define la región del mapa sobre la cual existen obstáculos
    obmap, minx, miny, maxx, maxy, xw, yw = calc_obstacle_map(ox, oy, reso, rr)

    # Modelo base para G (ver descripción de la función)
    motion = get_motion_model()

    # Inicialización de las listas abierta y cerrada (diccionario vacío)
    openset, closedset = dict(), dict()
    openset[calc_index(nstart, xw, minx, miny)] = nstart # El nodo de partida se anexa a la lista abierta (Paso 0)

 
    while 1:
        # Se extrae el primer nodo de la lista abierta y se define como el nodo actual(Paso 1)
        c_id = min(
            openset, key=lambda o: openset[o].cost + calc_h(ngoal, openset[o].x, openset[o].y))
        current = openset[c_id]
        #  print("current", current)

        # Mostrar animación (NO ES PARTE DEL ALGORITMO)
        if show_animation:
            plt.plot(current.x * reso, current.y * reso, "xc")
            if len(closedset.keys()) % 10 == 0:
                plt.pause(0.001)
         

        # Paso 3, caso a (Se ha llegado a la meta, finaliza la construcción de la trayectoria)
        if current.x == ngoal.x and current.y == ngoal.y:
            print("Find goal")
            ngoal.pind = current.pind
            ngoal.cost = current.cost
            break

        # Se extrae el primer elemento de la lista abierta y se inserta en la lista cerrada (Paso 2)
        del openset[c_id]
        closedset[c_id] = current

        # Se asigna al nodo el modelo de movimiento
        for i in range(len(motion)):
            node = Node(current.x + motion[i][0], current.y + motion[i][1],
                        current.cost + motion[i][2], c_id)
            n_id = calc_index(node, xw, minx, miny)

            # Paso 3, caso b (el nodo es un obstáculo)
            if not verify_node(node, obmap, minx, miny, maxx, maxy):
                continue

            # Paso 3, caso c (el nodo ya fue evaluado)
            if n_id in closedset:
                continue

            # Paso 3, caso d (el nodo ya está en la lisa abierta y se recalculan F,G y H)
            if n_id in openset:
                if openset[n_id].cost > node.cost:
                    openset[n_id].cost = node.cost
                    openset[n_id].pind = c_id
            else:
                openset[n_id] = node

    # Se reconstruye la trayectoria final una vez hallado el nodo de meta
    rx, ry = calc_fianl_path(ngoal, closedset, reso)

    return rx, ry

def calc_h(ngoal, x, y):
    w = 10.0  # weight of heuristic
    d = w * math.sqrt((ngoal.x - x)**2 + (ngoal.y - y)**2)
    return d

def verify_node(node, obmap, minx, miny, maxx, maxy):

    if node.x < minx:
        return False
    elif node.y < miny:
        return False
    elif node.x >= maxx:
        return False
    elif node.y >= maxy:
        return False

    if obmap[node.x][node.y]:
        return False

    return True

def calc_obstacle_map(ox, oy, reso, vr):

    minx = round(min(ox))
    miny = round(min(oy))
    maxx = round(max(ox))
    maxy = round(max(oy))
    #  print("minx:", minx)
    #  print("miny:", miny)
    #  print("maxx:", maxx)
    #  print("maxy:", maxy)

    xwidth = round(maxx - minx)
    ywidth = round(maxy - miny)
    #  print("xwidth:", xwidth)
    #  print("ywidth:", ywidth)

    # obstacle map generation

    
    # Genera un mapa de xwidth x ywidth de ceros
    obmap = [[False for i in range(xwidth)] for i in range(ywidth)]

    
    for ix in range(xwidth):
        x = ix + minx
        for iy in range(ywidth):
            y = iy + miny
            #  print(x, y)
            for iox, ioy in zip(ox, oy):
                # Condición para evitar colisión
                d = math.sqrt((iox - x)**2 + (ioy - y)**2)
                if d <= vr / reso:
                    obmap[ix][iy] = True
                    break

    return obmap, minx, miny, maxx, maxy, xwidth, ywidth

def calc_index(node, xwidth, xmin, ymin):
    return (node.y - ymin) * xwidth + (node.x - xmin)

def get_motion_model():
    # dx, dy, cost
    motion = [[1, 0, 1],
              [0, 1, 1],
              [-1, 0, 1],
              [0, -1, 1],
              [-1, -1, math.sqrt(2)],
              [-1, 1, math.sqrt(2)],
              [1, -1, math.sqrt(2)],
              [1, 1, math.sqrt(2)]]

    return motion

"""
OBSTACLES, BALLS FUNCTIONS
"""
def semiSquares(obstacles , balls):

    ox, oy = [], []

    for k in range(len(balls.x)):
        
        dist = int(100*balls.cordinates[k])
        x1 = balls.x[k] - 2*dist
        if(x1 < 0):
            x1 = 0
        x2 = balls.x[k] + 2*dist
        if(x2 > 100):
            x2 = 100
        y1 = balls.y[k] - dist
        if(x1 < 0):
            y1 = 0
        y2 = balls.y[k] + dist
        if(y2 > 100):
            y2 = 100
        x = x2 - x1
        y = y2 - y1
        
        # Recuadros de los objetivos
        for i in range((int)(y)):
            ox.append(x1)
            oy.append(y1 + i)
        for i in range((int)(y) + 1):
            ox.append(x2)
            oy.append(y1 + i)
        for i in range((int)(x)):
            ox.append(x1 + i)
            oy.append(y1)
    
    for j in range(len(obstacles.x)):    

        dist1 = int(100*obstacles.cordinates[j]) + 2
        x3 = obstacles.x[j] - dist1
        x4 = obstacles.x[j] + dist1
        y3 = obstacles.y[j] - dist1
        y4 = obstacles.y[j] + dist1
        x_1 = x4 - x3
        y_1 = y4 - y3

        # Recuadros de los obstaculos
        for i in range((int)(y_1)):
            ox.append(x3)
            oy.append(y3 + i)
        for i in range((int)(y_1) + 1):
            ox.append(x4)
            oy.append(y3 + i)
        for i in range((int)(x_1)):
            ox.append(x3 + i)
            oy.append(y3)
        for i in range((int)(x_1)):
            ox.append(x3 + i)
            oy.append(y4)
    return ox, oy

def squares(obstacles):

    ox, oy = [], []

    for k in range(len(obstacles.x)):
        
        dist = int(100*obstacles.cordinates[k])

        x1 = obstacles.x[k] - 2*dist
        x2 = obstacles.x[k] + 2*dist
        y1 = obstacles.y[k] - 2*dist
        y2 = obstacles.y[k] + 2*dist
        x = x2 - x1
        y = y2 - y1
        
        for i in range((int)(y)):
            ox.append(x1)
            oy.append(y1 + i)
        for i in range((int)(y) + 1):
            ox.append(x2)
            oy.append(y1 + i)
        for i in range((int)(x)):
            ox.append(x1 + i)
            oy.append(y1)
        for i in range((int)(x)):
            ox.append(x1 + i)
            oy.append(y2)

    return ox, oy

def comparationColors(balls , obstacles , car):

    c = cordinates()

    for k in range(len(car.color)):

        if car.color[k] == "YELLOW":

            c.x.append(car.x[k])
            c.y.append(car.y[k])
            c.cordinates.append(car.cordinates[k])
            c.color.append(car.color[k])

            if(k == 0):
                g = 1
            elif(k==1):
                g = 0

    if(g == 1):

        position = distanceStart_Target (c, car.x[1], car.y[1])

    elif(g == 0):

        position = distanceStart_Target (c, car.x[0], car.y[0])
        
    return g, c.x[position], c.y[position]

def comparationLists(balls, ballsAnt, obstacles, obstaclesAnt):

    for i in range(len(balls.x)):

        x1Max = ballsAnt.x[i]*100 + 1
        x1Min = ballsAnt.x[i]*100 - 1
        y1Max = ballsAnt.y[i]*100 + 1
        y1Min = ballsAnt.y[i]*100 - 1

        if (balls.x[i] > x1Max) or (balls.x[i] < x1Min):
            return True
        elif (balls.y[i] > y1Max) or (balls.y[i] < y1Min):
            return True

    for i in range(len(obstacles.x)):

        x2Max = obstaclesAnt.x[i]*100 + 1
        x2Min = obstaclesAnt.x[i]*100 - 1
        y2Max = obstaclesAnt.y[i]*100 + 1
        y2Min = obstaclesAnt.y[i]*100 - 1

        if obstacles.x[i] > x2Max or obstacles.x[i] < x2Min:
            return True
        elif obstacles.y[i] > y2Max or obstacles.y[i] < y2Min:
            return True
    
    return False

def distanceStart_Target ( balls, mediaX, mediaY ):

    dst, position, distance = 0, 0, 0
    distance = math.sqrt( math.pow((mediaX - balls.x[0]), 2) + math.pow((mediaY - balls.y[0]), 2) )

    for i in range(len(balls.x)):

        dst = math.sqrt( math.pow((mediaX - balls.x[i]), 2) + math.pow((mediaY - balls.y[i]), 2) )
        
        if dst <= distance:
            distance = dst
            position = i

    return position

def main():

    print(__file__ + " start!!")

    ip_address = "localhost"
    port = "8000"

    bluetoothS = serial.Serial("COM6", 9600)

    responseAnt = []
    errorAnt = 0

    while True: 

        r = requests.get("http://" + ip_address + ":" + port)
        response = r.json()
                 
        balls, obstacles, car = colorClassification(response)

        if car.color[0] == "BLUE":
            car.x[0], car.x[1] = car.x[1], car.x[0]
            car.y[0], car.y[1] = car.y[1], car.y[0]
            car.cordinates[0], car.cordinates[1] = car.cordinates[1], car.cordinates[0]
            car.color[0], car.color[1] = car.color[1], car.color[0]
        
        if responseAnt != []:
            ballsAnt, obstaclesAnt, car.Ant = colorClassification(responseAnt)

        mediaX, mediaY = carMedia(car.x, car.y)
        
        """
        key = input("Ingrese letra: ")
        bluetoothS = serial.Serial("COM5", 9600)
        remoteController(key, bluetoothS)
        bluetoothS.close()
        """
        
        # Start position
        sx = mediaX * 100 # [m]
        sy = mediaY * 100  # [m]

        for j in range(len(balls.x)):
            balls.x[j] = balls.x[j]*100
            balls.y[j] = balls.y[j]*100

        # Goal position
        position = distanceStart_Target ( balls, sx, sy )
        gx = balls.x[position]
        gy = balls.y[position]

        for k in range(len(obstacles.x)):
            obstacles.x[k] = obstacles.x[k] * 100
            obstacles.y[k] = obstacles.y[k] * 100

        # Tamaño de la cuadrícula que divide la pista
        grid_size = 1.0 # [m]
        # Tamaño del carrito (simulación)
        robot_size = 7.0 # [m]

        ox, oy = squares(obstacles)  ## REVISAR!! 

        for i in range(100):
            ox.append(i)
            oy.append(0.0)
        for i in range(100):
            ox.append(100.0)
            oy.append(i)
        for i in range(101):
            ox.append(i)
            oy.append(100.0)
        for i in range(101):
            ox.append(0.0)
            oy.append(i)

        if show_animation:
            plt.plot(ox, oy, ".k")
            plt.plot(sx, sy, "xr")
            plt.plot(gx, gy, "xb")
            plt.grid(True)
            plt.axis("equal")
        
        # Trayectoria
        if responseAnt == []:

            rx, ry = a_star_planning(gx, gy, sx, sy, ox, oy, grid_size, robot_size)

            for k in range(len(rx)):
                rx[k] = rx[k] / 100
                ry[k] = ry[k] / 100

            finalPointX = rx[len(rx)-1]
            finalPointY = ry[len(ry)-1]

            for i in range(2):
                rx.pop(0)
                ry.pop(0)
        """
        #elif (comparationLists(balls, ballsAnt, obstacles, obstaclesAnt)): 
            
            com = communication(0, 0, 0, 0)
            print("COM DE NUEVA TRAYECTORIA ", com)
            bluetoothS.write(com.encode())

            rx, ry = a_star_planning(gx, gy, sx, sy, ox, oy, grid_size, robot_size)

            for k in range(len(rx)):
                rx[k] = rx[k] / 100
                ry[k] = ry[k] / 100

            for i in range(4):
                rx.pop(0)
                ry.pop(0)
        """

        responseAnt = response

        i = 0
        print(rx, ry)

        ##pwm, direccion, errorAnt = control(rx.pop(0), ry.pop(0), car, mediaX, mediaY, errorAnt, balls, obstacles)  ## CAMBIAR balls POR POSICIÓN DE LA TRAYECTORIA
        if rx != []:
            bit = control(rx.pop(0), ry.pop(0), car, mediaX, mediaY, errorAnt, balls, obstacles, bluetoothS)
        else:
            bit = control(finalPointX, finalPointY, car, mediaX, mediaY, errorAnt, balls, obstacles, bluetoothS) + 1
        
        if bit == 1:
            
            for i in range(2):
                if rx != []:
                    rx.pop(0)
                    ry.pop(0)

            distance = distControl(balls.x[0],balls.y[0],mediaX, mediaY)
            print("DISTANCIA A LA PELOTA: ", distance)

            if distance < 16:
                while rx != []:
                    rx.pop(0)
                    ry.pop(0)
                com = communication(0, 0, 0 ,0)
                bluetoothS.write(com.encode())

            bit = 0
            
        """
        if rx == []:
            com = communication(0, 0, 0, 0)
            print("COM DE LISTA VACIA: ", com)
            bluetoothS.write(com.encode())
            break         ## PUNTO DE LLEGADA A LA PELOTA, AQUI COMIENZA EL RETORNO
        """
        if show_animation:
            plt.plot(rx, ry, "-r")
            plt.show()

    bluetoothS.close() 


if __name__ == '__main__':
    main()