import math
import pandas as pd
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"({self.x}, {self.y})"

beacons = {
    'B1_min': Point(3.9, 0.0),
    'B2_min': Point(0.0, 3.1),
    'B3_min': Point(7.9, 3.1),
    'B4_min': Point(0.0, 7.0),
    'B5_min': Point(7.9, 7.0),
    'B6_min': Point(0.0, 11.4),
    'B7_min': Point(7.9, 11.4),
    'B8_min': Point(3.9, 14.1),
    'B9_min': Point(2.9, 14.1)
}
programmer_positions = [
    (6.27, 3.76), (3.92, 8.75), (1.57, 13.74), (1.57, 8.75), (1.57, 7.09),
    (6.27, 7.09), (1.57, 10.42), (1.57, 5.42), (3.92, 5.42), (3.92, 13.74),
    (6.27, 12.08), (1.57, 2.03), (3.92, 3.76), (1.57, 3.76), (6.27, 8.75),
    (3.92, 12.08), (3.92, 0.77), (3.92, 7.09), (1.57, 0.77), (1.57, 12.08),
    (3.92, 10.42), (6.27, 5.42), (6.27, 2.03), (3.92, 2.03), (6.27, 13.74),
    (6.27, 10.42), (6.27, 0.77)
]

largura_ombros = 0.50
metade_largura_ombros = largura_ombros / 2
distancia_peitoral_celular = 0.30

def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def calculate_phone_position(programmer_position, direction):
    if direction == 1:  # Norte
        return Point(programmer_position.x, programmer_position.y + distancia_peitoral_celular)
    elif direction == 2:  # Leste
        return Point(programmer_position.x + distancia_peitoral_celular, programmer_position.y)
    elif direction == 3:  # Sul
        return Point(programmer_position.x, programmer_position.y - distancia_peitoral_celular)
    elif direction == 4:  # Oeste
        return Point(programmer_position.x - distancia_peitoral_celular, programmer_position.y)

def check_intersection(programmer_position, direction, beacon_position):
    phone_position = calculate_phone_position(programmer_position, direction)

    if direction in [1, 3]:  # Norte ou Sul
        p1 = Point(programmer_position.x - metade_largura_ombros, programmer_position.y)
        p2 = Point(programmer_position.x + metade_largura_ombros, programmer_position.y)
    elif direction in [2, 4]:  # Leste ou Oeste
        p1 = Point(programmer_position.x, programmer_position.y - metade_largura_ombros)
        p2 = Point(programmer_position.x, programmer_position.y + metade_largura_ombros)

    return intersect(p1, p2, beacon_position, phone_position)


def run_tests():
    # Testar todas as posições do programador com todas as direções
    for position in programmer_positions:
        programmer_point = Point(position[0], position[1])
        for direction in [1, 2, 3, 4]: 
            print(f"Teste para posição {programmer_point} e direção {direction}:")
            intercepted_beacons = []
            for beacon_name, beacon_position in beacons.items():
                if check_intersection(programmer_point, direction, beacon_position):
                    intercepted_beacons.append(beacon_name)
            print(f"  Beacons interceptados: {intercepted_beacons}")

if __name__ == "__main__":
    run_tests()