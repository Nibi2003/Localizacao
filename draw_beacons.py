import sys
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import OpenGL.GLU as glu

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"({self.x}, {self.y})"

# Definir os beacons
beacons = {
    'b1': Point(3.9, 0.0),
    'b2': Point(0.0, 3.1),
    'b3': Point(7.9, 3.1),
    'b4': Point(0.0, 7.0),
    'b5': Point(7.9, 7.0),
    'b6': Point(0.0, 11.4),
    'b7': Point(7.9, 11.4),
    'b8': Point(3.9, 14.1),
    'b9': Point(2.9, 14.1)
}

# Dimensões da sala
width = 7.9
height = 14.1

def draw_rect():
    gl.glBegin(gl.GL_LINE_LOOP)
    gl.glVertex2f(0, 0)
    gl.glVertex2f(width, 0)
    gl.glVertex2f(width, height)
    gl.glVertex2f(0, height)
    gl.glEnd()

def draw_beacons():
    gl.glPointSize(10)
    gl.glBegin(gl.GL_POINTS)
    for beacon in beacons.values():
        gl.glVertex2f(beacon.x, beacon.y)
    gl.glEnd()

def display():
    gl.glClear(gl.GL_COLOR_BUFFER_BIT)
    gl.glColor3f(1.0, 1.0, 1.0)

    draw_rect()
    draw_beacons()

    glut.glutSwapBuffers()

def main():
    glut.glutInit(sys.argv)
    glut.glutInitDisplayMode(glut.GLUT_DOUBLE | glut.GLUT_RGB)
    glut.glutInitWindowSize(800, 600)
    glut.glutInitWindowPosition(100, 100)
    glut.glutCreateWindow(b'Sala com Beacons')
    gl.glClearColor(0.0, 0.0, 0.0, 0.0)
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()
    glu.gluOrtho2D(0, width, 0, height)
    glut.glutDisplayFunc(display)
    glut.glutMainLoop()

if __name__ == "__main__":
    main()
