#include <GL/glut.h>
#include <cmath> // Para usar fabs
#include <map>
#include <vector>
#include <string>
#include <sstream>


float larguraOmbros = 0.50;
float metadeLarguaOmbros = larguraOmbros / 2;
float distanciaPeitoralCelular = 0.30;

struct Point {
    float x, y;
    Point(float x, float y) : x(x), y(y) {}
};

std::map<std::string, Point> beacons = {
    {"b1", Point(3.9, 0.0)}, {"b2", Point(0.0, 3.1)}, {"b3", Point(7.9, 3.1)},
    {"b4", Point(0.0, 7.0)}, {"b5", Point(7.9, 7.0)}, {"b6", Point(0.0, 11.4)},
    {"b7", Point(7.9, 11.4)}, {"b8", Point(3.9, 14.1)}, {"b9", Point(2.9, 14.1)}
};

std::vector<Point> additional_points = {
    {6.27, 3.76}, {3.92, 8.75}, {1.57, 13.74}, {1.57, 8.75}, {1.57, 7.09},
    {6.27, 7.09}, {1.57, 10.42}, {1.57, 5.42}, {3.92, 5.42}, {3.92, 13.74},
    {6.27, 12.08}, {1.57, 2.03}, {3.92, 3.76}, {1.57, 3.76}, {6.27, 8.75},
    {3.92, 12.08}, {3.92, 0.77}, {3.92, 7.09}, {1.57, 0.77}, {1.57, 12.08},
    {3.92, 10.42}, {6.27, 5.42}, {6.27, 2.03}, {3.92, 2.03}, {6.27, 13.74},
    {6.27, 10.42}, {6.27, 0.77}
};

Point currentProgrammerPosition(-1.0, -1.0);
Point selectedBeaconPosition(-1.0, -1.0);
Point currentPhonePosition(-1.0, -1.0);
int programmerDirection = 1;

void drawText(float x, float y, const std::string& text) {
    glRasterPos2f(x, y);
    for (size_t i = 0; i < text.length(); ++i) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, text[i]);
    }
}

void drawProgrammer(float x, float y, float phoneX, float phoneY, int direction) {
    glColor3f(1.0, 0.0, 1.0);
    glBegin(GL_LINES);
    if (direction == 1 || direction == 3) {
        glVertex2f(x - metadeLarguaOmbros, y);
        glVertex2f(x + metadeLarguaOmbros, y);
    } else {
        glVertex2f(x, y - metadeLarguaOmbros);
        glVertex2f(x, y + metadeLarguaOmbros);
    }
    glEnd();
    glColor3f(1.0, 0.5, 0.0);
    glPointSize(5.0);
    glBegin(GL_POINTS);
    glVertex2f(phoneX, phoneY);
    glEnd();
}

bool lineSegmentsIntersect(Point p1, Point p2, Point p3, Point p4) {
    float Ax = p1.x, Ay = p1.y, Bx = p2.x, By = p2.y;
    float Cx = p3.x, Cy = p3.y, Dx = p4.x, Dy = p4.y;
    float denominator = ((Bx - Ax) * (Dy - Cy)) - ((By - Ay) * (Dx - Cx));
    if (denominator == 0) return false;
    float r = (((Ay - Cy) * (Dx - Cx)) - ((Ax - Cx) * (Dy - Cy))) / denominator;
    float s = (((Ay - Cy) * (Bx - Ax)) - ((Ax - Cx) * (By - Ay))) / denominator;
    return (r >= 0 && r <= 1 && s >= 0 && s <= 1);
}

void display() {
    glClearColor(0.9, 0.9, 0.8, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-1.0, 8.9, -1.0, 15.1, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glColor3f(1.0, 1.0, 1.0);
    glBegin(GL_POLYGON);
    glVertex2f(0.0, 0.0);
    glVertex2f(7.9, 0.0);
    glVertex2f(7.9, 14.1);
    glVertex2f(0.0, 14.1);
    glEnd();

    glColor3f(0.0, 0.0, 0.0);
    glPointSize(5.0);
    glBegin(GL_POINTS);
    for (const auto& beacon : beacons) glVertex2f(beacon.second.x, beacon.second.y);
    glEnd();

    for (const auto& beacon : beacons) drawText(beacon.second.x + 0.1, beacon.second.y + 0.1, beacon.first);

    glColor3f(0.0, 0.0, 1.0);
    glBegin(GL_POINTS);
    for (const auto& point : additional_points) glVertex2f(point.x, point.y);
    glEnd();

    for (const auto& point : additional_points) {
        std::stringstream ss;
        glColor3f(0.8, 0.8, 0.8);
        ss << "(" << point.x << ", " << point.y << ")";
        drawText(point.x, point.y + 0.1, ss.str());
    }

    if (currentProgrammerPosition.x != -1.0 && currentProgrammerPosition.y != -1.0 &&
        currentPhonePosition.x != -1.0 && currentPhonePosition.y != -1.0) {
        drawProgrammer(currentProgrammerPosition.x, currentProgrammerPosition.y,
                       currentPhonePosition.x, currentPhonePosition.y, programmerDirection);
    }

    if (selectedBeaconPosition.x != -1.0 && selectedBeaconPosition.y != -1.0 &&
        currentPhonePosition.x != -1.0 && currentPhonePosition.y != -1.0) {
        glColor3f(1.0, 0.0, 0.0);
        glBegin(GL_LINES);
        glVertex2f(selectedBeaconPosition.x, selectedBeaconPosition.y);
        glVertex2f(currentPhonePosition.x, currentPhonePosition.y);
        glEnd();

        Point p1(currentProgrammerPosition.x, currentProgrammerPosition.y);
        if (programmerDirection == 1 || programmerDirection == 3) p1.x -= metadeLarguaOmbros;
        else p1.y -= metadeLarguaOmbros;

        Point p2(currentProgrammerPosition.x, currentProgrammerPosition.y);
        if (programmerDirection == 1 || programmerDirection == 3) p2.x += metadeLarguaOmbros;
        else p2.y += metadeLarguaOmbros;

        Point p3(selectedBeaconPosition.x, selectedBeaconPosition.y);
        Point p4(currentPhonePosition.x, currentPhonePosition.y);

        std::string popupText = lineSegmentsIntersect(p1, p2, p3, p4) ? "INTERCEPTA" : "NAO INTERCEPTA";
        drawText(selectedBeaconPosition.x + 0.1, selectedBeaconPosition.y + 0.1, popupText);
    }

    glFlush();
}

void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        GLdouble modelview[16], projection[16];
        GLint viewport[4];
        GLdouble winX = (double)x, winY = (double)viewport[3] - (double)y, winZ, posX, posY, posZ;

        glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
        glGetDoublev(GL_PROJECTION_MATRIX, projection);
        glGetIntegerv(GL_VIEWPORT, viewport);
        glReadPixels(x, int(winY), 1, 1, GL_DEPTH_COMPONENT, GL_DOUBLE, &winZ);
        gluUnProject(winX, winY, winZ, modelview, projection, viewport, &posX, &posY, &posZ);

        for (const auto& point : additional_points) {
            if (fabs(point.x - posX) < 0.5 && fabs(point.y - posY) < 0.5) {
                currentProgrammerPosition = point;
                currentPhonePosition = Point(point.x, point.y + distanciaPeitoralCelular);
                selectedBeaconPosition = Point(-1.0, -1.0);
                programmerDirection = 1;
                glutPostRedisplay();
                return;
            }
        }

        for (const auto& beacon : beacons) {
            if (fabs(beacon.second.x - posX) < 0.5 && fabs(beacon.second.y - posY) < 0.5) {
                selectedBeaconPosition = beacon.second;
                glutPostRedisplay();
                return;
            }
        }
    }
}

void specialKeys(int key, int x, int y) {
    switch (key) {
        case GLUT_KEY_UP: programmerDirection = 1;
            currentPhonePosition = Point(currentProgrammerPosition.x, currentProgrammerPosition.y + distanciaPeitoralCelular); break;
        case GLUT_KEY_RIGHT: programmerDirection = 2;
            currentPhonePosition = Point(currentProgrammerPosition.x + distanciaPeitoralCelular, currentProgrammerPosition.y); break;
        case GLUT_KEY_DOWN: programmerDirection = 3;
            currentPhonePosition = Point(currentProgrammerPosition.x, currentProgrammerPosition.y - distanciaPeitoralCelular); break;
        case GLUT_KEY_LEFT: programmerDirection = 4;
            currentPhonePosition = Point(currentProgrammerPosition.x - distanciaPeitoralCelular, currentProgrammerPosition.y); break;
    }
    glutPostRedisplay();
}

int main(int argc, char** argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(800, 800);
    glutCreateWindow("Laboratório");
    glutDisplayFunc(display);
    glutMouseFunc(mouse);
    glutSpecialFunc(specialKeys);
    glutMainLoop();
    return 0;
}
