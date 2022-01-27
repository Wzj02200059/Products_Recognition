import numpy as np

class Point(object):
    def __init__(self, x=0.0, y=0.0, z=1.0):
        self.x = x
        self.y = y
        self.z = z

    def calculatenewpoint(self, homo):
        point_old = np.array([self.x, self.y, self.z]).reshape(3, 1)
        point_new = np.dot(homo, point_old)
        point_new /= point_new[2, 0]
        self.x = point_new[0, 0]
        self.y = point_new[1, 0]
        self.z = point_new[2, 0]


class Corner(object):
    def __init__(self):
        self.ltop = Point()
        self.lbottom = Point()
        self.rtop = Point()
        self.rbottom = Point()

    def calculatefromimage(self, img):
        rows = img.shape[0]
        cols = img.shape[1]
        self.ltop.x = 0.0
        self.ltop.y = 0.0
        self.lbottom.x = 0.0
        self.lbottom.y = float(rows)
        self.rtop.x = float(cols)
        self.rtop.y = 0.0
        self.rbottom.x = float(cols)
        self.rbottom.y = float(rows)

    def calculatefromhomo(self, homo):
        self.ltop.calculatenewpoint(homo)
        self.lbottom.calculatenewpoint(homo)
        self.rtop.calculatenewpoint(homo)
        self.rbottom.calculatenewpoint(homo)

    def getoutsize(self):
        lx = min(self.ltop.x, self.lbottom.x)
        rx = max(self.rtop.x, self.rbottom.x)
        uy = min(self.ltop.y, self.rtop.y)
        dy = max(self.lbottom.y, self.rbottom.y)
        return lx, rx, uy, dy


def calculatecorners(imgs, homos):
    c = Corner()
    c.calculatefromimage(imgs)
    c.calculatefromhomo(homos)

    return c
