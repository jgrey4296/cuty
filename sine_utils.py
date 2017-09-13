#import cairocffi as cairo
import math
from math import atan2
from random import choice
import numpy as np
from numpy import pi
from numpy.random import random
from numpy import sin,cos
from scipy.interpolate import splprep
from scipy.interpolate import splev
import IPython
import logging
import sys
import random
import pickle
from os.path import join,isfile,exists,isdir
from os import listdir


DRAW_TEXT = True

#constants:
TEXT = [0,1,1,1]
EDGE = [1,0,0,1]
VERTEX = [1,0,1,1]
FACE = [0,0,1,1]
START = [0,1,0,1]
END = [1,0,0,1]

SMALL_RADIUS = 0.003

ALPHA = 0.1
BACKGROUND = [0,0,0,1]
FRONT = [0.8,0.1,0.71,ALPHA]
TWOPI = 2 * pi
QUARTERPI = pi * 0.5
THREEFOURTHSTWOPI = 3/4 * TWOPI
EPSILON = sys.float_info.epsilon


def write_to_png(surface,filename,i=None):
    if i:
        surface.write_to_png("{}_{}.png".format(filename,i))
    else:
        surface.write_to_png("{}.png".format(filename))

def drawRect(ctx,x,y,sx,sy):
    #ctx.set_source_rgba(*FRONT)
    ctx.rectangle(x,y,sx,sy)
    ctx.fill()

#take a position and radius, get a set of random positions on that circle
def sampleCircle(x,y,radius,numOfSteps):
    randI = np.sort(np.random.random(numOfSteps)) * TWOPI
    xPos = x + (np.cos(randI) * radius)
    yPos = y + (np.sin(randI) * radius)
    return np.column_stack((xPos,yPos))
    
def drawCircle(ctx,x,y,r,fill=True):
    try:
        ctx.arc(x,y,r,0,TWOPI)
    except TypeError as e:
        print(x,y,r)
        raise e
    if fill:
        ctx.fill()
    else:
        ctx.stroke()

def drawDCEL(ctx,dcel):
    """ A top level function to draw a dcel  """
    ctx.set_source_rgba(0.2,0.2,0.9,1)
    #draw the faces
    draw_dcel_faces(ctx,dcel)
    ctx.set_source_rgba(0.4,0.8,0.1,1)
    #draw edges
    #draw_dcel_edges(ctx,dcel)
    ctx.set_source_rgba(0.9,0.1,0.1,1)
    #draw vertices
    #draw_dcel_vertices(ctx,dcel)
    
def draw_dcel_faces(ctx,dcel):
    for f in dcel.faces:
        draw_dcel_single_face(ctx,dcel,f,clear=False)

def draw_dcel_single_face(ctx,dcel,face,clear=True,force_centre=False):
    if clear:
        clear_canvas(ctx)
    if len(face.edgeList) < 2 :
        return
    if force_centre:
        centre = face.getCentroid()
        invCentre = -centre
        ctx.translate(*invCentre)
        ctx.translate(0.5,0.5)
    ctx.set_line_width(0.004)
    faceCentre = face.getCentroid()
    drawText(ctx,*faceCentre,str("F: {}".format(face.index)))
    ctx.set_source_rgba(*END)
    drawCircle(ctx,*faceCentre,SMALL_RADIUS)
    #Draw face edges:
    initial = True
    for x in face.getEdges():
        ctx.set_source_rgba(*FACE)            
        v1,v2 = x.getVertices()
        if v1 is not None and v2 is not None:
            logging.debug("Drawing Face {} edge {}".format(face.index,x.index))
            logging.debug("Drawing Face edge from ({},{}) to ({},{})".format(v1.x,v1.y,
                                                                             v2.x,v2.y))
            if initial:
                ctx.move_to(v1.x,v1.y)
                initial = False
            #ctx.set_source_rgba(*np.random.random(3),1)
            ctx.line_to(v2.x,v2.y)

            #additional things to draw:
            #ctx.set_source_rgba(*START)
            #drawCircle(ctx,v1.x,v1.y,SMALL_RADIUS)
    if face.data is None or 'fill' not in face.data:
        ctx.stroke()
    else:
        ctx.close_path()
        ctx.fill()
    if force_centre:
        ctx.translate(-0.5,-0.5)
        ctx.translate(*centre)
            
        
def draw_dcel_edges(ctx,dcel):
    drawnEdges = []
    ctx.set_line_width(0.004)
    for e in dcel.halfEdges:
        i = e.index
        #only draw if the end hasnt been drawn yet:
        if i in drawnEdges:
            continue
        ctx.set_source_rgba(*EDGE)
        v1,v2 = e.getVertices()
        if v1 is not None and v2 is not None:
            centre = get_midpoint(v1.toArray(),v2.toArray())
            
            ctx.move_to(v1.x,v1.y)
            ctx.line_to(v2.x,v2.y)
            ctx.stroke()
            drawText(ctx,*centre,"E: {}".format(i))
            drawText(ctx,v1.x,v1.y-0.05,"S{}".format(i))
            drawText(ctx,v2.x,v2.y+0.05,"{}E".format(i))
            #Record that this line has been drawn
            drawnEdges.append(e.index)
            #drawnEdges.append(e.twin.index)
        else:
            logging.warning("Trying to draw a line thats missing at least one vertex")

def draw_dcel_halfEdge(ctx,halfEdge,clear=True):
    if clear:
        clear_canvas(ctx)
    ctx.set_line_width(0.002)
    ctx.set_source_rgba(*EDGE)
    v1,v2 = halfEdge.getVertices()
    if v1 is not None and v2 is not None:
        centre = get_midpoint(v1.toArray(),v2.toArray())
        logging.debug("Drawing HalfEdge {} : {},{} - {},{}".format(halfEdge.index,v1.x,v1.y,v2.x,v2.y))
        ctx.move_to(v1.x,v1.y)
        ctx.line_to(v2.x,v2.y)
        ctx.stroke()
        ctx.set_source_rgba(*START)
        drawCircle(ctx,v1.x,v1.y,0.01)
        ctx.set_source_rgba(*END)
        drawCircle(ctx,v2.x,v2.y,0.01)

        if halfEdge.face is not None:
            centre = halfEdge.face.getCentroid()
            drawText(ctx,*centre,"F:{}.{}".format(halfEdge.face.index,halfEdge.index))
        else:
            drawText(ctx,*centre,"HE: {}".format(halfEdge.index))
            
def draw_dcel_vertices(ctx,dcel):
    """ Draw all the vertices in a dcel as dots """
    for v in dcel.vertices:
        ctx.set_source_rgba(*VERTEX)
        if v is not None:
            drawCircle(ctx,v.x,v.y,0.01)
        

def clear_canvas(ctx):
    ctx.set_source_rgba(*BACKGROUND)
    ctx.rectangle(0,0,1,1)
    ctx.fill()
    ctx.set_source_rgba(*FRONT)

#takes array of [[x1,y1]] to smooth
def _interpolate(xy,num_points,smoothing=0.2):
    splineTuple,splineValues = splprep([xy[:,0],xy[:,1]],s=smoothing)
    interpolatePoints = np.linspace(0,1,num_points)
    smoothedXY = np.column_stack(splev(interpolatePoints, splineTuple))
    return smoothedXY

def getDirections(xys):
    #xys.shape = (n,2)
    #convert to vectors:
    #xysPrime.shape = (n,4)
    xysPrime = np.column_stack((xys[1:,:],xys[:-1,:]))
    
    dx = xysPrime[:,2] - xysPrime[:,0]
    dy = xysPrime[:,3] - xysPrime[:,1]

    #radians:
    arc = np.arctan2(dy,dx)
    directions = np.column_stack([np.cos(arc),np.sin(arc)])

    #hypotenuse
    dd = np.sqrt(np.square(dx)+np.square(dy))
    
    return (directions,dd)
    

def granulate(xys,grains=10,mult=2):
    #xys.shape = (n,2)
    #directions.shape = (n,3)
    #dd.shape = (n,)
    directions,dd = getDirections(xys)
    granulated = None
    for i,d in enumerate(dd):
        subGranules = xys[i,:] + (d * directions[i,:]*(np.random.random((grains,1))) * mult)
        if granulated is None:
            granulated = subGranules
        else:
            granulated = np.row_stack((granulated,subGranules))
    return granulated


def vary(xys,stepSize,pix):
    r = (1.0-2.0 * random((len(xys),1)))
    scale = np.reshape(np.arange(len(xys)).astype('float'), (len(xys),1))
    noise = (r*scale*stepSize)
    a = random(len(xys))
    rnd = np.column_stack((np.cos(a), np.sin(a)))
    rndNoise = rnd * noise
    rndNoisePix = rndNoise * pix
    xysPrime = xys + rndNoisePix
    return xysPrime


def displace_along_line(xys,amnt,num):
    t = np.linspace(0,2*pi,num)
    rotation = np.column_stack((sin(t),cos(t)))
    rnd = np.random.random(num)
    combined = np.column_stack((rotation[:,0] * rnd, rotation[:,1] * rnd))
    scaled = amnt * combined
    mod_points = xys + scaled
    all_points = np.concatenate((xys,mod_points))
    return all_points
    
    

def sampleAlongLine(x,y,ex,ey,t):
    o_x = (1 - t) * x + t * ex
    o_y = (1 - t) * y + t * ey
    return np.column_stack((o_x,o_y))

def createLine(x,y,ex,ey,t):
    lin = np.linspace(0,1,t)
    line = sampleAlongLine(x,y,ex,ey,lin)

    return line

def bezier1cp(start,cp,end,t):
    samplePoints = np.linspace(0,1,t)
    line1 = createLine(*start,*cp,t)
    line2 = createLine(*cp,*end,t)

    out = sampleAlongLine(line1[:,0],line1[:,1],line2[:,0],line2[:,1],samplePoints)
    return out

def bezier2cp(start,cp1,cp2,end,t):
    samplePoints = np.linspace(0,1,t)
    line1 = createLine(*start,*cp1,t)
    line2 = createLine(*cp1,*cp2,t)
    line3 = createLine(*cp2,*end,t)

    s2cp_interpolation = sampleAlongLine(line1[:,0],line1[:,1],line2[:,0],line2[:,1],samplePoints)
    cp2e_interpolation = sampleAlongLine(line2[:,0],line2[:,1],line3[:,0],line3[:,1],samplePoints)
    out = sampleAlongLine(s2cp_interpolation[:,0],s2cp_interpolation[:,1],cp2e_interpolation[:,0],cp2e_interpolation[:,1],samplePoints)
    
    return out
        

#--------------------
#from pgkelley4's line-segments-intersect on github
def line_segment_intersection(p,pr,q,qs):
    r = pr - p
    s = qs - q

    t = np.cross((q-p),s) / np.cross(r,s)
    u = np.cross((q-p),r) / np.cross(r,s)

    rs = np.cross(r,s)
    qpr = np.cross((q-p),r)

    if rs == 0 or qpr == 0:
        t0 = (q-p) * r / (r * r)
        t1 = t0 + s * r / (r * r)

    elif rs == 0 or qpr != 0:
        return None
    elif rs != 0 and t >= 0 and t <= 1 and u >= 0 and u <= 1:
        return None
    else:
        return None



def get_distance_raw(p1,p2):
    p1 = p1.reshape(-1,2)
    p2 = p2.reshape(-1,2)
    dSquared = pow(p2-p1,2)
    summed = dSquared[:,0] + dSquared[:,1]
    return summed
    
def get_distance(p1,p2):
    summed = get_distance_raw(p1,p2)
    sqrtd = np.sqrt(summed)
    return sqrtd

def get_distance_xyxy(x1,y1,x2,y2):
    return get_distance_raw(np.array([x1,y1]),np.array([x2,y2]))[0]


def get_normal(p1,p2):
    """ Get the normalized direction from two points """
    v = (p2-p1)
    norm = np.linalg.norm(v)
    if np.allclose(norm,0):
        return np.array([0,0])
    normalized = v / norm
    return normalized


def get_bisector(p1,p2,r=False):
    """ With a normalised line, rotate 90 degrees """
    n = get_normal(p1,p2)
    if r:
        nPrime = n.dot([[0,-1],
                        [1 ,0]])
    else:
        nPrime = n.dot([[0 ,1],
                        [-1,0]])
    return nPrime

def get_circle_3p(p1,p2,p3):
    """
    Given 3 points, treat them as defining two chords on a circle,
    intersect them to find the centre, then calculate the radius
    Thus: circumcircle
    """
    arb_height = 200
    #mid points and norms:
    m1 = get_midpoint(p2,p1)
    n1 = get_bisector(m1,p1,r=True)
    m2 = get_midpoint(p2,p3)        
    n2 = get_bisector(m2,p3,r=True)
    #extended norms:
    v1 = m1 + (1 * arb_height * n1)
    v2 = m2 + (1 * arb_height * n2)
    v1I = m1 + (-1 * arb_height * n1)
    v2I = m2 + (-1 * arb_height * n2)
    #resulting lines:
    l1 = np.column_stack((m1,v1))
    l2 = np.column_stack((m2,v2))
    l1i = np.column_stack((m1,v1I))
    l2i = np.column_stack((m2,v2I))
    #intersect extended norms:
    #in the four combinations of directions
    i_1 = intersect(l1[0],l2[0])
    i_2 = intersect(l1i[0],l2i[0])
    i_3 = intersect(l1[0],l2i[0])
    i_4 = intersect(l1i[0],l2[0])
    #get the intersection:
    the_intersect = [x for x in [i_1,i_2,i_3,i_4] if x is not None]
    if len(the_intersect) != 1:
        return None
    r1 = get_distance(p1,the_intersect[0])
    r2 = get_distance(p2,the_intersect[0])
    r3 = get_distance(p3,the_intersect[0])

    #a circle only if they are have the same radius
    if np.isclose(r1,r2) and np.isclose(r2,r3):
        return [the_intersect[0],r1]
    else:
        return None
    


def extend_line(p1,p2,m):
    """ Extend a line by m units """
    n = get_normal(p1,p2)
    el = p1 + (n * m)
    return el

def get_midpoint(p1,p2):
    m = (p1 + p2) / 2
    return m

def rotatePoint(p,cen,rads):
    c = np.cos(rads)
    s = np.sin(rads)
    centred = p - cen
    cosP = centred * c
    sinP = centred * s
    nx = cosP[:,0] - sinP[:,1]
    ny = sinP[:,0] + cosP[:,1]
    unCentred = np.column_stack((nx,ny)) + cen
    return unCentred

def checksign(a,b):
    return math.copysign(a,b) == a

def intersect(l1,l2):
    #From the line intersection stack overflow post
    #The points
    p0 = l1[0:2]
    p1 = l1[2:]
    p2 = l2[0:2]
    p3 = l2[2:]
    #The vectors of the lines
    s1 = p1 - p0
    s2 = p3 - p2
    #origins vectors
    s3 = p0 - p2

    numerator_1 = np.cross(s1,s3)
    numerator_2 = np.cross(s2,s3)
    denominator = np.cross(s1,s2)

    if denominator == 0:
         return None
    
    s = numerator_1 / denominator
    t = numerator_2 / denominator
    
    if 0 < s and s <= 1 and 0 < t and t <= 1:
        return np.array([p0[0] + (t * s1[0]), p0[1] + t * s1[1]])
        
    return None
    
def random_points(n):
    """ utility to get n 2d points """
    return np.random.random(n*2)

def bound_line_in_bbox(line,bbox):
    #todo: take in line, intersect with lines of bbox,
    #replace original line endpoint with intersection point
    return line


def makeHorizontalLine():
    """ Describe a horizontal line as a vector of start and end points  """
    x = random.random()
    x2 = random.random()
    y = random.random()
    if x < x2:    
        return np.array([x,y,x2,y])
    else:
        return np.array([x2,y,x,y])


def makeVerticalLine():
    """ Describe a vertical line as a vector of start and end points """
    x = random.random()
    y = random.random()
    y2 = random.random()
    if y < y2:
        return np.array([x,y,x,y2])
    else:
        return np.array([x,y2,x,y])

def makeParabola(focus,directrix,xs):
    """ Return the xy coords for a given range of xs, with a focus point and bounding line """
    p = Parabola(focus,directrix)
    ys = p(xs)
    xys = np.column_stack((xs,ys))
    return xys


def get_lowest_point_on_circle(centre,radius):
    """ given a point and a radius, rotate round 3/4 of 2PI """
    #return centre + np.array([np.cos(THREEFOURTHSTWOPI) * radius,
    #                          np.sin(THREEFOURTHSTWOPI) * radius])
    return centre + np.array([0,radius])

def sort_coords(arr):
    ind = np.lexsort((arr[:,1],arr[:,0]))
    return arr[ind]

def inCircle(centre,radius,point):
    d = get_distance(centre,point)
    return d < radius

def isClockwise(*args,cartesian=True):
    #based on stackoverflow.
    #sum over edges, if positive: CW. negative: CCW
    #assumes normal cartesian of y bottom = 0
    sum = 0
    p1s = args
    p2s = list(args[1:])
    p2s.append(args[0])
    pairs = zip(p1s,p2s)
    for p1,p2 in pairs:
        a = (p2[0,0]-p1[0,0]) * (p2[0,1]+p1[0,1])
        sum += a
    if cartesian:
        return sum >= 0
    else:
        return sum < 0
    
def getMinRangePair(p1,p2):
    d1 = get_distance(p1,p2)
    fp2 = np.flipud(p2)
    d2 = get_distance(p1,fp2)
    d1_min = d1.min()
    d2_min = d2.min()
    if d1_min < d2_min:
        i = d1.tolist().index(d1_min)
        #get the right xs
        return np.array([p1[i][0],p2[i][0]])        
    else:
        i = d2.tolist().index(d2_min)
        return np.array([p1[i][0],fp2[i][0]])

def getClosestToFocus(focus,possiblePoints):
    ds = get_distance(focus,possiblePoints)
    m_d = ds.min()
    i = ds.tolist().index(m_d)
    return possiblePoints[i]

def get_closest_on_side(refPoint,possiblePoints,left=True):
    subbed = possiblePoints - refPoint
    if left:
        onSide = subbed[:,0] < 0
    else:
        onSide = subbed[:,0] > 0
    try:
        i = onSide.tolist().index(True)
        return possiblePoints[i]
    except ValueError as e:
        return None

#TODO: rename for more accuracy
#should be radians_between_points
def angle_between_points(a,b):
    """ takes np.arrays
        return the radian relation of b to a (source)
        ie: if > 0: anti-clockwise, < 0: clockwise
    """
    c = b - a
    return atan2(c[1],c[0])

def drawText(ctx,x,y,string):
    if not DRAW_TEXT:
        return
    offset = random.random() * 0.005
    ctx.set_source_rgba(*TEXT)
    ctx.move_to(x+offset,y+offset)
    ctx.show_text(str(string))


def randomRad(min=-TWOPI,max=TWOPI):
    return min + (np.random.random() * (max-min)) 

def rotMatrix(rad):
    return np.array([[cos(rad),-sin(rad)],
                     [sin(rad),cos(rad)]])

def rotatePoint(p1,p2,radMin=-QUARTERPI,radMax=QUARTERPI):
    rad = randomRad(min=radMin,max=radMax)
    return p1 + ((p2-p1) @ rotMatrix(rad))

def clamp(n,minn,maxn):
    return max(min(maxn,n),minn)

def pickle_graph(node_dict,nx_graph,filename,directory="."):
    if not exists(directory):
        raise Exception("Directory not found: {}".format(directory))
    with open(join(directory,"{}_nxgraph.pickle".format(filename)),'wb') as f:
        pickle.dump(nx_graph,f)
    with open(join(directory,"{}_node_dict.pickle".format(filename)),'wb') as f:
        pickle.dump(node_dict,f)

def load_pickled_graph(filename,directory="."):
    if not ( exists(directory) and \
             exists(join(directory,"{}_nxgraph.pickle".format(filename))) and \
             exists(join(directory,"{}_node_dict.pickle".format(filename))) ):
        raise Exception("Missing file or directory for reconstruction")
    node_dict = None
    nx_graph = None
    with open(join(directory,"{}_nxgraph.pickle".format(filename)),'rb') as f:
        nx_graph = pickle.load(f)
    with open(join(directory,"{}_node_dict.pickle".format(filename)),'rb') as f:
        node_dict = pickle.load(f)
    return (node_dict,nx_graph)
