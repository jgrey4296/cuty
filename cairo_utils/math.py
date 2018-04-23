from numpy import cos, sin
from numpy import pi
from functools import partial
import logging as root_logger
import IPython
from scipy.interpolate import splprep, splev
import numpy as np
import numpy.random
import random
from .constants import TWOPI, QUARTERPI, EPSILON, TOLERANCE, IntersectEnum

logging = root_logger.getLogger(__name__)

#Partial Matrix Mul constructor for use in rotate point
#for slices of a 2d array:
def constructMatMul(a):
    assert(isinstance(a, np.ndarray))
    return partial(lambda x,y: x @ y, a)


def sampleCircle(x, y, radius, numOfSteps):
    """ take a position and radius,  get a set of random positions on that circle """
    randI = np.sort(np.random.random(numOfSteps)) * TWOPI
    xPos = x + (np.cos(randI) * radius)
    yPos = y + (np.sin(randI) * radius)
    return np.column_stack((xPos, yPos))

def _interpolate(xy, num_points, smoothing=0.2):
    """ given a set of points, generate values between those points """
    assert(isinstance(xy, np.ndarray))
    assert(len(xy.shape) == 2)
    assert(xy.shape[0] >= 4)
    assert(xy.shape[1] == 2)
    splineTuple, splineValues = splprep([xy[:, 0], xy[:, 1]], s=smoothing)
    interpolatePoints = np.linspace(0, 1, num_points)
    smoothedXY = np.column_stack(splev(interpolatePoints, splineTuple))
    return smoothedXY

def getDirections(xys):
    """ Given a set of points, get the unit direction
    from each point to the next point
    """
    assert(isinstance(xys, np.ndarray))
    assert(len(xys.shape) == 2)
    #convert to vectors:
    #xysPrime.shape = (n, 4)
    xysPrime = np.column_stack((xys[1:, :], xys[:-1, :]))

    dx = xysPrime[:, 2] - xysPrime[:, 0]
    dy = xysPrime[:, 3] - xysPrime[:, 1]

    #radians:
    arc = np.arctan2(dy, dx)
    directions = np.column_stack([np.cos(arc), np.sin(arc)])

    #hypotenuse
    dd = np.sqrt(np.square(dx)+np.square(dy))
    return (directions, dd)

def granulate(xys, grains=10, mult=2):
    """ Given a set of points, duplicate each point and offset each slightly
    by the direction between the points
    """
    assert(isinstance(xys, np.ndarray))
    assert(len(xys.shape) == 2)
    directions, dd = getDirections(xys)
    granulated = None
    for i, d in enumerate(dd):
        subGranules = xys[i, :] + (d * directions[i, :]*(np.random.random((grains, 1))) * mult)
        if granulated is None:
            granulated = subGranules
        else:
            granulated = np.row_stack((granulated, subGranules))
    return granulated


def vary(xys, stepSize, pix):
    """ 
    for a given set of points, wiggle them slightly
    """
    assert(isinstance(xys, np.ndarray))
    assert(len(xys.shape) == 2)
    r = (1.0-2.0 * np.random.random((len(xys), 1)))
    scale = np.reshape(np.arange(len(xys)).astype('float'), (len(xys), 1))
    noise = (r*scale*stepSize)
    a = np.random.random(len(xys))
    rnd = np.column_stack((np.cos(a), np.sin(a)))
    rndNoise = rnd * noise
    rndNoisePix = rndNoise * pix
    xysPrime = xys + rndNoisePix
    return xysPrime


def sampleAlongLine(x, y, ex, ey, t):
    """ Get a point as a ratio along the given start and end points,
    returns as a 2d np array """
    o_x = (1 - t) * x + t * ex
    o_y = (1 - t) * y + t * ey
    return np.column_stack((o_x, o_y))

def createLine(x, y, ex, ey, t):
    """ Given a start and end, create t number of points along that line """
    lin = np.linspace(0, 1, t)
    line = sampleAlongLine(x, y, ex, ey, lin)
    return line

def bezier1cp(start, cp, end, t):
    """ Given the start, end, and a control point, create t number of points along that bezier """
    assert(hasattr(start, '__len__'))
    assert(hasattr(cp, '__len__'))
    assert(hasattr(end, '__len__'))
    samplePoints = np.linspace(0, 1, t)
    line1 = createLine(*start, *cp, t)
    line2 = createLine(*cp, *end, t)
    out = sampleAlongLine(line1[:, 0], line1[:, 1], line2[:, 0], line2[:, 1], samplePoints)
    return out

def bezier2cp(start, cp1, cp2, end, t):
    """ Given a start, end, and two control points along the way, create t number of points along that bezier """
    assert(all([hasattr(a, '__len__') for a in [start, cp1, cp2, end]]))
    samplePoints = np.linspace(0, 1, t)
    line1 = createLine(*start, *cp1, t)
    line2 = createLine(*cp1, *cp2, t)
    line3 = createLine(*cp2, *end, t)

    s2cp_interpolation = sampleAlongLine(line1[:, 0],
                                         line1[:, 1],
                                         line2[:, 0],
                                         line2[:, 1],
                                         samplePoints)
    cp2e_interpolation = sampleAlongLine(line2[:, 0],
                                         line2[:, 1],
                                         line3[:, 0],
                                         line3[:, 1],
                                         samplePoints)
    out = sampleAlongLine(s2cp_interpolation[:, 0],
                          s2cp_interpolation[:, 1],
                          cp2e_interpolation[:, 0],
                          cp2e_interpolation[:, 1],
                          samplePoints)

    return out

def get_distance_raw(p1, p2):
    """ Get the non-square-root distance for pairs of points """
    assert(isinstance(p1, np.ndarray))
    assert(isinstance(p2, np.ndarray))
    p1 = p1.reshape(-1, 2)
    p2 = p2.reshape(-1, 2)
    dSquared = pow(p2-p1, 2)
    #summed = dSquared[:, 0] + dSquared[:, 1]
    summed = dSquared.sum(axis=1)
    return summed

def get_distance(p1, p2):
    """ Get the square-root distance of pairs of points """
    assert(isinstance(p1, np.ndarray))
    assert(isinstance(p2, np.ndarray))
   
    summed = get_distance_raw(p1, p2)
    sqrtd = np.sqrt(summed)
    return sqrtd

def get_distance_xyxy(x1,y1,x2,y2):
    """ Utility to get the raw distance of points as separate x's and y's  """
    return get_distance_raw(np.array([x1,y1]),np.array([x2,y2]))[0]


def get_normal(p1, p2):
    """ Get the normalized direction from two points """
    raise Exception("Misnamed: Use get_unit_vector")

def get_unit_vector(p1, p2):    
    assert(isinstance(p1, np.ndarray))
    assert(isinstance(p2, np.ndarray))
    d = get_distance(p1, p2)
    if np.allclose(d, 0):
        return np.array([0, 0])
    n = (p2-p1)
    normalized = n / d
    return normalized


def get_bisector(p1, p2, r=False):
    """ With a normalised line,  rotate 90 degrees,
    r=True : to the right
    r=False : to the left
    """
    n = get_unit_vector(p1, p2)
    if r:
        nPrime = n.dot([[0, -1],
                        [1, 0]])
    else:
        nPrime = n.dot([[0, 1],
                        [-1, 0]])
    return nPrime

def get_circle_3p(p1, p2, p3, arb_intersect=20000):
    """
    Given 3 points,  treat them as defining two chords on a circle,
    intersect them to find the centre,  then calculate the radius
    Thus: circumcircle
    """
    sortedPoints = sort_coords(np.array([p1,p2,p3]))
    p1 = sortedPoints[0]
    p2 = sortedPoints[1]
    p3 = sortedPoints[2]
    
    #TODO: assert that p1,2 and 3 are arrays
    arb_height = arb_intersect
    #mid points and norms:
    m1 = get_midpoint(p1, p2)
    n1 = get_bisector(m1, p2)
    m2 = get_midpoint(p2, p3)
    n2 = get_bisector(m2, p3)
    #extended norms:
    v1 = m1 + (1 * arb_height * n1)
    v2 = m2 + (1 * arb_height * n2)
    v1I = m1 + (-1 * arb_height * n1)
    v2I = m2 + (-1 * arb_height * n2)
    #resulting lines:
    l1 = np.row_stack((m1, v1))
    l2 = np.row_stack((m2, v2))
    l1i = np.row_stack((m1, v1I))
    l2i = np.row_stack((m2, v2I))
    #intersect extended norms:
    #in the four combinations of directions
    i_1 = intersect(l1, l2)
    i_2 = intersect(l1i, l2i)
    i_3 = intersect(l1, l2i)
    i_4 = intersect(l1i, l2)
    #get the intersection:
    the_intersect = [x for x in [i_1, i_2, i_3, i_4] if x is not None]
    if the_intersect is None or len(the_intersect) == 0:
        return None
    r1 = get_distance(p1, the_intersect[0])
    r2 = get_distance(p2, the_intersect[0])
    r3 = get_distance(p3, the_intersect[0])

    #a circle only if they are have the same radius
    if np.isclose(r1, r2) and np.isclose(r2, r3):
        return (the_intersect[0], r1)
    else:
        return None


def extend_line(p1, p2, m, fromStart=True):
    """ Extend a line by m units """
    n = get_unit_vector(p1, p2)
    if fromStart:
        el = p1 + (n * m)
    else:
        el = p2 + (n * m)
    return el

def get_midpoint(p1, p2):
    """ Given two points, get the point directly between them """
    m = (p1 + p2) / 2
    return m



def rotatePoint(p,cen=None,rads=None, radMin=-QUARTERPI,radMax=QUARTERPI):
    """ Given a point, rotate it around a centre point by either radians,
    or within a range of radians
    """
    #p1 = cen, p2=point, @ is matrix mul
    if cen is None:
        cen = np.array([0,0])
    if rads is None:
        useRads = randomRad(min=radMin,max=radMax)
        if isinstance(useRads, np.ndarray):
            useRads = useRads[0]
    else:
        useRads = rads
    #apply to 1d slices, this allows multiple points to be passed into the function together,
    #without messing up the rotation matmul
    rotM = rotMatrix(useRads)
    offset = (p - cen)
    if len(p.shape) == 1:
        applied = rotM @ offset
    else:
        applied = np.apply_along_axis(constructMatMul(rotM), 1, offset)
    result = cen + applied
    return result


def __rotatePoint_obsolete(p, cen, rads):
    """ Does what rotate point does, explicitly instead
    of with matrix multiplication """
    assert(len(p.shape) == 2)
    c = np.cos(rads)
    s = np.sin(rads)
    centred = p - cen
    cosP = centred * c
    sinP = centred * s
    nx = cosP[:, 0] - sinP[:, 1]
    ny = sinP[:, 0] + cosP[:, 1]
    unCentred = np.column_stack((nx, ny)) + cen
    return unCentred


def randomRad(min=-TWOPI,max=TWOPI):
    """ Get a random value within the range of radians -2pi -> 2pi """ 
    return min + (np.random.random() * (max-min)) 

def rotMatrix(rad):
    """ Get a matrix for rotating a point by an amount of radians """
    return np.array([[cos(rad),-sin(rad)],
                     [sin(rad),cos(rad)]])


def checksign(a, b):
    """ Test whether two numbers have the same sign """
    return math.copysign(a, b) == a

def intersect(l1, l2, tolerance=TOLERANCE):
    """ Get the intersection points of two line segments
    so l1:(start, end), l2:(start, end)
    returns np.array([x,y]) of intersection or None
    """
    assert(isinstance(l1, np.ndarray))
    assert(isinstance(l2, np.ndarray))
    assert(l1.shape == (2,2))
    assert(l2.shape == (2,2))
    #see: http://ericleong.me/research/circle-line/
    #The points
    p0 = l1[0]
    p1 = l1[1]
    p2 = l2[0]
    p3 = l2[1]

    a1 = p1[1] - p0[1]
    b1 = p0[0] - p1[0]
    c1b = a1 * p0[0] + b1 * p0[1]
    
    a2 = p3[1] - p2[1]
    b2 = p2[0] - p3[0]
    c2b = a2*p2[0] + b2*p2[1]

    detb = a1 * b2 - a2 * b1
    
    if detb == 0:
        return None

    xb = ((c1b * b2) - (b1 * c2b)) / detb
    yb = ((a1 * c2b) - (c1b * a2)) / detb
    xyb = np.array([xb,yb])

    l1mins = np.min((p0, p1), axis=0) - tolerance
    l2mins = np.min((p2,p3), axis=0) - tolerance
    l1maxs = np.max((p0, p1), axis=0) + tolerance
    l2maxs = np.max((p2,p3), axis=0) + tolerance


    
    if (l1mins <= xyb).all() and (l2mins <= xyb).all() and \
       (xyb <= l1maxs).all() and (xyb <= l2maxs).all():
        return xyb

    return None
    


def is_point_on_line(p, l):
    """ Test to see if a point is on a line """
    assert(isinstance(p, np.ndarray))
    assert(isinstance(l, np.ndarray))
    assert(p.shape == (2,))
    assert(l.shape == (2,2))
    the_line = l
    l_min_y = the_line[:,1].min()
    l_max_y = the_line[:,1].max()
    l_min_x = the_line[:,0].min()
    l_max_x = the_line[:,0].max()

    in_bounds_x =  l_min_x <= p[0] <= l_max_x
    in_bounds_y = l_min_y <= p[1] <= l_max_y
    
    if (l[0,0] - l[1,0]) == 0:
        return in_bounds_y and in_bounds_x
    slope = (l[0,1] - l[1,1]) / (l[0,0] - l[1,0])
    y_intersect = - slope * l[0,0] + l[0,1]
    line_y = slope * p[0] + y_intersect
    return line_y == p[1] and in_bounds_y and in_bounds_x
        

def random_points(n):
    """ utility to get n 2d points """
    return np.random.random(n*2)

def bound_line_in_bbox(line, bbox):
    #todo: take in line,  intersect with lines of bbox,
    #replace original line endpoint with intersection point
    bbl = bbox_to_lines(bbox)
    intersections = [x for x in [intersect(line, x) for x,y in bbl] if x is not None]
    if len(intersections) == 0:
        return [line]
    return [np.array([line[0], x]) for x in intersections]

def makeHorizontalLine():
    """ Utility to Describe a horizontal line as a vector of start and end points  """
    x = random.random()
    x2 = random.random()
    y = random.random()
    if x < x2:
        return np.array([x, y, x2, y])
    else:
        return np.array([x2, y, x, y])


def makeVerticalLine():
    """ utility Describe a vertical line as a vector of start and end points """
    x = random.random()
    y = random.random()
    y2 = random.random()
    if y < y2:
        return np.array([x, y, x, y2])
    else:
        return np.array([x, y2, x, y])

def get_lowest_point_on_circle(centre, radius):
    """ given the centre of a circle and a radius, get the lowest y point on that circle """
    #return centre + np.array([np.cos(THREEFOURTHSTWOPI) * radius,
    #                          np.sin(THREEFOURTHSTWOPI) * radius])
    return centre - np.array([0, radius])

def sort_coords(arr):
    """ Sort a list of points by x then y value  """
    ind = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[ind]

def inCircle(centre, radius, points):
    """ Test a set of points to see if they are within a circle's radius """
    d = get_distance_raw(centre, points)
    return d < pow(radius, 2)

def isClockwise(*args, cartesian=True):
    """ Test whether a set of points are in clockwise order  """
    #based on stackoverflow.
    #sum over edges,  if positive: CW. negative: CCW
    #assumes normal cartesian of y bottom = 0
    sum = 0
    p1s = args
    p2s = list(args[1:])
    p2s.append(args[0])
    pairs = zip(p1s, p2s)
    for p1, p2 in pairs:
        a = (p2[0, 0]-p1[0, 0]) * (p2[0, 1]+p1[0, 1])
        sum += a
    if cartesian:
        return sum >= 0
    else:
        return sum < 0

def getMinRangePair(p1, p2):
    """ TODO: Can't remember, test this """
    d1 = get_distance(p1, p2)
    fp2 = np.flipud(p2)
    d2 = get_distance(p1, fp2)
    d1_min = d1.min()
    d2_min = d2.min()
    if d1_min < d2_min:
        i = d1.tolist().index(d1_min)
        #get the right xs
        return np.array([p1[i][0], p2[i][0]])
    else:
        i = d2.tolist().index(d2_min)
        return np.array([p1[i][0], fp2[i][0]])

def getClosestToFocus(focus, possiblePoints):
    """ Given a set of points, return the point closest to the focus """
    ds = get_distance(focus, possiblePoints)
    m_d = ds.min()
    i = ds.tolist().index(m_d)
    return possiblePoints[i]

def get_closest_on_side(refPoint, possiblePoints, left=True):
    """ 
    given a reference point and a set of candidates, get the closest 
    point on either the left or right of that reference
    """
    subbed = possiblePoints - refPoint
    if left:
        onSide = subbed[:, 0] < 0
    else:
        onSide = subbed[:, 0] > 0
    try:
        i = onSide.tolist().index(True)
        return possiblePoints[i]
    except ValueError as e:
        return None

#TODO: rename for more accuracy
#should be radians_between_points
def angle_between_points(a, b):
    """ takes np.arrays
        return the radian relation of b to a (source)
        ie: if > 0: anti-clockwise,  < 0: clockwise
    """
    c = b - a
    return atan2(c[1], c[0])


def displace_along_line(xys,amnt,num):
    """ TODO: Can't remember, test """
    t = np.linspace(0,2*pi,num)
    rotation = np.column_stack((sin(t),cos(t)))
    rnd = np.random.random(num)
    combined = np.column_stack((rotation[:,0] * rnd, rotation[:,1] * rnd))
    scaled = amnt * combined
    mod_points = xys + scaled
    all_points = np.concatenate((xys,mod_points))
    return all_points
    
def clamp(n,minn=0,maxn=1):
    """ Clamp a number between min and max,
    could be replaced with np.clip
    """
    return max(min(maxn,n),minn)

def bbox_to_lines(bbox, epsilon=EPSILON):
    assert(isinstance(bbox, np.ndarray))
    assert(bbox.shape == (4,))
    bbox_e = bbox + np.array([-epsilon, -epsilon, epsilon, epsilon])
    bbox_t = bbox.reshape((2,2)).transpose()
    #convert the bbox to bounding lines
    selX = np.array([1,0])
    selY = np.array([0,1])
    mins = bbox_t[:,0]
    maxs = bbox_t[:,1]
    minXmaxY = mins * selX + maxs * selY
    maxXminY =  maxs * selX + mins * selY
    lines = [ (np.row_stack((mins, maxXminY)), IntersectEnum.HBOTTOM),
              (np.row_stack((minXmaxY, maxs)), IntersectEnum.HTOP),
              (np.row_stack((mins, minXmaxY)), IntersectEnum.VLEFT),
              (np.row_stack((maxXminY, maxs)), IntersectEnum.VRIGHT) ]

    return lines
