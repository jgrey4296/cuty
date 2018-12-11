from numpy import cos, sin
from numpy import pi
from functools import partial
import logging as root_logger
import IPython
from scipy.interpolate import splprep, splev
import numpy as np
from .constants import TWOPI, QUARTERPI, EPSILON, TOLERANCE, IntersectEnum

logging = root_logger.getLogger(__name__)

def constructMatMul(a):
    """ Partial Matrix Mul constructor for use in rotate point
    for slices of a 2d array: """
    assert(isinstance(a, np.ndarray))
    return partial(lambda x,y: x @ y, a)

#------------------------------
# def circle functions
#------------------------------

def displace_around_circle(xys,scale,n):
    """ displace the data around a scaled noisy circle """
    #Create a circle:
    t = np.linspace(0,2*pi,n)
    rotation = np.column_stack((sin(t),cos(t))).transpose()
    #create some noise:
    noise = np.random.random(n)
    #jitter the rotation:
    jittered = (rotation * noise)
    #control the amount of this noise to apply
    scaled = (jittered * scale).transpose()
    #apply the noise to the data
    mod_points = xys + scaled
    return mod_points

def sampleCircle(xyrs, n, diameter=True, sort_rads=True, sort_radi=True, rad_amnt=TWOPI):
    """ 
    Given circles xy and radius in array shape (i,3)
    produce n samples for each circle.
    diameter=True : on the edge boundary
    diameter=False : up to the edge boundary
    Returns: np.array.shape = (len(xyrs), n, 2)
    """
    #duplicate the points:
    xyrs_r = xyrs.reshape((-1,1,3)).repeat(n, axis=0).reshape((-1,n,3)).transpose((0,1,2))
    #get random rotations
    randI = np.random.random((len(xyrs_r), n))
    if sort_rads:
        randI.sort(axis=1)
    randI_t = randI.transpose()
    randI_ts = randI_t * rad_amnt
    #create the circle transform
    circ = np.array([np.cos(randI_ts), np.sin(randI_ts)])
    circ_t = circ.transpose((2,1,0))
    #Add radius:
    radi = xyrs_r[:,:,2].reshape((len(xyrs_r),n,1))
    if not diameter:
        rand_radi = np.random.random(radi.shape)
        if sort_radi:
            rand_radi.sort(axis=1)
        radi *= rand_radi
    offset = circ_t * radi
    #apply the transforms
    result = xyrs_r[:,:,:2] + offset
    return result

def get_circle_3p(p1, p2, p3, arb_intersect=20000):
    """
    Given 3 points,  treat them as defining two chords on a circle,
    intersect them to find the centre,  then calculate the radius
    Thus: circumcircle
    """
    assert(all([isinstance(x, np.ndarray) for x in [p1, p2, p3]]))
    sortedPoints = sort_coords(np.array([p1,p2,p3]))
    p1 = sortedPoints[0]
    p2 = sortedPoints[1]
    p3 = sortedPoints[2]
    
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

def get_lowest_point_on_circle(centre, radius):
    """ given the centre of a circle and a radius, get the lowest y point on that circle """
    #return centre + np.array([np.cos(THREEFOURTHSTWOPI) * radius,
    #                          np.sin(THREEFOURTHSTWOPI) * radius])
    return centre - np.array([0, radius])

def inCircle(centre, radius, points):
    """ Test a set of points to see if they are within a circle's radius """
    d = get_distance_raw(centre, points)
    return d < pow(radius, 2)

#------------------------------
# def interpolation functions
#------------------------------

def granulate(xys, grains=10, mult=2):
    """ Given a set of points, offset each slightly
    by the direction between the points
    """
    assert(isinstance(xys, np.ndarray))
    assert(len(xys.shape) == 2)
    directions, dd = getDirections(xys)
    granulated = np.zeros((1,2))
    for i, d in enumerate(dd):
        subGranules = xys[i, :] + (d * directions[i, :]*(np.random.random((grains, 1))) * mult)
        granulated = np.row_stack((granulated, subGranules))
    return granulated[1:]


def vary(xys, stepSize, pix):
    """ 
    TODO : investigate
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

#------------------------------
# def direction functions
#------------------------------
def getRandomDirections(n=1):
    """ Choose a direction of cardinal and intercardinal directions """ 
    dirs = [-1,0,1]
    result = np.random.choice(dirs, size=n*2, replace=True, p=None).reshape((n,2))
    return result


def getDirections(xys):
    """ Given a set of points, get the unit direction
    from each point to the next point
    """
    assert(isinstance(xys, np.ndarray))
    assert(len(xys.shape) == 2)
    #convert to vectors:
    #xysPrime.shape = (n, 4)
    #Leading point first to prevent wrap deformation
    xysPrime = np.column_stack((xys[1:, :], xys[:-1, :]))
    dx = xysPrime[:, 2] - xysPrime[:, 0]
    dy = xysPrime[:, 3] - xysPrime[:, 1]
    #radians:
    arc = np.arctan2(dy, dx)
    directions = np.column_stack([np.cos(arc), np.sin(arc)])
    #hypotenuse
    dd = np.sqrt(np.square(dx)+np.square(dy))
    return np.column_stack(directions,dd)


#------------------------------
# def line functions
#------------------------------
def intersect(l1, l2, tolerance=TOLERANCE):
    """ Get the intersection points of two line segments
    see: http://ericleong.me/research/circle-line/
    so l1:(start, end), l2:(start, end)
    returns np.array([x,y]) of intersection or None
    """
    assert(isinstance(l1, np.ndarray))
    assert(isinstance(l2, np.ndarray))
    assert(l1.shape == (2,2))
    assert(l2.shape == (2,2))
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

def get_unit_vector(p1, p2):
    """ Given two points, get the normalized direction """
    assert(isinstance(p1, np.ndarray))
    assert(isinstance(p2, np.ndarray))
    d = get_distance(p1, p2)
    if np.allclose(d, 0):
        return np.array([0, 0])
    n = (p2-p1)
    normalized = n / d
    return normalized

def extend_line(p1, p2, m, fromStart=True):
    """ Extend a line by m units 
    Returns the new end points only
    """
    n = get_unit_vector(p1, p2)
    if fromStart:
        el = p1 + (n * m)
    else:
        el = p2 + (n * m)
    return el

def is_point_on_line(p, l):
    """ Test to see if a point is on a line """
    assert(isinstance(p, np.ndarray))
    assert(isinstance(l, np.ndarray))
    points = p.reshape((-1,2))
    the_lines = l.reshape((-1,2,2))
    l_mins = the_lines.min(axis=1)
    l_maxs = the_lines.max(axis=1)

    in_bounds_xs =  l_mins[:,0] <= points[:,0] <= l_maxs[:,0]
    in_bounds_ys = l_mins[:,1] <= points[:, 1] <= l_maxs[:,1]
    
    if np.allclose((the_lines[:,0,0] - the_lines[:,1,0]), 0):
        return in_bounds_ys and in_bounds_xs
    slopes = (the_lines[:,0,1] - the_lines[:,1,1]) / (the_lines[:,0,0] - the_lines[:,1,0])
    y_intersects = - slopes * the_lines[:,0,0] + the_lines[:,0,1]
    line_ys = slopes * points[:,0] + y_intersects
    return np.allclose(line_ys, points[0,1]) and in_bounds_ys and in_bounds_xs
 
def makeHorizontalLines(n=1):
    """ Utility to Describe a horizontal line as a vector of start and end points  """
    x = np.random.random((n,2)).sort()
    y = np.random.random(n).reshape((-1,1))
    return np.column_stack((x[:,0], y, x[:,1], y))


def makeVerticalLines(n=1):
    """ utility Describe a vertical line as a vector of start and end points """
    x = random.random(n).reshape((-1,1))
    y = random.random((n,2)).sort()
    return np.colum_stack((x, y[:,0], x, y[:,1]))

def sampleAlongLine(xys, t):
    """ For a set of lines, sample along them len(t) times,
    with t's distribution
    """
    if not isinstance(t,np.ndarray):
        t = np.array([t])    
    len_t = len(t)
    lines = xys.reshape((-1,2,2))
    t_inv = 1 - t
    fade = np.row_stack((t_inv, t))
    xs = lines[:,:,0].repeat(len_t).reshape((-1,2,len_t)) * fade
    ys = lines[:,:,1].repeat(len_t).reshape((-1,2,len_t)) * fade
    s_xs = xs.sum(axis=1)
    s_ys = ys.sum(axis=1)
    paired = np.column_stack((s_xs,s_ys))
    reshaped = paired.reshape((-1,len_t,2), order='F')
    return reshaped    

def createLine(xys, t):
    """ Given a start and end, create t number of points along that line """
    lin = np.linspace(0, 1, t)
    line = sampleAlongLine(xys, lin)
    return line

def bezier1cp(aCb, t, f=None, p=None):
    """ Given the start, end, and a control point, create t number of points along that bezier
    t : the number of points to linearly create used to sample along the bezier
    f : a transform function for the sample points prior to calculate bezier
    p : an overriding set of arbitrary sample points for calculate bezier
    """
    assert(isinstance(aCb, np.ndarray))
    if p is not None:
        assert(isinstance(p, np.ndarray))
        samplePoints = p
    else:
        samplePoints = np.linspace(0, 1, t)
        if f is not None:    
            assert(callable(f))
            #f is an easing lookup function
            samplePoints = f(t)
    a2C = createLine(aCb[:,:4], t).reshape((-1,4))
    C2b = createLine(aCb[:,2:], t).reshape((-1,4))
    out = np.zeros((1,a2C.shape[1],2))
    for ((i,ac),(j,cb)) in zip(enumerate(a2C),enumerate(C2b)):
        out = np.row_stack((out,sampleAlongLine(np.column_stack((aC,Cb)), samplePoints)))
    return out[1:]

def bezier2cp(aCCb, t, f=None, p=None):
    """ Given a start, end, and two control points along the way, create t number of points along that bezier
    t : The number of points to sample linearly
    f : the transform function for the linear sampling
    p : arbitrary points to use for sampling instead    
    """
    assert(isinstance(abCC,np.ndarray))
    if p is not None:
        assert(isinstance(p, np.ndarray))
        samplePoints = p
    else:
        samplePoints = np.linspace(0, 1, t)
        if f is not None:
            assert(callable(f))
            samplePoints = f(samplePoints)
    aC = createLine(aCCb[:,:4], t)
    CC = createLine(aCCb[:,2:6], t)
    Cb = createLine(aCCb[:,4:], t)

    f_interp = np.zeros((1,aC.shape[1],2))
    for ((i,a),(j,b)) in enumerate(aC,CC):
        first_interp = np.row_stack((f,sampleAlongLine(np.column_stack((a,b)),
                                                       samplePoints)))

    s_interp = np.zeros((1,CC.shape[1],2))
    for ((i2,a2),(j2,b2)) in enumerate(CC,Cb):
        s_interp = np.row_stack((s_interp, sampleAlongLine(np.column_stack((a2,b2)),
                                                           samplePoints)))

    t_interp = np.zeros((1,f_interp.shape[1],2))
    for ((i3,a3),(j3,b3)) in enumerate((f_interp[1:],s_interp[1:])):
        t_interp = np.row_stack((t_interp, sampleAlongLine(np.column_stack((a3,b3)),
                                                           samplePoints)))


    return t_interp[1:]

#------------------------------
# def distance functions
#------------------------------
def checkDistanceFromPoints(point,quadTree):
    bbox = [point[0] - HALFDELTA,point[1] - HALFDELTA,point[0] + HALFDELTA, point[1] + HALFDELTA]
    area = quadTree.intersect(bbox)
    insideCanvas = point[0] > 0 and point[0] < 1.0 and point[1] > 0 and point[1] < 1.0
    return len(area) == 0 and insideCanvas

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

def get_midpoint(p1, p2):
    """ Given two points, get the point directly between them """
    m = (p1 + p2) / 2
    return m

#------------------------------
# def rotate functions
#------------------------------

#TODO: rename for more accuracy
#should be radians_between_points
def angle_between_points(a, b):
    """ takes np.arrays
        return the radian relation of b to a (source)
        ie: if > 0: anti-clockwise,  < 0: clockwise
    """
    c = b - a
    return atan2(c[1], c[0])



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
        a = (p2[0]-p1[0]) * (p2[1]+p1[1])
        sum += a
    if cartesian:
        return sum >= 0
    else:
        return sum < 0

def isCounterClockwise(a, b, c):
    assert(all([isinstance(x, np.ndarray) for x in [a,b,c]]))
    offset_b = b - a
    offset_c = c - a
    crossed = np.cross(offset_b, offset_c)
    return crossed >= 0

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
    #apply to 1d slices, this allows multiple points to be
    #passed into the function together,
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


def randomRad(min=-TWOPI,max=TWOPI, shape=(1,)):
    """ Get a random value within the range of radians -2pi -> 2pi """ 
    return min + (np.random.random(shape) * (max-min)) 

def rotMatrix(rad):
    """ Get a matrix for rotating a point by an amount of radians """
    return np.array([[cos(rad),-sin(rad)],
                     [sin(rad),cos(rad)]])

#------------------------------
# def point functions
#------------------------------
def nodeToPosition(x,y):
    return [NODE_RECIPROCAL * x, NODE_RECIPROCAL * y]


def calculateSinglePoint(point):
    #only a single point passed in, move in a random direction
    d = np.array([sin(random()*TWOPI),cos(random()*TWOPI)]) * (2 * DELTA)
    return point + d

def calculateVectorPoint(p1,p2):
    #passed in a pair of points, move in the direction of the vector
    #get the direction:
    vector = p2 - p1
    mag = np.sqrt(np.sum(np.square(vector)))
    normalizedVector = vector / mag
    moveVector = normalizedVector * (2 * DELTA)
    jiggledVector = moveVector * np.array([sin(random()*BALPI),cos(random()*BALPI)])
    return p2 + jiggledVector 

def sort_coords(arr):
    """ Sort a list of points by x then y value  """
    ind = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[ind]

def random_points(n):
    """ utility to get n 2d points """
    return np.random.random(n*2)

#------------------------------
# def bbox functions
#------------------------------
def bbox_to_lines(bbox, epsilon=EPSILON):
    """ take in the min and max values of a bbox,
    return back a list of 4 lines with the enum designating their position """
    assert(isinstance(bbox, np.ndarray))
    assert(bbox.shape == (4,))
    bbox_e = bbox + np.array([-epsilon, -epsilon, epsilon, epsilon])
    # [[minx, miny],[maxx, maxy]] -> [[minx,maxx], [miny, maxy]]
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

def bound_line_in_bbox(line, bbox):
    #todo: take in line,  intersect with lines of bbox,
    #replace original line endpoint with intersection point
    bbl = bbox_to_lines(bbox)
    intersections = [x for x in [intersect(line, x) for x,y in bbl] if x is not None]
    if len(intersections) == 0:
        return [line]
    return [np.array([line[0], x]) for x in intersections]

def calc_bbox_corner(bbox, ies, epsilon=EPSILON):
    assert(isinstance(bbox, np.ndarray))
    assert(bbox.shape == (4,))
    assert(isinstance(ies, set))
    hb = IntersectEnum.HBOTTOM
    ht = IntersectEnum.HTOP
    vl = IntersectEnum.VLEFT
    vr = IntersectEnum.VRIGHT
    bbox_e = bbox + np.array([-epsilon, -epsilon, epsilon, epsilon])
    # [[minx, miny],[maxx, maxy]] -> [[minx,maxx], [miny, maxy]]
    bbox_t = bbox.reshape((2,2)).transpose()
    #convert the bbox to bounding lines
    selX = np.array([1,0])
    selY = np.array([0,1])
    mins = bbox_t[:,0]
    maxs = bbox_t[:,1]
    minXmaxY = mins * selX + maxs * selY
    maxXminY =  maxs * selX + mins * selY
    
    if ies.issubset([hb, vl]):
        return mins
    elif ies.issubset([hb, vr]):
        return maxXminY
    elif ies.issubset([ht, vl]):
        return minXmaxY
    elif ies.issubset([ht, vr]):
        return maxs
    else:
        raise Exception("Calculating box corner failed for: {}".format(ies))

    
def within_bbox(point, bbox, tolerance=TOLERANCE):
    assert(isinstance(bbox, np.ndarray))
    assert(bbox.shape == (4,))
    assert(isinstance(point, np.ndarray))
    assert(point.shape == (2,))
    modBbox = bbox + np.array([-tolerance, -tolerance, tolerance, tolerance])
    inXBounds = bbox[0] < point[0] and point[0] < bbox[2]
    inYBounds = bbox[1] < point[1] and point[1] < bbox[3]
    return inXBounds and inYBounds

def bbox_centre(bbox):
    assert(isinstance(bbox, np.ndarray))
    assert(bbox.shape == (4,))
    bbox_t = bbox.reshape((2,2)).transpose()
    mins = bbox_t[:,0]
    maxs = bbox_t[:,1]
    ranges = maxs - mins
    midPoint = ranges * 0.5
    return midPoint

def makeBBoxFromPoint(point,i):
    border = HALFDELTA * (1/(i+1))
    return [point[0] - border, point[1] - border, point[0] + border, point[1] + border]

def checksign(a, b):
    """ Test whether two numbers have the same sign """
    return math.copysign(a, b) == a

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

def clamp(n,minn=0,maxn=1):
    """ Clamp a number between min and max,
    could be replaced with np.clip
    """
    return max(min(maxn,n),minn)

def getRanges(a):
    assert(isinstance(a, np.ndarray))
    assert(a.shape[1] == 2)
    ranges = np.array([a.min(axis=0), a.max(axis=0)])
    return ranges.T


#------------------------------
# def DEPRECATED
#------------------------------


def get_normal(p1, p2):
    """ Get the normalized direction from two points """
    raise Exception("Deprecated: Use get_unit_vector")
