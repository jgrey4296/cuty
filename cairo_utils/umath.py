"""
Math Utilities
TODO: Refactor into submodules
"""
import IPython
import logging as root_logger
from functools import partial
from math import atan2, copysign
import numpy as np
from numpy import cos, sin, pi
from scipy.interpolate import splprep, splev

from .constants import PI, TWOPI, QUARTERPI, EPSILON, TOLERANCE
from .constants import IntersectEnum, DELTA, HALFDELTA, NODE_RECIPROCAL
from .constants import SAMPLE_DATA_LEN, LINE_DATA_LEN, BEZIER_DATA_LEN, CIRCLE_DATA_LEN


logging = root_logger.getLogger(__name__)

def construct_matrix_multiplier(a):
    """ Partial Matrix Mul constructor for use in rotate point
    for slices of a 2d array: """
    assert(isinstance(a, np.ndarray))
    return partial(lambda x, y: x @ y, a)

#------------------------------
# def circle functions
#------------------------------
def displace_around_circle(xys, scale, n, random=None):
    """ displace the data around a scaled noisy circle """
    #pylint: disable=invalid-name
    #Create a circle:
    if random is None:
        random = np.random.random
    t = np.linspace(0, 2*pi, n)
    rotation = np.column_stack((sin(t), cos(t))).transpose()
    #create some noise:
    noise = random(n)
    #jitter the rotation:
    jittered = (rotation * noise)
    #control the amount of this noise to apply
    scaled = (jittered * scale).transpose()
    #apply the noise to the data
    mod_points = xys + scaled
    return mod_points

def sample_circle(xyrs, n, sort_rads=True, sort_radi=True, easing=None, random=None):
    """
    https://stackoverflow.com/questions/5837572
    """
    if random is None:
        random = np.random.random

    xyrs_r = xyrs.reshape((-1, 1, CIRCLE_DATA_LEN)).repeat(n, axis=1)

    r = scale_ndarray(np.sqrt(random((n, 1))), xyrs_r[0,:,4:])
    if sort_radi:
        r.sort(axis=1)

    theta = scale_ndarray(random((n, 1)), xyrs_r[0,:,2:4])
    if sort_rads:
        theta.sort(axis=1)

    r_shaped = r.reshape((xyrs_r.shape[0],n,1))
    theta_shaped = theta.reshape((xyrs_r.shape[0],n,1))

    rot = np.dstack((np.cos(theta_shaped), np.sin(theta_shaped)))
    r_x_rot = r_shaped * rot

    result = xyrs_r[:,:,:2] + r_x_rot
    flattened = result.reshape((-1, 2))
    return flattened

def get_circle_3p(p1, p2, p3, arb_intersect=20000):
    """
    Given 3 points, treat them as defining two chords on a circle,
    intersect them to find the centre, then calculate the radius
    Thus: circumcircle
    """
    #pylint: disable=too-many-locals
    assert(all([isinstance(x, np.ndarray) for x in [p1, p2, p3]]))
    sorted_points = sort_coords(np.array([p1, p2, p3]))
    p1 = sorted_points[0]
    p2 = sorted_points[1]
    p3 = sorted_points[2]

    arb_height = arb_intersect
    #mid points and norms:
    m1 = get_midpoint(p1, p2)
    n1 = get_bisector(m1, p2)
    m2 = get_midpoint(p2, p3)
    n2 = get_bisector(m2, p3)
    #extended norms:
    v1 = m1 + (1 * arb_height * n1)
    v2 = m2 + (1 * arb_height * n2)
    v1_i = m1 + (-1 * arb_height * n1)
    v2_i = m2 + (-1 * arb_height * n2)
    #resulting lines:
    l1 = np.row_stack((m1, v1))
    l2 = np.row_stack((m2, v2))
    l1_i = np.row_stack((m1, v1_i))
    l2_i = np.row_stack((m2, v2_i))
    #intersect extended norms:
    #in the four combinations of directions
    i_1 = intersect(l1, l2)
    i_2 = intersect(l1_i, l2_i)
    i_3 = intersect(l1, l2_i)
    i_4 = intersect(l1_i, l2)
    #get the intersection:
    the_intersect = [x for x in [i_1, i_2, i_3, i_4] if x is not None]
    if the_intersect is None or not bool(the_intersect):
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

def in_circle(centre, radius, points):
    """ Test a set of points to see if they are within a circle's radius """
    d = get_distance_raw(centre, points)
    return d < pow(radius, 2)


#------------------------------
# def interpolation functions
#------------------------------
def granulate(xys, grains=10, mult=2, random=None):
    """ Given a set of points, offset each slightly
    by the direction between the points
    """
    assert(isinstance(xys, np.ndarray))
    assert(len(xys.shape) == 2)
    if random is None:
        random = np.random.random
    directions, hypos = get_directions(xys)
    granulated = np.zeros((1, 2))
    for i, d in enumerate(hypos):
        sub_granules = xys[i, :] + (d * directions[i, :]*(random((grains, 1))) * mult)
        granulated = np.row_stack((granulated, sub_granules))
    return granulated[1:]

def vary(xys, step_size, pix, random=None):
    """
    FIXME : investigate
    for a given set of points, wiggle them slightly
    """
    assert(isinstance(xys, np.ndarray))
    assert(len(xys.shape) == 2)
    if random is None:
        random = np.random.random
    r = (1.0-2.0 * random((len(xys), 1)))
    scale = np.reshape(np.arange(len(xys)).astype('float'), (len(xys), 1))
    noise = (r* scale * step_size)
    a = random(len(xys))
    rnd = np.column_stack((np.cos(a), np.sin(a)))
    rnd_noise = rnd * noise
    rnd_noise_pix = rnd_noise * pix
    xys_prime = xys + rnd_noise_pix
    return xys_prime

def _interpolate(xy, num_points, smoothing=0.2):
    """ given a set of points, generate values between those points """
    assert(isinstance(xy, np.ndarray))
    assert(len(xy.shape) == 2)
    assert(xy.shape[0] >= 4)
    assert(xy.shape[1] == 2)
    spline_tuple, _ = splprep([xy[:, 0], xy[:, 1]], s=smoothing)
    interpolate_points = np.linspace(0, 1, num_points)
    smoothed_xy = np.column_stack(splev(interpolate_points, spline_tuple))
    return smoothed_xy

def scale(xs, minmax):
    result = minmax[0] + (xs * (minmax[1] - minmax[0]))
    return result

def scale_ndarray(xs, minmaxs):
    mins = minmaxs[:,0].reshape((-1,1))
    ranges = (minmaxs[:,1] - minmaxs[:,0]).reshape((-1,1))
    result = mins + (xs * ranges)
    return result


#------------------------------
# def direction functions
#------------------------------
def get_random_directions(n=1, random=None):
    """ Choose a direction of cardinal and intercardinal directions """
    dirs = [-1, 0, 1]
    if random is None:
        random = lambda a, x: np.random.choice(a, size=x*2, replace=True, p=None)
    result = random(dirs, n).reshape((n, 2))
    return result

def get_directions(xys):
    """ Given a set of points, get the unit direction
    from each point to the next point
    """
    assert(isinstance(xys, np.ndarray))
    assert(len(xys.shape) == 2)
    #convert to vectors:
    #xysPrime.shape = (n, 4)
    #Leading point first to prevent wrap deformation
    # xys_prime = np.column_stack((xys[1:, :], xys[:-1, :]))
    # dx = xys_prime[:, 2] - xys_prime[:, 0]
    # dy = xys_prime[:, 3] - xys_prime[:, 1]
    ds = xys[:-1] - xys[1:]
    #radians:
    arc = np.arctan2(ds[:,1], ds[:,0])
    directions = np.column_stack([np.cos(arc), np.sin(arc)])
    #hypotenuse
    hypos = np.sqrt(np.square(ds[:,0])+np.square(ds[:,1]))
    return np.column_stack((directions, hypos))


#------------------------------
# def line functions
#------------------------------
def intersect(line_1, line_2, tolerance=TOLERANCE):
    """ Get the intersection points of two line segments
    see: http://ericleong.me/research/circle-line/
    so line_1:(start, end), line_2:(start, end)
    returns np.array([x, y]) of intersection or None
    """
    #pylint: disable=too-many-locals
    assert(isinstance(line_1, np.ndarray))
    assert(isinstance(line_2, np.ndarray))
    assert(line_1.shape == (2, 2))
    assert(line_2.shape == (2, 2))
    #The points
    p0 = line_1[0]
    p1 = line_1[1]
    p2 = line_2[0]
    p3 = line_2[1]

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
    xyb = np.array([xb, yb])

    l1mins = np.min((p0, p1), axis=0) - tolerance
    l2mins = np.min((p2, p3), axis=0) - tolerance
    l1maxs = np.max((p0, p1), axis=0) + tolerance
    l2maxs = np.max((p2, p3), axis=0) + tolerance

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

def extend_line(p1, p2, m=1, from_start=True):
    """ Extend a line by m units
    Returns the new end points only
    """
    n = get_unit_vector(p1, p2)
    if from_start:
        el = p1 + (n * m)
    else:
        el = p2 + (n * m)
    return el

def is_point_on_line(p, l):
    """ Test to see if a point is on a line """
    assert(isinstance(p, np.ndarray))
    assert(isinstance(l, np.ndarray))
    points = p.reshape((-1, 2))
    the_lines = l.reshape((-1, 2, 2))
    l_mins = the_lines.min(axis=1)
    l_maxs = the_lines.max(axis=1)

    in_bounds_xs = l_mins[:, 0] <= points[:, 0] <= l_maxs[:, 0]
    in_bounds_ys = l_mins[:, 1] <= points[:, 1] <= l_maxs[:, 1]

    if np.allclose((the_lines[:, 0, 0] - the_lines[:, 1, 0]), 0):
        return in_bounds_ys and in_bounds_xs
    slopes = (the_lines[:, 0, 1] - the_lines[:, 1, 1]) / (the_lines[:, 0, 0] - the_lines[:, 1, 0])
    y_intersects = - slopes * the_lines[:, 0, 0] + the_lines[:, 0, 1]
    line_ys = slopes * points[:, 0] + y_intersects
    return np.allclose(line_ys, points[0, 1]) and in_bounds_ys and in_bounds_xs

def make_horizontal_lines(n=1, random=None):
    """ Utility to Describe a horizontal line as a vector of start and end points  """
    if random is None:
        random = np.random.random
    x = random((n, 2)).sort()
    y = random(n).reshape((-1, 1))
    return np.column_stack((x[:, 0], y, x[:, 1], y))

def make_vertical_lines(n=1, random=None):
    """ utility Describe a vertical line as a vector of start and end points """
    if random is None:
        random =np.random.random
    x = random(n).reshape((-1, 1))
    y = random((n, 2)).sort()
    return np.column_stack((x, y[:, 0], x, y[:, 1]))

def sample_along_lines(xys, n, easing=None, override=None):
    """ For a set of lines, sample along them n times,
    with easings distribution. can be hard overriden with override
    """
    t = np.linspace(0,1,n)
    if override is not None:
        t = override
    if easing is not None:
        t = easing(t)
    s_points = t
    s_invert = (1 - s_points)
    num_points = s_points.shape[0]
    fade = np.vstack((s_invert,s_points))
    lines = xys.reshape((-1, 2, 2))
    xs = lines[:, :, 0]
    ys = lines[:, :, 1]
    xsr = xs.repeat(num_points).reshape((-1, 2, num_points))
    ysr = ys.repeat(num_points).reshape((-1, 2, num_points))
    xsrf = xsr * fade
    ysrf = ysr * fade
    s_xs = xsrf.sum(axis=1)
    s_ys = ysrf.sum(axis=1)
    paired = np.hstack((s_xs, s_ys))
    reshaped = paired.reshape((-1, num_points, 2), order='F').reshape((-1,2))
    return reshaped

def create_line(xys, n, easing=None):
    """ Given a start and end, create t number of points along that line """
    line = sample_along_lines(xys.reshape((-1,LINE_DATA_LEN)), n, easing=easing)
    return line

def bezier1cp(a_cp_b, t, f=None, p=None):
    """ Given the start, end, and a control point, create t number of points along that bezier
    t : the number of points to linearly create used to sample along the bezier
    f : a transform function for the sample points prior to calculate bezier
    p : an overriding set of arbitrary sample points for calculate bezier
    """
    #pylint: disable=unused-variable
    assert(isinstance(a_cp_b, np.ndarray))
    if p is not None:
        assert(isinstance(p, np.ndarray))
        sample_points = p
    else:
        sample_points = np.linspace(0, 1, t)
        if f is not None:
            assert(callable(f))
            #f is an easing lookup function
            sample_points = f(t)

    len_samples = len(sample_points)
    quadratic_matrix = np.array([[[1,0,0],[-2,2,0],[1,-2,1]]])
    t_matrix = lambda x: np.array([1, x,  pow(x,2)])

    points = a_cp_b.reshape((-1,3,2))
    results = np.zeros((len(a_cp_b),1,2))
    for t in sample_points:
        t_mul_mat = t_matrix(t) @ quadratic_matrix
        t_lines = t_mul_mat @ points
        results = np.hstack((results, t_lines.reshape(-1,1,2)))
    return results[:,1:,:].reshape((-1,2))

def bezier2cp(a_cpcp_b, n=None, p=None, easing=None):
    """ Given a start, end, and two control points along the way,
    create n number of points along that bezier
    n : The number of points to sample linearly
    f : the transform function for the linear sampling
    p : arbitrary points to use for sampling instead
    Matrix Multiplication algorithm from:
    https://pomax.github.io/bezierinfo/#decasteljau
    """
    #pylint: disable=too-many-locals
    #pylint: disable=unused-variable
    assert(isinstance(a_cpcp_b, np.ndarray))
    if p is not None:
        assert(isinstance(p, np.ndarray))
        sample_points = p
    elif n is not None:
        sample_points = np.linspace(0, 1, n)
    else:
        raise Exception("Neither arbitrary points or n given")
    if easing is not None:
        assert(callable(easing))
        sample_points = easing(sample_points)

    len_samples = len(sample_points)
    cubic_matrix = np.array([[1,0,0,0],[-3,3,0,0],[3,-6,3,0],[-1,3,-3,1]])
    t_matrix = lambda x: np.array([1, x,  pow(x,2), pow(x,3)])

    points = a_cpcp_b.reshape((-1,4,2))
    results = np.zeros((len(a_cpcp_b),1,2))
    for t in sample_points:
        t_mul_mat = t_matrix(t) @ cubic_matrix
        t_lines = t_mul_mat @ points
        results = np.hstack((results, t_lines.reshape(-1,1,2)))
    return results[:,1:,:].reshape((-1,2))


#------------------------------
# def distance functions
#------------------------------
def check_distance_from_points(point, quad_tree, dist=HALFDELTA):
    """ Given a point and a quadtree, return true if the point is within
    the bounds of the quadtree but not near any points """
    bbox = [point[0] - dist, point[1] - dist, point[0] + dist, point[1] + dist]
    area = quad_tree.intersect(bbox)
    inside_canvas = point[0] > 0 and point[0] < 1.0 and point[1] > 0 and point[1] < 1.0
    return (not bool(area)) and inside_canvas

def get_closest_to_focus(focus, possible_points):
    """ Given a set of points, return the point closest to the focus """
    ds = get_distance(focus, possible_points)
    m_d = ds.min()
    i = ds.tolist().index(m_d)
    return possible_points[i]

def get_closest_on_side(ref_point, possible_points, left=True):
    """
    given a reference point and a set of candidates, get the closest
    point on either the left or right of that reference
    """
    subbed = possible_points - ref_point
    if left:
        on_side = subbed[:, 0] < 0
    else:
        on_side = subbed[:, 0] > 0
    try:
        i = on_side.tolist().index(True)
        return possible_points[i]
    except ValueError:
        return None

def get_distance_raw(p1, p2):
    """ Get the non-square-root distance for pairs of points """
    assert(isinstance(p1, np.ndarray))
    assert(isinstance(p2, np.ndarray))
    p1 = p1.reshape(-1, 2)
    p2 = p2.reshape(-1, 2)
    d_squared = pow(p2-p1, 2)
    #summed = dSquared[:, 0] + dSquared[:, 1]
    summed = d_squared.sum(axis=1)
    return summed

def get_distance(p1, p2):
    """ Get the square-root distance of pairs of points """
    assert(isinstance(p1, np.ndarray))
    assert(isinstance(p2, np.ndarray))

    summed = get_distance_raw(p1, p2)
    sqrtd = np.sqrt(summed)
    return sqrtd

def get_distance_xyxy(x1, y1, x2, y2):
    """ Utility to get the raw distance of points as separate x's and y's  """
    return get_distance_raw(np.array([x1, y1]), np.array([x2, y2]))[0]

def get_midpoint(p1, p2):
    """ Given two points, get the point directly between them """
    m = (p1 + p2) / 2
    return m


#------------------------------
# def rotate functions
#------------------------------
def radians_between_points(a, b):
    """ takes np.arrays
        return the radian relation of b to a (source)
        ie: if > 0: anti-clockwise, < 0: clockwise
    """
    c = b - a
    return atan2(c[1], c[0])

def is_clockwise(*args, cartesian=True):
    """ Test whether a set of points are in clockwise order  """
    #based on stackoverflow.
    #sum over edges, if positive: CW. negative: CCW
    #assumes normal cartesian of y bottom = 0
    the_sum = 0
    p1s = args
    p2s = list(args[1:])
    p2s.append(args[0])
    pairs = zip(p1s, p2s)
    for p1, p2 in pairs:
        a = (p2[0]-p1[0]) * (p2[1]+p1[1])
        the_sum += a
    if cartesian:
        return the_sum >= 0
    else:
        return the_sum < 0

def is_counter_clockwise(a, b, c):
    """ Given 3 points, do they form a counter clockwise turn """
    assert(all([isinstance(x, np.ndarray) for x in [a, b, c]]))
    offset_b = b - a
    offset_c = c - a
    crossed = np.cross(offset_b, offset_c)
    return crossed >= 0

def get_bisector(p1, p2, r=False):
    """ With a normalised line, rotate 90 degrees,
    r=True : to the right
    r=False : to the left
    """
    n = get_unit_vector(p1, p2)
    if r:
        n_prime = n.dot([[0, -1],
                         [1, 0]])
    else:
        n_prime = n.dot([[0, 1],
                         [-1, 0]])
    return n_prime

def rotate_point(p, cen=None, rads=None, rad_min=-QUARTERPI, rad_max=QUARTERPI, random=None):
    """ Given a point, rotate it around a centre point by either radians,
    or within a range of radians
    """
    #p1 = cen, p2=point, @ is matrix mul
    if cen is None:
        cen = np.array([0, 0])
    if rads is None:
        use_radians = random_radian(min_v=rad_min, max_v=rad_max, random=random)
        if isinstance(use_rads, np.ndarray):
            use_radians = use_rads[0]
    else:
        use_radians = rads
    #apply to 1d slices, this allows multiple points to be
    #passed into the function together,
    #without messing up the rotation matmul
    rot_m = rotation_matrix(use_radians)
    offset = (p - cen)
    if len(p.shape) == 1:
        applied = rot_m @ offset
    else:
        applied = np.apply_along_axis(construct_matrix_multiplier(rot_m), 1, offset)
    result = cen + applied
    return result

def __rotate_point_obsolete(p, cen, rads):
    """ Does what rotate point does, explicitly instead
    of with matrix multiplication """
    assert(len(p.shape) == 2)
    c = np.cos(rads)
    s = np.sin(rads)
    centred = p - cen
    cos_p = centred * c
    sin_p = centred * s
    nx = cos_p[:, 0] - sin_p[:, 1]
    ny = sin_p[:, 0] + cos_p[:, 1]
    un_centered = np.column_stack((nx, ny)) + cen
    return un_centered

def random_radian(min_v=-TWOPI, max_v=TWOPI, shape=(1,), random=None):
    """ Get a random value within the range of radians -2pi -> 2pi """
    if random is None:
        random = np.random.random
    return min_v + (random(shape) * (max_v-min_v))

def rotation_matrix(rad):
    """ Get a matrix for rotating a point by an amount of radians """
    return np.array([[cos(rad), -sin(rad)],
                     [sin(rad), cos(rad)]])


#------------------------------
# def point functions
#------------------------------
def node_to_position(x, y):
    """ Convert a nodes XY Position to its actual position """
    return [NODE_RECIPROCAL * x, NODE_RECIPROCAL * y]

def calculate_single_point(points, d=DELTA, random=None):
    """ points passed in, move in a random direction """
    if random is None:
        random = np.random.random
    arr = random((points.shape[0], 2)) * TWOPI
    delta = np.array([sin(arr[:, 0]), cos(arr[:, 1])]) * (2 * d)
    return points + delta

def calculate_vector_point(ps, d=DELTA, random=None):
    """ passed in pairs of points, move in the direction of the vector """
    if random is None:
        random = np.random.random
    vector = ps[:, 2:] - ps[:, :2]
    rand_amnt = random((ps.shape[0], 2)) * TWOPI
    mag = np.sqrt(np.sum(np.square(vector)))
    norm_vector = vector / mag
    move_vector = norm_vector * (2 * d)
    jiggled_vector = move_vector * np.array([sin(rand_amnt[:, 0]), cos(rand_amnt[:, 1])])
    return ps[:, 2:] + jiggled_vector

def sort_coords(arr):
    """ Sort a list of points by x then y value  """
    ind = np.lexsort((arr[:, 1], arr[:, 0]))
    return arr[ind]

def random_points(n, random=None):
    """ utility to get n 2d points """
    if random is None:
        random = np.random.random
    return random(n*2)


#------------------------------
# def bbox functions
#------------------------------
def bbox_to_lines(bbox, epsilon=EPSILON):
    """ take in the min and max values of a bbox,
    return back a list of 4 lines with the enum designating their position """
    assert(isinstance(bbox, np.ndarray))
    assert(bbox.shape == (4, ))
    bbox_e = bbox + np.array([-epsilon, -epsilon, epsilon, epsilon])
    # [[minx, miny], [maxx, maxy]] -> [[minx, maxx], [miny, maxy]]
    bbox_t = bbox_e.reshape((2, 2)).transpose()
    #convert the bbox to bounding lines
    select_x = np.array([1, 0])
    select_y = np.array([0, 1])
    mins = bbox_t[:, 0]
    maxs = bbox_t[:, 1]
    min_x_max_y = mins * select_x + maxs * select_y
    max_x_min_y = maxs * select_x + mins * select_y
    lines = [(np.row_stack((mins, max_x_min_y)), IntersectEnum.HBOTTOM),
             (np.row_stack((min_x_max_y, maxs)), IntersectEnum.HTOP),
             (np.row_stack((mins, min_x_max_y)), IntersectEnum.VLEFT),
             (np.row_stack((max_x_min_y, maxs)), IntersectEnum.VRIGHT)]

    return lines

def bound_line_in_bbox(line, bbox):
    """ takes in a line, limits it to be within a bbox """
    #replace original line endpoint with intersection point
    bbl = bbox_to_lines(bbox)
    intersections = [x for x in [intersect(line, x) for x, y in bbl] if x is not None]
    if not bool(intersections):
        return [line]
    return [np.array([line[0], x]) for x in intersections]

def calc_bbox_corner(bbox, ies, epsilon=EPSILON):
    """ Calculate the nearest corner of a bbox for set of existing intersections  """
    assert(isinstance(bbox, np.ndarray))
    assert(bbox.shape == (4, ))
    assert(isinstance(ies, set))
    hb = IntersectEnum.HBOTTOM
    ht = IntersectEnum.HTOP
    vl = IntersectEnum.VLEFT
    vr = IntersectEnum.VRIGHT
    bbox_e = bbox + np.array([-epsilon, -epsilon, epsilon, epsilon])
    # [[minx, miny], [maxx, maxy]] -> [[minx, maxx], [miny, maxy]]
    bbox_t = bbox_e.reshape((2, 2)).transpose()
    #convert the bbox to bounding lines
    select_x = np.array([1, 0])
    select_y = np.array([0, 1])
    mins = bbox_t[:, 0]
    maxs = bbox_t[:, 1]
    min_x_max_y = mins * select_x + maxs * select_y
    max_x_min_y = maxs * select_x + mins * select_y

    if ies.issubset([hb, vl]):
        return mins
    elif ies.issubset([hb, vr]):
        return max_x_min_y
    elif ies.issubset([ht, vl]):
        return min_x_max_y
    elif ies.issubset([ht, vr]):
        return maxs
    else:
        raise Exception("Calculating box corner failed for: {}".format(ies))

def within_bbox(point, bbox, tolerance=TOLERANCE):
    """ Test whether a point is within the given bbox """
    assert(isinstance(bbox, np.ndarray))
    assert(bbox.shape == (4, ))
    assert(isinstance(point, np.ndarray))
    assert(point.shape == (2, ))
    mod_bbox = bbox + np.array([-tolerance, -tolerance, tolerance, tolerance])
    in_x_bounds = mod_bbox[0] < point[0] and point[0] < mod_bbox[2]
    in_y_bounds = mod_bbox[1] < point[1] and point[1] < mod_bbox[3]
    return in_x_bounds and in_y_bounds

def bbox_centre(bbox):
    """ Get the centre of a bbox """
    assert(isinstance(bbox, np.ndarray))
    assert(bbox.shape == (4, ))
    bbox_t = bbox.reshape((2, 2)).transpose()
    mins = bbox_t[:, 0]
    maxs = bbox_t[:, 1]
    ranges = maxs - mins
    mid = ranges * 0.5
    return mid

def make_bbox_from_point(point, i):
    """ Given a centre point and a length, create a bbox """
    border = HALFDELTA * (1/(i+1))
    return [point[0] - border, point[1] - border, point[0] + border, point[1] + border]

def check_sign(a, b):
    """ Test whether two numbers have the same sign """
    return copysign(a, b) == a

def get_min_range_pair(p1, p2):
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

def clamp(n, minn=0, maxn=1):
    """ Clamp a number between min and max,
    could be replaced with np.clip
    """
    return max(min(maxn, n), minn)

def get_ranges(a):
    """ Given pairs, get the ranges of them """
    assert(isinstance(a, np.ndarray))
    assert(a.shape[1] == 2)
    ranges = np.array([a.min(axis=0), a.max(axis=0)])
    return ranges.T


#------------------------------
# def SAMPLING WRAPPER
#------------------------------
def sample_wrapper(func, data, n, radius, colour, easing=None, random=None):
    """ A Wrapper for sample_circle, sample_along_lines,
    bezier1cp and bezier2cp
    Takes the function, applies the data and samp_sig_or_count to it,
    then duplicates the radius and colours to the appropriate count,
    and combines
    """
    assert(callable(func))
    sampled = func(data, n, easing=easing, random=random)
    radius_size = len(data) * n
    repeated_radius = np.repeat(radius, radius_size).reshape((radius_size, 1))
    repeated_colours = np.repeat(colour.reshape((-1,4)), len(sampled), axis=0)
    combined = np.column_stack((sampled, repeated_radius, repeated_colours))
    return combined


#------------------------------
# def DEPRECATED
#------------------------------
def get_normal(p1, p2):
    """ Get the normalized direction from two points """
    raise Exception("Deprecated: Use get_unit_vector")

def angle_between_points(a, b):
    raise Exception("Deprecated: Use radians_between_points")
