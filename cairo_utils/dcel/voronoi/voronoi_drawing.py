"""
Utility module to assist with drawing and debugging voronoi
"""
import logging as root_logger
from math import nan
from os.path import isdir, join
import numpy as np
import cairo_utils as utils

logging = root_logger.getLogger(__name__)

#Constants:
COLOUR = [0.2, 0.1, 0.6, 1.0]
COLOUR_TWO = [1.0, 0.2, 0.4, 0.5]
SITE_COLOUR = [1, 0, 0, 1]
SITE_RADIUS = 10
CIRCLE_COLOUR = [1, 1, 0, 1]
CIRCLE_COLOUR_INACTIVE = [0, 0, 1, 1]
CIRCLE_RADIUS = 10
BEACH_LINE_COLOUR = [0, 1, 0]
BEACH_LINE_COLOUR_TWO = [1, 1, 0]
BEACH_NO_INTERSECT_COLOUR = [1, 0, 0, 1]
BEACH_RADIUS = 10
SWEEP_LINE_COLOUR = [0, 0, 1, 1]
LINE_WIDTH = 3
NUM_POINTS = 2000

class VoronoiDebug:
    """ Contains methods to draw various stages of the voronoi calculation """

    def __init__(self, n, directory, instance):
        surf, ctx, size, n = utils.drawing.setup_cairo(n=n, scale=False, cartesian=True)
        assert(isdir(directory))
        self.instance = instance
        self.surface = surf
        self.ctx = ctx
        self.draw_size = size
        self.draw_n = n
        self.save_dir = directory
        self.bbox = self.instance.bbox

    def draw_voronoi_diagram(self,
                             i=None,
                             clear=True,
                             faces=True,
                             edges=False,
                             verts=False,
                             text=False):
        """ Draw the final diagram """
        logging.debug("------------------------------")
        logging.debug("Drawing final voronoi diagram")
        dcel = self.instance.finalise_DCEL()
        if clear:
            utils.drawing.clear_canvas(self.ctx, bbox=self.bbox)
        self.ctx.set_source_rgba(*COLOUR)
        #draw sites
        for site in self.instance.sites:
            utils.drawing.draw_circle(self.ctx, *site.loc, 0.007)
        #draw faces
        utils.dcel.dcel_drawing.draw_dcel(self.ctx, dcel,
                                          text=text,
                                          faces=faces, edges=edges, verts=verts)
        utils.drawing.write_to_png(self.surface, join(self.save_dir, "voronoi_debug"))

    def draw_intermediate_states(self, i,
                                 sites=True,
                                 beachline=True,
                                 sweepline=True,
                                 circles=True,
                                 dcel=False,
                                 text=False,
                                 face=None,
                                 ind_face=False):
        """ Top level function to draw intermediate state of the algorithm """
        logging.info("Drawing intermediate state: {}".format(i))
        if ind_face:
            for f in self.instance.dcel.faces:
                self.draw_intermediate_states(f.index,
                                              face=f,
                                              text=True,
                                              sites=sites,
                                              beachline=beachline,
                                              sweepline=sweepline,
                                              circles=circles)

        utils.drawing.clear_canvas(self.ctx, bbox=self.bbox)

        if sites:
            self.draw_sites()
        if beachline:
            self.draw_beach_line_components()
        if sweepline:
            self.draw_sweep_line()
        if circles:
            self.draw_circle_events()
        if dcel:
            #TODO: draw the incomplete lines better
            #backup_dcel_data = self.instance.dcel.export_data()
            #dcelInstance = self.instance.finalise_DCEL()
            utils.dcel.dcel_drawing.draw_dcel(self.ctx, self.instance.dcel,
                                              background_colour=[0, 0, 0, 0],
                                              faces=False,
                                              edges=True,
                                              verts=True,
                                              text=text)
        if face is not None:
            face.draw(self.ctx, clear=False)

        #self.instance.dcel.import_data(backup_dcel_data)
        utils.drawing.write_to_png(self.surface, join(self.save_dir, "voronoi_intermediate"), i=i)

    def draw_sites(self):
        """ Draw the individual sites """
        self.ctx.set_source_rgba(*SITE_COLOUR)
        for site in self.instance.sites:
            utils.drawing.draw_circle(self.ctx, *site.loc, SITE_RADIUS)

    def draw_circle_events(self):
        """ Draw the circle events """
        for event in self.instance.circles:
            if event.active:
                self.ctx.set_source_rgba(*CIRCLE_COLOUR)
                utils.drawing.draw_circle(self.ctx, *event.loc, CIRCLE_RADIUS)
            else:
                self.ctx.set_source_rgba(*CIRCLE_COLOUR_INACTIVE)
                utils.drawing.draw_circle(self.ctx, *event.loc, CIRCLE_RADIUS)

    def draw_beach_line_components(self):
        """ Draw the arcs of the beach line """
        #the arcs themselves
        self.ctx.set_source_rgba(*BEACH_LINE_COLOUR, 0.1)
        xs = np.linspace(self.bbox[0], self.bbox[2], NUM_POINTS)
        for node in self.instance.beachline.nodes:
            arc = node.value
            xys = arc(xs)
            for x, y in xys:
                utils.drawing.draw_circle(self.ctx, x, y, BEACH_RADIUS)
        #--------------------
        #the frontier:
        # Essentially a horizontal travelling sweep line to draw segments
        self.ctx.set_source_rgba(*BEACH_LINE_COLOUR_TWO, 1)
        leftmost_x = nan
        ##Get the chain of arcs:
        chain = self.instance.beachline.get_chain()
        if len(chain) > 1:
            enumerated = list(enumerate(chain))
            logging.debug("Chain: {}".format("".join([str(x) for x in chain])))
            pairs = zip(enumerated[0:-1], enumerated[1:])
            for (i, a), (j, b) in pairs:
                logging.debug("Drawing {} -> {}".format(a, b))
                intersections = a.value.intersect(b.value, self.instance.sweep_position.y())
                logging.debug("Intersections: {}".format(intersections))
                if not bool(intersections):
                    logging.exception("NO INTERSECTION: {} - {}".format(i, j))
                    #Draw the non-intersecting line as red
                    self.ctx.set_source_rgba(*BEACH_NO_INTERSECT_COLOUR)
                    xs = np.linspace(self.bbox[0], self.bbox[2], NUM_POINTS)
                    axyrs = np.column_stack((a.value(xs), BEACH_RADIUS))
                    bxyrs = np.column_stack((b.value(xs), BEACH_RADIUS))

                    utils.drawing.draw_circle(self.ctx, axyrs)
                    utils.drawing.draw_circle(self.ctx, bxyrs)
                    self.ctx.set_source_rgba(*BEACH_LINE_COLOUR_TWO, 1)
                    continue
                #----------
                #intersection xs:
                i_xs = intersections[:, 0]
                #xs that are further right than what we've drawn
                if leftmost_x is nan:
                    valid_xs = i_xs
                else:
                    valid_xs = i_xs[leftmost_x < i_xs]
                if not bool(valid_xs):
                    #nothing valid, try the rest of the arcs
                    logging.debug("No valid xs, continuing")
                    continue
                left_most_intersection = valid_xs.min()
                logging.debug("Arc {0} from {1:.2f} to {2:.2f}".format(i,
                                                                       leftmost_x,
                                                                       left_most_intersection))
                if leftmost_x is nan:
                    leftmost_x = left_most_intersection - 1
                xs = np.linspace(leftmost_x, left_most_intersection, NUM_POINTS)
                #update the position
                leftmost_x = left_most_intersection
                frontier_arc = np.column_stack((a.value(xs), BEACH_RADIUS))
                utils.drawing.draw_circle(self.ctx, frontier_arc)

        if bool(chain) and (leftmost_x is nan or leftmost_x < self.bbox[2]):
            if leftmost_x is nan:
                leftmost_x = 0
            #draw the last arc:
            logging.debug("Final Arc: {}".format(str(chain[-1])))
            logging.debug("Final Arc from {0:.2f} to {1:.2f}".format(leftmost_x, self.bbox[2]))
            xs = np.linspace(leftmost_x, self.bbox[2], NUM_POINTS)
            frontier_arc = np.column_stack((chain[-1].value(xs), BEACH_RADIUS))
            utils.drawing.draw_circle(self.ctx, frontier_arc)

    def draw_sweep_line(self):
        """ Draw the horizontal sweep line """
        if self.instance.sweep_position is None:
            return
        self.ctx.set_source_rgba(*SWEEP_LINE_COLOUR)
        self.ctx.set_line_width(LINE_WIDTH)
        #a tuple
        sweep_event = self.instance.sweep_position
        self.ctx.move_to(self.bbox[0], sweep_event.y())
        self.ctx.line_to(self.bbox[2], sweep_event.y())
        self.ctx.close_path()
        self.ctx.stroke()
