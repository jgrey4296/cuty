import logging as root_logger
import numpy as np
from ..constants import START, END, SMALL_RADIUS, FACE, EDGE, VERTEX, WIDTH
from ..drawing import drawRect, drawCircle, clear_canvas, drawText
from .Face import Face
from .dcel import DCEL
from ..math import get_midpoint
from .constants import FaceE, EdgeE, VertE


logging = root_logger.getLogger(__name__)

def drawDCEL(ctx, dcel, text=False, faces=True, edges=False, verts=False,
             face_colour=[0.2, 0.2, 0.9, 1],
             edge_colour=[0.4, 0.8, 0.1, 1],
             vert_colour=[0.9, 0.1, 0.1, 1],
             background_colour=[0,0,0,1],
             edge_width=WIDTH):

    """ A top level function to draw a dcel  """
    clear_canvas(ctx, colour=background_colour, bbox=dcel.bbox)
    
    if faces:
        ctx.set_source_rgba(*face_colour)
        draw_dcel_faces(ctx, dcel, text=text)

    if edges:
        ctx.set_source_rgba(*edge_colour)
        draw_dcel_edges(ctx, dcel, text=text, width=edge_width)

    if verts:
        ctx.set_source_rgba(*vert_colour)
        draw_dcel_vertices(ctx, dcel)

def draw_dcel_faces(ctx, dcel, text=True, clear=False):
    for f in dcel.faces:
        sample_data = f.draw(ctx, clear=clear, text=text)

def draw_dcel_edges(ctx, dcel, text=True, width=WIDTH):
    originalTextState = text
    drawnTexts = set()
    for edge in dcel.halfEdges:
        if edge.index not in drawnTexts:
            text = originalTextState
            drawnTexts.add(edge.index)
            drawnTexts.add(edge.twin.index)
        else:
            text = False
        edge.draw(ctx, text=text, width=width)
        
def draw_dcel_vertices(ctx, dcel):
    """ Draw all the vertices in a dcel as dots """
    for v in dcel.vertices:
        sample_data = v.draw(ctx)
        
        
