"""
Provides broad drawing support for a dcel
"""
#pylint: disable=too-many-arguments
import logging as root_logger
from ..constants import WIDTH
from ..drawing import clear_canvas
from .constants import FACE_COLOUR, EDGE_COLOUR, VERT_COLOUR, BACKGROUND_COLOUR


logging = root_logger.getLogger(__name__)


def draw_dcel(ctx, dcel, text=False, faces=True, edges=False, verts=False,
              face_colour=None, edge_colour=None, vert_colour=None,
              background_colour=None, edge_width=WIDTH):
    """ A top level function to draw a dcel  """
    if face_colour is None:
        face_colour = FACE_COLOUR
    if edge_colour is None:
        edge_colour = EDGE_COLOUR
    if vert_colour is None:
        vert_colour = VERT_COLOUR
    if background_colour is None:
        background_colour = BACKGROUND_COLOUR

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
    """ Draw ll faces in a dcel """
    for f in dcel.faces:
        f.draw(ctx, clear=clear, text=text)

def draw_dcel_edges(ctx, dcel, text=True, width=WIDTH):
    """ Draw all edges in a dcel """
    original_text_state = text
    drawn_texts = set()
    for edge in dcel.halfEdges:
        if edge.index not in drawn_texts:
            text = original_text_state
            drawn_texts.add(edge.index)
            drawn_texts.add(edge.twin.index)
        else:
            text = False
        edge.draw(ctx, text=text, width=width)

def draw_dcel_vertices(ctx, dcel):
    """ Draw all the vertices in a dcel as dots """
    for v in dcel.vertices:
        v.draw(ctx)
