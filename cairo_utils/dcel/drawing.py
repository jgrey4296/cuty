"""
Provides broad drawing support for a dcel
"""
##-- imports
from __future__ import annotations
import logging as root_logger
from ..constants import WIDTH
from ..drawing import clear_canvas
from .constants import FACE_COLOUR, EDGE_COLOUR, VERT_COLOUR, BACKGROUND_COLOUR

##-- end imports

logging = root_logger.getLogger(__name__)

@dataclass
class DCELDrawSettings:
    text     : bool = field(default=False)
    faces    : bool = field(default=True)
    edges    : bool = field(default=False)
    _vertices : bool = field(default=False)

    face_colour : Colour = field(default=FACE_COLOUR)
    edge_colour : Colour = field(default=EDGE_COLOUR)
    vert_colour : Colour = field(default=VERT_COLOUR)
    background  : Colour = field(default=BACKGROUND_COLOUR)

    edge_width  : float  = field()


class DCELDraw:

    @staticmethod
    def draw(ctx, dcel, settings:DCELDrawSettings):
        """ A top level function to draw a dcel  """

        clear_canvas(ctx, colour=settings.background, bbox=dcel.bbox)

        if settings.faces:
            ctx.set_source_rgba(*face_colour)
            DCELDraw._faces(ctx, dcel, text=text)

        if settings.edges:
            ctx.set_source_rgba(*edge_colour)
            DCELDraw._edges(ctx, dcel, text=text, width=edge_width)

        if settings.verts:
            ctx.set_source_rgba(*vert_colour)
            DCELDraw._vertices(ctx, dcel)

    @staticmethod
    def _faces(ctx, dcel, text=True, clear=False):
        """ Draw ll faces in a dcel """
        for f in dcel.faces:
            f.draw(ctx, clear=clear, text=text)

    @staticmethod
    def _edges(ctx, dcel, text=True, width=WIDTH):
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

    @staticmethod
    def _vertices(ctx, dcel):
        """ Draw all the vertices in a dcel as dots """
        for v in dcel.vertices:
            v.draw(ctx)
