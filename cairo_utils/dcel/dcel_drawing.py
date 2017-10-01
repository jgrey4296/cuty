import logging as root_logger
import numpy as np
from cairo_utils.constants import START, END, SMALL_RADIUS, FACE, EDGE, VERTEX
from cairo_utils.drawing import drawRect, drawCircle, clear_canvas, drawText
from cairo_utils.math import get_midpoint
from .constants import FaceE, EdgeE, VertE

logging = root_logger.getLogger(__name__)

def drawDCEL(ctx, dcel, text=False, faces=True, edges=False, verts=False,
             face_colour=[0.2, 0.2, 0.9, 1],
             edge_colour=[0.4, 0.8, 0.1, 1],
             vert_colour=[0.9, 0.1, 0.1, 1]):
    """ A top level function to draw a dcel  """
    if faces:
        ctx.set_source_rgba(*face_colour)
        draw_dcel_faces(ctx, dcel, text=text)

    if edges:
        ctx.set_source_rgba(*edge_colour)
        draw_dcel_edges(ctx, dcel, text=text)

    if verts:
        ctx.set_source_rgba(*vert_colour)
        draw_dcel_vertices(ctx, dcel)

def draw_dcel_faces(ctx, dcel, text=True):
    for f in dcel.faces:
        draw_dcel_single_face(ctx, dcel, f, clear=False, text=text)

def draw_dcel_single_face(ctx, dcel, face, clear=True, force_centre=False, text=True, data_override=None):
    """ Draw a single Face from a dcel. 
    Can be the only thing drawn (clear=True),
    Can be drawn in the centre of the context for debugging (force_centre=True)
    """
    data = face.data.copy()
    if data_override is not None:
        assert(isinstance(data, dict))
        data.update(data_override)
        
    #early exits:
    if len(face.edgeList) < 2:
        return
    if FaceE.NULL in data:
        return
    #Custom Clear
    if clear:
        clear_canvas(ctx)

    #Data Retrieval:
    lineWidth = 0.004
    vertColour = START
    vertRad = SMALL_RADIUS
    faceCol = FACE
    radius = SMALL_RADIUS
    text_string = "F: {}".format(face.index)
    should_offset_text = FaceE.TEXT_OFFSET in data
    drawCentroid = FaceE.CENTROID in data
    
    if drawCentroid and isinstance(data[FaceE.CENTROID], (list, np.ndarray)):
        centroidCol = data[FaceE.CENTROID]
    if FaceE.STARTVERT in data and isinstance(data[FaceE.STARTVERT], (list, np.ndarray)):
        vertColour = data[FaceE.STARTVERT]
    if FaceE.STARTRAD in data:
        vertRad = data[FaceE.STARTRAD]
    if FaceE.FILL in data and isinstance(data[FaceE.FILL], (list, np.ndarray)):
        faceCol = data[FaceE.FILL]
    if FaceE.CEN_RADIUS in data:
        radius = data[FaceE.CEN_RADIUS]
    if FaceE.TEXT in data:
        text_string = data[FaceE.TEXT]
    if FaceE.WIDTH in data:
        lineWidth = data[FaceE.WIDTH]
        
    #Centre to context
    faceCentre = face.getCentroid()
    if force_centre:
        invCentre = -faceCentre
        ctx.translate(*invCentre)
        ctx.translate(0.5, 0.5)
        
    ctx.set_line_width(lineWidth)
    ctx.set_source_rgba(*faceCol)
    #Setup Edges:
    initial = True
    for x in face.getEdges():
        v1, v2 = x.getVertices()
        assert(v1 is not None)
        assert(v2 is not None)
        logging.debug("Drawing Face {} edge {}".format(face.index, x.index))
        logging.debug("Drawing Face edge from ({}, {}) to ({}, {})".format(v1.x, v1.y,
                                                                               v2.x, v2.y))
        if initial:
            ctx.move_to(v1.x, v1.y)
            initial = False
        ctx.line_to(v2.x, v2.y)

        if FaceE.STARTVERT in data:
            ctx.set_source_rgba(*vertColour)
            drawCircle(ctx, v1.x, v1.y, vertRad)

    #****Draw*****
    if FaceE.FILL not in data:
        ctx.stroke()
    else:
        ctx.close_path()
        ctx.fill()

    #Drawing the Centroid point
    ctx.set_source_rgba(*END)
    if drawCentroid:
        ctx.set_source_rgba(*centroidCol)
        drawCircle(ctx, *faceCentre, radius)
        
    #Text Retrieval and drawing
    if text or FaceE.TEXT in data:
        drawText(ctx, *faceCentre, text_string, offset=should_offset_text)
        
    #Reset the forced centre
    if force_centre:
        ctx.translate(-0.5, -0.5)
        ctx.translate(*centre)


def draw_dcel_edges(ctx, dcel, text=True):
    drawnEdges = []
    for edge in dcel.halfEdges:
        if edge.index in drawnEdges or edge.twin.index in drawnEdges:
            continue
        draw_dcel_halfEdge(ctx, edge, clear=False, text=text)
        drawnEdges.append(edge.index)
        drawnEdges.append(edge.twin.index)

def draw_dcel_halfEdge(ctx, halfEdge, clear=True, text=True, data_override=None):
    if clear:
        clear_canvas(ctx)
    data = halfEdge.data.copy()
    if data_override is not None:
        assert(isinstance(data, dict))
        data.update(data_override)
    if EdgeE.NULL in data:
        return
    
    width = 0.002
    colour = EDGE
    startEndPoints = False
    startCol = START
    endCol = END
    startRad = 0.01
    endRad = 0.01
    writeText = "HE:{}.{}".format(halfEdge.index, halfEdge.twin.index)
    
    if EdgeE.WIDTH in data:
        width = data[EdgeE.WIDTH]
    if EdgeE.COLOUR in data:
        colour = data[EdgeE.COLOUR]
    if EdgeE.START in data and isinstance(data[EdgeE.START], (list, np.ndarray)):
        startCol = data[EdgeE.START]
    if EdgeE.END in data and isinstance(data[EdgeE.END], (list, np.ndarray)):
        endCol = data[EdgeE.END]
    if EdgeE.START in data and EdgeE.END in data:
        startEndPoints = True
    if EdgeE.STARTRAD in data:
        startRad = data[EdgeE.STARTRAD]
    if EdgeE.ENDRAD in data:
        endRad = data[EdgeE.ENDRAD]
    if EdgeE.TEXT in data:
        text = True
        if isinstance(data[EdgeE.TEXT], str):
            writeText = data[EdgeE.TEXT]

        
    ctx.set_line_width(width)
    ctx.set_source_rgba(*colour)
    v1, v2 = halfEdge.getVertices()
    assert(v1 is not None)
    assert(v2 is not None)
    centre = get_midpoint(v1.toArray(), v2.toArray())
    logging.debug("Drawing HalfEdge {} : {}, {} - {}, {}".format(halfEdge.index,
                                                                 v1.x,
                                                                 v1.y,
                                                                 v2.x,
                                                                 v2.y))
    ctx.move_to(v1.x, v1.y)
    ctx.line_to(v2.x, v2.y)
    ctx.stroke()
    if startEndPoints:
        ctx.set_source_rgba(*startCol)
        drawCircle(ctx, v1.x, v1.y, startRad)
        ctx.set_source_rgba(*endCol)
        drawCircle(ctx, v2.x, v2.y, endRad)

    if text:
        drawText(ctx, *centre, writeText)

def draw_dcel_vertices(ctx, dcel):
    """ Draw all the vertices in a dcel as dots """
    for v in dcel.vertices:
        draw_dcel_vertex(ctx, v)
        
def draw_dcel_vertex(ctx, vertex, data_override=None):
    data = vertex.data.copy()
    if data_override is not None:
        data.update(data_override)
    if VertE.NULL in data:
        return
        
    vertCol = VERTEX
    vertRad = 0.005
    if VertE.STROKE in data and isinstance(data[VertE.STROKE], (list, np.ndarray)):
        vertCol = data[VertE.STROKE]
    if VertE.RADIUS in data:
        vertRad = data[VertE.RADIUS]
        
    ctx.set_source_rgba(*vertCol)
    drawCircle(ctx, vertex.x, vertex.y, vertRad)
    
        
