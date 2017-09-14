import logging as root_logger
from .constants import START, END, SMALL_RADIUS, FACE
from .drawing import drawRect, drawCircle, clear_canvas, drawText

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

def draw_dcel_single_face(ctx, dcel, face, clear=True, force_centre=False, text=True):
    if clear:
        clear_canvas(ctx)
    if len(face.edgeList) < 2:
        return
    
    if force_centre:
        centre = face.getCentroid()
        invCentre = -centre
        ctx.translate(*invCentre)
        ctx.translate(0.5, 0.5)
    ctx.set_line_width(0.004)
    faceCentre = face.getCentroid()
    
    if text:
        drawText(ctx, *faceCentre, str("F: {}".format(face.index)))
        
    ctx.set_source_rgba(*END)
    drawCircle(ctx, *faceCentre, SMALL_RADIUS)
    #Draw face edges:
    initial = True
    for x in face.getEdges():
        if 'fill' in face.data and isinstance(face.data['fill'], list):
            ctx.set_source_rgba(*face.data['fill'])
        else:
            ctx.set_source_rgba(*FACE)
        v1, v2 = x.getVertices()
        if v1 is not None and v2 is not None:
            logging.debug("Drawing Face {} edge {}".format(face.index, x.index))
            logging.debug("Drawing Face edge from ({}, {}) to ({}, {})".format(v1.x, v1.y,
                                                                               v2.x, v2.y))
            if initial:
                ctx.move_to(v1.x, v1.y)
                initial = False
            #ctx.set_source_rgba(*np.random.random(3), 1)
            ctx.line_to(v2.x, v2.y)

            #additional things to draw:
            #ctx.set_source_rgba(*START)
            #drawCircle(ctx, v1.x, v1.y, SMALL_RADIUS)
    if face.data is None or 'fill' not in face.data:
        ctx.stroke()
    else:
        ctx.close_path()
        ctx.fill()
    if force_centre:
        ctx.translate(-0.5, -0.5)
        ctx.translate(*centre)


def draw_dcel_edges(ctx, dcel, text=True):
    drawnEdges = []
    ctx.set_line_width(0.004)
    for e in dcel.halfEdges:
        i = e.index
        #only draw if the end hasnt been drawn yet:
        if i in drawnEdges:
            continue
        ctx.set_source_rgba(*EDGE)
        v1, v2 = e.getVertices()
        if v1 is not None and v2 is not None:
            centre = get_midpoint(v1.toArray(), v2.toArray())
            ctx.move_to(v1.x, v1.y)
            ctx.line_to(v2.x, v2.y)
            ctx.stroke()
            if text:
                drawText(ctx, *centre, "E: {}".format(i))
                drawText(ctx, v1.x, v1.y-0.05, "S{}".format(i))
                drawText(ctx, v2.x, v2.y+0.05, "{}E".format(i))
            #Record that this line has been drawn
            drawnEdges.append(e.index)
            #drawnEdges.append(e.twin.index)
        else:
            logging.warning("Trying to draw a line thats missing at least one vertex")

def draw_dcel_halfEdge(ctx, halfEdge, clear=True, text=True):
    if clear:
        clear_canvas(ctx)
    ctx.set_line_width(0.002)
    ctx.set_source_rgba(*EDGE)
    v1, v2 = halfEdge.getVertices()
    if v1 is not None and v2 is not None:
        centre = get_midpoint(v1.toArray(), v2.toArray())
        logging.debug("Drawing HalfEdge {} : {}, {} - {}, {}".format(halfEdge.index,
                                                                     v1.x,
                                                                     v1.y,
                                                                     v2.x,
                                                                     v2.y))
        ctx.move_to(v1.x, v1.y)
        ctx.line_to(v2.x, v2.y)
        ctx.stroke()
        ctx.set_source_rgba(*START)
        drawCircle(ctx, v1.x, v1.y, 0.01)
        ctx.set_source_rgba(*END)
        drawCircle(ctx, v2.x, v2.y, 0.01)

        if halfEdge.face is not None:
            centre = halfEdge.face.getCentroid()
            if text:
                drawText(ctx, *centre, "F:{}.{}".format(halfEdge.face.index, halfEdge.index))
        elif text:
            drawText(ctx, *centre, "HE: {}".format(halfEdge.index))

def draw_dcel_vertices(ctx, dcel):
    """ Draw all the vertices in a dcel as dots """
    for v in dcel.vertices:
        ctx.set_source_rgba(*VERTEX)
        if v is not None:
            drawCircle(ctx, v.x, v.y, 0.01)

