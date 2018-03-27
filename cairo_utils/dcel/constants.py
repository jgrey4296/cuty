"""
	Constant strings for data held in face/halfedge/vertex .data fields
"""
from enum import Enum

FaceE = Enum("Face Data Enum", "FILL TEXT TEXT_OFFSET STROKE CENTROID CEN_RADIUS NULL STARTVERT STARTRAD WIDTH")
EdgeE = Enum("HalfEdge Data Enum", "STROKE WIDTH START END STARTRAD ENDRAD NULL TEXT")
VertE = Enum("Vertex Data Enum", "STROKE RADIUS NULL TEXT")

IntersectEnum = Enum("BBox Intersect Edge", "LEFT RIGHT TOP BOTTOM")



