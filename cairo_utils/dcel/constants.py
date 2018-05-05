"""
	Constant strings for data held in face/halfedge/vertex .data fields
"""
from enum import Enum

FaceE = Enum("Face Data Enum", "FILL TEXT TEXT_OFFSET STROKE CENTROID CEN_RADIUS NULL STARTVERT STARTRAD WIDTH")
EdgeE = Enum("HalfEdge Data Enum", "STROKE WIDTH START END STARTRAD ENDRAD NULL TEXT")
VertE = Enum("Vertex Data Enum", "STROKE RADIUS NULL TEXT")

EditE = Enum("Edit return type enum", "MODIFIED NEW")

EDGE_FOLLOW_GUARD = 200

#The amount used to nudge forwards/down the sweep line in the y direction
#for line segment intersection in dcel.intersect_halfedges
#Default: -0.1 for cartesian bboxs of larger than 0-1
SWEEP_NUDGE = -0.1
