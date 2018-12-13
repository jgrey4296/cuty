"""
	Constant strings for data held in face/halfedge/vertex .data fields
"""
#pylint: disable=line-too-long
from enum import Enum

FaceE = Enum("Face Data Enum", "FILL TEXT TEXT_OFFSET STROKE CENTROID CEN_RADIUS NULL STARTVERT STARTRAD WIDTH SAMPLE SAMPLE_POST")
EdgeE = Enum("HalfEdge Data Enum", "STROKE WIDTH START END STARTRAD ENDRAD NULL TEXT BEZIER SAMPLE BEZIER_SIMPLIFY SAMPLE_POST")
VertE = Enum("Vertex Data Enum", "STROKE RADIUS NULL TEXT SAMPLE SAMPLE_POST")

EditE = Enum("Edit return type enum", "MODIFIED NEW")
SampleE = Enum("Ways to sample a drawing", "CIRCLE VECTOR TARGET ANGLE TRANSFER")
SampleFormE = Enum("How the SampleFunction should treat the target", "FACE EDGE VERTEX")

EDGE_FOLLOW_GUARD = 200

#The amount used to nudge forwards/down the sweep line in the y direction
#for line segment intersection in dcel.intersect_halfedges
#Default: -0.1 for cartesian bboxs of larger than 0-1
SWEEP_NUDGE = - 1e-3

FACE_COLOUR=[0.2, 0.2, 0.9, 1]
EDGE_COLOUR=[0.4, 0.8, 0.1, 1]
VERT_COLOUR=[0.9, 0.1, 0.1, 1]
BACKGROUND_COLOUR=[0, 0, 0, 1]
