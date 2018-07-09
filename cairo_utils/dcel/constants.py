"""
	Constant strings for data held in face/halfedge/vertex .data fields
"""
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
