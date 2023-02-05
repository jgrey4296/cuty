"""
	Constant strings for data held in face/halfedge/vertex .data fields
"""
from typing import Final
from enum import Enum

FaceE              : Final = Enum("Face Data Enum", "FILL TEXT TEXT_OFFSET STROKE CENTROID CEN_RADIUS NULL STARTVERT STARTRAD WIDTH SAMPLE SAMPLE_POST")
EdgeE              : Final = Enum("HalfEdge Data Enum", "STROKE WIDTH START END STARTRAD ENDRAD NULL TEXT BEZIER SAMPLE BEZIER_SIMPLIFY SAMPLE_POST")
VertE              : Final = Enum("Vertex Data Enum", "STROKE RADIUS NULL TEXT SAMPLE SAMPLE_POST")

EditE              : Final = Enum("Edit return type enum", "MODIFIED NEW")
SampleE            : Final = Enum("Ways to sample a drawing", "CIRCLE VECTOR TARGET ANGLE TRANSFER")
SampleFormE        : Final = Enum("How the SampleFunction should treat the target", "FACE EDGE VERTEX")

EDGE_FOLLOW_GUARD  : Final = 200

#The amount used to nudge forwards/down the sweep line in the y direction
#for line segment intersection in dcel.intersect_halfedges
#Default: -0.1 for cartesian bboxs of larger than 0-1
SWEEP_NUDGE        : Final = - 1e-3

