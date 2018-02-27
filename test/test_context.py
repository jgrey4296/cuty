"""
Utility to simplify importing of the module to test, by modifying the search path
"""
import os
import sys
sys.path.insert(0,os.path.abspath('..'))
import cairo_utils
