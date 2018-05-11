#to enable custom logging, switch to python main_tests.py
all: main math quadratic parabola tree dcel rbtree

main:
	python -m unittest main_tests.py -v


dcel:
	-rm dcel_actual_save_test.dcel
	python test_dcel.py -v

math:
	python -m unittest test_math.py -v

parabola:
	python -m unittest test_parabola.py -v

quadratic:
	python -m unittest test_quadratic.py -v

rbtree:
	python -m unittest test_rbtree.py -v
	python -m unittest test_rbtree_node.py -v

tree:
	python -m unittest test_tree.py -v
