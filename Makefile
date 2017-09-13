INSTALL_DIR=$(shell echo $(JG_PYLIBS))
LIBNAME=cairo_utils

install: clean
	cp -r ./${LIBNAME} ${INSTALL_DIR}/${LIBNAME}

uninstall:
	rm -r ${INSTALL_DIR}/${LIBNAME}

clean :
	find . -name "__pycache__" | xargs rm -r
	find . -name "*.pyc" | xargs rm

dependencies:
	pip install pyqtree numpy
