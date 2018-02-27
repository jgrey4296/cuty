INSTALL_DIR=$(shell echo $(JG_PYLIBS))
LIBNAME=cairo_utils

install: test clean uninstall finstall

finstall : clean uninstall
	cp -rf ./${LIBNAME} ${INSTALL_DIR}/${LIBNAME}

test:
	make -f ./test/Makefile

uninstall:
	-rm -r ${INSTALL_DIR}/${LIBNAME}

clean :
	find . -name "__pycache__" | xargs rm -r
	find . -name "*.pyc" | xargs rm

dependencies:
	pip install pyqtree numpy
