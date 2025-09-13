#!/bin/bash

if [ ! -e /workspace/pybind11_branch ]; then
	if [ -e /workspace/include ]; then
		ln -s /workspace /workspace/pybind11_branch
	else
		echo "Mount a pybind11 branch to /workspace/pybind11_branch"
		exit 1
	fi
fi

if [ ! -e /workspace/pybind11 ]; then
	git clone https://github.com/pybind/pybind11.git /workspace/pybind11
fi

if [ ! -e /usr/include/nanobind ]; then
	git clone --recursive https://github.com/wjakob/nanobind.git /workspace/nanobind
	cd /workspace/nanobind/
	mkdir build
	cd build
	cmake ..
	make -j4
fi

cd /workspace
python3 microbench.py "$@"

if [ -e /out ]; then
	cp *.png *.svg /out/
fi

