#!/bin/sh

docker build . -t pybind-bench
exec docker run --rm -it -v `pwd`:/out/ -v `pwd`/../..:/workspace/pybind11_branch pybind-bench ./run-bench.sh "$@"
