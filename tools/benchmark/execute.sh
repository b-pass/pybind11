#!/bin/sh

docker build . -t pybind-bench
exec docker run --rm -it -v `pwd`:/out/ pybind-bench ./run-bench.sh
