#!/bin/sh

# this script creates lib/chainer-5.4.0-py3-none-any.whl

cd lib
git clone --depth 1 --branch v5 https://github.com/chainer/chainer
cd chainer_patch
find . -type f -exec cp {} ../chainer/{} \;
cd ../chainer
pip install build
python -m build
cp dist/chainer-5.4.0-py3-none-any.whl ../
