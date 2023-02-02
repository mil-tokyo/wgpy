#!/bin/sh

# rewrites version number of this library.

if [ $# != 1 ]; then
  echo "usage: ./scripts/rewrite-version.sh VERSION"
  echo "VERSION is like 1.2.3"
  exit 1
fi

VERSION=$1
VERSION_REGEX='[0-9]\+\(\.[0-9]\+\)*'

set -ex

sed -i -e "s/\"version\": \"${VERSION_REGEX}\"/\"version\": \"${VERSION}\"/g" package.json
sed -i -e "s/version='${VERSION_REGEX}'/version='${VERSION}'/g" setup_webgl.py setup_webgpu.py setup_test.py
sed -i -e "s/__version__ = \"${VERSION_REGEX}\"/__version__ = \"${VERSION}\"/g" wgpy/__init__.py

sed -i -e "s/wgpy\\(_test\\|_webgl\\|_webgpu\\)-${VERSION_REGEX}-/wgpy\\1-${VERSION}-/g" test/worker.js samples/hello/worker.js samples/mnist/worker.js
