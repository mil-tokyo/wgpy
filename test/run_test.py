import os
import pytest
import micropip
import wgpy_test
from js import pythonIO
await micropip.install('/lib/chainer-5.4.0-py3-none-any.whl')

test_dir = os.path.dirname(os.path.abspath(wgpy_test.__file__))
pytest.main([test_dir+pythonIO.testPath])
