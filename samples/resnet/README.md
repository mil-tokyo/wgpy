# ResNet-18 training on CIFAR-100

This example also shows how to use user-defined package in Pyodide.

# Setup

You need to build wheel for resnet model definition.

```bash
python setup.py bdist_wheel
```

This outputs `dist/resnet-0.0.1-py3-none-any.whl`.
The file is referenced in `code.py`.
