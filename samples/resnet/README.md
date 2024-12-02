# ResNet-18 training on CIFAR-100

This example also shows how to use user-defined package in Pyodide.

By default, it uses subset of CIFAR-100 dataset (5000 images for training and 1000 images for validation) to be able to run on wide range of devices.
To use full set, replace `"/lib/cifar100-subset-5k.zip"` by `"/lib/cifar100.zip"` in `code.py`.

# Setup

You need to build wheel for resnet model definition.

```bash
python setup.py bdist_wheel
```

This outputs `dist/resnet-0.0.1-py3-none-any.whl`.
The file is referenced in `code.py`.
