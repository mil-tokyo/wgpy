from setuptools import setup

setup(name='wgpy-webgpu',
      version='1.0.0',
      install_requires=['numpy'],
      description='cupy-like GPU linear algebra module on WebGL',
      packages=['cupy', 'cupy.cuda', 'cupyx', 'cupyx.scipy', 'cupy_backends', 'cupy_backends.runtime', 'cupy_backends.webgpu', 'wgpy', 'wgpy.common', 'wgpy_backends', 'wgpy_backends.runtime', 'wgpy_backends.webgpu'],
      package_dir={'cupy_backends': 'webgpu/cupy_backends', 'wgpy_backends': 'webgpu/wgpy_backends'}
      )
