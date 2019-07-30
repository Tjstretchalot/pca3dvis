"""Uses setuptools to install the pca3dvis module"""
import setuptools
import os

setuptools.setup(
    name='pca3dvis',
    version='0.0.1',
    author='Timothy Moore',
    author_email='mtimothy984@gmail.com',
    description='Visualize 3d matrices or their 3d projections easily',
    license='CC0',
    keywords='pca3dvis animations video mp4 3d',
    url='https://github.com/tjstretchalot/pca3dvis',
    packages=['pca3dvis'],
    long_description=open(
        os.path.join(os.path.dirname(__file__), 'README.md')).read(),
    long_description_content_type='text/markdown',
    install_requires=['pympanim', 'pytypeutils', 'numpy', 'matplotlib',
                      'hdbscan', 'pytweening'],
    classifiers=(
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
        'Topic :: Utilities'),
    python_requires='>=3.6',
)
