from setuptools import setup

setup(
    name='common-nlp',
    version='0.1.0',
    packages=[
        'core',
        'core.nn',
        'core.nn.losses',
        'core.data',
        'core.data.batching',
        'core.data.encoders',
        'core.data.converters',
        'core.fileio',
    ],
    url='',
    license='',
    author='Ngoc Do',
    author_email='',
    description='',
    install_requires=['numpy', 'tensorflow'],
)
