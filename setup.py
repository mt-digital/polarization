from setuptools import setup

setup(
    name='polarization',
    version='0.1',
    author='Matthew A. Turner',
    author_email='maturner01@gmail.com',

    entry_points='''
        [console_scripts]
        polexp=scripts.runner:cli
    ''',

    install_requires=[
        'appdirs==1.4.3',
        'appnope==0.1.0',
        'click==6.7',
        'decorator==4.0.11',
        'joblib==0.11',
        'networkx==1.11',
        'nose==1.3.7',
        'numpy==1.12.1',
        'packaging==16.8',
        'pexpect==4.2.1',
        'pickleshare==0.7.4',
        'prompt-toolkit==1.0.14',
        'ptyprocess==0.5.1',
        'Pygments==2.2.0',
        'pyparsing==2.2.0',
        'simplegeneric==0.8.1',
        'six==1.10.0',
        'traitlets==4.3.2',
        'wcwidth==0.1.7'
    ]
)
