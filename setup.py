import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    dependencies = [line.rstrip() for line in f]

setuptools.setup(
    name="Tuning",
    version="3.5.0",
    author="Ralf Antonius Timmermann",
    author_email="rtimmermann@astro.uni-bonn.de",
    description="Harpsichord/Piano Tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tamburasca/HarpsichordTuning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='=3.9',
    # install_requires=['pynput       >=1.7.3',
    #                   'PyAudio      >=0.2.11',
    #                   'numpy        >=1.22.3',
    #                   'scipy        >=1.7.3',
    #                   'matplotlib   >=3.5.2',
    #                   'scikit-image >=0.19.2',
    #                   'numdifftools >=0.9.39'],
    install_requires=dependencies
)
