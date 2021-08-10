import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="Tuning",
    version="0.6",
    author="Ralf Antonius Timmermann",
    author_email="rtimmermann@astro.uni-bonn.de",
    description="Harpsichord Tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tamburasca/HarpsichordTuning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=['pynput       >=1.7.1',
                      'PyAudio      >=0.2.11',
                      'numpy        >=1.19.1',
                      'scipy        >=1.5.1',
                      'matplotlib   >=3.3.1',
                      'scikit-image >=0.17.2'],
)
