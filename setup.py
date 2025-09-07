import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    dependencies = [line.rstrip() for line in f]

setuptools.setup(
    name="Tuning",
    version="3.5.1",
    author="Ralf Antonius Timmermann",
    author_email="rtimmermann@astro.uni-bonn.de",
    description="Harpsichord/Piano Tuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tamburasca/HarpsichordTuning",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause",
        "Operating System :: OS Independent",
    ],
    python_requires='=3.12',
    install_requires=dependencies,
    license='BSD 3-Clause',
)
