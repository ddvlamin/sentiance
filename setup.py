import os
import setuptools

with open("README.md", "r") as fh:
  long_description = fh.read()

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + '/requirements.txt'
install_requires = [] 
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
  name="sentiance-ddvlamin", # Replace with your own username
  version="0.0.1",
  author="Dieter Devlaminck",
  author_email="ddvlamin@gmail.com",
  description="processing gps sensor data for sentiance",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/ddvlamin/sentiance",
  packages=setuptools.find_packages(),
  classifiers=[
      "Programming Language :: Python :: 3",
      "License :: OSI Approved :: MIT License",
      "Operating System :: OS Independent",
  ],
  python_requires='>=3.8',
  install_requires=install_requires,
  include_package_data=True
)
