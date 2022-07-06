
import io
import os
import sys
from shutil import rmtree

from setuptools import Command
from setuptools import find_packages
from setuptools import setup


NAME = "srgan_pytorch"

REQUIRED = ["torch"]


EXTRAS = {}


here = os.path.abspath(os.path.dirname(__file__))




class UploadCommand(Command):
    description = " for building and publishing"
    user_options = []

    @staticmethod
    def status(s):
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status(" remove previous version")
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass

        self.status("building")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status(" upload")
        os.system("twine upload dist/*")

        sys.exit()


setup(name=NAME,
      packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
      install_requires=REQUIRED,
      extras_require=EXTRAS,
      include_package_data=True,
      license="Apache",
      classifiers=[

          "License :: OSI Approved :: Apache Software License",
          "Programming Language :: Python :: 3 :: Only"
      ],
      cmdclass={
          "upload": UploadCommand,
      },
      )
