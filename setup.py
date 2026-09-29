from mypyc.build import mypycify
from setuptools import setup

setup(
    name="quatint",

    # mypyc docs say to just set packages simply like this:
    #   packages=['quatint'],
    #
    # However: When I do that, quatint/__init__.py *itself* is included in the wheel which we don't want,
    #   because then the python version will be used instead of the mypyc-compiled pyd version.
    packages=["quatint-stubs"],
    include_package_data=True,
    package_data={'quatint-stubs': ["*.pyi"]},

    ext_modules=mypycify([
        "quatint/__init__.py",
        "quatint/quat.py",
    ]),

    license="MIT",
)
