from setuptools import setup


setup(
    name="rqcopt",
    version="1.0.0",
    author="Christian B. Mendl",
    author_email="christian.b.mendl@gmail.com",
    packages=["rqcopt"],
    url="https://github.com/qc-tum/rqcopt",
    install_requires=[
        "numpy",
        "scipy",
    ],
)
