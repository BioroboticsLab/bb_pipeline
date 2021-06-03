from distutils.core import setup

def parse_requirements(filename):
    with open(filename, "r") as file:
        lines = (line.strip() for line in file)
        return [line for line in lines if line and not line.startswith("#")]

reqs = parse_requirements("requirements.txt")
dep_links = [url for url in reqs if "http" in url]
reqs = [req for req in reqs if "http" not in req]
reqs += [url.split("egg=")[-1] for url in dep_links if "egg=" in url]

setup(
    name="bb_pipeline",
    version="2.0.0",
    description="",
    entry_points={
        "console_scripts": [
            "bb_pipeline = pipeline.scripts.bb_pipeline:main",
            "bb_pipeline_api = pipeline.scripts.bb_pipeline_api:main",
            "bb_pipeline_batch = pipeline.scripts.bb_pipeline_batch:main",
            "bb_pipeline_mpi = pipeline.scripts.bb_pipeline_mpi:main",
        ]
    },
    install_requires=reqs,
    dependency_links=dep_links,
    extras_require={
        "ResultCrownVisualizer": ["cairocffi"],
        "bb_pipeline_mpi": ["mpi4py"],
    },
    packages=["pipeline", "pipeline.scripts", "pipeline.stages"],
    package_data={"pipeline": ["config.ini"]},
)
