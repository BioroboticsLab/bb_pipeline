from distutils.core import setup

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements

install_reqs = parse_requirements("requirements.txt", session=False)
reqs = [str(ir.req) for ir in install_reqs]
dep_links = [str(req_line.url) for req_line in install_reqs]

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
