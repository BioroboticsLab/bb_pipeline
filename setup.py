from distutils.core import setup
from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]
dep_links = [str(req_line.url) for req_line in install_reqs]

setup(
    name='bb_pipeline',
    version='0.0.1',
    description='',
    entry_points={
        'console_scripts': [
            'bb_pipeline = pipeline.scripts.bb_pipeline:main',
            'bb_pipeline_batch = pipeline.scripts.bb_pipeline_batch:main',
        ]
    },
    install_requires=reqs,
    dependency_links=dep_links,
    packages=[
        'pipeline',
        'pipeline.scripts',
    ],
    package_data={
        '': ['pipeline/config.ini']
    }
)
