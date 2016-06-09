from distutils.core import setup
from pip.req import parse_requirements

install_reqs = parse_requirements('requirements.txt', session=False)
reqs = [str(ir.req) for ir in install_reqs]
dep_links = [str(req_line.url) for req_line in install_reqs]

setup(
    name='deeppipeline',
    version='0.0.1',
    description='',
    install_requires=reqs,
    dependency_links=dep_links,
    packages=['pipeline']
)
