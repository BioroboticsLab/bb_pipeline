# bb_pipeline

[![Build Status](https://travis-ci.org/BioroboticsLab/bb_pipeline.svg?branch=master)](https://travis-ci.org/BioroboticsLab/bb_pipeline) [![Coverage Status](https://coveralls.io/repos/github/BioroboticsLab/bb_pipeline/badge.svg?branch=master)](https://coveralls.io/github/BioroboticsLab/bb_pipeline?branch=master)

The beesbook pipeline is used to detect and decode the tags of honeybees.

## Usage example

```python
from pipeline import Pipeline
from pipeline.objects import Filename, LocalizerPositions, Saliencies, IDs
from pipeline.pipeline import get_auto_config

pipeline = Pipeline([Filename],  # inputs
                    [LocalizerPositions, Saliencies, IDs],  # outputs
                    **get_auto_config())
results = pipeline(['/local/image/file.jpeg'])
```

```get_auto_config()``` will automatically download and cache the required model files for you. After calling ```pipeline()```, ```results[LocalizerPositions]``` will contain the coordinates of all detected tags, ```results[IDs]``` will contain their IDs and so on.

Read the full documentation at [bb-pipeline.readthedocs.org](http://bb-pipeline.readthedocs.org/).

## License

Copyright 2016 Biorobotics Lab, Berlin

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
