# abdo-deform
## Description

![Pipeline Diagram](pipeline_diagram.png)

## Getting started
```python
# Set up the virtual environment (tested with Python 3.9.5)
python -m venv deform_venv
source deform_venv/bin/activate

# Install required packages
pip install -r requirements.txt
```
## Run deformation pipeline

The deformation pipeline consists of three steps, located in the `deformation_pipeline` folder.
```
├── abdo-deform
│   ├── deformation_pipeline
│       ├── deformation_utils.py
│       ├── step1_calibration.py
│       ├── step2_deformation.py
│       ├── step3_scaling.py
│   ├── data
```

You may place any required input data inside a `data` folder, as the code is structured to load files from there.

### Step 1: Bowel gas calibration
From the root directory (`abdo-deform`), run:

```bash
python deformation_pipeline/step1_calibration.py
```

### Step 2: Apply deformation

```bash
python deformation_pipeline/step1_calibration.py
```

### Step 3: Apply scaling

```bash
python deformation_pipeline/step1_calibration.py
```

## Publication
