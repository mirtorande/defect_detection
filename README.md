# Defect Detection Project

This project focuses on developing a machine learning model to detect defects in various materials. The goal is to automate the inspection process and improve the accuracy and efficiency of defect detection.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Defect detection is a critical task in manufacturing and quality control. This project leverages machine learning techniques to identify defects in materials, reducing the need for manual inspection and minimizing errors.

## Installation
To get started with this project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/mirtorande/defect_detection.git
cd defect_detection
pip install -r requirements.txt
```

## Usage
To train the model, use the following command:

```bash
python train.py
```

## Model
The model is built using a pre-trained image classification models as a backbone (either ResNet18, ResNet50 or MobileNetV2). Details about the model architecture and training process can be found in the `model_factory.py` file.

## Results
The results of the model, including accuracy and loss metrics, will be saved in the `results` directory. Visualizations of the training process can be found in the `plots` directory.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.
