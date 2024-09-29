# Project Title
EnhancedBERT: A Python Software Tailored for Arabic Word Sense Disambiguation

# Project Overview

The project title is "EnhancedBERT: A Python Software Tailored for Arabic Word Sense Disambiguation". This is the code used in the article entitled "EnhancedBERT: A feature-rich ensemble model for Arabic word sense disambiguation with statistical analysis and optimized data collection" (https://www.sciencedirect.com/science/article/pii/S1319157823004652). The dataset used is "A comprehensive dataset for Arabic word sense disambiguation" (https://www.data-in-brief.com/article/S2352-3409(24)00558-4/fulltext), the data files are available here: https://data.mendeley.com/datasets/pmdbs9tby8/1

## Project Installation and Setup
This guide will walk you through the installation and setup process for the project, which focuses on disambiguating Arabic words using an ensemble of AraBERT models.

__Prerequisites__  
Before getting started, ensure that you have the following dependencies installed:  

Python 3.x  
pip package manager  

__Dependencies__  
The project requires the following libraries to be installed:  

transformers
farasapy
pyarabic
arabert
camel-tools
NLTK
scikit-learn  
torch
emoji
Numpy
Pandas
re
string 

To install the dependencies, run the following command:  
_pip install transformers farasapy pyarabic arabert camel-tools NLTK scikit-learn torch emoji Numpy Pandas re string_  

Then install the datasets required by CAMeL Tools components, run the following command:
camel_data -i all

__Installation Steps__  
Follow the steps below to install and set up the project:  

Clone the repository from codeocean:
_git clone https://git.codeocean.com/capsule-xxx.git_  
  
Navigate to the project directory:  
_cd your-project_  
  
Create a virtual environment (optional but recommended):  
_python -m venv env_  
  
Activate the virtual environment:  
For Windows: _env\Scripts\activate_  
  
For macOS and Linux: _source env/bin/activate_  


## Usage

Run the main script to start disambiguating words:
_python main.py_

##Fine-tuning:
In addition to testing on WSD data, you can finetune the models saved on your own dataset and then test the model.

## Configuration

No specific configuration is required for this project.

## Contributing

We welcome contributions from the community! To contribute to this project, please follow the guidelines below:

- Clone the repository and create your branch.
- Make your changes and test them thoroughly.
- Submit a pull request clearly explaining the changes you've made.

Please adhere to our code style conventions and ensure your code is well-documented.

## License

This project is licensed under the Apache License 2.0. For more details, see the [License](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) file.

## Support and Contact

If you have any questions, issues, or suggestions, please feel free to open an issue at (https://huggingface.co/research-s/enhancedBERTv01) or contact us at sanaa.kaddoura@zu.ac.ae  

