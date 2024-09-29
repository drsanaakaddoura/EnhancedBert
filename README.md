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

Download the Data and Models
Go to: https://codeocean.com/capsule/2747745/tree/v1
Go to Capsule on top left
Click Export
Checkbox the Include Data
Extract the downloaded zip file
Copy the data folder to working directory

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

## Citation

If you utilize this code, you are implicitly using the associated data. We kindly ask that you cite the following papers in your work:

1) Kaddoura, S., & Nassar, R. (2024). A Comprehensive Dataset for Arabic Word Sense Disambiguation. Data in Brief, 110591. https://doi.org/10.1016/j.dib.2024.110591

2) Kaddoura, S., & Nassar, R. (2024). EnhancedBERT: A Feature-Rich Ensemble Model for Arabic Word Sense Disambiguation with Statistical Analysis and Optimized Data Collection. Journal of King Saud University - Computer and Information Sciences, 36(1), 101911. https://doi.org/10.1016/j.jksuci.2023.101911
