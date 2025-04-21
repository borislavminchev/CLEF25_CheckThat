# Task 2: Claim Normalization

Given a noisy, unstructured social media post, the task is to simplify it into a concise form.
This is a text generation task in which systems have to generate the normlized claims for the goven social media posts.

The task comprises two settings:
- **Monolingual**: In this setup, the training, development, and test datasets are all provided for the specific language. The model is trained, validated, and tested using data exclusively from this single language, meaning that all the stages (training, validation, and testing) are confined to the same language. This setup ensures that the model learns language-specific patterns and structures. Languages: English, German, French, Spanish, Portugese, Hindi, Marathi, Punjabi, Tamil, Arabic, Thai, Indonesian, and Polish. 
- **Zero-shot**: In this case, only the test data is available for the specific language, and you are not provided with any training or development data for that language. You are free to use data from other languages for training your models, or you can choose to conduct a zero-shot experiment using LLMs where the model is tested on the target language without being exposed to any training data. This approach evaluates how well the model can generalize to unseen languages. Languages: Dutch, Romanian, Bengali, Telugu, Korean, Greek, and Czech. 

__Table of contents:__

<!-- - [Evaluation Results](#evaluation-results) -->
- [List of Versions](#list-of-versions)
- [Contents of the Task 2 Directory](#contents-of-the-repository)
- [Input Data Format](#input-data-format)
- [Output Data Format](#output-data-format)
- [Evaluation Metrics](#evaluation-metrics)
- [Credits](#credits)

<!-- ## Evaluation Results


## List of Versions
- [20/01/2025] Data released.

<!-- * **subtask-2A-english**
  - [03/05/2023] (unlabeled) test data are released.
  - [21/02/2023] previously released training data contained also validation data, they are now split in two separate files.
  - [30/01/2023] training data are released.
* **subtask-2A-arabic**
  - [03/05/2023] (unlabeled) test data are released.
  - [10/03/2023] training and validation data are released.
* **subtask-2A-dutch**
  - [03/05/2023] (unlabeled) test data are released.
  - [16/03/2023] training and validation data are released.
* **subtask-2A-german**
  - [03/05/2023] (unlabeled) test data are released.
  - [02/03/2023] training and validation data are released.
* **subtask-2A-italian**
  - [03/05/2023] (unlabeled) test data are released.
  - [21/02/2023] validation data are released.
  - [30/01/2023] training data are released.
* **subtask-2A-turkish**
  - [03/05/2023] (unlabeled) test data are released.
  - [02/03/2023] training and validation data are released.
* **subtask-2A-multilingual**
  - [03/05/2023] (unlabeled) test data are released.
  - [23/03/2023] training and validation data are released. -->

## Contents of the Task 2 Directory

- Data folder: [data](./data)
  - Contains a subfolder for  train, test and dev sets. 
  - Each split has subfolders for each language, in the csv format.
<!-- - Main folder: [baseline](./baseline)<br/>
  - Contains a single file, baseline.py, used to train a baseline and provide predictions.
- Main folder: [scorer](./scorer)<br/>
  - Contains a single file, evaluate.py, that checks the format of a submission and evaluate the various metrics. -->
- [README.md](./README.md)

## Input Data Format

The data will be provided as a CSV file with two columns:
> post, <TAB> normalized claim 

## Output Data Format

The output must be a CSV format with only one column:
> normalized claim.

DO NOT shuffle the test data. 

## Evaluation Metrics

We will use the METEOR measure for the ranking of teams.

<!--
There is a limit of 5 runs (total and not per day), and only one person from a team is allowed to submit runs.

Submission Link: Coming Soon

Evaluation File task3/evaluation/CLEF_-_CheckThat__Task3ab_-_Evaluation.txt -->

<!-- ## Scorers

To evaluate the output of your model which should be in the output format required, please run the script below:

> python evaluate.py -g dev_truth.tsv -p dev_predicted.tsv

where dev_predicted.tsv is the output of your model on the dev set, and dev_truth.tsv is the golden label file provided by us.

The file can be used also to validate the format of the submission, simply use the provided test file as gold data.
The evaluation will not be performed, but the format of your input will be checked.


## Baselines

The script to train the baseline is provided in the related directory.
The script can be run as follow:

> python baseline.py -trp train_data.tsv -ttp dev_data.tsv

where train_data.tsv is the file to be used for training and dev_data.tsv is the file on which doing the prediction.

The baseline is a logistic regressor trained on a Sentence-BERT multilingual representation of the data.

<!-- ### Task 3: Multi-Class Fake News Detection of News Articles

For this task, we have created a baseline system. The baseline system can be found at https://zenodo.org/record/6362498
 --> 

## Submission

TBA

## Related Work

Information regarding the task and data can be found in the following paper:

> Megha Sundriyal, Tanmoy Chakraborty, and Preslav Nakov. [From Chaos to Clarity: Claim Normalization to Empower Fact-Checking.](https://aclanthology.org/2023.findings-emnlp.439/) Findings of the Association for Computational Linguistics: EMNLP 2023. 2023. pp. 6594 - 6609.


## Credits
Please find it on the task website: https://checkthat.gitlab.io/clef2025/task2/
