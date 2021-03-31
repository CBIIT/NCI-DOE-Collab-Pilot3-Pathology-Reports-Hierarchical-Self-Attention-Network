### Model Description
The Pathology Reports Hierarchical Self-Attention Network (HiSAN) is a deep learning model composed of hierarchies where the computationally-expensive recurrent neural network (RNN) layers are replaced with self-attention mechanisms. The model has two "hierarchies". The lower hierarchy takes in one sentence at a time, broken into word embeddings. This hierarchy outputs a weighted sentence embedding based on the words in the sentence that are most relevant to the classification. The upper hierarchy takes in one document at a time, broken into the sentence embeddings from the lower hierarchy. This hierarchy outputs a weighted document embedding based on the sentences in the document that are most relevant to the classification. Dropout is applied to this final document embedding, and it is then fed into a softmax classifier. 

### Completed Model Trans_Validate Template
| Attribute  | Value |
| ------------- | ------------- |
| Model Developer / Point of Contact  | Hong-Jun Yoon |
| Model Name | HiSAN |
| Inputs  | Indices of tokenized text  |
| Outputs  | classification, softmax  |
| Training Data  | sample data available in the repo  |
| Uncertainty Quantification  | N/A  |
| Platform  | TensorFlow   |

### Software Setup
To set up the Python environment needed to train and run this model:
1. Install [conda](https://docs.conda.io/en/latest/) package manager.
2. Clone this repository.
3. Create the environment as shown below.
```bash
   conda env create -f environment.yml -n hisan 
   conda activate hisan 
   ```
### Data Setup

To set up the data, run the following commands from the top level directory of the repository.
1. Download the reports needed to train and test the model, and the trained model file:
   1. Create an account on the Model and Data Clearinghouse ([MoDaC](https://modac.cancer.gov)). 
   2. Run the script  [./data_utils/download_data.py](./data_utils/download_data.py). This script downloads the reports and their corresponding metadata from MoDaC.
   3. When prompted by the training and test scripts, enter your MoDaC credentials.
2. Generate training/validaton/test datasets by running the script [./data_utils/trainTestSplitMetaData.py](./data_utils/trainTestSplitMetaData.py). This script splits the data into training/validation/test datasets and maps the site and histology to integer categories. 
3. Process reports and generate features by running the script [./data_utils/data_handler.py](./data_utils/data_handler.py). This script does the following: 
   * Cleans up punctuation and unecessary tokens from the reports.
   * Generates a dictionary that maps words in the corpus with a least five appearances to a unique index. 
   * Uses the word to index dictionary to encode every report into a numpy vector of size 1500, where 1500 is the maximum number of words in a pathology report. Every element in that array represents the index of the word in the corpus.
   * Generates the corresponding numpy arrays for the training/validation/test datasets.

For more information about the original, cleaned, and generated data, refer to this [README](./data/README.md) file. The system generates all artifacts after you run the data setup commands above.

### Training

To train an HiSAN  model with the sample data, execute the script [tf_mthisan.py](./tf_mthisan.py). 

Here is example output from running the script:

```
$ python tf_mthisan.py 
training network on 4579 documents, validation on 509 documents
...
task 0 test micro: 0.9724842767295597
task 0 test macro: 0.9320871926345243
task 1 test micro: 0.7688679245283019
task 1 test macro: 0.3562566082744524
```

### Inference on Test Dataset
To test the trained model in inference:
1. Download the trained model by running the script (download_model.py)[./data_utils/download_model.py]. 
2. Run the script (tf_mthisan.py)[./tf_mthisan.py] with the --test option set. The script performs the following:
   * Performs inference on the test dataset.
   * Reports the micro, macro F1 scores of the model on the test dataset.

Here is example output from running the script:

```bash
   python tf_mthisan.py --test 
   .....

   task 0 test micro: 0.9661949685534591
   task 0 test macro: 0.9228817377788936
   task 1 test micro: 0.7672955974842768
   task 1 test macro: 0.3501320612252872
```

### Disclaimer
UT-Battelle, LLC and the government make no representations and disclaim all warranties, both expressed and implied. There are no express or implied warranties:
* Of merchantability or fitness for a particular purpose, 
* Or that the use of the software will not infringe any patent, copyright, trademark, or other proprietary rights, 
* Or that the software will accomplish the intended results, 
* Or that the software or its use will not result in injury or damage. 

The user assumes responsibility for all liabilities, penalties, fines, claims, causes of action, and costs and expenses, caused by, resulting from or arising out of, in whole or in part the use, storage or disposal of the software.


### Acknowledgments
This work has been supported in part by the Joint Design of Advanced Computing Solutions for Cancer (JDACS4C) program established by the U.S. Department of Energy (DOE) and the National Cancer Institute (NCI) of the National Institutes of Health.
