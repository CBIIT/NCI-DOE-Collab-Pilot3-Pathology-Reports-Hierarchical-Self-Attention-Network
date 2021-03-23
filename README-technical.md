### Model Description
Multi Task-Convolutional Neural Networks (MT-CNN) is a CNN for Natural Language Processing (NLP) and information extraction from free-form texts. The Biomedical Sciences, Engineering, and Computing (BSEC) group designed the model for information extraction from cancer pathology reports.

### Completed Model Trans_Validate Template
&#x1F534;_**(Question: When I formatted this info as a table, I added generic column headings. Are these column headings ok?)**_
| Attribute  | Value |
| ------------- | ------------- |
| Model Developer / Point of Contact  | Hong-Jun Yoon |
| Model Name | MT-CNN |
| Inputs  | Indices of tokenized text  |
| Outputs  | softmax  |
| Training Data  | sample data available in the repo  |
| Uncertainty Quantification  | N/A  |
| Platform  | Keras/Tensorflow   |

### Software Setup
To set up the Python environment needed to train and run this model:
1. Install [conda](https://docs.conda.io/en/latest/) package manager.
2. Clone this repository.
3. Create the environment as shown below.
```bash
   conda env create -f environment.yml -n mt-cnn
   conda activate mt-cnn
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

To train an MT-CNN model with the sample data, execute the script [mt_cnn_exp.py](./mt_cnn_exp.py). This script calls MT-CNN implementation in [keras_mt_shared_cnn.py](./keras_mt_shared_cnn.py). 

Here is example output from running the script:

```
$ python mt_cnn_exp.py
Using TensorFlow backend.
....
Number of classes:  [25, 117]
Model file:  mt_cnn_model.h5
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
Input (InputLayer)              (None, 1500)         0                                            
__________________________________________________________________________________________________
embedding (Embedding)           (None, 1500, 300)    6948900     Input[0][0]                      
__________________________________________________________________________________________________
0_thfilter (Conv1D)             (None, 1500, 100)    90100       embedding[0][0]                  
__________________________________________________________________________________________________
1_thfilter (Conv1D)             (None, 1500, 100)    120100      embedding[0][0]                  
__________________________________________________________________________________________________
2_thfilter (Conv1D)             (None, 1500, 100)    150100      embedding[0][0]                  
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 100)          0           0_thfilter[0][0]                 
__________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalM (None, 100)          0           1_thfilter[0][0]                 
__________________________________________________________________________________________________
global_max_pooling1d_3 (GlobalM (None, 100)          0           2_thfilter[0][0]                 
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 300)          0           global_max_pooling1d_1[0][0]     
                                                                 global_max_pooling1d_2[0][0]     
                                                                 global_max_pooling1d_3[0][0]     
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 300)          0           concatenate_1[0][0]              
__________________________________________________________________________________________________
site (Dense)                    (None, 25)           7525        dropout_1[0][0]                  
__________________________________________________________________________________________________
histology (Dense)               (None, 117)          35217       dropout_1[0][0]                  
==================================================================================================
Total params: 7,351,942
Trainable params: 7,351,942
Non-trainable params: 0
__________________________________________________________________________________________________
None
Train on 4579 samples, validate on 509 samples
.....
.....
Epoch 00024: val_loss did not improve from 1.37346
Epoch 25/100
 - 19s - loss: 0.6999 - site_loss: 0.0430 - histology_loss: 0.2128 - site_acc: 0.9886 - histology_acc: 0.9393 - val_loss: 1.5683 - val_site_loss: 0.1621 - val_histology_loss: 0.9508 - val_site_acc: 0.9607 - val_histology_acc: 0.7937

Epoch 00025: val_loss did not improve from 1.37346
Prediction on test set 
task site test f-score: 0.9599,0.9389
task histology test f-score: 0.8184,0.4192
```

### Inference on Test Dataset
To test the trained model in inference:
1. Download the trained model by running the script (download_model.py)[./data_utils/download_model.py]. 
2. Run the script (mt_cnn_infer.py)[mt_cnn_infer.py] which performs the following:
   * Performs inference on the test dataset.
   * Reports the micro, macro F1 scores of the model on the test dataset.

Here is example output from running the script:

```bash
   python mt_cnn_infer.py
   .....
   Prediction on test set 
   task site test f-score: 0.9662,0.9421
   task histology test f-score: 0.8168,0.4098
   ```

### Inference on a Single Report
To test the model in inference model for a single report, run the script (predictions.py_[./predictions.py]. &#x1F534;_**(Question: The word "model" is mentioned twice. Is that intentional?)**_

This script accepts as input a single txt report, runs inference, and displays the true labels and the inferenced labels. The script uses a default report for prediction. &#x1F534;_**(Questions: Do we want to mention the name of the default report? Does the script use that default report only when no report is specified?)**_

Here is example output from running the script:

```
   python predictions.py
   
   MTCNN Prediction
   prostate
   8140.0
   8141.
   Original Labels
         filename                                             site            histology
   3555  TCGA-2A-AAYO.3889AA76-F350-4DA4-987B-79E8D2349...    "prostate"      8140.0

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
