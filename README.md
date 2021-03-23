## Multi Task-Convolutional Neural Networks (MT-CNN)

##### Author: Biomedical Sciences, Engineering, and Computing (BSEC) Group; Computer Sciences and Engineering Division; Oak Ridge National Laboratory

### Description
MT-CNN is a CNN for natural language processing (NLP) and information Extraction from free-form texts. BSEC group designed the model for information extraction from cancer pathology reports.

### User Community
Data scientists interested in classifying free form texts (such as pathology reports, clinical trials, abstracts, and so on). 

### Usability
The provided untrained model can be used by data scientists to be trained on their own data, or use the trained model to classify the provided test samples. The provided scripts use a pathology report that has been downloaded from the Genomics Data Commons, converted to text format, cleaned, and preprocessed. Here is an example [report](https://portal.gdc.cancer.gov/legacy-archive/files/a9a42650-4613-448d-895e-4f904285f508).

### Uniqueness
Classification of unstructured text is a classical problem in natural language processing. The community has developed state-of-the-art models like BERT, Bio-BERT, and Transformer. This model has the advantage of working on a relatively long report (that is, over 400 words) and shows scalability in terms of accuracy and speed with relatively small number of unstructured pathology reports. 
&#x1F534;_**(Question: Are you saying it should be scalable to a larger number of reports?)**_

### Components
* Original and processed training, validation, and test data.
* Untrained neural network model.
* Trained model weights and topology to be used in inference.

