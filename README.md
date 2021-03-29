## Pathology Reports Hierarchical Self-Attention Networks (HiSAN)

### Description
This purpose of the HiSAN model is to automate information abstraction (e.g. Cancer site, laterality, behavior, etc.) from unstructured cancer pathology text reports using natural language processing (NLP) and information extraction from free-form texts. 

### User Community
Data scientists interested in classifying free form texts (such as pathology reports, clinical trials, abstracts, and so on). 

### Usability
Data scientists can train the provided untrained model on their own data, or use the trained model to classify the provided test samples. The provided scripts use a pathology report that has been downloaded from the Genomics Data Commons (GDC), converted to text format, cleaned, and preprocessed. Here is an example [report](https://portal.gdc.cancer.gov/legacy-archive/files/a9a42650-4613-448d-895e-4f904285f508).

### Uniqueness
Classification of unstructured text is a classical problem in natural language processing. The community has developed state-of-the-art models like BERT, Bio-BERT, and Transformer. This model has the advantage of working on a relatively long report (that is, over 400 words) and shows robustness in terms of accuracy and speed with relatively small number of unstructured pathology reports. 

### Components
The following components are in the Model and Data Clearinghouse ([MoDaC-TODO](https://modac.cancer.gov/searchTab?dme_data_id=)):
* Original and processed training, validation, and test data.
* Untrained neural network model.
* Trained model weights and topology to be used in inference.

### Technical Details
Refer to this [README](README-technical.md). 

### Author
Biomedical Sciences, Engineering, and Computing (BSEC) Group; Computer Sciences and Engineering Division; Oak Ridge National Laboratory
