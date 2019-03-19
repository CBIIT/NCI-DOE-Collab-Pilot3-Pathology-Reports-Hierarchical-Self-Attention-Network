## Multi Task-Convolutional Neural Networks (MT-CNN)

##### Author: Biomedical Sciences, Engineering and Computing Group, Computer Sciences and Engineering Division, Oak Ridge National Laboratory

MT-CNN is a CNN for Natural Language Processing and Information Extraction from free-form texts. BSEC group designed the model for information extraction from cancer pathology reports.

### Installation

MT-CNN is written and tested in `Python 3.6` with the following dependencies.

- Keras: The Python Deep Learning library
    - `pip install keras`
- TensorFlow: An open source machine learning framework
    - `pip install tensorflow`
- scikit-learn: Machine Learning in Python
    - `pip install scikit-learn`
- NumPy: The fundamental package for scientific computing with Python
    - `pip install numpy`
- SciPy: The Python-based ecosystem of open-source software for mathematics, science, and engineering.
    - `pip install scipy`

### Run with sample data

To train a MT-CNN model with the sample data, execute the script `mt_cnn_exp.py`. This script calls MT-CNN implementation in `keras_mt_shared_cnn.py`

```
$ python mt_cnn_exp.py
Using TensorFlow backend.

__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
Input (InputLayer)              (None, 1500)         0
__________________________________________________________________________________________________
embedding (Embedding)           (None, 1500, 300)    1396200     Input[0][0]
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
Dense0 (Dense)                  (None, 6)            1806        dropout_1[0][0]
__________________________________________________________________________________________________
Dense1 (Dense)                  (None, 2)            602         dropout_1[0][0]
__________________________________________________________________________________________________
Dense2 (Dense)                  (None, 2)            602         dropout_1[0][0]
__________________________________________________________________________________________________
Dense3 (Dense)                  (None, 3)            903         dropout_1[0][0]
==================================================================================================
Total params: 1,760,413
Trainable params: 1,760,413
Non-trainable params: 0
__________________________________________________________________________________________________
None
Train on 1000 samples, validate on 100 samples
Epoch 1/100
496/1000 [=============>................] - ETA: 2:07 - loss: 3.1325 - Dense0_loss: 0.8384 - Dense1_loss: 0.3084 - Dense2_loss: 0.2827 - Dense3_loss: 0.9116 - Dense0_acc: 0.7106 - Dense1_acc: 0.8434 - Dense2_acc: 0.8941 - Dense3_acc: 0.5377

...

```

### Disclaimer
UT-BATTELLE, LLC AND THE GOVERNMENT MAKE NO REPRESENTATIONS AND DISCLAIM ALL WARRANTIES, BOTH EXPRESSED AND IMPLIED. THERE ARE NO EXPRESS OR IMPLIED WARRANTIES OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE USE OF THE SOFTWARE WILL NOT INFRINGE ANY PATENT, COPYRIGHT, TRADEMARK, OR OTHER PROPRIETARY RIGHTS, OR THAT THE SOFTWARE WILL ACCOMPLISH THE INTENDED RESULTS OR THAT THE SOFTWARE OR ITS USE WILL NOT RESULT IN INJURY OR DAMAGE. THE USER ASSUMES RESPONSIBILITY FOR ALL LIABILITIES, PENALTIES, FINES, CLAIMS, CAUSES OF ACTION, AND COSTS AND EXPENSES, CAUSED BY, RESULTING FROM OR ARISING OUT OF, IN WHOLE OR IN PART THE USE, STORAGE OR DISPOSAL OF THE SOFTWARE.


### Acknowledgments
This work has been supported in part by the Joint Design of Advanced Computing Solutions for Cancer (JDACS4C) program established by the U.S. Department of Energy (DOE) and the National Cancer Institute (NCI) of the National Institutes of Health.
