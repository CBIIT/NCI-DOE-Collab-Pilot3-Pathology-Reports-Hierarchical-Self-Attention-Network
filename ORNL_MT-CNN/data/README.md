## Tokenized Cancer Pathology Report dataset

Provided is the sample cancer pathology report dataset stored as tokenized sequences. Files are stored in as `numpy` array file format.

```
$ python
>>> import numpy as np
>>> x = np.load( 'train_X.npy' )
>>> y = np.load( 'train_Y.npy' )
>>> a = np.load( 'test_X.npy' )
>>> b = np.load( 'test_Y.npy' )
>>> x.shape
(1000, 1500)
>>> y.shape
(1000, 4)
>>> a.shape
(100, 1500)
>>> b.shape
(100, 4)
>>> x[ 0 ]
array([ 35, 197, 232, ...,   0,   0,   0])
>>> y[ 0 ]
array([14,  1,  2,  2])
>>>
```

- Includes 1,000 training samples and 100 testing samples
- There are four annotated tasks per each case (subsite, laterality, behavior and grade)

### Dataset Description

The dataset was created by the synthesized cancer pathology reports. We synthesized text data from Long Short-Term Memory model trained by the actual de-identified report corpus. We tokenized the words in the reports, then truncated/zero padded to 1,500 tokens of words to get fit to the CNN. 
