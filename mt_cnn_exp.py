"""
DISCLAIMER
UT-BATTELLE, LLC AND THE GOVERNMENT MAKE NO REPRESENTATIONS AND DISCLAIM ALL WARRANTIES,
BOTH EXPRESSED AND IMPLIED.  THERE ARE NO EXPRESS OR IMPLIED WARRANTIES OF MERCHANTABILITY
OR FITNESS FOR A PARTICULAR PURPOSE, OR THAT THE USE OF THE SOFTWARE WILL NOT INFRINGE ANY
PATENT, COPYRIGHT, TRADEMARK, OR OTHER PROPRIETARY RIGHTS, OR THAT THE SOFTWARE WILL
ACCOMPLISH THE INTENDED RESULTS OR THAT THE SOFTWARE OR ITS USE WILL NOT RESULT IN INJURY
OR DAMAGE.  THE USER ASSUMES RESPONSIBILITY FOR ALL LIABILITIES, PENALTIES, FINES, CLAIMS,
CAUSES OF ACTION, AND COSTS AND EXPENSES, CAUSED BY, RESULTING FROM OR ARISING OUT OF, IN
WHOLE OR IN PART THE USE, STORAGE OR DISPOSAL OF THE SOFTWARE.
"""
"""
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from keras_mt_shared_cnn import init_export_network
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import f1_score, confusion_matrix
from keras.models import load_model
import math

import numpy as np
import time
import os

import urllib.request
import ftplib


def main():
    # mtcnn parameters
    wv_len = 300
    seq_len = 1500
    batch_size = 16
    epochs = 100
    filter_sizes = [3, 4, 5]
    num_filters = [100, 100, 100]
    concat_dropout_prob = 0.5
    emb_l2 = 0.001
    w_l2 = 0.01
    optimizer = 'adam'
    tasks = ['site', 'histology']
    num_tasks = len(tasks)

    train_x = np.load(r'./data/npy/train_X.npy')
    train_y = np.load(r'./data/npy/train_Y.npy')
    val_x = np.load(r'./data/npy/val_X.npy')
    val_y = np.load(r'./data/npy/val_Y.npy')
    test_x = np.load(r'./data/npy/test_X.npy')
    test_y = np.load(r'./data/npy/test_Y.npy')

    max_classes = []
    for task in range(num_tasks):
        cat1 = np.unique(train_y[:, task])
        print(task, len(cat1), cat1)
        cat2 = np.unique(val_y[:, task])
        print(task, len(cat2), cat2)
        cat3 = np.unique(test_y[:, task])
        print(task, len(cat3), cat3)
        cat12 = np.union1d(cat1, cat2)
        cat = np.union1d(cat12, cat3)
        print(task, len(cat), cat)
        max_classes.append(len(cat))

    max_vocab = np.max(train_x)
    max_vocab2 = np.max(test_x)
    if max_vocab2 > max_vocab:
        max_vocab = max_vocab2

    wv_mat = np.random.randn(int(max_vocab) + 1, wv_len).astype('float32') * 0.1

    num_classes = []
    for i in range(num_tasks):
        num_classes.append(max_classes[i])

    print("Number of classes: ", num_classes)

    cnn = init_export_network(
        task_names=tasks,
        num_classes=num_classes,
        in_seq_len=seq_len,
        vocab_size=len(wv_mat),
        wv_space=len(wv_mat[0]),
        filter_sizes=filter_sizes,
        num_filters=num_filters,
        concat_dropout_prob=concat_dropout_prob,
        emb_l2=emb_l2,
        w_l2=w_l2,
        optimizer=optimizer)

    model_name = 'mt_cnn_model.h5'
    print("Model file: ", model_name)

    print(cnn.summary())

    validation_data = (
        {'Input': np.array(val_x)},
        {
            tasks[0]: val_y[:, 0],  # Dense Layer associated with site
            tasks[1]: val_y[:, 1],  # Dense Layer associated with histology
        }
    )

    checkpointer = ModelCheckpoint(filepath=model_name, verbose=1, save_best_only=True)
    stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='auto')

    _ = cnn.fit(x=np.array(train_x),
                y=[
                    np.array(train_y[:, 0]),
                    np.array(train_y[:, 1]),
                ],
                batch_size=batch_size,
                epochs=epochs,
                verbose=2,
                validation_data=validation_data,
                callbacks=[checkpointer, stopper])

    # Predict on Test data
    model = load_model(model_name)
    pred_probs = model.predict(np.array(test_x))
    print('Prediction on test set')
    for t in range(len(tasks)):
        preds = [np.argmax(x) for x in pred_probs[t]]
        pred_max = [np.max(x) for x in pred_probs[t]]
        y_pred = preds
        y_true = test_y[:, t]
        y_prob = pred_max
        micro = f1_score(y_true, y_pred, average='micro')
        macro = f1_score(y_true, y_pred, average='macro')
        print('task %s test f-score: %.4f,%.4f' % (tasks[t], micro, macro))
        #print(confusion_matrix(y_true,y_pred))


if __name__ == "__main__":
    main()
