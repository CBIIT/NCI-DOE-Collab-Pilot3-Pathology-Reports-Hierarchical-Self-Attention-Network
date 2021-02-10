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

from sklearn.metrics import f1_score, confusion_matrix
from keras.models import load_model

import numpy as np
import time

import urllib.request
import ftplib


def main():

    tasks = ['site', 'histology']
    test_x = np.load(r'./data/npy/test_X.npy')
    test_y = np.load(r'./data/npy/test_Y.npy')

    model_name = 'mt_cnn_model.h5'
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


if __name__ == "__main__":
    main()
