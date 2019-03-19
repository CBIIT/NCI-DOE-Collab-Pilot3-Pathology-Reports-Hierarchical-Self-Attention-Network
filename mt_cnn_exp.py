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
from keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.metrics import f1_score, confusion_matrix
from keras.models import load_model
import math

import numpy as np
import time
import os

import urllib.request
import ftplib



def main():
    train_x = np.load( './data/train_X.npy' )
    train_y = np.load('./data/train_Y.npy')

    test_x = np.load( './data/test_X.npy' )
    test_y = np.load( './data/test_Y.npy' )

    for task in range(len(train_y[0, :])):
        cat = np.unique(train_y[:, task])
        train_y[:, task] = [np.where(cat == x)[0][0] for x in train_y[:, task]]
        test_y[:, task] = [np.where(cat == x)[0][0] for x in test_y[:, task]]

    max_vocab = np.max( train_x )
    max_vocab2 = np.max( test_x )
    if max_vocab2 > max_vocab:
        max_vocab = max_vocab2

    wv_len = 300
    seq_len = 1500

    wv_mat = np.random.randn( max_vocab + 1, wv_len ).astype( 'float32' ) * 0.1

    #num_classes = np.max( train_y ) + 1

    num_classes = []
    num_classes.append(np.max( train_y[:,0] ) + 1)
    num_classes.append(np.max( train_y[:,1] ) + 1)
    num_classes.append(np.max( train_y[:,2] ) + 1)
    num_classes.append(np.max( train_y[:,3] ) + 1)


    cnn = init_export_network(
        num_classes= num_classes,
        in_seq_len= 1500,
        vocab_size= len( wv_mat ),
        wv_space= len( wv_mat[ 0 ] ),
        filter_sizes= [ 3, 4, 5 ],
        num_filters= [ 100, 100, 100 ],
        concat_dropout_prob = 0.5,
        emb_l2= 0.001,
        w_l2= 0.01,
        optimizer= 'adadelta')

    model_name = 'mt_cnn_model.h5'
    print( model_name )

    print( cnn.summary() )

    validation_data = (
        { 'Input': np.array( test_x ) },
        {
            'Dense0': test_y[ :, 0 ],
            'Dense1': test_y[ :, 1 ],
            'Dense2': test_y[ :, 2 ],
            'Dense3': test_y[ :, 3 ],
        }
    )


    checkpointer = ModelCheckpoint( filepath= model_name, verbose= 1, save_best_only= True )
    stopper = EarlyStopping( monitor= 'val_loss', min_delta= 0, patience= 10, verbose= 0, mode= 'auto' )

    _ = cnn.fit( x= np.array( train_x ),
                 y= [
                     np.array( train_y[ :, 0 ] ),
                     np.array( train_y[ :, 1 ] ),
                     np.array( train_y[ :, 2 ] ),
                     np.array( train_y[ :, 3 ] )
                 ],
                 batch_size= 16,
                 epochs= 100,
                 verbose= 1,
                 validation_data= validation_data, callbacks= [ checkpointer, stopper ] )


if __name__ == "__main__":
    main()
