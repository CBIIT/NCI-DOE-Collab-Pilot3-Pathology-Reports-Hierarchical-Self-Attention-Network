import os

import numpy as np
import tensorflow as tf
import sys
import time
from sklearn.metrics import f1_score
import random
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default="data", type=str,
                        help='Provide the path to processed data')
    parser.add_argument('--saved_model', default="model/savedmodels", type=str,
                        help='Provide the path to save or load the model')

    parser.add_argument('--test', action = 'store_true', 
                        help='Use the save model to run inference in test mode')

    args = parser.parse_args()
    return args




class hisan(object):
    '''
    hierarchical self-attention network for text classification
    
    parameters:
      - embedding_matrix: numpy array
        numpy array of word embeddings
        each row should represent a word embedding
        NOTE: the word index 0 is dropped, so the first row is ignored
      - num_classes: list(ints)
        number of unique classes per task
      - max_sents: int
        maximum number of sentences per document
      - max_words: int
        maximum number of words per sentence
      - attention_heads: int (default: 8)
        number of attention heads to use in multihead attention
      - attention_size: int (default: 512)
        dimension size of output embeddings from attention 
      - dropout_keep: float (default: 0.9)
        dropout keep rate for embeddings and attention softmax
      - activation: tensorflow activation function (default: tf.nn.elu)
        activation function to use for convolutional feature extraction
      - lr: float (default: 0.0001)
        learning rate to use for adam optimizer
        
    methods:
      - train(data,labels,batch_size=64,epochs=30,patience=5,validation_data=None,savebest=False,filepath=None)
        train network on given data
      - predict(data,batch_size=64)
        return the predicted labels on each task for given data
      - score(data,labels,batch_size=64)
        calculate the micro and macro f1 scores on each task for given data
      - save(filepath)
        save the model weights to a file
      - load(filepath)
        load model weights from a file
    '''
    def __init__(self,embedding_matrix,num_classes,max_sents,max_words,attention_heads=8,
                 attention_size=512,dropout_keep=0.9,activation=tf.nn.elu,lr=0.0001):

        self.dropout_keep = dropout_keep
        self.dropout = tf.placeholder(tf.float32)
        self.ms = max_sents
        self.mw = max_words
        self.embedding_matrix = embedding_matrix.astype(np.float32)
        self.attention_size = attention_size
        self.attention_heads = attention_heads
        self.activation = activation
        self.num_tasks = len(num_classes)
        self.unk_tok = embedding_matrix.shape[0] - 1
        self.vocab_size = embedding_matrix.shape[0]

        #doc input
        self.doc_input = tf.placeholder(tf.int32, shape=[None,max_sents,max_words])
        batch_size = tf.shape(self.doc_input)[0]
        words_per_line = tf.math.count_nonzero(self.doc_input,2,dtype=tf.int32)
        max_words_ = tf.reduce_max(words_per_line)
        lines_per_doc = tf.math.count_nonzero(words_per_line,1,dtype=tf.int32)
        max_lines_ = tf.reduce_max(lines_per_doc)
        num_words_ = words_per_line[:,:max_lines_]
        num_words_ = tf.reshape(num_words_,(-1,))
        doc_input_reduced = self.doc_input[:,:max_lines_,:max_words_]

        #masks
        skip_lines = tf.not_equal(num_words_,0)
        count_lines = tf.reduce_sum(tf.cast(skip_lines,tf.int32))
        mask_words = tf.cast(tf.sequence_mask(num_words_,max_words_),tf.float32)[skip_lines]
        mask_words1 = tf.tile(tf.expand_dims(mask_words,2),[1,1,self.attention_size])
        mask_words2 = tf.tile(tf.expand_dims(mask_words,2),[self.attention_heads,1,max_words_])
        mask_words3 = tf.tile(tf.expand_dims(mask_words,1),[self.attention_heads,1,1])
        mask_lines = tf.cast(tf.sequence_mask(lines_per_doc,max_lines_),tf.float32)
        mask_lines1 = tf.tile(tf.expand_dims(mask_lines,2),[1,1,self.attention_size])
        mask_lines2 = tf.tile(tf.expand_dims(mask_lines,2),[self.attention_heads,1,max_lines_])
        mask_lines3 = tf.tile(tf.expand_dims(mask_lines,1),[self.attention_heads,1,1])

        #word embeddings
        doc_input_reduced = tf.reshape(doc_input_reduced,(-1,max_words_))[skip_lines]
        word_embeds = tf.gather(tf.get_variable('embeddings',
                      initializer=self.embedding_matrix,dtype=tf.float32),
                      doc_input_reduced)    #batch*max_lines x max_words x embed_dim
        word_embeds = tf.nn.dropout(word_embeds,self.dropout)

        #word self attention
        Q = tf.layers.conv1d(word_embeds,self.attention_size,1,
            padding='same',activation=self.activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        K = tf.layers.conv1d(word_embeds,self.attention_size,1,
            padding='same',activation=self.activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        V = tf.layers.conv1d(word_embeds,self.attention_size,1,
            padding='same',activation=self.activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        Q = tf.multiply(mask_words1,Q)
        K = tf.multiply(mask_words1,K)
        V = tf.multiply(mask_words1,V)

        Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(K,self.attention_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(V,self.attention_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        outputs = tf.add(outputs,(mask_words2-1)*1e10)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.multiply(outputs,mask_words2)
        outputs = tf.matmul(outputs,V_)
        outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
        outputs = tf.multiply(outputs,mask_words1)

        #word target attention
        Q = tf.get_variable('word_Q',(1,1,self.attention_size),
            tf.float32,tf.orthogonal_initializer())
        Q = tf.tile(Q,[count_lines,1,1])

        Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(outputs,self.attention_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(outputs,self.attention_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        outputs = tf.add(outputs,(mask_words3-1)*1e10)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.multiply(outputs,mask_words3)
        outputs = tf.matmul(outputs,V_)
        outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
        sent_embeds = tf.scatter_nd(tf.where(skip_lines),
                            tf.reshape(outputs,(count_lines,self.attention_size)),
                            (batch_size*max_lines_,self.attention_size))
        sent_embeds = tf.reshape(sent_embeds,(batch_size,max_lines_,self.attention_size))
        sent_embeds = tf.nn.dropout(sent_embeds,self.dropout)

        #sent self attention
        Q = tf.layers.conv1d(sent_embeds,self.attention_size,1,
            padding='same',activation=self.activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        K = tf.layers.conv1d(sent_embeds,self.attention_size,1,
            padding='same',activation=self.activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer())
        V = tf.layers.conv1d(sent_embeds,self.attention_size,1,
            padding='same',activation=self.activation,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        Q = tf.multiply(mask_lines1,Q)
        K = tf.multiply(mask_lines1,K)
        V = tf.multiply(mask_lines1,V)

        Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(K,self.attention_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(V,self.attention_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        outputs = tf.add(outputs,(mask_lines2-1)*1e10)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.multiply(outputs,mask_lines2)
        outputs = tf.matmul(outputs,V_)
        outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
        outputs = tf.multiply(outputs,mask_lines1)

        #sent target attention
        Q = tf.get_variable('sent_Q',(1,1,self.attention_size),
            tf.float32,tf.orthogonal_initializer())
        Q = tf.tile(Q,[batch_size,1,1])

        Q_ = tf.concat(tf.split(Q,self.attention_heads,axis=2),axis=0)
        K_ = tf.concat(tf.split(outputs,self.attention_heads,axis=2),axis=0)
        V_ = tf.concat(tf.split(outputs,self.attention_heads,axis=2),axis=0)

        outputs = tf.matmul(Q_,tf.transpose(K_,[0, 2, 1]))
        outputs = outputs/(K_.get_shape().as_list()[-1]**0.5)
        outputs = tf.add(outputs,(mask_lines3-1)*1e10)
        outputs = tf.nn.dropout(tf.nn.softmax(outputs),self.dropout)
        outputs = tf.multiply(outputs,mask_lines3)
        outputs = tf.matmul(outputs,V_)
        outputs = tf.concat(tf.split(outputs,self.attention_heads,axis=0),axis=2)
        self.doc_embeds = tf.nn.dropout(tf.squeeze(outputs,[1]),self.dropout)

        #classification functions
        self.logits = []
        self.predictions = []
        for t in range(self.num_tasks):
            logit = tf.layers.dense(self.doc_embeds,num_classes[t],
                     kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.logits.append(logit)
            self.predictions.append(tf.nn.softmax(logit))

        #loss, accuracy, and training functions
        self.labels = []
        self.loss = 0
        for t in range(self.num_tasks):
            label = tf.placeholder(tf.int32,shape=[None])
            self.labels.append(label)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                                  logits=self.logits[t],labels=label))
            self.loss += loss/self.num_tasks                                      
        self.optimizer = tf.train.AdamOptimizer(lr,0.9,0.99).minimize(self.loss)

        #init op
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _split_heads(self,x,batch_size):
        x = tf.reshape(x,(batch_size,-1,self.attention_heads,
                        int(self.attention_size/self.attention_heads)))
        return tf.transpose(x,perm=[0, 2, 1, 3])

    def _list_to_numpy(self,inputval,noise=False):
    
        batch_size = len(inputval)
        retval = np.zeros((batch_size,self.ms,self.mw))
        for i,doc in enumerate(inputval):
            doc_ = doc
            doc_ = list(doc[doc.nonzero()])
            
            #randomly add padding to front
            if noise:
                pad_amt = np.random.randint(0,self.mw)
                doc_ = [int(self.unk_tok) for i in range(pad_amt)] + doc_
            tokens = len(doc_)

            for j,line in enumerate([doc_[i:i+self.mw] for i in range(0,tokens,self.mw)]):
                line_ = line
                l = len(line_)
                
                #randomly replace tokens
                if noise and np.count_nonzero(line) == self.mw:
                    r_idx = np.random.randint(0,self.mw)
                    line_[r_idx] = np.random.randint(1,self.vocab_size)
                retval[i,j,:l] = line_
        return retval
    
    def train(self,data,labels,batch_size=64,epochs=30,patience=5,
              validation_data=None,savebest=False,filepath=None):
        '''
        train network on given data
        
        parameters:
          - data: numpy array
            2d numpy array (doc x word ids) of input data
          - labels: numpy array
            2d numpy array (task x label ids) of corresponding labels for each task
          - batch_size: integer (default: 64)
            batch size to use during training
          - epochs: int (default: 30)
            number of epochs to train for
          - validation_data: tuple (optional)
            tuple of numpy arrays (X,y) representing validation data
          - savebest: boolean (default: False)
            set to True to save the best model based on validation score per epoch
          - filepath: string (optional)
            path to save model if savebest is set to True
        
        outputs:
            None
        '''
        if savebest==True and filepath==None:
            raise Exception("Please enter a path to save the network")

        if validation_data:
            validation_size = len(validation_data[0])
        else:
            validation_size = len(data)

        print('training network on %i documents, validation on %i documents' \
              % (len(data), validation_size))

        #track best model for saving
        bestloss = np.inf
        pat_count = 0

        for ep in range(epochs):

            #shuffle data
            labels.append(data)
            xy = list(zip(*labels))
            random.shuffle(xy)
            shuffled = list(zip(*xy))
            data = list(shuffled[-1])
            labels = list(shuffled[:self.num_tasks])

            y_preds = [[] for i in range(self.num_tasks)]
            y_trues = [[] for i in range(self.num_tasks)]
            start_time = time.time()

            #train
            for start in range(0,len(data),batch_size):

                #get batch index
                if start+batch_size < len(data):
                    stop = start+batch_size
                else:
                    stop = len(data)

                inputvals = self._list_to_numpy(data[start:stop],noise=True)
                feed_dict = {self.doc_input:inputvals,self.dropout:self.dropout_keep}
                for t in range(self.num_tasks):
                    feed_dict[self.labels[t]] = labels[t][start:stop]
                retvals = self.sess.run(self.predictions + [self.optimizer,self.loss],
                                        feed_dict=feed_dict)
                loss = retvals[-1]

                #track correct predictions
                for t in range(self.num_tasks):
                    y_preds[t].extend(np.argmax(retvals[t],1))
                    y_trues[t].extend(labels[t][start:stop])
                print("epoch %i, sample %i of %i, loss: %f        \r"\
                                 % (ep+1,stop,len(data),loss))
                #sys.stdout.flush()

            #checkpoint after every epoch
            print("\ntraining time: %.2f" % (time.time()-start_time))
            
            for t in range(self.num_tasks):
                micro = f1_score(y_trues[t],y_preds[t],average='micro')
                macro = f1_score(y_trues[t],y_preds[t],average='macro')
                print("epoch %i task %i training micro/macro: %.4f, %.4f" % (ep+1,t+1,micro,macro))

            scores,loss = self.score(validation_data[0],validation_data[1],batch_size=batch_size)
            for t in range(self.num_tasks):
                micro,macro = scores[t]
                print("epoch %i task %i validation micro/macro: %.4f, %.4f" % (ep+1,t+1,micro,macro))
            print("epoch %i validation loss: %.8f" % (ep+1,loss))

            #save if performance better than previous best
            if loss < bestloss:
                bestloss = loss
                pat_count = 0
                if savebest:
                    self.save(filepath)
            else:
                pat_count += 1
                if pat_count >= patience:
                    break

            #reset timer
            start_time = time.time()

    def score(self,data,labels,batch_size=64):
        '''
        calculate the micro and macro f1 scores on each task for given data
        parameters:
          - data: numpy array
            2d numpy array (doc x word ids) of input data
          - labels: numpy array
            2d numpy array (task x label ids) of corresponding labels for each task
          - batch_size: integer (default: 64)
            batch size to use during prediction
        
        outputs:
            list(tuple(floats)), float
            micro and macro f1 scores for each task
            mean prediction loss across all tasks
        '''     
        y_preds = [[] for t in range(self.num_tasks)]
        loss = []
        for start in range(0,len(data),batch_size):
        
            #get batch index
            if start+batch_size < len(data):
                stop = start+batch_size
            else:
                stop = len(data)

            inputvals = self._list_to_numpy(data[start:stop])
            feed_dict = {self.doc_input:inputvals,self.dropout:1.0}
            for t in range(self.num_tasks):
                feed_dict[self.labels[t]] = labels[t][start:stop]
            preds = self.sess.run(self.predictions + [self.loss],feed_dict=feed_dict)
            loss.append(preds[-1])
            for t in range(self.num_tasks):
                y_preds[t].append(np.argmax(preds[t],1))
                
            print("processed %i of %i records        \r" % (stop,len(data)))
            #sys.stdout.flush()
            
        print()
        for t in range(self.num_tasks):
            y_preds[t] = np.concatenate(y_preds[t],0)
            
        scores = []
        for t in range(self.num_tasks):  
            micro = f1_score(labels[t],y_preds[t],average='micro')
            macro = f1_score(labels[t],y_preds[t],average='macro')
            scores.append((micro,macro))
            print('task '+str(t)+' test micro: '+ str(micro))
            print('task '+str(t)+' test macro: '+ str(macro))

        
        return scores,np.mean(loss)

    def predict(self,data,batch_size=64):
        '''
        return the predicted labels on each task for given data
        
        parameters:
          - data: numpy array
            2d numpy array (doc x word ids) of input data
        
        outputs:
            list(numpy array)
            predicted labels for each task
        '''
        y_preds = [[] for t in range(self.num_tasks)]
        
        for start in range(0,len(data),batch_size):
        
            #get batch index
            if start+batch_size < len(data):
                stop = start+batch_size
            else:
                stop = len(data)

            inputvals = self._list_to_numpy(data[start:stop])
            feed_dict = {self.doc_input:inputvals,self.dropout:1.0}
            preds = self.sess.run(self.predictions,feed_dict=feed_dict)
            for t in range(self.num_tasks):
                y_preds[t].append(np.argmax(preds[t],1))
                
            print("processed %i of %i records        \r" % (stop,len(data)))
            #sys.stdout.flush()
            
        print()
        for t in range(self.num_tasks):
            y_preds[t] = np.concatenate(y_preds[t],0)
        
        return y_preds

    def save(self,filename):
        '''
        save the model weights to a file
        
        parameters:
          - filepath: string
            path to save model weights
        
        outputs:
            None
        '''
        print("Saving model to:", filename)
        self.saver.save(self.sess,filename)

    def load(self,filename):
        '''
        load model weights from a file
        
        parameters:
          - filepath: string
            path from which to load model weights
        
        outputs:
            None
        '''
        self.saver.restore(self.sess,filename)

if __name__ == "__main__":


    args = parse_arguments()
    #params
    batch_size = 128
    epochs = 100
    max_lines = 151
    max_words = 10
    num_classes = [25,117]
    embedding_size = 300
    attention_heads = 8
    attention_size = 400
    savepath = args.saved_model
    vocab_size=279836
    
    #create data
    data_dir = args.data_dir
    vocab = np.load(os.path.join(data_dir, 'vocab.npy'))
    npy_dir = os.path.join(data_dir, 'npy')

    train_x = np.load(os.path.join(npy_dir, 'train_X.npy'))
    train_y = np.load(os.path.join(npy_dir, 'train_Y.npy'))
    print(train_y.shape)
    train_y = [train_y[:,i] for i in range(train_y.shape[1])]

    val_x = np.load(os.path.join(npy_dir, 'val_X.npy'))
    val_y = np.load(os.path.join(npy_dir, 'val_Y.npy'))
    val_y = [val_y[:, i] for i in range(val_y.shape[1])]

    test_x = np.load(os.path.join(npy_dir, 'test_X.npy'))
    test_y = np.load(os.path.join(npy_dir, 'test_Y.npy'))
    test_y = [test_y[:,i] for i in range(test_y.shape[1])]

    #make sure save dir exists
    print(os.path.dirname(os.path.realpath(savepath)))
    if not os.path.exists(os.path.dirname(os.path.realpath(savepath))):
        os.makedirs(os.path.dirname(os.path.realpath(savepath)))


    model = hisan(vocab,num_classes,max_lines,max_words,
              attention_heads,attention_size)

    if not args.test:
        #train model

        model.train(train_x,train_y,batch_size,epochs,
            validation_data=(val_x,val_y),
            savebest=True,filepath=savepath)
        model.score(test_x,test_y,batch_size)
    else: 
        #Load the saved model and run inference
        model.load(savepath)
        model.score(test_x,test_y,batch_size)
