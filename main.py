import glob
import os
import librosa
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wf
import os
import pyaudio
import wave
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    print("sample_rate")
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
    sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            str=(fn.split('/')[2])[0]
            if str=='n':
                labels=np.append(labels,0)
            else:
                labels=np.append(labels,1)

    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

def parse_audio_files_pred(parent_dir,sub_dirs,file_ext="*.wav"):
    features = np.empty((0,193))
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
    return np.array(features)

parent_dir = 'db'
tr_sub_dirs=["train"]
ts_sub_dirs=["test"]
pred_sub_dirs=["test1"]
tr_features, tr_labels = parse_audio_files(parent_dir,tr_sub_dirs)
ts_features, ts_labels = parse_audio_files(parent_dir,ts_sub_dirs)

pred_features=parse_audio_files_pred(parent_dir,pred_sub_dirs)

tr_labels = one_hot_encode(tr_labels)
ts_labels = one_hot_encode(ts_labels)



training_epochs = 50
n_dim = tr_features.shape[1]
n_classes = 2
n_hidden_units_one = 280 
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01



X = tf.placeholder(tf.float32,[None,n_dim],name="x")
Y = tf.placeholder(tf.float32,[None,n_classes],name="y")

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd),name="W_1")
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd),name="b_1")
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1,name="h_1")

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], 
mean = 0, stddev=sd),name="W_2")
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd),name="b_2")
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2,name="h_2")

W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd),name="W")
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd),name="b")
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b,name='pred')

init = tf.global_variables_initializer()

cost_function = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):            
        _,cost = sess.run([optimizer,cost_function],feed_dict={X:tr_features,Y:tr_labels})
        cost_history = np.append(cost_history,cost)
    
    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})
     
     
    y_true = sess.run(tf.argmax(ts_labels,1))
    print("Test accuracy: ",sess.run(accuracy, 
    	feed_dict={X: ts_features,Y: ts_labels}),3)
    
    saver.save(sess, '/home/fati/Desktop/model/',global_step=1000)
    




saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.import_meta_graph('/home/fati/Desktop/model/-1000.meta')
    graph = tf.get_default_graph()
    
    X=graph.get_tensor_by_name("x:0")
    Y=graph.get_tensor_by_name("y:0")
    W_1 = graph.get_tensor_by_name("W_1:0")
    W_2= graph.get_tensor_by_name("W_2:0")
    b_1=graph.get_tensor_by_name("b_1:0")
    b_2=graph.get_tensor_by_name("b_2:0")
    W=graph.get_tensor_by_name("W:0")
    b=graph.get_tensor_by_name("b:0")


    h_1=graph.get_tensor_by_name("h_1:0")
    h_2=graph.get_tensor_by_name("h_2:0")
    y_=graph.get_tensor_by_name("pred:0")

    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: pred_features})
    print("pred: ", y_pred[0])
