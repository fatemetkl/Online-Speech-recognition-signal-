import glob
import os
import librosa
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io.wavfile as wf
# from matplotlib.pyplot import specgram
import os
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










parent_dir = 'db'
ts_sub_dirs=["test2"]

ts_features = parse_audio_files(parent_dir,ts_sub_dirs)



training_epochs = 50
n_dim = ts_features.shape[1]
n_classes = 2
n_hidden_units_one = 280 
n_hidden_units_two = 300
sd = 1 / np.sqrt(n_dim)
learning_rate = 0.01



X = tf.placeholder(tf.float32,[None,n_dim])
Y = tf.placeholder(tf.float32,[None,n_classes])

W_1 = tf.Variable(tf.random_normal([n_dim,n_hidden_units_one], mean = 0, stddev=sd))
b_1 = tf.Variable(tf.random_normal([n_hidden_units_one], mean = 0, stddev=sd))
h_1 = tf.nn.tanh(tf.matmul(X,W_1) + b_1)

W_2 = tf.Variable(tf.random_normal([n_hidden_units_one,n_hidden_units_two], 
mean = 0, stddev=sd))
b_2 = tf.Variable(tf.random_normal([n_hidden_units_two], mean = 0, stddev=sd))
h_2 = tf.nn.sigmoid(tf.matmul(h_1,W_2) + b_2)

W = tf.Variable(tf.random_normal([n_hidden_units_two,n_classes], mean = 0, stddev=sd))
b = tf.Variable(tf.random_normal([n_classes], mean = 0, stddev=sd))
y_ = tf.nn.softmax(tf.matmul(h_2,W) + b)

init = tf.global_variables_initializer()

cost_function = -tf.reduce_sum(Y * tf.log(y_))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

cost_history = np.empty(shape=[1],dtype=float)
y_true, y_pred = None, None

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.import_meta_graph('/home/fati/Desktop/-1000.meta')
    # saver.restore(sess,tf.train.latest_checkpoint('./'))
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

    y_pred = sess.run(tf.argmax(y_,1),feed_dict={X: ts_features})