from os import walk
import librosa
from sklearn.metrics import confusion_matrix
from sklearn import mixture
from sklearn.metrics import accuracy_score
import numpy as np
trainpath = "train"
testpath = "test"
sr = 16000
for file in walk(trainpath):
    train_labels = [x[10] for x in list(file)[2]]
train_filenames = np.asarray(list(file)[2])
for file in walk(testpath):
    test_labels_actual = [int(x[10]) for x in list(file)[2]]
test_filenames = np.asarray(list(file)[2])
posterior = np.zeros(10*len(test_filenames)).reshape(len(test_filenames), 10)
for digit in range(0,10):
    mfcc_train = []
    train_dif = []
    train_digit_files_index = np.argwhere(np.asarray(train_labels) == str(digit))
    train_digit_files = train_filenames[train_digit_files_index]   
    for x in train_digit_files:
        x = "train"+"/"+x[0]
        y, sr1 = librosa.core.load(x, sr, mono=True)
        yt, index = librosa.effects.trim(y, top_db = 30)
        train_dif.append(len(y) - len(yt))
        q = librosa.feature.mfcc(yt, sr, S = None, n_mfcc = 13, win_length = 320, hop_length = 160).T
        dell = librosa.feature.delta(q)
        deldel = librosa.feature.delta(dell)
        q = np.append(q, dell, axis = 1)
        q = np.append(q, deldel, axis = 1)
        mfcc_train.append(q)
    mfcc_train_matrix = np.concatenate(mfcc_train , axis = 0)
    g = mixture.GaussianMixture(n_components = 16, max_iter = 500)
    g.fit(mfcc_train_matrix)
    test_digit_files = test_filenames
    for i,x in zip(range(len(test_digit_files)), test_digit_files):
        x = "test"+"/"+x
        y, sr1 = librosa.core.load(x, sr, mono=True)
        yt, index = librosa.effects.trim(y, top_db = 30 )
        p = librosa.feature.mfcc(yt, sr, S = None, n_mfcc = 13, win_length = 320, hop_length = 160).T
        dell = librosa.feature.delta(p)
        deldel = librosa.feature.delta(dell)
        p = np.append(p, dell, axis = 1)
        p = np.append(p, deldel, axis = 1)
        posterior[i][digit] = g.score(p)
test_labels_predicted = np.argmax(posterior, axis = 1)
c_m = confusion_matrix(test_labels_actual, test_labels_predicted, labels = [0,1,2,3,4,5,6,7,8,9])       
a_s = accuracy_score(test_labels_actual, test_labels_predicted)
    
