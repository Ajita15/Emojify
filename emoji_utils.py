import csv
import numpy as np
import emoji
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

def read_glove_files(glove_file):
    with open(glove_file) as f:
        words=set()
        words_to_vec_map={}
        for line in f:
            line=line.strip().split()
            curr_word=line[0]
            words.add(curr_word)
            words_to_vec_map[curr_word]=np.array(line[1:],dtype=np.float64)
        i=1
        words_to_index={}
        index_to_words={}
        for word in words:
            words_to_index[word]=i
            index_to_words[i]=word
            i+=1
        return words_to_index,index_to_words,words_to_vec_map

def softmax1(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def softmax2(x):
    e_x=np.exp(x)
    return e_x/e_x.sum()

def read_csv(filename='data/emojify_data.csv'):
    with open(filename) as f:
        csvfile=csv.reader(f)
        phrases=[]
        emoji=[]
        for row in csvfile:
            phrases.append(row[0])
            emoji.append(row[1])
    X=np.asarray(phrases)
    Y=np.asarray(emoji,dtype=int)
    return X,Y

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y

emoji_dictionary = {"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def label_to_emoji(label):
    #converts label to corresponding emoji
    return(emoji.emojize(emoji_dictionary[str(label)]))

def print_predictions(X,pred):
    for i in range(X.shape[0]):
        print(X[i],label_to_emoji(pred[i]))
        
def plot_confusion_matrix(y_actu, y_pred, title='Confusion matrix', cmap=plt.cm.gray_r):
    
    df_confusion = pd.crosstab(y_actu, y_pred.reshape(y_pred.shape[0],), rownames=['Actual'], colnames=['Predicted'], margins=True)
    
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

def predict(X,Y,W,b,word_to_vec_map):
    pred=np.zeros((X.shape[0],1))
    for i in range(X.shape[0]):
        avg=np.zeros((50,))
        line=X[i].lower().split()
        for words in line:
            avg+=words_to_vec_map[words]
        avg=avg/len(line)
        
        a=np.dot(W,avg)+b
        z=softmax1(a)
        pred[i,0]=argmax(z)
    print("Accuracy:{0}".format(np.mean(pred[:]==Y.reshape(Y.shape[0],1)[:])*100))
    return pred