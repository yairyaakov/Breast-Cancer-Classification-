import keras
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import shuffle
from xgboost import XGBClassifier
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.compat.v1 import disable_v2_behavior
import tensorflow.compat.v1 as tf
disable_v2_behavior()

# Hide TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == '__main__':

    address = r'Breast_Cancer.csv'
    df = pd.read_csv(address)

    # Change features with words to numbers
    labelencoder = LabelEncoder()

    df['Race'] = labelencoder.fit_transform(df['Race'])
    df['Marital Status'] = labelencoder.fit_transform(df['Marital Status'])
    df['N Stage'] = labelencoder.fit_transform(df['N Stage'])
    df['T Stage '] = labelencoder.fit_transform(df['T Stage '])
    df['6th Stage'] = labelencoder.fit_transform(df['6th Stage'])
    df['differentiate'] = labelencoder.fit_transform(df['differentiate'])
    df['A Stage'] = labelencoder.fit_transform(df['A Stage'])
    df['Estrogen Status'] = labelencoder.fit_transform(df['Estrogen Status'])
    df['Progesterone Status'] = labelencoder.fit_transform(df['Progesterone Status'])
    df['Status'] = labelencoder.fit_transform(df['Status'])
    df['Grade'] = labelencoder.fit_transform(df['Grade'])

    # Shuffle the data
    df = shuffle(df)

    x = df.drop('Status', axis=1)
    y = df.Status

    xtrain, xtest, ytrain, ytest = train_test_split(x, y)

    ytrain = ytrain.values[:, np.newaxis]
    ytest = ytest.values[:, np.newaxis]

    features = 15
    classes = 1
    epochs = 10000

    X = tf.placeholder(tf.float32, [None, features])

    Y = tf.placeholder(tf.float32, [None, classes])

    W = tf.Variable(tf.zeros([features, classes], dtype=tf.dtypes.float32, name="weight"))
    b = tf.Variable(tf.zeros([classes], dtype=tf.dtypes.float32, name="bias"))

    pred = tf.nn.sigmoid(tf.matmul(X, W) + b)
    logistic = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=pred)
    loss = tf.reduce_mean(logistic)
    update = tf.compat.v1.train.AdamOptimizer(0.0001).minimize(loss)

    with tf.compat.v1.Session() as sess:

        sess.run(tf.global_variables_initializer())
        for i in range(epochs):

            sess.run(update, feed_dict={X: xtrain, Y: ytrain})

            if i % 100 == 0:
                print("i = ", i, "loss =", loss.eval({X: xtrain, Y: ytrain}))

        print("\nOptimization Finished!\n")

        # Calculate the predictions for the test data
        predictions = sess.run(pred, feed_dict={X: xtest, Y: ytest})

        # Convert the predictions to binary class labels
        predictions_binary = (predictions > 0.5).astype(int)

        # Calculate the accuracy
        accuracy = (predictions_binary == ytest).mean()
        print(f'Accuracy: {accuracy:.2f}')

    # Add layers manually, similar to original code

    XLayers = tf.placeholder(tf.float32, [None, features])
    YLayers = tf.placeholder(tf.float32, [None, classes])

    # Layer 1
    layer_1 = 15
    W1 = tf.Variable(tf.truncated_normal([features, layer_1], dtype=tf.dtypes.float32, stddev=0.01))
    b1 = tf.Variable(tf.constant(0.1, dtype=tf.dtypes.float32, shape=[layer_1]))
    z1 = tf.add(tf.matmul(XLayers, W1), b1)
    a1 = tf.nn.elu(z1)

    # Layer 2
    layer_2 = 15
    W2 = tf.Variable(tf.truncated_normal([layer_1, layer_2], dtype=tf.dtypes.float32, stddev=0.01))
    b2 = tf.Variable(tf.constant(0.1, dtype=tf.dtypes.float32, shape=[layer_2]))
    z2 = tf.add(tf.matmul(XLayers, W2), b2)
    a2 = tf.nn.elu(z2)

    # Layer 3
    layer_3 = 15
    W3 = tf.Variable(tf.truncated_normal([layer_2, layer_3], dtype=tf.dtypes.float32, stddev=0.01))
    b3 = tf.Variable(tf.constant(0.1, dtype=tf.dtypes.float32, shape=[layer_3]))
    z3 = tf.add(tf.matmul(XLayers, W3), b3)
    a3 = tf.nn.elu(z3)

    # Layer 4
    layer_4 = 15
    W4 = tf.Variable(tf.truncated_normal([layer_3, layer_4], dtype=tf.dtypes.float32, stddev=0.01))
    b4 = tf.Variable(tf.constant(0.1, dtype=tf.dtypes.float32, shape=[layer_4]))
    z4 = tf.add(tf.matmul(XLayers, W4), b4)
    a4 = tf.nn.elu(z4)

    W5 = tf.Variable(tf.truncated_normal([layer_4, classes], dtype=tf.dtypes.float32, stddev=0.01))
    b5 = tf.Variable(tf.constant(0.1, dtype=tf.dtypes.float32, shape=[classes]))

    predLayers = tf.add(tf.matmul(z4, W5), b5)
    logisticLayers = tf.nn.sigmoid_cross_entropy_with_logits(labels=YLayers, logits=predLayers)
    lossLayers = tf.reduce_mean(logisticLayers)
    updateLayers = tf.compat.v1.train.AdamOptimizer(0.0001).minimize(lossLayers)

    with tf.compat.v1.Session() as sess:

        sess.run(tf.global_variables_initializer())
        for i in range(epochs):

            sess.run(updateLayers, feed_dict={XLayers: xtrain, YLayers: ytrain})

            if i % 100 == 0:
                print("i = ", i, "loss =", lossLayers.eval({XLayers: xtrain, YLayers: ytrain}))

        print("\nOptimization Finished!\n")

        # Calculate the predictions for the test data
        predictionsLayers = sess.run(predLayers, feed_dict={XLayers: xtest, YLayers: ytest})

        # Convert the predictions to binary class labels
        predictions_binary_Layers = (predictionsLayers > 0.5).astype(int)

        # Calculate the accuracy
        accuracyLayers = (predictions_binary_Layers == ytest).mean()
        print(f'Layers accuracy: {accuracyLayers:.2f}')

    # Adding dropout layers

    model = keras.Sequential([
        keras.layers.Input(shape=(features,)),  # First layer
        keras.layers.Dense(15, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(15, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(15, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(xtrain, ytrain, epochs=1000, batch_size=100)

    print(model.evaluate(xtest, ytest))

    # Adaboost model
    adb = AdaBoostClassifier()
    adb_model = adb.fit(xtrain, ytrain)
    print("Adaboost accuracy", adb_model.score(xtest, ytest))

    # XGBoost model
    xgb = XGBClassifier()
    xgb_model = xgb.fit(xtrain, ytrain)
    print("XGB accuracy", xgb_model.score(xtest, ytest))
