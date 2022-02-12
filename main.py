from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import numpy as np
import os
import time

def split_input_target(textChunk):
    inputText = textChunk[:-1]
    targetText = textChunk[1:]
    return inputText, targetText

def build_model(vocabSize, embeddingDim, rnnUnits, batchSize):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocabSize, embeddingDim,
                                  batch_input_shape=[batchSize, None]),
        tf.keras.layers.GRU(rnnUnits,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocabSize)
    ])
    return model
  
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def generate_text(model, startString):
    lenGen = 25000
    inputEval = [char2idx[s] for s in startString]
    inputEval = tf.expand_dims(inputEval, 0)
    text = []
    temp = 1.0

    model.reset_states()
    for i in range(lenGen):
        if (i % 5000 == 0):
            print("{0}/{1}".format(i, lenGen))
        predictions = model(inputEval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temp 
        predictedId = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        inputEval = tf.expand_dims([predictedId], 0)
        text.append(idx2char[predictedId])
    return startString + ''.join(text)

BATCH_SIZE = 64
BUFFER_SIZE = 10000
SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 256
RNN_UNITS = 1024
EPOCHS = 30 

text = open('williamblake.txt').read().decode(encoding="utf-8")
vocab = sorted(set(text))
VOCABULARY_SIZE = len(vocab)

idx2char = np.array(vocab)
char2idx = {u:i for i, u in enumerate(vocab)}
textAsInt = np.array([char2idx[c] for c in text])

examplesPerEpoch = len(text)//(SEQUENCE_LENGTH+1)
charDataset = tf.data.Dataset.from_tensor_slices(textAsInt)
sequences = charDataset.batch(SEQUENCE_LENGTH+1, drop_remainder=True)
dataset = sequences.map(split_input_target)
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

model = build_model(
        vocab_size = VOCABULARY_SIZE,
        embedding_dim = EMBEDDING_DIM,
        rnn_units = RNN_UNITS,
        batch_size = BATCH_SIZE)

model.compile(optimizer="adam", loss=loss)

checkpointDir = './training_checkpoints'
checkpointPrefix = os.path.join(checkpointDir, "ckpt_{epoch}")

checkpointCallback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpointPrefix,
    save_weights_only=True)

tf.train.latest_checkpoint(checkpointDir)
model = build_model(VOCABULARY_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpointDir))
model.build(tf.TensorShape([1, None]))
model.summary()

START_WORDS = [u"The ", u"My ", u"When ", u"To ", u"I ", u"Can "]
for word in START_WORDS:
    filename = "outputs/{0}_out.txt".format(word[:-2])
    print(word, filename)
    out_text = open(filename, "w+")
    out_text.write(generate_text(model, start_string = word))
    out_text.close()
