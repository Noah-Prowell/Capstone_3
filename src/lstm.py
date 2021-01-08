from enum import auto
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers


dataset = keras.preprocessing.text_dataset_from_directory('training_data', label_mode=None, batch_size=256)
dataset = dataset.map(lambda x: tf.strings.regex_replace(x, "<br />", " "))


sequence_length = 100
vectorize_layer = TextVectorization(
    max_tokens=15000,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
)
vectorize_layer.adapt(dataset)
vocab = vectorize_layer.get_vocabulary()
tokens_index = dict(enumerate(vocab))


def make_lm_dataset(text_batch):
    tokenized = vectorize_layer(text_batch)
    return tokenized[:, :-1], tokenized[:, 1:]

lm_dataset = dataset.map(make_lm_dataset).prefetch(tf.data.experimental.AUTOTUNE)


inputs = keras.Input(shape=(None,), dtype='int64')
x = layers.Embedding(len(vocab), 256)(inputs)
x = layers.LSTM(256, return_sequences=True)(x)
x = layers.LSTM(256, return_sequences=True)(x)
outputs = layers.Dense(len(vocab), activation='softmax')(x)
model = keras.Model(inputs, outputs)



def decode_token_indices(indices):
    return ' '.join([tokens_index[i] for i in indices])

def sample_next(preds, temperature=1.0):
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.argmax(np.random.multinomial(1, preds, 1))


class TextGenerator(keras.callbacks.Callback):

    def __init__(self,
                prompt,
                generate_length,
                model_input_length,
                temperatures=(1.,)):
        self.prompt = prompt
        self.generate_length = generate_length
        self.model_input_length = model_input_length
        self.temperatures = temperatures
        self.words_list = []

    def on_epoch_end(self, epoch, logs=None):
        for temperature in self.temperatures:
            token_sequence = self.prompt[:]
            tokens_generated = []
            while len(token_sequence) - len(self.prompt) < self.generate_length:
                model_input = tf.convert_to_tensor([token_sequence])
                preds = self.model.predict(model_input).astype('float64')
                next_token = sample_next(preds[0, -1], temperature=temperature)
                token_sequence.append(next_token)
            self.words_list.append(decode_token_indices(token_sequence))


text_prompt = "Hey guys"
prompt = list(vectorize_layer([text_prompt]).numpy()[0])[:2]
text_gen_callback = TextGenerator(
    prompt,
    generate_length=50,
    model_input_length=sequence_length,
    temperatures=(0.1, 0.2, 0.5, 0.7, 1., 1.5))

if __name__ == "__main__":
    # model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')
    # model.fit(lm_dataset, epochs=700, callbacks=[text_gen_callback])
    # print('Training')
    model = keras.models.load_model('stan_500')
    text_prompt = "Hey guys"
    prompt = list(vectorize_layer([text_prompt]).numpy()[0])[:2]
    text_gen_callback = TextGenerator(
        prompt,
        generate_length=15,
        model_input_length=sequence_length,
        temperatures=(.9, 1., 1.3, 1.5, 1.8, 1.9, 2., 2.2, 2.5, 2.8))
    print(model.summary())
    # model.fit(lm_dataset, epochs=1, callbacks=[text_gen_callback])
    words_out = text_gen_callback.words_list
    # (.9, 1., 1.3, 1.5, 1.8, 1.9, 2., 2.2, 2.5, 2.8)