import tensorflow as tf
import tensorflow.keras as keras


########################################################################################

class InferNER(tf.keras.layers.Layer):

    def __init__(self, vocab_size, two_stack_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.two_stack_size = two_stack_size
        self.window_size = window_size
        self.inception = tf.keras.applications.InceptionV3(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

        # TODO:

        self.image_embedding = tf.keras.layers.Dense(two_stack_size)

        # Define embedding layers:
        self.stacked_embedding = tf.keras.layers.Embedding(vocab_size, 1024)
        initializer = tf.keras.initializers.RandomUniform(minval=-tf.sqrt(3/30), maxval=tf.sqrt(3/30))


        self.character_embedding = tf.keras.layers.Embedding(vocab_size, 30, embeddings_initializer=initializer)
        self.sentence_embedding = tf.keras.layers.Embedding(vocab_size, 512)

        # Define decoder layer that handles language and image context:     
        self.decoder = tf.keras.layers.GRU(two_stack_size, return_sequences=True, activation='relu')
        self.encoder = None # can use our hw implementation
        # Define classification layer (LOGIT OUTPUT)
        self.classifier = tf.keras.layers.Dense(vocab_size)


        self.BLSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(10, return_sequences=True),
                             input_shape=(5, 10))

        self.ReLU = tf.keras.layers.ReLU()
    def call(self, encoded_images, captions):
        # TODO:
        # 1) Embed the encoded images into a vector of the correct dimension for initial state (299x299)
        # 2) 2-stack, character, and sentence level encoders
        # 3) Apply dense layer(s) to the decoder to generate prediction **logits**
        embedded_image = tf.keras.applications.inception_v3.preprocess_input(encoded_images)
        embedded_caption = self.stacked_embedding(captions)
        ro = self.BLSTM(embedded_caption)
        rpo = self.BLSTM(ro)
        wt = tf.add(ro, rpo) #(1)
        decoded = self.decoder(embedded_caption, initial_state=embedded_image)
        logits = self.classifier(decoded)
        return logits


########################################################################################

class TransformerDecoder(tf.keras.Model):

    def __init__(self, vocab_size, hidden_size, window_size, **kwargs):

        super().__init__(**kwargs)
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.window_size = window_size

        # TODO: Define image and positional encoding, transformer decoder, and classification layers

        # Define feed forward layer to embed image features into a vector 
        self.image_embedding = tf.keras.layers.Dense(hidden_size)

        # Define positional encoding to embed and offset layer for language:
        self.encoding = PositionalEncoding(vocab_size, hidden_size, window_size)

        # Define transformer decoder layer:
        self.decoder = TransformerBlock(hidden_size)

        # Define classification layer (logits)
        self.classifier = tf.keras.layers.Dense(vocab_size)

    def call(self, encoded_images, captions):
        # TODO:
        # 1) Embed the encoded images into a vector (HINT IN NOTEBOOK)
        # 2) Pass the captions through your positional encoding layer
        # 3) Pass the english embeddings and the image sequences to the decoder
        # 4) Apply dense layer(s) to the decoder out to generate logits
        embedded_images = self.image_embedding(encoded_images)
        embedded_images = tf.expand_dims(embedded_images, 1)
        captions = self.encoding(captions)
        probs = self.classifier(self.decoder(captions, embedded_images))
        return probs
