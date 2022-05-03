from keras import backend as K
from keras.layers import Dense, Lambda, Dot, Activation, Concatenate
from keras.layers import Layer


class ManyToOneAttention(Layer):
    def __init__(self, units=128, **kwargs):
        super(ManyToOneAttention, self).__init__(**kwargs)
        self.units = units
        self.attention_score_vec = None
        self.h_t = None
        self.attention_score = None
        self.attention_weight = None
        self.context_vector = None
        self.attention_output = None
        self.attention_vector = None

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        with K.name_scope('attention'):
            self.attention_score_vec = Dense(input_dim, use_bias=False, name='attention_score_vec')
            self.h_t = Lambda(lambda x: x[:, -1, :], output_shape=(input_dim,), name='last_hidden_state')
            self.attention_score = Dot(axes=[1, 2], name='attention_score')
            self.attention_weight = Activation('softmax', name='attention_weight')
            self.context_vector = Dot(axes=[1, 1], name='context_vector')
            self.attention_output = Concatenate(name='attention_output')
            self.attention_vector = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def __call__(self, inputs, training=None, **kwargs):
        return self.call(inputs, training, **kwargs)

    def call(self, inputs, training=None, **kwargs):
        self.build(inputs.shape)
        score_first_part = self.attention_score_vec(inputs)
        h_t = self.h_t(inputs)
        score = self.attention_score([h_t, score_first_part])
        attention_weights = self.attention_weight(score)
        context_vector = self.context_vector([inputs, attention_weights])
        pre_activation = self.attention_output([context_vector, h_t])
        attention_vector = self.attention_vector(pre_activation)
        return attention_vector
