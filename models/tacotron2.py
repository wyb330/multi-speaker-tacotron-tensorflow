# Code from https://github.com/keithito/tacotron/blob/tacotron2-work-in-progress/models/tacotron.py

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMBlockCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, AttentionWrapper
from text.symbols import symbols
from utils.infolog import log
from .attention import LocationSensitiveAttention
from .helpers import TacoTestHelper, TacoTrainingHelper
from .modules import conv_and_lstm, postnet
from .rnn_wrappers2 import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper
import numpy as np


# Code from https://www.github.com/kyubyong/dc_tts/utils.py
def guided_attention(max_N, max_T, g=0.2):
    '''Guided attention. Refer to page 3 on the paper.
       W = 1 - exp{-(n/N - t/T)^2 / 2G^2)}
    '''
    W = np.zeros((max_N, max_T), dtype=np.float32)
    for n_pos in range(W.shape[0]):
        for t_pos in range(W.shape[1]):
            W[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(max_T) - n_pos / float(max_N)) ** 2 / (2 * g * g))
    return W


class Tacotron2():
    def __init__(self, hparams):
        self._hparams = hparams

    def initialize(self, inputs, input_lengths, num_speakers, speaker_id,
            mel_targets=None, linear_targets=None, loss_coeff=None,
            rnn_decoder_test_mode=False, is_randomly_initialized=False):
        '''Initializes the model for inference.

        Sets "mel_outputs", "linear_outputs", and "alignments" fields.

        Args:
          inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
            steps in the input time series, and values are character IDs
          input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
            of each sequence in inputs.
          mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
            of steps in the output time series, M is num_mels, and values are entries in the mel
            spectrogram. Only needed for training.
          linear_targets: float32 Tensor with shape [N, T_out, F] where N is batch_size, T_out is number
            of steps in the output time series, F is num_freq, and values are entries in the linear
            spectrogram. Only needed for training.
        '''
        with tf.variable_scope('inference') as scope:
            is_training = linear_targets is not None
            self.batch_size = tf.shape(inputs)[0]
            hp = self._hparams

            # Embeddings
            embedding_table = tf.get_variable(
                'embedding', [len(symbols), hp.embedding_dim], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.5))
            embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)  # [N, T_in, 512]

            # Encoder
            encoder_outputs = conv_and_lstm(
                embedded_inputs,
                input_lengths,
                conv_layers=hp.encoder_conv_layers,
                conv_width=hp.encoder_conv_width,
                conv_channels=hp.encoder_conv_channels,
                lstm_units=hp.encoder_lstm_units,
                is_training=is_training,
                scope='encoder')  # [N, T_in, 512]

            # Attention
            # For manaul control of attention
            self.is_manual_attention = tf.placeholder(
                tf.bool, shape=(), name='is_manual_attention',
            )
            self.manual_alignments = tf.placeholder(
                tf.float32, shape=[None, None, None], name="manual_alignments",
            )

            attention_cell = AttentionWrapper(
                DecoderPrenetWrapper(LSTMBlockCell(hp.attention_depth), is_training),
                LocationSensitiveAttention(hp.attention_depth, encoder_outputs),
                alignment_history=True,
                output_attention=False)  # [N, T_in, 128]

            # Concatenate attention context vector and RNN cell output into a 512D vector.
            concat_cell = ConcatOutputAndAttentionWrapper(attention_cell)  # [N, T_in, 512]

            # Decoder (layers specified bottom to top):
            decoder_cell = MultiRNNCell([
                concat_cell,
                LSTMBlockCell(hp.decoder_lstm_units),
                LSTMBlockCell(hp.decoder_lstm_units)
            ], state_is_tuple=True)  # [N, T_in, 1024]

            # Project onto r mel spectrograms (predict r outputs at each RNN step):
            output_cell = OutputProjectionWrapper(decoder_cell, hp.num_mels * hp.outputs_per_step)

            if is_training:
                helper = TacoTrainingHelper(inputs, mel_targets, hp.num_mels, hp.outputs_per_step)
            else:
                helper = TacoTestHelper(self.batch_size, hp.num_mels, hp.outputs_per_step)

            decoder_init_state = output_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32)
            (multi_decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                BasicDecoder(output_cell, helper, decoder_init_state),
                maximum_iterations=hp.max_iters)  # [N, T_out/r, M*r]

            # Reshape outputs to be one output per entry                                [N, T_out, M]
            decoder_outputs = tf.reshape(multi_decoder_outputs, [self.batch_size, -1, hp.num_mels])

            # Postnet: predicts a residual
            postnet_outputs = postnet(
                decoder_outputs,
                layers=hp.postnet_conv_layers,
                conv_width=hp.postnet_conv_width,
                channels=hp.postnet_conv_channels,
                is_training=is_training)
            mel_outputs = decoder_outputs + postnet_outputs

            # Convert to linear using a similar architecture as the encoder:
            expand_outputs = conv_and_lstm(
                mel_outputs,
                None,
                conv_layers=hp.expand_conv_layers,
                conv_width=hp.expand_conv_width,
                conv_channels=hp.expand_conv_channels,
                lstm_units=hp.expand_lstm_units,
                is_training=is_training,
                scope='expand')  # [N, T_in, 512]
            linear_outputs = tf.layers.dense(expand_outputs, hp.num_freq)  # [N, T_out, F]

            # Grab alignments from the final decoder state:
            alignments = tf.transpose(final_decoder_state[0].alignment_history.stack(), [1, 2, 0])

            self.inputs = inputs
            self.input_lengths = input_lengths
            self.decoder_outputs = decoder_outputs
            self.mel_outputs = mel_outputs
            self.linear_outputs = linear_outputs
            self.alignments = alignments
            self.mel_targets = mel_targets
            self.linear_targets = linear_targets
            log('Initialized Tacotron model. Dimensions: ')
            log('  embedding:               %d' % embedded_inputs.shape[-1])
            log('  encoder out:             %d' % encoder_outputs.shape[-1])
            log('  attention out:           %d' % attention_cell.output_size)
            log('  concat attn & out:       %d' % concat_cell.output_size)
            log('  decoder cell out:        %d' % decoder_cell.output_size)
            log('  decoder out (%d frames):  %d' % (hp.outputs_per_step, decoder_outputs.shape[-1]))
            log('  decoder out (1 frame):   %d' % mel_outputs.shape[-1])
            log('  expand out:              %d' % expand_outputs.shape[-1])
            log('  linear out:              %d' % linear_outputs.shape[-1])

    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        with tf.variable_scope('loss') as scope:
            hp = self._hparams
            self.decoder_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.decoder_outputs))
            self.mel_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.mel_outputs))

            # Prioritize loss for frequencies under 2000 Hz.
            l1 = tf.abs(self.linear_targets - self.linear_outputs)
            n_priority_freq = int(2000 / (hp.sample_rate * 0.5) * hp.num_freq)
            self.linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:, :, 0:n_priority_freq])

            # guided_attention loss
            pad_alignment = tf.pad(self.alignments, [(0, 0), (0, hp.max_N), (0, hp.max_T)], mode="CONSTANT",
                                   constant_values=-1.)[:, :hp.max_N, :hp.max_T]
            attention_masks = tf.to_float(tf.not_equal(pad_alignment, -1))
            gts = tf.convert_to_tensor(guided_attention(hp.max_N, hp.max_T))
            self.alignments_guide = tf.abs(pad_alignment * gts) * attention_masks
            self.loss_attention = tf.reduce_sum(self.alignments_guide) / tf.to_float(self.batch_size)
            # self.loss_attention = tf.clip_by_value(self.loss_attention, 0, 10)

            self.loss = self.decoder_loss + self.mel_loss + self.linear_loss

    def add_optimizer(self, global_step):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

        Args:
          global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.variable_scope('optimizer') as scope:
            hp = self._hparams
            self.learning_rate = tf.train.exponential_decay(
                hp.initial_learning_rate, global_step, hp.learning_rate_decay_halflife, 0.5)
            optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step)

    def get_dummy_feed_dict(self):
        feed_dict = {
            self.is_manual_attention: False,
            self.manual_alignments: np.zeros([1, 1, 1]),
        }
        return feed_dict
