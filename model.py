import tensorflow.compat.v1 as tf
import modeling
# from crf import crf_log_likelihood
tf.disable_v2_behavior()


class POSModel:
    def __init__(self, config, out_shape):
        self.activation = modeling.get_activation(config.hidden_act)
        self.hidden_dim = config.hidden_size
        self.initializer_range = config.initializer_range
        self.output_shape = out_shape
        self.max_length = config.max_length
        self.trans = None

    def __call__(self, x, y, sequence_length):
        x = tf.reshape(x, (-1, self.hidden_dim))

        self.logits = tf.layers.dense(x, self.output_shape,
                                      activation=self.activation,
                                      kernel_initializer=modeling.create_initializer(self.initializer_range))
        self.targets = y
        self.preds = tf.reshape(tf.argmax(self.logits, axis=-1), [-1, self.max_length])
        istarget = tf.to_float(sequence_length)
        self.accuracy = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.targets)) * istarget) / (tf.reduce_sum(istarget))

        self.loss = self.loss_layer(self.logits, self.targets, sequence_length)
        # self.loss = self.crf_loss_layer(self.logits, self.targets, sequence_length)
        return

    def loss_layer(self, x, y, sequence_length):

        sequence_length = tf.to_float(sequence_length)

        loss = tf.reduce_mean(sequence_loss_by_example(
            [x],
            [tf.reshape(y, [-1], name='reshaped_target')],
            [tf.reshape(sequence_length, [-1], name='sequence_length')],
            average_across_timesteps=True,
            softmax_loss_function=tf.nn.sparse_softmax_cross_entropy_with_logits,
        ))

        return loss

    # def crf_loss_layer(self, x, y, sequence_length):
    #     sequence_length = tf.reduce_sum(sequence_length, axis=1)
    #     self.trans = tf.get_variable("transitions", shape=[self.output_shape, self.output_shape],
    #                                  initializer=modeling.create_initializer(self.initializer_range))
    #     x = tf.reshape(x, (-1, self.max_length, self.output_shape))
    #     log_likelihood, trans = crf_log_likelihood(inputs=x, tag_indices=y,
    #                                                transition_params=self.trans, sequence_lengths=sequence_length)
    #     return tf.reduce_mean(-log_likelihood)


def sequence_loss_by_example(logits,
                             targets,
                             weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None):

    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
        if softmax_loss_function is None:
            # TODO(irving,ebrevdo): This reshape is needed because
            # sequence_loss_by_example is called with scalars sometimes, which
            # violates our general scalar strictness policy.
            target = tf.reshape(target, [-1])
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target, logits=logit)
        else:
            # calculate the cross entropy between target and logit
            crossent = softmax_loss_function(labels=target, logits=logit)
            # 1D tensor
            # shape(crossent) = [batch_size * num_step]
            # shape(weight) = [batch_size * num_step]
            # * is elementwise product
        log_perp_list.append(crossent * weight)

    log_perps = tf.add_n(log_perp_list)
    # shape(log_perps) = [batch_size * num_step]
    if average_across_timesteps:
        total_size = tf.add_n(weights)
        total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
        log_perps /= total_size
    return log_perps

