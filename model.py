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
        istarget = tf.to_float(tf.not_equal(self.targets, 0))
        self.accuracy = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.targets)) * istarget) / (
            tf.reduce_sum(istarget))

        istargetv2 = tf.to_float(sequence_length)
        self.accuracy2 = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.targets)) * istargetv2) / (
            tf.reduce_sum(istargetv2))

        self.loss = self.loss_layer(self.logits, self.targets, sequence_length)
        # self.loss = self.crf_loss_layer(self.logits, self.targets, sequence_length)
        return

    def loss_layer(self, x, y, sequence_length):

        sequence_length = tf.to_float(sequence_length)

        weight = tf.to_float(tf.equal(y, 0))*0. + tf.to_float(tf.not_equal(y, 0))*1.
        sequence_length = sequence_length * weight

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


class POSRegModel:
    def __init__(self, config, out_shape=1):
        self.activation = None
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
        self.targets = tf.to_float(y)
        self.preds = tf.reshape(self.logits, [-1, self.max_length])
        istarget = tf.to_float(tf.not_equal(self.targets, 0))
        self.accuracy = tf.reduce_sum(tf.to_float(tf.square(tf.subtract(self.preds, self.targets))) * istarget) / (
            tf.reduce_sum(istarget))

        istargetv2 = tf.to_float(sequence_length)
        self.accuracy2 = tf.reduce_sum(tf.to_float(tf.square(tf.subtract(self.preds, self.targets))) * istargetv2) / (
            tf.reduce_sum(istargetv2))

        self.loss = self.loss_layer(self.logits, self.targets, sequence_length)
        # self.loss = self.crf_loss_layer(self.logits, self.targets, sequence_length)
        return

    def loss_layer(self, x, y, sequence_length):

        sequence_length = tf.to_float(sequence_length)

        weight = tf.to_float(tf.equal(y, 0))*0.1 + tf.to_float(tf.not_equal(y, 0))*1.
        sequence_length = sequence_length * weight

        loss = tf.reduce_mean(sequence_loss_by_example(
            [tf.reshape(x, [-1], name='reshaped_input')],
            [tf.reshape(y, [-1], name='reshaped_target')],
            [tf.reshape(sequence_length, [-1], name='sequence_length')],
            average_across_timesteps=True,
            softmax_loss_function=ms_error,
        ))

        return loss


class AttenModel:
    def __init__(self, config):
        self.activation = None
        self.hidden_dim = config.hidden_size
        self.initializer_range = config.initializer_range
        self.max_length = config.max_length_node
        self.batch_size = config.batch_size
        self.trans = None

    def __call__(self, x, relation_graph, node_length, node2posl, node2posr):
        x = tf.reduce_mean(x, axis=-3)  # B*N*N
        node_length = tf.to_float(node_length)
        relation_graph = tf.nn.softmax(tf.to_float(relation_graph), axis=-1)
        total_x = list()
        total_loss = list()
        for ind in range(self.batch_size):
            x_, l_, r_, relation_graph_, len_ = x[ind], node2posl[ind], node2posr[ind], relation_graph[ind], node_length[ind]
            x_temp = list()
            # relation_graph_ = tf.reshape(relation_graph_, [-1])
            for ind_ in range(self.max_length):
                l, r, lent = l_[ind_], r_[ind_], len_[ind_]
                for ind__ in range(self.max_length):
                    l__, r__, lent__ = l_[ind__], r_[ind__], len_[ind__]
                    temp = tf.reduce_mean(x_[l:r, l__:r__]) * lent * lent__
                    x_temp.append(temp)
            x_temp = tf.to_float(x_temp)
            total_loss.append(tf.reduce_mean(ms_error(relation_graph_, x_temp)))

            total_x.append(x_temp)
        self.loss = tf.add_n(total_loss)


class AttenModel2:
    def __init__(self, config):
        self.activation = None
        self.hidden_dim = config.hidden_size
        self.initializer_range = config.initializer_range
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.trans = None

    def __call__(self, x, relation_graph, sequence_length):
        x = tf.reduce_mean(x, axis=-3)  # B*N*N
        sequence_length = tf.expand_dims(tf.to_float(sequence_length), axis=1)
        sequence_length = tf.tile(sequence_length, (1, self.max_length, 1))
        adder = (1.0 - tf.cast(sequence_length, tf.float32)) * -10000.0
        sequence_length = tf.reshape(sequence_length, [-1, self.max_length * self.max_length])
        relation_graph = tf.to_float(relation_graph)
        relation_graph = tf.reshape(relation_graph, [-1, self.max_length, self.max_length]) * 5
        relation_graph += adder

        relation_graph = tf.nn.softmax(relation_graph)
        relation_graph = tf.reshape(relation_graph, [-1, self.max_length * self.max_length])

        self.relation_graph = relation_graph
        x = tf.reshape(x, [-1, self.max_length * self.max_length])
        # self.loss = tf.reduce_sum(tf.multiply(ms_error(relation_graph, x), sequence_length))
        self.loss = tf.reduce_mean(sequence_loss_by_example(
            [tf.reshape(x, [-1], name='reshaped_input')],
            [tf.reshape(relation_graph, [-1], name='reshaped_target')],
            [tf.reshape(sequence_length, [-1], name='sequence_length')],
            average_across_timesteps=True,
            softmax_loss_function=tf.nn.softmax_cross_entropy_with_logits_v2,
        ))

def ms_error(labels, logits):
    return tf.square(tf.subtract(labels, logits))


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

