import modeling as bm
import model as m
from optimization import create_optimizer
from data2tfrecord import *
import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()

bert_config = bm.BertConfig.from_json_file(BERT_CONFIG_PATH)  # 配置文件地址。


def run_train():
    if GLOVE:
        embedding = np.load(os.path.join(ROOT_PATH, EMBEDDING_NAME))
        bert_config.vocab_size = embedding.shape[0]

    with tf.Session() as sess:
        padding_shape = ([bert_config.max_length], [bert_config.max_length], [bert_config.max_length])
        data_set_train = get_dataset(TRAIN_DATA_NAME)
        data_set_train = data_set_train.shuffle(bert_config.shuffle_pool_size).repeat(). \
            padded_batch(bert_config.batch_size, padded_shapes=padding_shape)
        data_set_train_iter = data_set_train.make_one_shot_iterator()
        train_handle = sess.run(data_set_train_iter.string_handle())

        data_set_test = get_dataset(os.path.join(TEST_DATA_NAME))

        data_set_test = data_set_test.shuffle(bert_config.shuffle_pool_size). \
            padded_batch(bert_config.batch_size, padded_shapes=padding_shape)

        data_set_test_iter = data_set_test.make_one_shot_iterator()
        test_handle = sess.run(data_set_test_iter.string_handle())

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, data_set_train.output_types,
                                                       data_set_train.output_shapes)
        input_ids, input_mask, targets = iterator.get_next()

        model = bm.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=False  # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
        )

        transformer_output = model.get_sequence_output()

        posmodel = m.POSModel(bert_config, 4)

        posmodel(transformer_output, targets, input_mask)

        tvars = tf.trainable_variables()
        num_train_steps = int((322*bert_config.num_train_epochs)/bert_config.batch_size)
        num_warmup_steps = int(num_train_steps * bert_config.warmup_proportion)
        train_op = create_optimizer(posmodel.loss, bert_config.init_lr, num_train_steps, num_warmup_steps, False)

        logger("**** Trainable Variables ****")

        # graph = tf.get_default_graph()
        # summary_write = tf.summary.FileWriter("./tensorboard/", graph)
        # summary_write.close()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        if GLOVE:
            tf.assign(model.embedding_table, embedding)

        for iter_ in range(num_train_steps):
            # a, b, c = sess.run([input_ids, input_mask, targets], feed_dict={handle: train_handle})
            sess.run(train_op, feed_dict={handle: train_handle})

            if iter_ % 50 == 0:
                loss, acc, preds, targets = sess.run([posmodel.loss, posmodel.accuracy, posmodel.preds, posmodel.targets], feed_dict={handle: train_handle})
                logger("[AM-POS] iter: {}\tloss: {}\tacc: {}".format(iter_, loss, acc))
                logger("[AM-POS] iter: {}\ntarget: {}\npreds: {}".format(iter_, targets[0, :], preds[0, :]))
        saver.save(sess, MODEL_PATH)


if __name__ == "__main__":
    run_train()










