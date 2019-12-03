import modeling as bm
import model as m
import plot_utils as pu
from optimization import create_optimizer
from data2tfrecord import *
import tensorflow.compat.v1 as tf
import numpy as np
import pickle
tf.disable_v2_behavior()

bert_config = bm.BertConfig.from_json_file(BERT_CONFIG_PATH)  # 配置文件地址。


def run_train():
    if GLOVE:
        embedding = np.load(os.path.join(ROOT_PATH, EMBEDDING_NAME))
        bert_config.vocab_size = embedding.shape[0]
    with open(os.path.join(ROOT_PATH, WORD_DICT_NAME), 'rb') as f:
        word_dict = pickle.load(f)
    with tf.Session() as sess:
        padding_shape = ([bert_config.max_length], [bert_config.max_length], [bert_config.max_length], [bert_config.max_length])
        data_set_train = get_dataset(TRAIN_DATA_NAME)
        data_set_train = data_set_train.shuffle(bert_config.shuffle_pool_size).repeat(). \
            padded_batch(bert_config.batch_size, padded_shapes=padding_shape)
        data_set_train_iter = data_set_train.make_one_shot_iterator()
        train_handle = sess.run(data_set_train_iter.string_handle())

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, data_set_train.output_types,
                                                       data_set_train.output_shapes)
        input_ids, input_mask, targets, targets_pos = iterator.get_next()

        model = bm.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=False  # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
        )

        # transformer_middle_output = model.get_all_encoder_layers()[bert_config.num_hidden_layers//2]
        transformer_middle_output = model.get_sequence_output()
        pos_model = m.POSModel(bert_config, 5)
        pos_model(transformer_middle_output, targets_pos, input_mask)

        transformer_output = model.get_sequence_output()
        entity_model = m.POSModel(bert_config, 4)
        entity_model(transformer_output, targets, input_mask)

        entities_weight = 1
        pos_weight = 0.1
        joint_loss = pos_weight*pos_model.loss + entities_weight*entity_model.loss

        tvars = tf.trainable_variables()
        num_train_steps = int((322*bert_config.num_train_epochs)/bert_config.batch_size)
        num_warmup_steps = int(num_train_steps * bert_config.warmup_proportion)

        # pos_train_op = create_optimizer(pos_model.loss, bert_config.init_lr,
        #                                 num_train_steps, num_warmup_steps, False)
        # entity_train_op = create_optimizer(entity_model.loss, bert_config.init_lr,
        #                                    num_train_steps, num_warmup_steps, False)
        joint_train_op = create_optimizer(joint_loss, bert_config.init_lr, num_train_steps, num_warmup_steps, False)

        # graph = tf.get_default_graph()
        # summary_write = tf.summary.FileWriter("./tensorboard/", graph)
        # summary_write.close()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        if GLOVE:
            tf.assign(model.embedding_table, embedding)
        if False:
            saver.restore(sess, MODEL_PATH)
            data_set_test = get_dataset(os.path.join(TEST_DATA_NAME))

            data_set_test = data_set_test. \
                padded_batch(bert_config.batch_size, padded_shapes=padding_shape)

            data_set_test_iter = data_set_test.make_one_shot_iterator()
            test_handle = sess.run(data_set_test_iter.string_handle())

            ids, entity_labels, in_masks, attention_output = sess.run([input_ids, targets, input_mask,
                                                             model.attention_output], feed_dict={handle: test_handle})
            ids = ids[0]
            attention_output = attention_output[0]
            attention_output = np.mean(attention_output, axis=0)
            entity_labels = entity_labels[0]
            in_masks = in_masks[0]
            pu.plot_attention(attention_output, ids, ids, entity_labels, in_masks, word_dict)

        logger("**** Trainable Variables ****")
        # saver.restore(sess, MODEL_PATH)

        best_score = 0.
        for iter_ in range(num_train_steps):
            # sess.run(pos_train_op, feed_dict={handle: train_handle})
            sess.run(joint_train_op, feed_dict={handle: train_handle})

            if iter_ % 50 == 0:
                loss, acc, preds, targets = sess.run([pos_model.loss, pos_model.accuracy, pos_model.preds, pos_model.targets], feed_dict={handle: train_handle})
                logger("[AM-POS] iter: {}\tloss: {}\tacc_pos: {}".format(iter_, loss, acc))
                # logger("[AM-POS] iter: {}\ntarget: {}\npreds: {}".format(iter_, targets[0, :], preds[0, :]))

                loss, acc, preds, targets = sess.run(
                    [entity_model.loss, entity_model.accuracy, entity_model.preds, entity_model.targets],
                    feed_dict={handle: train_handle})
                logger("[AM-Entities] iter: {}\tloss: {}\tacc_entity: {}".format(iter_, loss, acc))

                data_set_test = get_dataset(os.path.join(TEST_DATA_NAME))

                data_set_test = data_set_test.shuffle(bert_config.shuffle_pool_size). \
                    padded_batch(bert_config.batch_size, padded_shapes=padding_shape)

                data_set_test_iter = data_set_test.make_one_shot_iterator()
                test_handle = sess.run(data_set_test_iter.string_handle())
                try:
                    total_acc_pos = 0
                    total_acc_entity = 0
                    total_num = 0
                    while True:
                        tacc_pos, tacc_entity = sess.run([pos_model.accuracy, entity_model.accuracy], {handle: test_handle})
                        total_acc_pos += tacc_pos
                        total_acc_entity += tacc_entity
                        total_num += 1
                except tf.errors.OutOfRangeError:
                    logger("[AM-POS-Test] iter: {}\tacc_pos: {}\t[AM-Entities-Test]\tacc_entity: {}".
                           format(iter_, total_acc_pos/total_num, total_acc_entity/total_num))
                if acc > best_score:
                    best_score = acc
                    saver.save(sess, MODEL_PATH)
        logger("Best Score: {}".format(best_score))
        logger("**** Training End ****")


if __name__ == "__main__":
    run_train()










