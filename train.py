import modeling as bm
import model as m
import plot_utils as pu
from optimization import create_optimizer
from data2tfrecord import *
from data2normal import *
import tensorflow.compat.v1 as tf
import numpy as np
import pickle
import operator
tf.disable_v2_behavior()

bert_config = bm.BertConfig.from_json_file(BERT_CONFIG_PATH)  # 配置文件地址。


def run_train():
    if GLOVE:
        embedding = np.load(os.path.join(ROOT_PATH, EMBEDDING_NAME))
        bert_config.vocab_size = embedding.shape[0]
    with open(os.path.join(ROOT_PATH, WORD_DICT_NAME), 'rb') as f:
        word_dict = pickle.load(f)
    data = EssayV2()
    with tf.Session() as sess:
        padding_shape = ([bert_config.max_length], [bert_config.max_length],
                         [bert_config.max_length], [bert_config.max_length],
                         [bert_config.max_length], [bert_config.max_length],
                         # ------------------------------------------------
                         [MAX_LEN_NODE], [math.pow(MAX_LEN_NODE, 2)],
                         [MAX_LEN_NODE], [MAX_LEN_NODE],
                         [bert_config.max_length*bert_config.max_length])

        data_set_train = get_dataset(TRAIN_DATA_NAME)
        data_set_train = data_set_train.shuffle(bert_config.shuffle_pool_size).repeat(). \
            padded_batch(bert_config.batch_size, padded_shapes=padding_shape)
        data_set_train_iter = data_set_train.make_one_shot_iterator()
        train_handle = sess.run(data_set_train_iter.string_handle())

        handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.data.Iterator.from_string_handle(handle, data_set_train.output_types,
                                                       data_set_train.output_shapes)
        input_ids, input_mask, targets, targets_pos, targets_relation, targets_distance, \
        node_mask, relation_graph, node2pos_l, node2pos_r, relation_graph_word = iterator.get_next()

        model = bm.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=False  # 这里如果使用TPU 设置为True，速度会快些。使用CPU 或GPU 设置为False ，速度会快些。
        )

        transformer_first_output = model.get_all_encoder_layers()[0]
        transformer_second_output = model.get_all_encoder_layers()[1]
        transformer_third_output = model.get_all_encoder_layers()[0]
        transformer_output = model.get_sequence_output()
        transformer_first_attention = model.get_all_attention_layers()[0]  # B*H*N*N

        atten_model = m.AttenModel2(bert_config)
        atten_model(transformer_first_attention, relation_graph_word, input_mask)

        entity_model = m.POSModel(bert_config, data.num_classes_entities)
        entity_model(transformer_second_output, targets, input_mask)

        pos_model = m.POSModel(bert_config, data.num_classes_pos)
        pos_model(transformer_first_output, targets_pos, input_mask)

        rel_model = m.POSModel(bert_config, data.num_classes_relations)
        rel_model(transformer_second_output, targets_relation, input_mask)

        dis_model = m.POSRegModel(bert_config, data.num_classes_distances)
        dis_model(transformer_third_output, targets_distance, input_mask)

        entities_weight = 1
        pos_weight = 0
        rel_weight = 0
        dis_weight = 0
        atten_weight = 0
        logger("entities_weight: {}\tpos_weight: {}\trel_weight: {}\tdis_weight: {}\tatten_weight: {}".
               format(entities_weight, pos_weight, rel_weight, dis_weight, atten_weight))
        joint_loss = pos_weight*pos_model.loss + entities_weight*entity_model.loss + \
                     rel_weight*rel_model.loss + dis_weight*dis_model.loss + atten_weight*atten_model.loss

        tvars = tf.trainable_variables()
        num_train_steps = int((data.num_train_set*bert_config.num_train_epochs)/bert_config.batch_size)
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
            data_set_test = get_dataset(os.path.join(TRAIN_DATA_NAME))

            data_set_test = data_set_test. \
                padded_batch(bert_config.batch_size, padded_shapes=padding_shape)

            data_set_test_iter = data_set_test.make_one_shot_iterator()
            test_handle = sess.run(data_set_test_iter.string_handle())

            ids, entity_labels, in_masks, in_relation, node2posl, node2posr, attention_output = \
                sess.run([input_ids, targets, input_mask, relation_graph, node2pos_l, node2pos_r,
                          model.all_attention],
                         feed_dict={handle: test_handle})

            show_inds = [0, 1]
            for show_ind in show_inds:
                ids_ = ids[show_ind]
                entity_labels_ = entity_labels[show_ind]
                in_masks_ = in_masks[show_ind]
                in_relation_ = in_relation[show_ind]
                node2pos = list(zip(node2posl[show_ind], node2posr[show_ind]))

                attention_output_ = np.array(attention_output)[0:, show_ind, :, :, :]

                attention_output_ = np.mean(attention_output_, axis=-3)
                for ind_, attention_output__ in enumerate(attention_output_):
                    pu.plot_attention(attention_output__, ids_, ids_, entity_labels_, in_masks_, word_dict,
                                      node2pos=node2pos, relation_graph=in_relation_,
                                      name="test_{}_attention_layer_{}.png".format(show_ind, ind_))
                pu.plot_attention(attention_output_, ids_, ids_, entity_labels_, in_masks_, word_dict,
                                  node2pos=node2pos, relation_graph=in_relation_,
                                  name="test_{}attention_layer_avg.png".format(show_ind))
            exit()

        logger("**** Trainable Variables ****")
        # saver.restore(sess, MODEL_PATH)
        best_score = 0.
        for iter_ in range(num_train_steps):
            if iter_ == 0:
                a = sess.run(atten_model.relation_graph, feed_dict={handle: train_handle})
                print(max(a[0]), min(a[0]))

            sess.run(joint_train_op, feed_dict={handle: train_handle})

            if iter_ % 10 == 0:
                loss, acc, preds, targets = sess.run([pos_model.loss, pos_model.accuracy, pos_model.preds,
                                                      pos_model.targets], feed_dict={handle: train_handle})
                logger("[AM-POS] iter: {}\tloss: {}\tacc_pos: {}".format(iter_, loss, acc))
                # logger("[AM-POS] iter: {}\ntarget: {}\npreds: {}".format(iter_, targets[0, :], preds[0, :]))

                loss, acc, preds, targets = sess.run(
                    [entity_model.loss, entity_model.accuracy, entity_model.preds, entity_model.targets],
                    feed_dict={handle: train_handle})
                logger("[AM-Entities] iter: {}\tloss: {}\tacc_entity: {}".format(iter_, loss, acc))

                loss, acc, preds, targets = sess.run(
                    [dis_model.loss, dis_model.accuracy, dis_model.preds, dis_model.targets],
                    feed_dict={handle: train_handle})
                logger("[AM-Distances] iter: {}\tloss: {}\tacc_distance: {}".format(iter_, loss, acc))

                loss, acc, preds, targets = sess.run(
                    [rel_model.loss, rel_model.accuracy, rel_model.preds, rel_model.targets],
                    feed_dict={handle: train_handle})
                logger("[AM-Relations] iter: {}\tloss: {}\tacc_relation: {}".format(iter_, loss, acc))

                loss, = sess.run(
                    [atten_model.loss],
                    feed_dict={handle: train_handle})
                logger("[AM-Atten] iter: {}\tloss: {}".format(iter_, loss))

                data_set_test = get_dataset(os.path.join(TEST_DATA_NAME))

                data_set_test = data_set_test.shuffle(bert_config.shuffle_pool_size). \
                    padded_batch(bert_config.batch_size, padded_shapes=padding_shape)

                data_set_test_iter = data_set_test.make_one_shot_iterator()
                test_handle = sess.run(data_set_test_iter.string_handle())

                total_acc_pos = 0
                total_acc_entity = 0
                total_num = 0
                try:
                    while True:
                        tacc_pos, tacc_entity = sess.run([pos_model.accuracy, entity_model.accuracy], {handle: test_handle})
                        total_acc_pos += tacc_pos
                        total_acc_entity += tacc_entity
                        total_num += 1
                except tf.errors.OutOfRangeError:
                    logger("[AM-POS-Test] iter: {}\tacc_pos: {}\t[AM-Entities-Test]\tacc_entity: {}".
                           format(iter_, total_acc_pos/total_num, total_acc_entity/total_num))
                acc = total_acc_entity/total_num
                if acc > best_score:
                    best_score = acc
                    saver.save(sess, MODEL_PATH)
                    logger("Saved at Score: {}".format(best_score))
        saver.save(sess, MODEL_PATH)
        logger("Best Score: {}".format(best_score))
        logger("**** Training End ****")


if __name__ == "__main__":
    run_train()










