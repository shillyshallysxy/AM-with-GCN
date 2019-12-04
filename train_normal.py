import modeling as bm
import model as m
import plot_utils as pu
from optimization import create_optimizer
from data2normal import *
import tensorflow.compat.v1 as tf
import numpy as np
import pickle
tf.disable_v2_behavior()

bert_config = bm.BertConfig.from_json_file(BERT_CONFIG_PATH)  # 配置文件地址。


# Define placeholders
placeholders = {
    'input_ids': tf.placeholder(tf.int64, shape=(None, bert_config.max_length)),
    'input_mask': tf.placeholder(tf.int64, shape=(None, bert_config.max_length)),
    'targets': tf.placeholder(tf.int64, shape=(None, bert_config.max_length)),
    'targets_pos': tf.placeholder(tf.int64, shape=(None, bert_config.max_length)),
    'adj_graph': [tf.sparse_placeholder(tf.float32) for _ in range(1)],
}


def construct_feed_dict(input_ids, input_mask, targets, targets_pos, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['input_ids']: input_ids})
    feed_dict.update({placeholders['input_mask']: input_mask})
    feed_dict.update({placeholders['targets']: targets})
    feed_dict.update({placeholders['targets_pos']: targets_pos})
    # feed_dict.update({placeholders['adj_graph'][i]: support[i] for i in range(len(support))})
    return feed_dict


def run_train():
    if GLOVE:
        embedding = np.load(os.path.join(ROOT_PATH, EMBEDDING_NAME))
        bert_config.vocab_size = embedding.shape[0]
    with open(os.path.join(ROOT_PATH, WORD_DICT_NAME), 'rb') as f:
        word_dict = pickle.load(f)

    data = EssayV2()
    with tf.Session() as sess:

        input_ids, input_mask, targets, targets_pos, _ = placeholders.values()

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
        pos_model = m.POSModel(bert_config, data.num_classes_pos)
        pos_model(transformer_middle_output, targets_pos, input_mask)

        transformer_output = model.get_sequence_output()
        entity_model = m.POSModel(bert_config, data.num_classes_entities)
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
            for iter_, (batch_x, batch_y, batch_others, _) in enumerate(data.gen_next_batch(batch_size=bert_config.batch_size,
                                                                                            is_train_set=True,
                                                                                            epoch=bert_config.num_train_epochs)):
                feed_dict = construct_feed_dict(batch_x[0], batch_x[1], batch_y[0], batch_y[1], placeholders)
                ids, entity_labels, in_masks, attention_output = sess.run([input_ids, targets, input_mask,
                                                                           model.attention_output],
                                                                          feed_dict=feed_dict)
                ids = ids[0]
                attention_output = attention_output[0]
                attention_output = np.mean(attention_output, axis=0)
                entity_labels = entity_labels[0]
                in_masks = in_masks[0]
                relation_graph = batch_others[0][0]
                node2pos = batch_others[1][0]
                pu.plot_attention(attention_output, ids, ids, entity_labels, in_masks, word_dict,
                                  node2pos=node2pos, relation_graph=relation_graph)

        logger("**** Trainable Variables ****")
        # saver.restore(sess, MODEL_PATH)

        best_score = 0.
        for iter_, (batch_x, batch_y, _) in enumerate(data.gen_next_batch(batch_size=bert_config.batch_size,
                                                                          is_train_set=True,
                                                                          epoch=bert_config.num_train_epochs)):
            # sess.run(pos_train_op, feed_dict={handle: train_handle})
            feed_dict = construct
            _feed_dict(batch_x[0], batch_x[1], batch_y[0], batch_y[1], placeholders)
            sess.run(joint_train_op, feed_dict=feed_dict)

            if iter_ % 50 == 0:
                loss, acc, preds, targets = sess.run([pos_model.loss,
                                                      pos_model.accuracy,
                                                      pos_model.preds,
                                                      pos_model.targets], feed_dict=feed_dict)
                logger("[AM-POS] iter: {}\tloss: {}\tacc_pos: {}".format(iter_, loss, acc))
                # logger("[AM-POS] iter: {}\ntarget: {}\npreds: {}".format(iter_, targets[0, :], preds[0, :]))

                loss, acc, preds, targets = sess.run(
                    [entity_model.loss, entity_model.accuracy, entity_model.preds, entity_model.targets],
                    feed_dict=feed_dict)
                logger("[AM-Entities] iter: {}\tloss: {}\tacc_entity: {}".format(iter_, loss, acc))

                if acc > best_score:
                    best_score = acc
                    saver.save(sess, MODEL_PATH)
        logger("Best Score: {}".format(best_score))
        logger("**** Training End ****")


if __name__ == "__main__":
    run_train()










