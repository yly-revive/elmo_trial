import tensorflow_hub as hub
import tensorflow as tf
from argparse import ArgumentParser
import json
from progressbar import ProgressBar
import numpy as np

ELMO = "https://tfhub.dev/google/elmo/2"


def main():
    parser = ArgumentParser()
    parser.add_argument("--train-file", "-t", type=str, default="", help="input file for embedding")
    parser.add_argument("--dev-file", "-d", type=str, default="", help="input file for embedding")
    parser.add_argument("--context-file", "-c", type=str, default="", help="context embedding file")
    parser.add_argument("--question-file", "-q", type=str, default="", help="question embedding file")
    parser.add_argument("--train-ratio", "-tr", type=float, default=0.005, help="ratio of training data")
    parser.add_argument("--dev-ratio", "-dr", type=float, default=0.05, help="ratio of dev data")

    args = parser.parse_args()

    context_elmo_embedding = {}
    question_elmo_embedding = {}

    """
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    """
    """
    with tf.Graph().as_default():
        elmo = hub.Module(ELMO, trainable=False)  # モジュール用意
        
        embeddings = elmo(
            ["the cat is on the mat", "dogs are in the fog"],
            signature="default",
            as_dict=True)["elmo"]  # モジュール適用
    """

    print("process train file\n")
    with open(args.train_file) as f:

        json_list = [json.loads(line) for line in f]

        p_len = int(args.train_ratio * len(json_list))

        json_list = json_list[:p_len]

        pbar = ProgressBar()

        with tf.Graph().as_default():
            elmo = hub.Module(ELMO, trainable=False)  # モジュール用意

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())

                for json_item in pbar(json_list):

                    json_item["document"] = json_item["document"][:300]
                    tokens_length = [len(json_item['document'])]
                    tokens_input = [json_item["document"]]

                    embeddings_context = elmo(
                        inputs={
                            "tokens": tokens_input,
                            "sequence_len": tokens_length
                        },
                        signature="tokens",
                        as_dict=True)["elmo"]

                    if json_item["qid2cid"] not in context_elmo_embedding:
                        context_elmo_embedding[json_item['qid2cid']] = \
                            sess.run(embeddings_context)

                    tokens_length_q = [len(json_item['question'])]
                    tokens_input_q = [json_item["question"]]

                    embeddings_question = elmo(
                        inputs={
                            "tokens": tokens_input_q,
                            "sequence_len": tokens_length_q
                        },
                        signature="tokens",
                        as_dict=True)["elmo"]

                    question_elmo_embedding[json_item["id"]] = \
                        sess.run(embeddings_question)

    print("process dev file\n")
    with open(args.dev_file) as f:
        json_list = [json.loads(line) for line in f]

        p_len = int(args.dev_ratio * len(json_list))

        json_list = json_list[:p_len]

        pbar = ProgressBar()

        with tf.Graph().as_default():
            elmo = hub.Module(ELMO, trainable=False)  # モジュール用意

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())

                for json_item in pbar(json_list):

                    json_item["document"] = json_item["document"][:300]
                    tokens_length = [len(json_item['document'])]
                    tokens_input = [json_item["document"]]

                    embeddings_context = elmo(
                        inputs={
                            "tokens": tokens_input,
                            "sequence_len": tokens_length
                        },
                        signature="tokens",
                        as_dict=True)["elmo"]

                    if json_item["qid2cid"] not in context_elmo_embedding:
                        context_elmo_embedding[json_item['qid2cid']] = \
                            sess.run(embeddings_context)

                    tokens_length_q = [len(json_item['question'])]
                    tokens_input_q = [json_item["question"]]

                    embeddings_question = elmo(
                        inputs={
                            "tokens": tokens_input_q,
                            "sequence_len": tokens_length_q
                        },
                        signature="tokens",
                        as_dict=True)["elmo"]

                    question_elmo_embedding[json_item["id"]] = \
                        sess.run(embeddings_question)

    np.save(args.context_file, context_elmo_embedding)
    np.save(args.question_file, question_elmo_embedding)


def load_elmo_embedding():
    parser = ArgumentParser()
    parser.add_argument("--train-file", "-t", type=str, default="", help="input file for embedding")
    parser.add_argument("--dev-file", "-d", type=str, default="", help="input file for embedding")
    parser.add_argument("--context-file", "-c", type=str, default="", help="context embedding file")
    parser.add_argument("--question-file", "-q", type=str, default="", help="question embedding file")

    args = parser.parse_args()

    context_file = args.context_file + ".npy"
    question_file = args.question_file + ".npy"

    c_emb = np.load(context_file)
    q_emb = np.load(question_file)

    print(len(c_emb.item()))
    for item in c_emb.item():
        print(item)
        print(c_emb.item()[item])

    print(len(q_emb.item()))
    for item in q_emb.item():
        print(item)
        print(q_emb.item()[item])


if __name__ == '__main__':
    main()
    # load_elmo_embedding()
