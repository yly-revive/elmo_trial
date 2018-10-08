from allennlp.modules.elmo import Elmo, batch_to_ids
# from allennlp.commands.elmo import ElmoEmbedder
from argparse import ArgumentParser
import json
from progressbar import ProgressBar
import numpy as np

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"


def main():
    parser = ArgumentParser()
    parser.add_argument("--train-file", "-t", type=str, default="", help="input file for embedding")
    parser.add_argument("--dev-file", "-d", type=str, default="", help="input file for embedding")
    parser.add_argument("--context-file", "-c", type=str, default="", help="context embedding file")
    parser.add_argument("--question-file", "-q", type=str, default="", help="question embedding file")

    args = parser.parse_args()

    elmo = Elmo(options_file, weight_file, 2, dropout=0)
    """
    # use batch_to_ids to convert sentences to character ids
    #sentences = [['First', 'sentence', '.'], ['Another', '.']]
    sentences = [['First', 'sentence', '.']]
    character_ids = batch_to_ids(sentences)

    embeddings = elmo(character_ids)
    embeddings_n = embeddings["elmo_representations"][0].detach().numpy()
    """

    """
    elmo = ElmoEmbedder()
    sentences = ['First', 'sentence', '.']
    embeddings = elmo.embed_sentence(sentences)
     """
    # print(embeddings)
    # print(len(embeddings))

    context_elmo_embedding = {}
    question_elmo_embedding = {}

    print("process train_file\n")
    with open(args.train_file) as f:
        json_list = [json.loads(line) for line in f]
        """
        count = 0
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
            count += 1
            if count >= 2:
                break
        """
        pbar = ProgressBar()

        for json_item in pbar(json_list):

            json_item["document"] = json_item["document"][:300]

            if json_item["qid2cid"] not in context_elmo_embedding:
                context_elmo_embedding[json_item['qid2cid']] = \
                    elmo(batch_to_ids([json_item["document"]]))["elmo_representations"][0].detach().numpy()

            question_elmo_embedding[json_item["id"]] = \
                elmo(batch_to_ids([json_item["question"]]))["elmo_representations"][0].detach().numpy()

    print("process dev_file\n")
    with open(args.dev_file) as f:

        json_list = [json.loads(line) for line in f]

        """
        count = 0
        json_list = []
        for line in f:
            json_list.append(json.loads(line))
            count += 1
            if count >= 2:
                break
        """

        pbar = ProgressBar()

        for json_item in pbar(json_list):

            json_item["document"] = json_item["document"][:300]

            if json_item["qid2cid"] not in context_elmo_embedding:
                context_elmo_embedding[json_item['qid2cid']] = \
                    elmo(batch_to_ids([json_item["document"]]))["elmo_representations"][0].detach().numpy()

            question_elmo_embedding[json_item["id"]] = \
                elmo(batch_to_ids([json_item["question"]]))["elmo_representations"][0].detach().numpy()

    print("process context_file\n")
    """
    with open(args.context_file, "w") as of:
        pbar = ProgressBar()
        for key in pbar(context_elmo_embedding):
            of.write(str(key))
            of.write(":")
            of.write(context_elmo_embedding[key])
            of.write("\n")
    """
    np.save(args.context_file, context_elmo_embedding)
    print("process question_file\n")

    """
    with open(args.question_file, "w") as of:
        pbar = ProgressBar()
        for key in pbar(question_elmo_embedding):
            of.write(str(key))
            of.write(":")
            of.write(question_elmo_embedding[key])
            of.write("\n")
            
    """

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
