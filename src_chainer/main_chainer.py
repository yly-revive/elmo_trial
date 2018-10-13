import h5py

from bilm import dump_bilm_embeddings
from argparse import ArgumentParser
import os,json
from progressbar import ProgressBar

def main():

    parser = ArgumentParser()
    parser.add_argument('--train-file', type=str, default="")
    parser.add_argument('--dev-file', type=str, default="")
    parser.add_argument('--save-dir', type=str, default="")
    parser.add_argument('--dataset-file', type=str, default="dataset_file.txt")
    parser.add_argument('--embedding-file', type=str, default="elmo_embeddings.hdf5")
    parser.add_argument('--sentence-mapping-file', type=str, default="sentence_map.txt")
    
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    
    args = parser.parse_args()

    # Create the dataset file.
    dataset_file = os.path.join(args.save_dir, args.dataset_file)
    
    sentence_mapping = []

    print("train file processing...\n")
    with open(args.train_file) as f, \
        open(dataset_file, 'w') as of:
        json_list = [json.loads(line) for line in f]
        
        pbar = ProgressBar()

        for json_item in pbar(json_list):

            json_item["document"] = json_item["document"][:300]
            #print(json_item["qid2cid"])
            if json_item["qid2cid"] not in sentence_mapping:
                of.write(' '.join(json_item["document"]) + '\n')
                sentence_mapping.append(json_item["qid2cid"])
            
            of.write(' '.join(json_item["question"]) + '\n')
            sentence_mapping.append(json_item["id"])
            #question_elmo_embedding[json_item["id"]] = elmo(batch_to_ids([json_item["question"]]))["elmo_representations"][0].detach().numpy()

    print("dev file processing...\n")
    with open(args.dev_file) as f, \
        open(dataset_file, 'a') as of:
        json_list = [json.loads(line) for line in f]
        
        pbar = ProgressBar()

        for json_item in pbar(json_list):

            json_item["document"] = json_item["document"][:300]

            if json_item["qid2cid"] not in sentence_mapping:
                of.write(' '.join(json_item["document"]) + '\n')
                sentence_mapping.append(json_item["qid2cid"])
            
            of.write(' '.join(json_item["question"]) + '\n')
            sentence_mapping.append(json_item["id"])


    print("sentence_mapping file processing...\n")
    sentence_mapping_file = os.path.join(args.save_dir, args.sentence_mapping_file)
    with open(sentence_mapping_file, 'w') as of:
        for item in sentence_mapping:
            of.write(str(item) + '\n')
    """

    with open(dataset_file, 'w') as fout:
        for sentence in tokenized_context + tokenized_question:
            fout.write(' '.join(sentence) + '\n')

    """

    # Location of pretrained LM.
    vocab_file = 'elmo-chainer/vocab-2016-09-10.txt'
    options_file = 'elmo-chainer/elmo_2x4096_512_2048cnn_2xhighway_options.json'
    weight_file = 'elmo-chainer/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'

    # Dump the embeddings to a file. Run this once for your dataset.
    embedding_file = os.path.join(args.save_dir, 'elmo_embeddings.hdf5')

    # gpu id
    # if you want to use cpu, set gpu=-1
    gpu = args.gpu
    # batchsize
    # encoding each token is inefficient
    # encoding too many tokens is difficult due to memory
    batchsize = 32


    print("elmo file processing...\n")
    dump_bilm_embeddings(
        vocab_file, dataset_file, options_file, weight_file, embedding_file,
        gpu=gpu, batchsize=batchsize
    )

if __name__ == "__main__":
    main()
"""
# Load the embeddings from the file -- here the 2nd sentence.
with h5py.File(embedding_file, 'r') as fin:
    second_sentence_embeddings = fin['1'][...]
    print(second_sentence_embeddings.shape)
    # (n_layers=3, sequence_length, embedding_dim)
    print(second_sentence_embeddings)
"""