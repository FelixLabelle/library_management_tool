import argparse
import os
import json
import math
import pickle
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util

def format_result(score, result):
    return f"Score {score:.3f}; Filepath: {result['filepath']}\n\nText: {result['text']}\n\n"

if __name__ == "__main__":
    # Languages/type of text present
    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument('-f', '--dataset_filename', type=str, required=True, help='Dataset file')
    parser.add_argument('--search_type', type=str, default="exact", choices = ['ann','exact'], help="Type of search to perform, ann is faster but may miss results. Exact is slower but more precise")
    parser.add_argument('--model_name', type=str, default='all-mpnet-base-v2', help='Model name, sentence transformer models only supported ATM')
    parser.add_argument('--top_k', type=int, default=10, help='Output k best results')

    args = parser.parse_args()

    model_name = args.model_name
    embedding_size = 768 # TODO: Get this directly from the model
    top_k = args.top_k
    dataset_filename = args.dataset_filename
    search_type = args.search_type
    filename = os.path.basename(dataset_filename)
    filename_without_extension = os.path.splitext(filename)[0] 
    embedding_cache_path = f'embeddings_{filename_without_extension}_{model_name}.pkl'
    

 
    model = SentenceTransformer(model_name)
    #Check if embedding cache path exists
    if not os.path.exists(embedding_cache_path):
        
        # Get all unique sentences from the file
        passages = []
        relevant_metadata = []
        with open(dataset_filename, encoding='utf8') as fh:
            for document in json.load(fh):
                if 'error' in document:
                    print(document['error'])
                    continue
                for passage in document['passages']:
                    passage_text = document['text'][passage['start_pos']:passage['end_pos']]
                    passages.append(passage_text)
                    metadatum = {}
                    def is_contained(a1,a2,b1,b2):
                        return (a1 <= b1 and b1 <= a2) or (a1 <= b2 and b2 <= a2)
                        
                    metadatum['pages'] = [idx+1 for idx, (start_pos, end_pos) in enumerate(document['page_char_mapping']) if is_contained(start_pos,end_pos, passage['start_pos'], passage['end_pos']) ]
                    metadatum['filepath'] = document['filepath']
                    metadatum['filename'] = document['filename']
                    metadatum['text'] = passage_text
                    relevant_metadata.append(metadatum)
        corpus_embeddings = model.encode(passages, show_progress_bar=True, convert_to_numpy=True, batch_size=256)
        # Normalize corpus embeddings to make inner product equivalent to cosine similarity
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]
        print("Store file on disc")
        print(len(relevant_metadata))
        with open(embedding_cache_path, "wb") as fh:
            pickle.dump({'metadata': relevant_metadata, 'embeddings': corpus_embeddings}, fh)
    else:
        print("Load pre-computed embeddings from disc")
        with open(embedding_cache_path, "rb") as fh:
            cache_data = pickle.load(fh)
            relevant_metadata = cache_data['metadata']
            corpus_embeddings = cache_data['embeddings']
        print(len(relevant_metadata))

    ### Create the FAISS index
    print("Start creating FAISS index")

    #Defining our FAISS index
    #Number of clusters used for faiss. Select a value 4*sqrt(N) to 16*sqrt(N) - https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    n_clusters = int(4*math.sqrt(len(relevant_metadata)))

    # Create index that uses inner product
    quantizer = faiss.IndexFlatIP(embedding_size)
    index = faiss.IndexIVFFlat(quantizer, embedding_size, n_clusters, faiss.METRIC_INNER_PRODUCT)

    #Number of clusters to explorer at search time. We will search for nearest neighbors in 3 clusters.
    index.nprobe = 3
    # Then we train the index to find a suitable clustering
    index.train(corpus_embeddings)
    # Finally we add all embeddings to the index
    index.add(corpus_embeddings)



    ######### Search in the index ###########
    print("Corpus loaded with {} sentences / embeddings".format(len(relevant_metadata)))
    while True:
        inp_question = input("Please enter a question: ")

        start_time = time.time()
        question_embedding = model.encode(inp_question)

        #FAISS works with inner product (dot product). When we normalize vectors to unit length, inner product is equal to cosine similarity
        question_embedding = question_embedding / np.linalg.norm(question_embedding)
        question_embedding = np.expand_dims(question_embedding, axis=0)
        if search_type == "ann":
            # Search in FAISS. It returns a matrix with distances and corpus ids.
            distances, corpus_ids = index.search(question_embedding, top_k)

            # We extract corpus ids and scores for the first query
            hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids[0], distances[0])]
            hits = sorted(hits, key=lambda x: x['score'], reverse=True)
            end_time = time.time()

            print("Input question:", inp_question)
            print("Results (after {:.3f} seconds):".format(end_time-start_time))
            for hit in hits[0:top_k]:
                result_str = format_result(hit['score'], relevant_metadata[hit['corpus_id']])
                print(result_str)
        
        elif search_type == "exact":
            # Approximate Nearest Neighbor (ANN) is not exact, it might miss entries with high cosine similarity
            hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)[0]
            hits = sorted(hits, key=lambda x: x['score'], reverse=True)
            end_time = time.time()

            print("Input question:", inp_question)
            print("Results (after {:.3f} seconds):".format(end_time-start_time))

            for hit in hits[0:top_k]:
                result_str = format_result(hit['score'], relevant_metadata[hit['corpus_id']])
                print(result_str)

            
        else:
            """
        
            recall = len(ann_corpus_ids.intersection(correct_hits_ids)) / len(correct_hits_ids)
        print("\nApproximate Nearest Neighbor Recall@{}: {:.2f}".format(top_k, recall * 100))

        if recall < 1:
            print("Missing results:")
            for hit in correct_hits[0:top_k]:
                if hit['corpus_id'] not in ann_corpus_ids:
                    print("\t{:.3f}\t{}".format(hit['score'], relevant_metadata[hit['corpus_id']]))
        print("\n\n========\n")
            """
            raise NotImplementedError