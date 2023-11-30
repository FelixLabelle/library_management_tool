import argparse
from datetime import datetime, timezone
import os
import json

from pypdf import PdfReader
from tqdm.auto import tqdm
import spacy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser")
    parser.add_argument('-r', '--root_dir', type=str, required=True, help='Path to where PDFs are stored')
    parser.add_argument('-t', '--tag', type=str, required=True, help='Versioning tag, used to keep track of different indices etc..')
    parser.add_argument('--write_count', type=int, default=20, help='How often to save temporary files, this can be increased to run slightly faster')
    parser.add_argument('--passage_size', type=int, default=250, help='Passage size to be used during retrieval')
    parser.add_argument('--tokenizer', type=str, default="en_core_web_sm", help='Sentence tokenizer to be used, currently only Spacy models are supported')
    
    args = parser.parse_args()
    root_dir = args.root_dir
    write_count = args.write_count
    tag = args.tag
    passage_size = args.passage_size
    tokenizer = args.tokenizer
    
    nlp = spacy.load(TOKENIZER)
    pdfs = []
    nlp.add_pipe('sentencizer')
    for dirpath, dirnames, filenames in tqdm(os.walk(ROOT_DIR)):
        for filename in tqdm(filenames):
            if not filename.lower().endswith(".pdf"):
                continue
            pdf = {}
            try:
                pdf_filepath = os.path.join(dirpath,filename)
                pdf['filename'] = filename
                pdf['filepath'] = pdf_filepath
                reader = PdfReader(pdf_filepath)
                page_char_mapping = []
                pdf_text = ""
                for page in reader.pages:
                    page_text = page.extract_text() 
                    page_char_mapping.append([len(pdf_text) , len(pdf_text)+len(page_text)])
                    pdf_text += page_text
                    
                pdf['page_char_mapping'] = page_char_mapping
                pdf['text'] = pdf_text
                nlp.max_length= len(pdf_text)+1
                doc = nlp(pdf_text, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer","ner"])
                # Split text...
                pdf['sentence_idxs'] = [{"start_pos" : sent.start_char, "end_pos" : sent.end_char, "num_words" : len(sent)} for sent in doc.sents]
                pdf['passages'] = []
                current_passage = pdf['sentence_idxs'][0]
                for sentence_idx in pdf['sentence_idxs'][1:]:
                    if current_passage['num_words'] + sentence_idx['num_words'] <= PASSAGE_SIZE:
                        current_passage['end_pos'] = sentence_idx['end_pos']
                        current_passage['num_words'] += sentence_idx['num_words']
                    else:
                        pdf['passages'].append(current_passage)
                        current_passage = sentence_idx
                        
                if current_passage:
                    pdf['passages'].append(current_passage)
                    
                pdf['tokenizer'] = TOKENIZER
                pdf['tag'] = TAG
                pdf['date'] = datetime.now(timezone.utc).strftime("%m/%d/%Y, %H:%M:%S")
                pdf['passage_size'] = PASSAGE_SIZE
                
            except Exception as e:
                print(str(e))
                pdf['error'] = str(e)
            pdfs.append(pdf)
            if len(pdfs) > 1 and (len(pdfs) % WRITE_COUNT) == 0:
               json.dump(pdfs,open(f"temp_pdfs_parse_{TAG}.json","w",encoding="utf-8"))

    json.dump(pdfs,open(f"pdfs_parse_{TAG}.json","w",encoding="utf-8"))
    os.remove(f"temp_pdfs_parse_{TAG}.json")