#!/usr/bin/env python
from wiki_dsmb.context import Context 
from collections import defaultdict
import time

class TrieMatcher:
    def __init__(self, path, wiki):
        self.path = path
        self.wiki = wiki

    def read_raw_tuple(self, token_file):
        line = token_file.readline()
        if(not line):
            raise EOFError()
        return line.strip().split('\t')

    def read_tuple(self, token_file):
        while(True):
            result = self.read_raw_tuple(token_file)
            if(len(result) == 4):
                result[0] = int(result[0])
                result[1] = int(result[1])
                return result

    def seek_to_first(self, first_doc, token_file):
        # The doc id might not be present in the index
        # thus we find the first id that is larger than first_doc,
        # and is present in the index.
        max_value = max(self.wiki.token_index.keys())
        while(first_doc < max_value):
            if(first_doc in self.wiki.token_index):
                token_file.seek(self.wiki.token_index[first_doc])
                break
            first_doc += 1

    # The method iterates over the tokens of the file. If the token is a potential link, the returned
    # tuple (context, level) has lavel > 0, indicating that the matched trie element has the
    # indicated lavel in the trie structure.
    #
    # The parameter first_doc, if given, indicates the first document that will processed. 
    # The trie matcher uses a simple index to find the first token in the doc, thus skipping documents
    # is pretty fast.
    # 
    # The parameter last_doc, if given, indicates the first document that *won't be* processed.
    def links(self, first_doc=None, last_doc=None):
        self.index = self.wiki.trie
        context = Context(self.index, None)
        with open(self.path) as token_file:
            try:
                if(first_doc is not None):
                    self.seek_to_first(first_doc, token_file)
                while(True):
                    context.document_id, context.first_token_index, space, token = self.read_tuple(token_file)
                    if(last_doc is not None and context.document_id >= last_doc):
                        break
                    recorded_pos = token_file.tell()
                    context.last_token_index = context.first_token_index
                    context.string = token
                    context.token = token
                    context.space = space
                    level = 1
                    while(True):
                        matched, next_trie = context.index.match_str(context.string)
                        if(matched):
                            yield (context, level)
                        else:
                            if(level == 1):
                                yield (context, -1)
                        level += 1
                        last_document_id, context.last_token_index, space, last_token = self.read_tuple(token_file)
                        if(context.document_id != last_document_id):
                            break
                        if(not next_trie and space == "1"):
                            break
                        if(space == "1"):
                            context.string += " "
                        context.string += last_token
                        context.token = last_token
                    token_file.seek(recorded_pos)
            except EOFError:
                pass

if __name__=='__main__':
    from wiki_dsmb.wiki import Wiki
    from wiki_dsmb.trie_element import TrieElement
    from wiki_dsmb.token_trie import TokenTrie
    import argparse

    parser = argparse.ArgumentParser(description='Ngram counter.')
    parser.add_argument('--data_path', required=True, help='the path to the directory with Wikipedia data')
    parser.add_argument('--first_doc', required=True, type=int, default=1, help='the id of the first document to process')
    parser.add_argument('--last_doc', required=True, type=int, default=10, help='the id of the first document *not to* to process')

    args = parser.parse_args()

    MIN_SENSE_PROB = 0.01

    #data_path = '/net/scratch/people/plgapohl/wiki/pl/2017/'
    wiki = Wiki(args.data_path, MIN_SENSE_PROB)
    wiki._load_token_trie()
    wiki._load_indices()
    counts = defaultdict(lambda: 0)
    start_time = time.time()
    matcher = TrieMatcher(args.data_path + 'tokens.tsv', wiki)
    i = 0
    for context, level in matcher.links(args.first_doc, args.last_doc):
        if(level > 0):
            i += 1
            counts[context.string] += 1
            
    fname = "counts_ngrams.{}-{}.txt".format(args.first_doc, args.last_doc)
    with open(args.data_path + fname, 'w') as f:
        for link, count in counts.items():
            f.write("{} {}\n".format(link, count))

    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    print("{} tokens/s".format(i / float(end_time - start_time) )) 
