#!/usr/bin/env python
from wiki_dsmb.context import Context
from wiki_dsmb.trie_matcher import TrieMatcher
from collections import defaultdict
import time

class ConllTrieMatcher(TrieMatcher):
    def __init__(self, path, wiki):
        super(TrieMatcher, self).__init__(path, wiki)
        self.token_index = 0
        self.doc_index = 0

    def read_tuple(self, token_file):
        while(True):
            result = self.read_raw_tuple(token_file)
            if(len(result) == 4):
                self.token_index += 1
                tuple = [self.doc_index, self.token_index, result[2], result[0]]
                return tuple
            else:
                if(len(result) == 0):
                    #self.doc_index += 1
                    pass
