import re
from wiki_dsmb.trie_element import TrieElement
from collections import deque

class TokenTrie:
    # in Python \p{Ll} is not supported yet
    # but by default in Python 3 regex support unicode classes,
    # e.g. the \s matches regular spaces and non-breaking spaces
    space_re = re.compile('\s',re.U)

    def __init__(self):
        self.dictionary = {}
        self.token_index = 1
        self.root = TrieElement()

    def add_str(self, string):
        self.add(self.str2tokens(string))

    def add(self, tokens):
        self.root.add(self.tokens2ids(tokens, True))

    def match_str(self, string, root=None):
        return self.match(self.str2tokens(string), root)

    def match(self, tokens, root=None):
        if(not root):
            root = self.root
        return root.match(self.tokens2ids(tokens))

    def to_str(self):
        reverse_dictionary = {v: k for k, v in self.dictionary.items()}
        return self.root.to_str(reverse_dictionary)

    def tokens2ids(self, tokens, add_flag=False):
        return deque([self.token2id(token, add_flag) for token in tokens])

    def token2id(self, token, add_flag=False):
        if(token not in self.dictionary):
            if(add_flag):
                self.dictionary[token] = self.token_index
                self.token_index += 1
            else:
                return 0
        return self.dictionary[token]

    def str2tokens(self, string):
        return self.space_re.split(string)
