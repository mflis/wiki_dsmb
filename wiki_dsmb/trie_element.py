class TrieElement:
    def __init__(self):
        self.children = {}
        self.final = False

    def add(self, tokens):
        if(len(tokens) > 0):
            token_id = tokens.popleft()
            if(token_id not in self.children):
                self.children[token_id] = TrieElement()
            self.children[token_id].add(tokens)
        else:
            self.final = True

    def to_str(self, dictionary, spaces=''):
        result = ""
        for token_id in self.children.keys():
            token = dictionary.get(token_id, str(token_id) + " missing!")
            result += spaces + token
            element = self.children[token_id]
            if(element.final):
                result += "(!)"
            result += ":\n"
            result += element.to_str(dictionary, spaces + '  ')
        return result

    def match(self, tokens):
        if(len(tokens) > 0):
            token_id = tokens.popleft()
            if(token_id in self.children):
                return self.children[token_id].match(tokens)
            else:
                return (False, None)
        else:
            return (self.final, self)

    def __str__(self):
        return "TrieElement[{}]".format(len(self.children))

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return self.__str__()
