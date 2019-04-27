class Context:
    def __init__(self, index, counts):
        self.index = index
        self.counts = counts
        self.output = None
        self.string = None
        self.first_token_index = 0
        self.last_token_index = 0
        self.document_id = None
        self.token = None
        self.space = "0"
