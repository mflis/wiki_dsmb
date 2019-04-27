class ConceptBlockGenerator:
    def __init__(self, path, wiki, min_concept_block=5, max_concept_block=50):
        self.path = path
        self.wiki = wiki
        self.min_concept_block = min_concept_block
        self.max_concept_block = max_concept_block
    
    def seek_to_first(self, first_doc, links_file):
        # The doc id might not be present in the index
        # thus we find the first id that is larger than first_doc,
        # and is present in the index.
        max_value = max(self.wiki.link_index.keys())
        while(first_doc < max_value):
            if(first_doc in self.wiki.link_index):
                links_file.seek(self.wiki.link_index[first_doc])
                break
            first_doc += 1
        
    # The method iterates over the links of Wikipedia. 
    #
    # The parameter first_doc, if given, indicates the first document that will processed. 
    # The generator uses a simple index to find the first link in the doc, thus skipping documents
    # is pretty fast.
    # 
    # The parameter last_doc, if given, indicates the first document that *won't be* processed.
    def blocks(self, first_doc=None, last_doc=None):
        last_doc_id = None
        concept_block = []
        with open(self.path) as links_file:
            if(first_doc is not None):
                self.seek_to_first(first_doc, links_file)
            while(True):
                try:
                    doc_id, start_token, end_token, sense, link = links_file.readline().rstrip().split('\t')
                    start_token = int(start_token)
                    end_token = int(end_token)
                except ValueError as ex:
                    continue
                doc_id = int(doc_id)
                self.add_self_link(concept_block, doc_id, start_token, end_token)
                sense = sense[0:1].upper() + sense[1:]
                try:
                    sense_id = self.wiki.pages_reverse[sense]
                    sense_type = self.wiki.pages[sense_id][1]
                    if(sense_type == 2):
                        sense_id = self.wiki.resolve_redirect(sense_id)
                        sense_type = self.wiki.pages[sense_id][1]
                    if(sense_type == 2):
                        continue
                    if(doc_id != last_doc_id or len(concept_block) >= self.max_concept_block):
                        if(len(concept_block) >= self.min_concept_block):
                            weights, goodness = self.wiki.compute_weights_and_goodness(concept_block)
                            yield(concept_block, weights, goodness, start_token, last_doc_id)
                        if(doc_id == last_doc_id):
                            # max concept block size reached
                            # TODO so far we just clear the block
                            concept_block = []
                        else:
                            # new document reached
                            concept_block = []
                    if(last_doc is not None and doc_id >= last_doc):
                        break
                    last_doc_id = doc_id
                    self.add_self_link(concept_block, doc_id, start_token, end_token)
                    concept_block.append([link, sense_id, start_token, end_token])
                except KeyError:
                    continue

    def add_self_link(self, concept_block, doc_id, start_token, end_token):
        if(len(concept_block) == 0 and doc_id in self.wiki.pages):
            # add artificial link to the document itself
            concept_block.append([self.wiki.pages[doc_id][0], doc_id, start_token, end_token])
