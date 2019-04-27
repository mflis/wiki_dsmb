import copy

class SegmentSelector:
    def __init__(self, matcher, wiki):
        self.buffer = []
        self.token_buffer = []
        self.wiki = wiki
        self.matcher = matcher
        self.last_tuple = None

    def selected_links(self, first_doc, last_doc):
        self.link = self.matcher.links(first_doc, last_doc)
        previous_doc = first_doc
        while(True):
            main_context, level = next(self.link)
            if(level == 1 or level == -1):
                self.token_buffer.append(main_context)
            main_prob = float(self.wiki.link_probability(main_context.string))
            #print("{:f}".format(main_prob), main_context.string, level)

            marked_for_passing = []
            marked_for_removal = []
            append = True
            for current_context, current_prob in self.buffer:
                if(current_context.last_token_index < main_context.first_token_index):
                    marked_for_passing.append((current_context, current_prob))
                else:
                    if(current_prob >= main_prob):
                        append = False
                        break
                    else:
                        marked_for_removal.append((current_context, current_prob))
                        #print("x " + current_context.string)
            if(append):
                self.buffer.append((copy.copy(main_context), main_prob))

            for tuple_to_remove in marked_for_removal:
                for current_token in self.token_buffer:
                    if(current_token.first_token_index >= tuple_to_remove[0].first_token_index and
                          current_token.first_token_index < tuple_to_remove[0].last_token_index and
                          (len(self.buffer) == 0 or
                            current_token.first_token_index < self.buffer[0][0].first_token_index)):
                        marked_for_passing.append((current_token, link_probability(current_token.string)))
                self.buffer.remove(tuple_to_remove)

            tokens_to_remove = []
            sorted_tuples = sorted(marked_for_passing, key=lambda x: x[0].first_token_index)
            for tuple_to_pass in sorted_tuples:
                if(tuple_to_pass in self.buffer):
                    self.buffer.remove(tuple_to_pass)
                for current_token in self.token_buffer:
                    if(current_token.first_token_index >= tuple_to_pass[0].first_token_index and
                          current_token.first_token_index < tuple_to_pass[0].last_token_index):
                        tokens_to_remove.append(current_token)
                self.last_tuple = tuple_to_pass
                yield tuple_to_pass[0]

            for token_to_remove in tokens_to_remove:
                if(token_to_remove in self.token_buffer):
                    self.token_buffer.remove(token_to_remove)
            if(previous_doc != main_context.document_id):
                self.buffer = []
                self.token_buffer = []
            previous_doc = main_context.document_id


    def push_back(self):
        if(self.last_tuple is not None):
            self.buffer.insert(0, self.last_tuple)
            self.last_tuple = None
