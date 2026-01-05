import json

class BasicTokenizer:
    def __init__(self):
        # self.vocab_size = 276    q                                # the desired final vocab size : we preset
        self.vocab = {idx: bytes([idx]) for idx in range(256)}   # vocab: raw byte values: utf-8 encoding + BPE encoding
        self.merges = {}                                         # merges dictionary: new id mapping for pair of tokens
    
    def get_pair_of_id(self, ids:list):
        '''
        find the most common pair in the corpus,
        - read the list of ids,
        - each consecutive pair is a candidate,
        - count to keep tract of total number of pair
        return dictionary (not sorted, pair is distinct)
        '''
        counts = {}                                     #initialize empty dictionary
        for pair in zip(ids, ids[1:]):                  #create pairs by zipping the list with itself offset by 1
            counts[pair] = counts.get(pair, 0) + 1      #count occurrences of each pair
        return counts

    def map_pair_with_new_id(self, ids, pair, idx):
        '''
        map the new token (pair) with new id (idx)
        - get the encoded text: list of id (ids) 
        - get the "pair" from the ids (pair), maximum count to be replaced first
        - replace the "pair" by a new index (idx)
        return the new list of ids after replacement
        '''
        new_ids = []                        #initialize empty list
        i = 0
        while i < len(ids):                 #iterate through the list of original ids
            if i < len(ids)-1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)         #append the new index, skip the next id
                i+=2
            else:
                new_ids.append(ids[i])      #append the current id
                i+=1
        return new_ids

    def encode(self, text:str):
        '''
            encode the human text to sequence of integers
            - inital encoder is utf-8
            - then apply BPE merges
            - integer is map to byte object, from vocabulary
            output: list of integers
        '''
        tokens = list(text.encode("utf-8"))                 #initial tokenization using utf-8 encoding

        # apply BPE merges
        while len(tokens) >= 2:                             #need at least 2 tokens to form a pair
            stats = self.get_pair_of_id(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))        #find the pair with the lowest index in merges
            if pair not in self.merges:
                break
                print('Nothing else can be merge!')
            idx = self.merges[pair]                              #get the new index for the pair from merges
            tokens = self.map_pair_with_new_id(tokens, pair, idx)
        return tokens

    def decode(self, ids:list):
        '''
            decode the sequence of integers to human text
            using vocabulary + merges that created in training stage 
            output: list of string
        '''
        decoded_list = []
        # print('tokens:',ids)
        for i in range(len(ids)):
            token = self.vocab[ids[i]].decode("utf-8", errors = 'replace')   #map each id to its corresponding byte object in vocab and decode to string
            decoded_list.append(token)                                       #append the decoded string to the list
        # text = ''.join(decoded_list)
        return decoded_list


    def train(self, training_text:str, vocab_size:int, verbose = False):
        '''
        train the BPE merges
        - starting from utf-8 encoding (initial vocab of size 256)
        - iteratively find the most common pair and merge them
        - continue until desired vocab size is reached
        output: merges dictionary
        '''
        num_merges = vocab_size - int(256)                           #256 is the initial vocab size (ASCII utf-8)
        ids = list(training_text.encode("utf-8"))               #initial tokenization using utf-8 encoding
        tokens = ids.copy()                                     #keep a copy of original tokens for statistics
        
        # Build merge dict: maintain child1(int1), child2(int2) maping to a new tokens(int3)
        print('Building BPE merges...')
        i = 0
        while i <= num_merges:
            stats = self.get_pair_of_id(ids)
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f"merge pair {top_pair} into new id {idx}")
            ids = self.map_pair_with_new_id(ids, top_pair, idx)
            self.merges[top_pair] = idx
            i+=1
        
        print('Complete merges: ', self.merges)                           # new extended merges
        
        #statistic check
        print("ori tokens lenght:", len(tokens))
        print("new tokens lenght:", len(ids))
        print(f"compression ratio: {len(tokens)/len(ids):.2f}X")

        # Build the vocab: from 256 raw byte + extended BPE tokens from merges
        print('Building vocabulary...')
        for (p0, p1), idx in self.merges.items():                # mapping for the new token id in 'merges' to pair
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]    # new token is the concatenation of the two child tokens

        print('Complete vocab: ', self.vocab)                             # new extended vocab


        
        # Save to JSON file
        vocab_json = {k: str(v) for k, v in self.vocab.items()}            # bytes -> list[int]
        merges_json = {f"({k0},{k1})": v for (k0, k1), v in self.merges.items()}  # tuple key -> string
        # merges_json = self.merges
        with open('vocab.json', 'w', encoding='utf-8') as vf:
            json.dump(vocab_json, vf, ensure_ascii=False)

        with open('merges.json', 'w', encoding='utf-8') as mf:
            json.dump(merges_json, mf, ensure_ascii=False)
