# from operator import itemgetter
import os
from math import log, sqrt
from collections import defaultdict
import nltk
import re
inverted_index = defaultdict(list)
nos_of_documents=0
vects_for_docs = {}  # we will need nos of docs number of vectors, each vector is a dictionary
vect7_for_docs={}
document_freq_vect = {}  # sort of equivalent to initializing the number of unique words to 0
names=[]
words=[]
# this is the first function that is executed.
# It updates the vects_for_docs variable with vectors of all the documents.
def iterate_over_all_docs():
    global nos_of_documents
    for fname in os.listdir():
        names.append(fname)
        nos_of_documents+=1
        doc_text = get_document_text_from_doc_id(fname)
        token_list = getlist(doc_text,fname)
        vect = create_vector(token_list)        
        vects_for_docs[fname]=vect
def getlist(doc_text,fname):
    vect5=defaultdict(list)
    lineno = 1
    global vect7_for_docs
    fo=open(fname).read().splitlines()
    #as vect7 has dictionary as values
    vect7=defaultdict(dict)
    a=[]
    ps = nltk.stem.PorterStemmer()
            
    for line in fo:
        line=line.lower()
        line=re.sub(r'[^a-z0-9 |]',' ',line)
        
        tokens = nltk.word_tokenize(line)
        i=1
        for word in tokens:
            
            if(word not in words): 
                if(word not in a):
                    vect7[ps.stem(word)]=defaultdict(list)
                    a.append(word)    
                vect7[ps.stem(word)][lineno].append(i)
 
            i+=1
        lineno+=1        
    vect7_for_docs[fname]=vect7
    return vect5
        
# this function returns a list of tokenized and stemmed words of any text
def get_tokenized_and_normalized_list(doc_text):
    # return doc_text.split()
    doc_text=doc_text.lower()
    doc_text =re.sub(r'[^a-z0-9 |]',' ',doc_text)  
     
    tokens = nltk.word_tokenize(doc_text)
    ps = nltk.stem.PorterStemmer()
    stemmed = []
    for word in tokens:
        if (word not in words):
            stemmed.append(ps.stem(word))
    return stemmed


# creates a vector from a list (l1) , vector is a dictionary, containing words:frequency pairs
# this function should not be called to parse the query given by the user
# because this function also updates the document frequency dictionary
def create_vector(l1):
    vect = {}  # this is a dictionary
    global document_freq_vect

    for token in l1.keys():
        if token in vect:
            vect[token] += 1
            
        else:
            vect[token] = 1
            if token in document_freq_vect:
                document_freq_vect[token] += 1
            else:
                document_freq_vect[token] = 1
    return vect


def get_document_text_from_doc_id(fname):
    # noinspection PyBroadException
    try:
        str1 = open(fname).read()
    except:
        str1 = ""
    return str1

# name is self explanatory, it generates and inverted index in the global variable inverted_index,
# however, precondition is that vects_for_docs should be completely initialized
def generate_inverted_index():
    
    
    for fname,vector in vects_for_docs.items():
        for word1 in vector:
            inverted_index[word1].append(fname)
        


# it updates the vects_for_docs global variable (the list of frequency vectors for all the documents)
# and changes all the frequency vectors to tf-idf unit vectors (tf-idf score instead of frequency of the words)
def create_tf_idf_vector():
    vect_length = 0.0
    for vect in vects_for_docs.values():
        for word1 in vect:
            word_freq = vect[word1]
            temp = calc_tf_idf(word1, word_freq)
            vect[word1] = temp
            vect_length += temp ** 2

        vect_length = sqrt(vect_length)
        for word1 in vect:
            vect[word1] /= vect_length

# precondition: word is in the document_freq_vect
# this function calculates the tf-idf score for a given word in a document
def calc_tf_idf(word1, word_freq):
    return log(1 + word_freq) * log(nos_of_documents / document_freq_vect[word1])

# creates a vector from a query in the form of a list (l1) , vector is a dictionary, containing words:frequency pairs
def create_vector_from_query(l1):
    vect = {}
    for token in l1:
        if token in vect:
            vect[token] += 1.0
        else:
            vect[token] = 1.0
    return vect

# note: even though you do not need to convert the query vector into a unit vector,
# I have done so because that would make all the dot products <= 1
# as the name suggests, this function converts a given query vector
# into a tf-idf unit vector(word:tf-idf vector given a word:frequency vector
def get_tf_idf_from_query_vect(query_vector1):
    vect_length = 0.0
    for word1 in query_vector1:
        word_freq = query_vector1[word1]
        if word1 in document_freq_vect:  # I have left out any term which has not occurred in any document because
            query_vector1[word1] = calc_tf_idf(word1, word_freq)
        else:
            query_vector1[word1] = log(1 + word_freq) * log(
                nos_of_documents)  # this additional line will ensure that if the 2 queries,
            # the first having all words in some documents,
            #   and the second having and extra word that is not in any document,
            # will not end up having the same dot product value for all documents
        vect_length += query_vector1[word1] ** 2
    vect_length = sqrt(vect_length)
    if vect_length != 0:
        for word1 in query_vector1:
            query_vector1[word1] /= vect_length



# define a number of functions,
# function to to read a given document word by word and
# 1. Start building the dictionary of the word frequency of the document,
#       2. Update the number of distinct words
#  function to :
#       1. create the dictionary of the term freqency (number of documents which have the terms);


# this function returns the dot product of vector1 and vector2
def get_dot_product(vector1, vector2):
    if len(vector1) > len(vector2):  # this will ensure that len(dict1) < len(dict2)
        temp = vector1
        vector1 = vector2
        vector2 = temp
    keys1 = vector1.keys()
    keys2 = vector2.keys()
    sum = 0
    for i in keys1:
        if i in keys2:
            sum += vector1[i] * vector2[i]
    return sum

# this function takes the dot product of the query with all the documents
#  and returns a sorted list of tuples of docId, cosine score pairs
def get_result_from_query_vect(query_vector1):
    parsed_list = []
    for i in names:
        dot_prod = get_dot_product(query_vector1, vects_for_docs[i])
        parsed_list.append((i, dot_prod))
        parsed_list = sorted(parsed_list, key=lambda x: x[1],reverse=True)
    return parsed_list

# to take care of words which are non essential
def stopword():
    f=open("stopwords.dat").read().splitlines()
    global words
    for i in f:
        words.append(i)
    
    return words
#the directory is changed
def directory():
    while 1:
        search_path = input("Enter directory path to search : ")
        if not os.path.exists(search_path):
            print("Invalid Directory try again")
        else:
            break
    os.chdir( search_path )
# initializing the vects_for_docs variable
def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


words=stopword()

directory()

iterate_over_all_docs()

# self explanatory
generate_inverted_index()

# changes the frequency values in vects_for_docs to tf-idf values
create_tf_idf_vector()

count = 1
print()
while True:
    query = input("Please enter your query....")
    if len(query) == 0:
        break
    query_list = get_tokenized_and_normalized_list(query)
    query_vector = create_vector_from_query(query_list)
    get_tf_idf_from_query_vect(query_vector)
    result_set = get_result_from_query_vect(query_vector)
    equal=defaultdict(list)
    number=[]
    j=1

    for tup in result_set:
        print( " Rank is " + str(j)+":The file is " + tup[0] )
        i=0
        flag=0
        temp=[]
        j+=1
        for word in query_list:
            if(i==0):
                temp=vect7_for_docs[tup[0]][word].keys()
            temp=intersection(temp,vect7_for_docs[tup[0]][word].keys())
            i=1
        for line in temp:
            index=[]
            i=0
            for word in query_list:
                if(i==0):
                    index=vect7_for_docs[tup[0]][word][line]
                else:
                    a=vect7_for_docs[tup[0]][word][line]
                    a[:] = [x - i for x in a]
                    index=intersection(index,a)
                i+=1
            for t in index:
                flag=1
                print ("  ["+str(line)+"]")
                equal[tup[0]].append(line)
                break
        if (flag==0):
            print("  No perfect matches in "+tup[0])
        print()
    print("Other matches:\n")
    j=1
    for tup in result_set:
        print( " Rank is " + str(j)+":The file is " + tup[0] )
        j+=1  
        flag=0         
        x=set()
        truth=defaultdict(int)
        for word in query_list:
            
            for i in vect7_for_docs[tup[0]][word].keys():
                
                if (truth[i]==0 and i not in equal[tup[0]]):
                    flag=1
                    truth[i]=1
                    print("  ["+str(i)+"]")
        if (flag==0):
            print("  No matches in "+tup[0])
        print()
    print()
            
        
        	
    
