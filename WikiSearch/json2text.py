# -*- coding: utf-8 -*-
import argparse
import gzip
import json
import multiprocessing
import time
import re

#
# python json2text.py -source simplewiki-20171120-cirrussearch-content.json.gz -target wiki.txt -lower 1
#

parser = argparse.ArgumentParser(description='Process json Wikipedia file to plain text')
parser.add_argument('-source', action='store', dest='source', 
                    help='Filename of json Wikipedia')
parser.add_argument('-target', action='store', dest='target', 
                    help='File name of text file')
parser.add_argument('-lower', action='store', dest='is_lower', 
                    help='Convert to lowercase if argument is 1 (default converts to lower case)',
                    type=int, default=1)
args = parser.parse_args()
IS_LOWER = True if args.is_lower == 1 else False

def tokenize(text):
    global IS_LOWER
    article = text.lower()
    return re.findall('\w+|[^\w\s]', article)  
     
def process(content):
    content = content.split('\n') 
    json1, json2 = json.loads(content[0]), json.loads(content[1])
    wiki_type = json1['index']['_type']   
   
    if wiki_type == 'page' and 'text' in json2:
        text = json2['text']
        tokens = tokenize(text)
        line = ' '.join(tokens)
        return json2['title'], line
    else:
        return '', ''
    
def yield2lines(filename):
    with gzip.open(filename, 'rt') as json_f:  
        for line in json_f:
            next_line = json_f.__next__()
            yield line + next_line

class Wiki():
    def __init__(self, filename):
        self.filename = filename
        self.processes = max(1, multiprocessing.cpu_count() - 1)
    def get_article(self):
        pool = multiprocessing.Pool(self.processes)
        all_articles = yield2lines(self.filename)
        for title, text in pool.imap(process, all_articles):
            yield title, text
        pool.terminate()

if __name__ == '__main__':
    multiprocessing.freeze_support()  
    count = 0
    t = time.time()
    wiki = Wiki(args.source)
    with open(args.target, 'w', encoding='utf8') as text_f, open(args.target + '.title', 'w', encoding='utf8') as id_f:
        for title, line in wiki.get_article():
            if len(line) > 150 and not line.startswith(":"):
                text_f.write(line + "\n")
                id_f.write(title + "\n")
                count += 1
                if count % 10000 == 0:
                    print(count, 'articles processed')
    print(count, 'total wiki articles processed & saved')   
    print((time.time() - t) / 60, "minutes")        