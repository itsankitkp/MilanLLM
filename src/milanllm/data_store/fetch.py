import mwparserfromhell
import requests
from tqdm import tqdm

def get_wikitext(topic: str)->str:
    data = requests.get(
        url='https://en.wikipedia.org/w/api.php',
        params={
            "action":"parse",
            "page":topic,
            "prop":"wikitext",
            "formatversion":2,
            "format":"json"
        }
    )
    data_json = data.json()
    wikitext = data_json['parse']['wikitext']
    final_text =mwparserfromhell.parse(wikitext).strip_code()
    return final_text 

def get_categories(title: str)-> list[str]:
    data = requests.get(
        url='https://en.wikipedia.org/w/api.php',
        params={
            'action': 'query',
            'format':'json',
            'titles': title,
            'prop': 'categories'
        }
    )

    data_json = data.json()
    pages = data_json['query']['pages']
    categories=[]
    
    for _, v in pages.items():
        categories=v['categories']
        break

    final_cat=[]

    for item in categories:
        final_cat.append(item['title'])

    return final_cat

def get_search(topic: str)->list[str]:
    data = requests.get(
        url='https://en.wikipedia.org/w/api.php',
        params={
            'action': 'query',
            'format':'json',
            'srsearch': topic,
            'utf8': None,
            'list':'search',
            'srlimit': 500
        }
    )
    data_json = data.json()
    
    search = data_json['query']['search']
    titles= []
    for item in search:
        titles.append(item['title'])

    return titles
    
def get_corpus(topic)->str:
    topics = get_search(topic)

    corpus = []
    for t in tqdm(topics):
        txt = get_wikitext(t)
        corpus.append(txt)

    return corpus

def write_to_disk(corpus, fname='corpus.txt'):
    with open(fname,'w+', encoding='utf-8') as f:
        f.write(corpus)

def get_corpus_from_disk(fname='corpus.txt')->str:
    with open(fname,'r',encoding='utf-8') as f:
        corpus = f.read()
    return corpus