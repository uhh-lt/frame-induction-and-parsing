# source :http://u.cs.biu.ac.il/~nlp/wp-content/uploads/extract_deps.py_.txt
import re
def read_conll(fh):
    root = (0,'*root*',-1,'rroot')
    tokens = [root]
    for line in fh.split('\n'):
        line.lower()
        tok = line.strip().split()
        try:
            if not tok:
                if len(tokens)>1: yield tokens
                tokens = [root]
            else:
                tokens.append((int(tok[0]),tok[1],int(tok[6]),tok[7]))
        except Exception as ex:
            print('---------------------------------------------')
            print(fh)
            print(ex)
            raise(ex)
    if len(tokens) > 1: 
        yield tokens


# stanford parser output example:  
# num(Years-3, Five-1)
line_extractor = re.compile('([a-z]+)\(.+-(\d+), (.+)-(\d+)\)')      
def read_stanford(fh):
    root = (0,'*root*',-1,'rroot')
    tokens = [root]
    for line in fh:
        if lower: line = line.lower()
        tok = line_extractor.match(line)
        if not tok:
            if len(tokens)>1: yield tokens
            tokens = [root]
        else:
            tokens.append((int(tok.group(4)),tok.group(3),int(tok.group(2)),tok.group(1)))
    if len(tokens) > 1:
        yield tokens

def read_sent(fh, format='conll'):
    if format == 'conll':
        return read_conll(fh)
    elif format == 'stanford':
        return read_stanford(fh)
    
    
'''
slight change after PaM
return index of word as well along with dep rel, to cater the situation where sentence may have multiple occurences of a word
'''
def dep_rels(parsed_text, vocab, format='conll'):
    """"
    parsed_text: either in conll format or stanford tree format
    """
#     =====================================
    deprels = []
    for i, sent in enumerate(read_sent(parsed_text, format=format)):
        for tok in sent[1:]:
            par_ind = tok[2] 
            par = sent[par_ind]
            m = tok[1]
            if m not in vocab: continue
            rel = tok[3]
            m_ind = tok[0]
    #      Stanford dependencies
            if rel == 'prep': 
                continue # this is the prep. we'll get there (or the PP is crappy)
            if rel == 'pobj' and par[0] != 0:
                ppar = sent[par[2]]
                rel = "%s:%s" % (par[3],par[1])
                h = ppar[1]
                h_ind = ppar[0]
            else:
                h = par[1]
                h_ind = par[0]

            if h not in vocab and h != '*root*': 
#                 print('Error...,' ,h, 'not in vocab') 
                continue

            if h != '*root*': 
                deprels.append((h_ind, h,"_".join((rel,m))))  

            deprels.append((m_ind, m,"I_".join((rel,h)))) 
    
    #     =====================================
    return deprels

