from glob import glob
import xml.etree.ElementTree as et
from os.path import join
import pandas as pd
import os
from pathlib import Path  


INPUT_DIR = '../parser_workdir/data/open_sesame_v1_data/fndata-1.7'
OUTPUT_DIR = "../workdir/framenet_data"

# all frame elements in one row v1
COLUMNS = ['frameID', 'frameName', 'sentence', 'luName', 'luID', 'luText', 'luPOS', 'luIndex', 'frameElements', 'sentenceID', 'annoID','file']
# one frame element per row v2
COLUMNS2 = ['frameID', 'frameName', 'sentence', 'luName', 'luID', 'luText', 'luPOS', 'luIndex', 'feName', 'feID', 'feText', 'feIndex', 'sentenceID', 'annoID', 'file']
# gold dataset for roles
ROLE_COLUMNS = ['frameID', 'frameName', 'feName', 'feID', 'feText']
# for fullText xml v3
COLUMNS3 = ['frameID', 'frameName', 'sentence', 'luName', 'luID', 'luText', 'luPOS', 'luIndex', 'frameElements', 'file', 'corpID', 'sentNo', 'paragNo', 'docID']
## ---------------------------------------------------------------------------------------------------- These two methods will extract one frame element per row [Written for roles dataset at later stage]
def get_luText(row): # to get masked text
    """get text marked by luIndex"""
    indices = row['luIndex']
    sent = row['sentence']
    text = ''
    if indices is not None:
        for start, end in indices:
            if end > len(sent): continue
            if text == '': text = sent[int(start): int(end)+1]
            else: text = text + ' ' + sent[int(start): int(end)+1]
    
    return text


      
def extract_fulltextXML2(xml_file, verbose=False):
    
    """Return a dataframe, which contains extracted records from input xml file."""

    df = pd.DataFrame(columns=COLUMNS2)
    tree = et.parse(xml_file)
    root = tree.getroot()
    i = 0
    for child in root:
        text = None
        text_annotated = False
        if child.tag.endswith("sentence"):
            text = None
            sent_id = int(child.attrib["ID"])
            for gchild in child:
                
                lu_start = None
                lu_end = None
                luIndex = None
                lu_id = None
                
                if gchild.tag.endswith("text"):
                    text = gchild.text
                    if verbose: print("=" * 50, "\n", text)
    
                if gchild.tag.endswith("annotationSet") and "frameID" in gchild.attrib:

                    anno_id = gchild.attrib["ID"] 
                    i = i+1
                    lu_name, lu_pos = gchild.attrib["luName"].split('.')  
                    lu_id = gchild.attrib["luID"] 

                    frame_id = gchild.attrib["frameID"]
                    frame_name = gchild.attrib["frameName"]

                    layers = [layer for layer in gchild]
                    Target_layers = [layer for layer in layers if layer.attrib["name"]=="Target"]
                    FE_layers = [layer for layer in layers if layer.attrib["name"]=="FE"]
                    # lu
                    for layer in Target_layers:

                        for label in layer:
                            if label.tag.endswith("label") and "end" in label.attrib:
                                lu_start = int(label.attrib["start"])
                                lu_end = int(label.attrib["end"])
                                if luIndex is None: luIndex = [(lu_start, lu_end)]
                                else: luIndex.append((lu_start, lu_end))
                    # FE
                    for layer in FE_layers:

                        for label in layer:
                            if label.tag.endswith("label") and "end" in label.attrib:
                                fe_start = int(label.attrib["start"])
                                fe_end = int(label.attrib["end"])
                                fe_text = text[fe_start:fe_end + 1]
                                fe_name = label.attrib["name"]
                                fe_id = int(label.attrib["feID"])
                                df.loc[len(df)] = [frame_id, frame_name, text, lu_name, lu_id, '', lu_pos, luIndex, fe_name, fe_id, fe_text, [(fe_start, fe_end)], sent_id, anno_id, Path(xml_file).name]

                            elif label.tag.endswith("label") and "itype" in label.attrib:
                                itype = label.attrib["itype"]
                                fe_name = label.attrib["name"]
                                fe_id = int(label.attrib["feID"])
                                df.loc[len(df)] = [frame_id, frame_name, text, lu_name, lu_id, '', lu_pos, luIndex, fe_name, fe_id, None, [(itype)], sent_id, anno_id, Path(xml_file).name]

                        if list(layer) == []: # No frame elements annotations
                            df.loc[len(df)] = [frame_id, frame_name, text, lu_name, lu_id, '', lu_pos, luIndex, None, None, None, None, sent_id, anno_id, Path(xml_file).name]
                            
    
    # sort the indices
    if not df.empty:
        df['luIndex'].apply(lambda x: x.sort(key=lambda tup: tup[0]) if x is not None else x) 
        df['luText'] = df.apply(get_luText, axis=1)  

    return df, i


def extract_luXML2(xml_file, verbose=False):
    
    """Return a dataframe, which contains extracted records from input xml file."""
    
    df = pd.DataFrame(columns=COLUMNS2)
    tree = et.parse(xml_file)
    root = tree.getroot()
    
    lu_name, lu_pos = root.attrib["name"].split('.')
    lu_id = root.attrib["ID"]
    
    frame_id = root.attrib["frameID"]
    frame_name = root.attrib["frame"]
    i = 0
    for child in root:
        
        if child.tag.endswith("subCorpus"):
            
            for gchild in child:
                
                text = None
                if gchild.tag.endswith("sentence"):
                    sent_id = int(gchild.attrib["ID"])
                    for ggchild in gchild:
                        
                        if ggchild.tag.endswith("text"):
                            text = ggchild.text
                            if verbose: print("=" * 50, "\n", text)

                        if ggchild.tag.endswith("annotationSet") and "status" in ggchild.attrib:
                            
                            lu_start = None
                            lu_end = None
                            luIndex = None
                            
                            if ggchild.attrib["status"] != "UNANN":
                                anno_id = int(ggchild.attrib["ID"])
                                i = i+1
                                layers = [layer for layer in ggchild]
                                Target_layers = [layer for layer in layers if layer.attrib["name"]=="Target"]
                                FE_layers = [layer for layer in layers if layer.attrib["name"]=="FE"]
                                # lu
                                for layer in Target_layers:
                                        
                                    for label in layer:
                                        if label.tag.endswith("label") and "end" in label.attrib:
                                            lu_start = int(label.attrib["start"])
                                            lu_end = int(label.attrib["end"])
                                            if luIndex is None: luIndex = [(lu_start, lu_end)]
                                            else: luIndex.append((lu_start, lu_end))
                                # FE
                                for layer in FE_layers:
                                    
                                    for label in layer:
                                        if label.tag.endswith("label") and "end" in label.attrib:
                                            fe_start = int(label.attrib["start"])
                                            fe_end = int(label.attrib["end"])
                                            fe_text = text[fe_start:fe_end + 1]
                                            fe_name = label.attrib["name"]
                                            fe_id = int(label.attrib["feID"])
                                            df.loc[len(df)] = [frame_id, frame_name, text, lu_name, lu_id, '', lu_pos, luIndex, fe_name, fe_id, fe_text, [(fe_start, fe_end)], sent_id, anno_id, Path(xml_file).name]

                                        elif label.tag.endswith("label") and "itype" in label.attrib:
                                            itype = label.attrib["itype"]
                                            fe_name = label.attrib["name"]
                                            fe_id = int(label.attrib["feID"])
                                            df.loc[len(df)] = [frame_id, frame_name, text, lu_name, lu_id, '', lu_pos, luIndex, fe_name, fe_id, None, [(itype)], sent_id, anno_id, Path(xml_file).name]

                                    if list(layer) == []: # No frame elements annotations
                                        df.loc[len(df)] = [frame_id, frame_name, text, lu_name, lu_id, '', lu_pos, luIndex, None, None, None, None, sent_id, anno_id, Path(xml_file).name]
                                           
    if not df.empty:
        df['luIndex'].apply(lambda x: x.sort(key=lambda tup: tup[0]) if x is not None else x) 
        df['luText'] = df.apply(get_luText, axis=1)
              
    return df, i

## ---------------------------------------------------------------------------------------------------- These functions will extract all frame elements of the sentence in single list [first implementation]
def extract_fulltextXML(xml_file, verbose=False):
    
    """Return a dataframe, which contains extracted records from input xml file."""

    df = pd.DataFrame(columns=COLUMNS)
    tree = et.parse(xml_file)
    root = tree.getroot()
    i = 0;
    for child in root:
        
        text = None
        
        if child.tag.endswith("sentence"):
            sent_id = int(child.attrib["ID"])

            for gchild in child:

                FEs = []
                lu_start = None
                lu_end = None
                luIndex = None

                if gchild.tag.endswith("text"):
                    text = gchild.text
                    if verbose: print("=" * 50, "\n", text)

                elif gchild.tag.endswith("annotationSet") and "frameID" in gchild.attrib:                    
                    anno_id = int(gchild.attrib["ID"])
                    lu_name, lu_pos = gchild.attrib["luName"].split('.')  
                    lu_id = gchild.attrib["luID"] 
                    frame_id = gchild.attrib["frameID"]
                    frame_name = gchild.attrib["frameName"]


                    for ggchild in gchild:
                        if ggchild.tag.endswith("layer") and ggchild.attrib["name"] == "Target":
    
                            for label in ggchild:
                                if label.tag.endswith("label") and "end" in label.attrib:
                                    lu_start = int(label.attrib["start"])
                                    lu_end = int(label.attrib["end"])
                                    if luIndex is None: luIndex = [(lu_start, lu_end)]
                                    else: luIndex.append((lu_start, lu_end))
                        if ggchild.tag.endswith("layer") and ggchild.attrib["name"] == "FE":

                            for label in ggchild:
                                if label.tag.endswith("label") and "end" in label.attrib:
                                    fe_start = int(label.attrib["start"])
                                    fe_end = int(label.attrib["end"])
                                    fe_text = text[fe_start:fe_end + 1]
                                    fe_name = label.attrib["name"]
                                    if FEs == []: FEs = [(fe_text, (fe_start, fe_end), fe_name)]
                                    else: FEs.append((fe_text, (fe_start, fe_end), fe_name))

                                elif label.tag.endswith("label") and "itype" in label.attrib:
                                    itype = label.attrib["itype"]
                                    fe_name = label.attrib["name"]
                                    if FEs == []: FEs = [("null", (itype), fe_name)]
                                    else: FEs.append(("null", (itype), fe_name))

                    df.loc[len(df)] = [frame_id, frame_name, text, lu_name, lu_id, '', lu_pos, luIndex, FEs,
                                       sent_id, anno_id, Path(xml_file).name]
                    i = i + 1

                else:
                    continue

    if not df.empty:
#         sort the indices
        df['luIndex'].apply(lambda x: x.sort(key=lambda tup: tup[0]) if x is not None else x) 
        df['luText'] = df.apply(get_luText, axis=1)
          
    return df, i


def extract_luXML(xml_file, verbose=False):
    """Return a dataframe, which contains extracted records from input xml file."""
    df = pd.DataFrame(columns=COLUMNS)
    tree = et.parse(xml_file)
    root = tree.getroot()
    i = 0;
    status = []
    lu_name, lu_pos = root.attrib["name"].split('.')
    lu_id = root.attrib["ID"]
    
    frame_id = root.attrib["frameID"]
    frame_name = root.attrib["frame"]
    for child in root:
        if child.tag.endswith("subCorpus"):
            for gchild in child:

                text = None
                if gchild.tag.endswith("sentence"):
                    sent_id = int(gchild.attrib["ID"])
                    
                    for ggchild in gchild:

                        if ggchild.tag.endswith("text"):
                            text = ggchild.text
                            if verbose: print("=" * 50, "\n", text)

                        if ggchild.tag.endswith("annotationSet") and "status" in ggchild.attrib:

                            FEs = []
                            lu_start = None
                            lu_end = None
                            luIndex = None
                            
                            if ggchild.attrib["status"] != "UNANN":
                                anno_id = int(ggchild.attrib["ID"])
                                for layer in ggchild:

                                    if layer.tag.endswith("layer") and layer.attrib["name"] == "Target":
                                        for label in layer:
                                            if label.tag.endswith("label") and "end" in label.attrib:
                                                lu_start = int(label.attrib["start"])
                                                lu_end = int(label.attrib["end"])
                                                if luIndex is None: luIndex = [(lu_start, lu_end)]
                                                else: luIndex.append((lu_start, lu_end))

                                    if layer.tag.endswith("layer") and layer.attrib["name"] == "FE":
                                        for label in layer:

                                            if label.tag.endswith("label") and "end" in label.attrib:
                                                fe_start = int(label.attrib["start"])
                                                fe_end = int(label.attrib["end"])
                                                fe_text = text[fe_start:fe_end + 1]
                                                fe_name = label.attrib["name"]
                                                if FEs == []: FEs=[(fe_text, (fe_start, fe_end), fe_name)]
                                                else: FEs.append((fe_text, (fe_start, fe_end), fe_name))

                                            elif label.tag.endswith("label") and "itype" in label.attrib:
                                                itype = label.attrib["itype"]
                                                fe_name = label.attrib["name"]
                                                if FEs == []: FEs=[("null", (itype), fe_name)]
                                                else: FEs.append(("null", (itype), fe_name))

                                df.loc[len(df)] = [frame_id, frame_name, text, lu_name, lu_id, '', lu_pos, luIndex, FEs, sent_id, anno_id, Path(xml_file).name]
                                i = i + 1
    if not df.empty:
        # sort the indices
        df['luIndex'].apply(lambda x: x.sort(key=lambda tup: tup[0]) if x is not None else x) 
        df['luText'] = df.apply(get_luText, axis=1)

    return df, i

## ---------------------------------------------------------------------------------------------------- extract just roles data for gold annotations
def extract_roles_fulltextXML(xml_file, verbose=False):
    
    """Return a dataframe, which contains extracted records from input xml file."""

    df = pd.DataFrame(columns=ROLE_COLUMNS)
    tree = et.parse(xml_file)
    root = tree.getroot()
    FEs = {}
    for child in root:
        
        if child.tag.endswith("sentence"):

            for gchild in child:
                
                if gchild.tag.endswith("text"):
                    text = gchild.text
                    if verbose: print("=" * 50, "\n", text)

                if gchild.tag.endswith("annotationSet") and "frameID" in gchild.attrib:                    
                    
                    frame_id = gchild.attrib["frameID"]
                    frame_name = gchild.attrib["frameName"]


                    for ggchild in gchild:
                        
                        if ggchild.tag.endswith("layer") and ggchild.attrib["name"] == "FE":

                            for label in ggchild:
                                if label.tag.endswith("label") and "end" in label.attrib:
                                    fe_start = int(label.attrib["start"])
                                    fe_end = int(label.attrib["end"])
                                    fe_text = text[fe_start:fe_end + 1]
                                    fe_name = label.attrib["name"]
                                    
                                    df.loc[len(df)] = [frame_id, frame_name, fe_name, fe_text, text]

                                    
                                elif label.tag.endswith("label") and "itype" in label.attrib:
                                    itype = label.attrib["itype"]
                                    fe_name = label.attrib["name"]
                                    
                                    df.loc[len(df)] = [frame_id, frame_name, fe_name, '', text]


                else:
                    continue


    return df


def extract_roles_luXML(xml_file, verbose=False):
    """Return a dataframe, which contains extracted records from input xml file."""
    df = pd.DataFrame(columns=ROLE_COLUMNS)
    tree = et.parse(xml_file)
    root = tree.getroot()
    FEs = {}

    
    frame_id = root.attrib["frameID"]
    frame_name = root.attrib["frame"]
    for child in root:
        if child.tag.endswith("subCorpus"):
            for gchild in child:

                if gchild.tag.endswith("sentence"):

                    for ggchild in gchild:
                        
                        if ggchild.tag.endswith("text"):
                            text = ggchild.text
                            if verbose: print("=" * 50, "\n", text)

                        if ggchild.tag.endswith("annotationSet") and "status" in ggchild.attrib:
                            
                            if ggchild.attrib["status"] != "UNANN":
                                for layer in ggchild:

                                    if layer.tag.endswith("layer") and layer.attrib["name"] == "FE":
                                        for label in layer:

                                            if label.tag.endswith("label") and "end" in label.attrib:
                                                fe_start = int(label.attrib["start"])
                                                fe_end = int(label.attrib["end"])
                                                fe_text = text[fe_start:fe_end + 1]
                                                fe_name = label.attrib["name"]
                                                
                                                df.loc[len(df)] = [frame_id, frame_name, fe_name, fe_text, text]

                                            elif label.tag.endswith("label") and "itype" in label.attrib:
                                                itype = label.attrib["itype"]
                                                fe_name = label.attrib["name"]
                                                
                                                df.loc[len(df)] = [frame_id, frame_name, fe_name, '', text]


       
    return df


## ---------------------------------------------------------------------------------------------------- implementation to extract context for short sentences
def extract_fulltextXML3(xml_file, verbose=False):
    
    """also extract corpID, sentNo, paragNo, docID"""


    df = pd.DataFrame(columns=COLUMNS3)
    tree = et.parse(xml_file)
    root = tree.getroot()
    i = 0;
    for child in root:
        
        text = None

        
        if child.tag.endswith("sentence"):
            
            corpID = int(child.attrib['corpID'])
            sentNo = int(child.attrib['sentNo'])
            paragNo = int(child.attrib['paragNo'])
            docID = int(child.attrib['docID'])
            
            for gchild in child:

                FEs = []
                lu_start = None
                lu_end = None
                luIndex = None

                if gchild.tag.endswith("text"):
                    text = gchild.text
                    if verbose: print("=" * 50, "\n", text)

                elif gchild.tag.endswith("annotationSet") and "frameID" in gchild.attrib:                    
                    lu_name, lu_pos = gchild.attrib["luName"].split('.')  
                    lu_id = gchild.attrib["luID"] 
                    frame_id = gchild.attrib["frameID"]
                    frame_name = gchild.attrib["frameName"]


                    for ggchild in gchild:
                        if ggchild.tag.endswith("layer") and ggchild.attrib["name"] == "Target":
    
                            for label in ggchild:
                                if label.tag.endswith("label") and "end" in label.attrib:
                                    lu_start = int(label.attrib["start"])
                                    lu_end = int(label.attrib["end"])
                                    if luIndex is None: luIndex = [(lu_start, lu_end)]
                                    else: luIndex.append((lu_start, lu_end))
                        if ggchild.tag.endswith("layer") and ggchild.attrib["name"] == "FE":

                            for label in ggchild:
                                if label.tag.endswith("label") and "end" in label.attrib:
                                    fe_start = int(label.attrib["start"])
                                    fe_end = int(label.attrib["end"])
                                    fe_text = text[fe_start:fe_end + 1]
                                    fe_name = label.attrib["name"]
                                    if FEs == []: FEs=[(fe_text, (fe_start, fe_end), fe_name)]
                                    else: FEs.append((fe_text, (fe_start, fe_end), fe_name))

                                elif label.tag.endswith("label") and "itype" in label.attrib:
                                    itype = label.attrib["itype"]
                                    fe_name = label.attrib["name"]
                                    if FEs == []: FEs=[("null", (itype), fe_name)]
                                    else: FEs.append(("null", (itype), fe_name))

                    df.loc[len(df)] = [frame_id, frame_name, text, lu_name, lu_id, '', lu_pos, luIndex, FEs, Path(xml_file).name
                                       ,corpID, sentNo, paragNo, docID]
                    i = i + 1

                else:
                    continue

    if not df.empty:
        # sort the indices
        df['luIndex'].apply(lambda x: x.sort(key=lambda tup: tup[0]) if x is not None else x) 
        df['luText'] = df.apply(get_luText, axis=1)
          
    return df, i

## ================================================================================================    
def extract_fulltext3(output_dir=OUTPUT_DIR, verbose=False):
    
    """also extract corpID, sentNo, paragNo, docID"""

    files = 0
    records = 0
    df = pd.DataFrame(columns = COLUMNS3)
    perFile_records = pd.DataFrame(columns = ['File', 'Extracted_Records'])
    xml_dir = '{}/fulltext'.format(INPUT_DIR)
    for xml_file in glob(xml_dir+'/*.xml'):
        files = files+1
        df11, sents = extract_fulltextXML3(xml_file)
        perFile_records.loc[len(perFile_records)] = [xml_file, sents]
        df = df.append(df11, sort=False, ignore_index=True)
        records = records + sents
        if verbose: print(xml_file.split("/")[-1], sents)
    
    print("# of fulltext records v3 = {}".format(len(df)))
    
    if output_dir != None:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        df.to_csv(output_dir+'/fulltext_v3.csv', index=False)
        perFile_records.to_csv(output_dir+'/fulltext_v3_perFile_records.csv', index=False)
    
    return df


def extract_fulltext(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, version=1, verbose=False):
    
    """Extract all records from the fulltext dir and also denotes records perFile and write them into a pickle and csv file"""

    extractor_funcs = {
        1:(extract_fulltextXML, COLUMNS),
        2:(extract_fulltextXML2, COLUMNS2)
    }
    ft_extractor_func, columns = extractor_funcs[version]
   
    files = 0
    records = 0
    df = pd.DataFrame(columns = columns)
    perFile_records = pd.DataFrame(columns = ['File', 'Extracted_Records'])
    xml_dir = '{}/fulltext'.format(input_dir)
    for xml_file in glob(xml_dir+'/*.xml'):
        files = files+1
        df1, sents = ft_extractor_func(xml_file)
        perFile_records.loc[len(perFile_records)] = [xml_file, sents]
        df = df.append(df1, sort=False, ignore_index=True)
        records = records + sents
        if verbose: print(xml_file.split("/")[-1], sents)
    
    print("# of fulltext records = {}".format(len(df)))
    
    if output_dir != None:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        
        df.to_csv('{}/fulltext_v{}.csv'.format(output_dir, version), index=False)
        df.to_pickle('{}/fulltext_v{}.pkl'.format(output_dir, version))

        perFile_records.to_csv('{}/fulltext_v{}_perFile_records.csv'.format(output_dir, version), index=False)

    return df


def extract_lu(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, version=1, verbose=False):   
   
    """Extract all records from the lu dir and also denotes records perFile and write them into a csv file."""
    
    extractor_funcs = {
        1:(extract_luXML, COLUMNS),
        2:(extract_luXML2, COLUMNS2)
    }
    lu_extractor_func, columns = extractor_funcs[version]
        
    files = 0
    records = 0
    df = pd.DataFrame(columns = columns)
    perFile_records = pd.DataFrame(columns = ['File', 'Extracted_Records'])
    xml_dir = '{}/lu'.format(input_dir)
    for xml_file in glob(xml_dir+'/*.xml'):
        files = files + 1
        df1, sents = lu_extractor_func(xml_file)
        perFile_records.loc[len(perFile_records)] = [xml_file, sents]
        df = df.append(df1, sort=False, ignore_index=True)
        records = records + sents
        if verbose: print(xml_file.split("/")[-1], sents)
            
    
    print("# of lu records = {}".format(len(df)))
    if output_dir != None:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
            
        df.to_csv('{}/lu_v{}.csv'.format(output_dir, version), index=False)
        df.to_pickle('{}/lu_v{}.pkl'.format(output_dir, version))

        perFile_records.to_csv('{}/lu_v{}_perFile_records.csv'.format(output_dir, version), index=False)

            
    return df
## -------------------------------------------------------------------------------- gold dataset for roles
def extract_roles(input_dir= INPUT_DIR, output_dir=OUTPUT_DIR ):
    
    df = pd.DataFrame(columns = ROLE_COLUMNS)
    xml_dir = '{}/fulltext'.format(input_dir)
    for xml_file in glob(xml_dir+'/*.xml'):
        df1 = extract_roles_fulltextXML(xml_file)

        df = df.append(df1, sort=False, ignore_index=True)

    xml_dir = '{}/lu'.format(input_dir)
    for xml_file in glob(xml_dir+'/*.xml'):
        df1 = extract_roles_luXML(xml_file)

        df = df.append(df1, sort=False, ignore_index=True)

    
    print('# of roles extracted in total:', len(df))
    # drop records where feText is null or empty
    df2 = df.loc[df['feText']!='']
    df2 = df2.loc[df2['feText']!='null']
    print('# of roles extracted after dropping null:', len(df2))

    df3 = df2.drop_duplicates(['frameID', 'frameName', 'feName', 'feText'])
    print('# of roles after dropping duplicates:', len(df3))

    df3.to_csv('{}/fn_roles.csv'.format(output_dir), index=False)
    
    return df3


def extract_all_data(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, version=1, verbose=False): # 2nd implementation
    """extract and write both fulltext and lu using version1"""
    
    print("=" * 50,"\n extracting fulltext dir \n" ,"=" * 50)
    dfft = extract_fulltext(input_dir,  output_dir, version, verbose)
    print("=" * 50,"\n extracting lu dir \n" ,"=" * 50)
    dflu = extract_lu(input_dir,  output_dir, version, verbose)
    
    dfa = dfft.append(dflu, sort=False, ignore_index=True)

    dfa.to_csv('{}/fn_data_v{}.csv'.format(output_dir, version), index=False)
    dfa.to_pickle('{}/fn_data_v{}.pkl'.format(output_dir, version))
    print('# of records extracted in total:', len(dfa))

    return dfft, dflu, dfa    



def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extracts frames, sentences, lexical units and frame elements from the framenet.')
    parser.add_argument('-i', '--input_dir', help='Directory with the framenet files.')
    parser.add_argument('-o', '--output_dir', help='Output directory.')
    parser.add_argument('-v', '--version', choices = [1,2], default = 2, help='which version to call, 1 will extract all frame elements in one list, 2 will extract one frame element per line.')

    args = parser.parse_args()
    print("Input: ", args.input_dir)
    print("Output: ", args.output_dir)
    print("Version: ", args.version)
    
    extract_all_data(args.input_dir, args.output_dir, args.version)
    

if __name__ == '__main__':
    main()