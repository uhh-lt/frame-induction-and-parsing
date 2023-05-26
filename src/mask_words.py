

def get_indices(text, indices):
    """get start, end index of word to be masked"""
    # for single and multi-token but contigous chunk
    start, end = indices[0][0], indices[-1][1]

    oldstr = text[int(start): int(end)+1]

    """halfcarried --> half carried"""
    bfr = ''
    aft = ''
    if int(start)!=0:
        if text[int(start) -1] != ' ': bfr = ' '
    if len(text) != int(end)+1: 
        if text[int(end) +1] != ' ': aft = ' '    

    return oldstr, start, end, bfr, aft

        
    

def mask_lu_withPatterns(row):
     
    text = row['sentence']
    index = row['luIndex']
    
    oldstr, start, end, bfr, aft = get_indices(text, index)
    
    
    row['T'] = mask_swv_T(row)
    row['Tand-'] = mask_swv_Tand(row)
    row['-andT'] = mask_swv_andT(row)
    row['Tandthen-'] = mask_swv_Tandthen(row)
    row['-andthenT'] = mask_swv_andthenT(row)
    
    return row


def mask_role_withPatterns(row):
     
    text = row['sentence']
    index = row['feIndex']
    
    oldstr, start, end, bfr, aft = get_indices(text, index)
    
    row['T'] = mask_role_T(row)
    row['Tand-'] = mask_role_Tand(row)
    row['-andT'] = mask_role_andT(row)
    row['Tandthen-'] = mask_role_Tandthen(row)
    row['-andthenT'] = mask_role_andthenT(row)
   

    return row

# -----------------------------------------    
def mask_lu_T(row):
    """mask single word verb"""

    text = row['sentence']
    index = row['luIndex']
    
    oldstr, start, end, bfr, aft = get_indices(text, index)
    
    if ' ' in oldstr: # if multi-token replace the oldstr with -
        oldstr = '-'
        
    return '{}{}__{}__{}{}'.format(text[0:int(start)], bfr, oldstr, aft, text[int(end)+1:])
    
    
def mask_lu_Tand(row):
    """mask single word verb"""

    text = row['sentence']
    index = row['luIndex']
    
    oldstr, start, end, bfr, aft = get_indices(text, index)
    return '{}{}{} and __-__{}{}'.format(text[0:int(start)], bfr, oldstr, aft, text[int(end)+1:])

    
    
def mask_lu_andT(row):
    """mask single word verb"""

    text = row['sentence']
    index = row['luIndex']
    
    oldstr, start, end, bfr, aft = get_indices(text, index)
    return '{}{}__-__ and {}{}{}'.format(text[0:int(start)], bfr, oldstr, aft, text[int(end)+1:])

    
    
def mask_lu_Tandthen(row):
    """mask single word verb"""
    
    text = row['sentence']
    index = row['luIndex']
    
    oldstr, start, end, bfr, aft = get_indices(text, index)
    return '{}{}{} and then __-__{}{}'.format(text[0:int(start)], bfr, oldstr, aft, text[int(end)+1:])


    
def mask_lu_andthenT(row):
    """mask single word verb"""
    
    text = row['sentence']
    index = row['luIndex']
    
    oldstr, start, end, bfr, aft = get_indices(text, index)
    return'{}{}__-__ and then {}{}{}'.format(text[0:int(start)], bfr, oldstr, aft, text[int(end)+1:])

# ----------------------------------------------------------------------------    

def mask_role_T(row):
    """mask role"""
    
    text = row['sentence']
    index = row['feIndex']
    
    oldstr, start, end, bfr, aft = get_indices(text, index)
    
    if ' ' in oldstr: # if multi-token replace the oldstr with -
        oldstr = '-'
    
    return '{}{}__{}__{}{}'.format(text[0:int(start)], bfr, oldstr, aft, text[int(end)+1:])
    

def mask_role_Tand(row):
    """mask role"""
    
    text = row['sentence']
    index = row['feIndex']
    
    oldstr, start, end, bfr, aft = get_indices(text, index)
    return '{}{}{} and __-__{}{}'.format(text[0:int(start)], bfr, oldstr, aft, text[int(end)+1:])


def mask_role_andT(row):
    """mask role"""
    
    text = row['sentence']
    index = row['feIndex']
    
    oldstr, start, end, bfr, aft = get_indices(text, index)
    return '{}{}__-__ and {}{}{}'.format(text[0:int(start)], bfr, oldstr, aft, text[int(end)+1:])



def mask_role_Tandthen(row):
    """mask role"""
    
    text = row['sentence']
    index = row['feIndex']
    
    oldstr, start, end, bfr, aft = get_indices(text, index)
    return '{}{}{} and then __-__{}{}'.format(text[0:int(start)], bfr, oldstr, aft, text[int(end)+1:])


def mask_role_andthenT(row):
    """mask role"""
    text = row['sentence']
    index = row['feIndex']
    
    oldstr, start, end, bfr, aft = get_indices(text, index)
    return '{}{}__-__ and then {}{}{}'.format(text[0:int(start)], bfr, oldstr, aft, text[int(end)+1:])



# ---------------------------
    
def mask_lu_TandT(row):
    """mask single word verb"""

    text = row['sentence']
    index = row['luIndex']
    
    oldstr, start, end, bfr, aft = get_indices(text, index)
    return '{}{}{} and __{}__{}{}'.format(text[0:int(start)], bfr, oldstr, oldstr, aft, text[int(end)+1:])


    
    
def mask_lu_TandthenT(row):
    """mask single word verb"""
    
    text = row['sentence']
    index = row['luIndex']
    
    oldstr, start, end, bfr, aft = get_indices(text, index)
    return '{}{}{} and then __{}__{}{}'.format(text[0:int(start)], bfr, oldstr, oldstr, aft, text[int(end)+1:])


    
# ----------------------------------------------------------------------------     

def mask_role_TandT(row):
    """mask role"""
    
    text = row['sentence']
    index = row['feIndex']
    
    oldstr, start, end, bfr, aft = get_indices(text, index)
    return '{}{}{} and __{}__{}{}'.format(text[0:int(start)], bfr, oldstr, oldstr, aft, text[int(end)+1:])


def mask_role_TandthenT(row):
    """mask role"""
    
    text = row['sentence']
    index = row['feIndex']
    
    oldstr, start, end, bfr, aft = get_indices(text, index)
    return '{}{}{} and then __{}__{}{}'.format(text[0:int(start)], bfr, oldstr, oldstr, aft, text[int(end)+1:])

