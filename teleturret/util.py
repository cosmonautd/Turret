import re
import string
import unicodedata

def normalize(word):
    """ Remove special characters
    """
    return "".join(c for c in unicodedata.normalize('NFKD', word) if c in string.ascii_letters).lower()

def replacement_func(match, repl_pattern):
    """ Lambda function for case input-insensitive/output-sensitive
        https://stackoverflow.com/questions/9208786/best-way-to-do-a-case-insensitive-replace-but-match-the-case-of-the-word-to-be-r
    """
    match_str = match.group(0)
    repl = ''.join([r_char if m_char.islower() else r_char.upper() for r_char, m_char in zip(repl_pattern, match_str)])
    repl += repl_pattern[len(match_str):]
    return repl

def full_replace(sentence, match, replace_pattern):
    """ Replace using above lambda function
    """
    return re.sub(match, lambda m: replacement_func(m, replace_pattern), sentence, flags=re.I)

def remove_contractions(sentence):
    """ Replace common contractions with their long versions
    """
    full_form = {
        "who's"     : "who is",
        "what's"    : "what is",
        "you're"    : "you are",
        "gonna"     : "going to"
    }
    for c in full_form.keys():
        sentence = full_replace(sentence, c, full_form[c])
    return sentence

def remove_punctuation(sentence):
    return sentence.translate(str.maketrans('','',string.punctuation))

def untokenize(tokens):
    output = str()
    for token in tokens:
        if token in list(string.punctuation): output += token
        else: output += ' ' + token
    return output.strip()

def preserve_entity_annotations(tokens):
    output = list()
    i = 0
    while i < len(tokens):
        if tokens[i] == '#' and i+2 < len(tokens) and tokens[i+2] == '#':
            output.append('#' + tokens[i+1] + '#')
            i += 3
        else:
            output.append(tokens[i])
            i += 1
    return output