import sys
import os
from lexer.unicodeManager import UnicodeWriter
from lexer.utilities import *
from lexer.fileUtilities import *
from pygments.lexers import get_lexer_for_filename
from pygments import lex
from subprocess import call
import lexer.Android as Android
import re
import lexer.dictUtils as dictUtils
import pickle

METADATAFLAG = True
SKIP_BIG = False
EXPLICIT_TYPE_WRITE = False

#string, string, list of tokens, int, string --> -----
#Given the output directory, the original file type, the tokens to write, and the
#file id, write out the tokens to file
#Precondition: files must be less than MAX_SIZE tokens (large files seem to
#cause problems with Zhaopeng's cache model.
def writeLexedFile(outputFile, lexedWoComments, flag, explicitWrite, keepLines):
    assert(len(lexedWoComments) <= MAX_SIZE)
    with open(outputFile, "w") as f:
        curr_line_empty = True
        for t in lexedWoComments:
            token = t[1]
            token_stripped = token.strip()

            if '\n' in token:
                if curr_line_empty and t[0] != Token.Text and token_stripped != '':
                    f.write(token_stripped + "\n")
                else:
                    f.write(token_stripped + "\n")
                curr_line_empty = True
            elif t[0] == Token.Text:
                continue
            else:
                curr_line_empty = False
                f.write(token)
                f.write(' ')

def main(sourcePath, outputPath, strFlag, token_split, SKIP_BIG, EXPLICIT_TYPE_WRITE, keepLines = False):    
    #Count of Error tokens
    errorCount = 0

    if(token_split.lower() == "api"):
        #Load in internally defined functions
        corpusDefinitions = pickle.load(open(os.path.join(basePath, "definitions.pickle"), 'r'))

    components = sourcePath.split(".")
    fileContents = ""
    with open(sourcePath, 'r') as f:
        fileContents = ''.join(f.readlines())

    lexer = get_lexer_for_filename(sourcePath)
    tokens = lex(fileContents, lexer) # returns a generator of tuples
    tokensList = list(tokens)
    language = languageForLexer(lexer)

    # Strip comments and alter strings
    lexedWoComments = tokensExceptTokenType(tokensList, Token.Comment)
    lexedWoComments = tokensExceptTokenType(lexedWoComments, Token.Literal.String.Doc)
    lexedWoComments = fixTypes(lexedWoComments, language) #Alter the pygments lexer types to be more comparable between our languages
    lexedWoComments = convertNamespaceTokens(lexedWoComments, language)

    if(token_split.lower() == "full" or token_split.lower() == "labelled" or token_split.lower() == "android" or token_split.lower() == "api"):
        if(strFlag == 0):
            lexedWoComments = modifyStrings(lexedWoComments, underscoreString)
        elif(strFlag == 1):
            lexedWoComments = modifyStrings(lexedWoComments, singleStringToken)
        elif(strFlag == 2):
            lexedWoComments = modifyStrings(lexedWoComments, spaceString)
        elif(strFlag == 3):
            lexedWoComments = modifyStrings(lexedWoComments, singleStringToken)
            #print(lexedWoComments)
            lexedWoComments = collapseStrings(lexedWoComments)
            lexedWoComments = modifyNumbers(lexedWoComments, singleNumberToken)
        else:
            print("Not a valid string handling flag. Valid types are currently 0, 1, 2, and 3")
            quit()

        if(token_split.lower() == "android"):
            lexedWoComments = labelAndroidTypes(lexedWoComments)

    elif(token_split.lower() == "keyword"):            
        lexedWoComments = getKeywords(lexedWoComments, language.lower())
    elif(token_split.lower() == "name"):
        lexedWoComments = getNameTypes(lexedWoComments, language.lower())
    elif(token_split.lower() == "nonname"):
        lexedWoComments = getNonNameTypes(lexedWoComments)
    elif(token_split.lower() == "collapsed"):
        lexedWoComments = modifyStrings(lexedWoComments, singleStringToken)
        lexedWoComments = collapseStrings(lexedWoComments)
        lexedWoComments = modifyNumbers(lexedWoComments, singleNumberToken)
        lexedWoComments = modifyNames(lexedWoComments, singleNameToken)
    else:
        print("Not a valid token split.")
        
    #Remove empty files (all comments).
    if(len(lexedWoComments) == 0):
        print("Skipping: " + sourcePath)
        return False

    
    (lineCount, ave, lineDict, lineLengths) = getLineMetrics(lexedWoComments)
    #print(lexedTokens)
    noWSTokens = []
    for t in lexedWoComments:
        noWS = t[1].strip()
        noWS = noWS.replace('\n', '') #Remove new lines
        if(noWS == "" or noWS[0] == Token.Text):
            continue
        noWSTokens.append((t[0],noWS))

    if keepLines:
        writeLexedFile(outputPath, lexedWoComments, token_split, EXPLICIT_TYPE_WRITE, keepLines)
    else:
        writeLexedFile(outputPath, noWSTokens, token_split, EXPLICIT_TYPE_WRITE, keepLines)    
    
    return len(noWSTokens) > 0

if __name__ == "__main__":
    if len(sys.argv) < 7:
        print('Usage: python lex.py source_file output_file flag token_split skip_big explicit_type')
        print("Example: python simplePyLex.py ~/CodeNLP/HaskellProjects/ *.hs tests/ 0 full True False")
        print("Flag is 0, 1, 2, or 3 currently")
        print("0 -> replace all spaces in strings with _")
        print("1 -> replace all strings with a <str> tag.")
        print("2 -> add spaces to the ends of the strings")
        print("3 -> collapse strings to <str> and collapses numbers to a type as well.")
        print("token_split is full, keyword, name, labelled, nonname, android, api, or collapsed currently")
        print("labelled keeps all tokens, but attaches labels or name, keyword, or other")
        print("the android option is exclusively for android, and attaches a Android.* to")
        print("All android api references.  Other tokens retain the type from the highlighter.")
        print("Collapsed replaces all the name types (plus Keyword.Type) with their label.")
        print("Collapsed option also forces the string option to 1.")
        print("And finally two y/n options on whether to keep files with over " + str(MAX_SIZE))
        print("tokens and if we want to explicitly have the type <Token|Type> in the output files.")
        quit()
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])