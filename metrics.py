
from nlg_eval_via_simi_measures.bary_score import BaryScoreMetric
from nlg_eval_via_simi_measures.depth_score import DepthScoreMetric
from nlg_eval_via_simi_measures.infolm import InfoLM
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from bert_score import score as BERTScore
from rouge import Rouge


def BaryScore(reference, candidate):
    metric = BaryScoreMetric()
    metric.prepare_idfs(reference, candidate)
    score = metric.evaluate_batch(reference, candidate)
    return score

def DepthScore(reference, candidate):
    metric = DepthScoreMetric()
    metric.prepare_idfs(reference, candidate)
    score = metric.evaluate_batch(candidate,reference )["depth_score"][0]
    return score

def InfoLM(reference, candidate):
    metric = InfoLM(reference, candidate)
    metric.prepare_idfs(reference, candidate)
    score = metric.evaluate_batch(reference, candidate)
    return score

# Try different n-grams
def BLUE(reference, hypothesis):
    for n in range(1, len(hypothesis.split())+1):
        weights = tuple(1/n for i in range(n))
        score = sentence_bleu([reference], hypothesis.split(), weights=weights)
        # Stop iterating if blue score is greater than 0.01
        if score > 0.01:
            break
    return score

def METEOR(reference, candidate):
    # tokenized reference and candidate
    reference_tokens = [reference.split()]
    story_tokens = candidate.split()
    score = meteor_score(reference_tokens, story_tokens)
    return score    

def ROUGE(reference, candidate):
    rouge = Rouge()
    score = rouge.get_scores(candidate, reference)
    return score[0]['rouge-1']['r']

def Rouge_L(pred, target):
    rouge = Rouge()
    scores = rouge.get_scores(pred, target, avg=True, ignore_empty=True)
    return scores[0]['rouge-l']['r']

def BertScore(reference, candidate):
    score = BERTScore(reference, candidate, lang='en', verbose=True)
    return score

def WER(ref, hyp ,debug=True):
    r = ref.split()
    h = hyp.split()
    #costs will holds the costs, like in the Levenshtein distance algorithm
    costs = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
    # backtrace will hold the operations we've done.
    # so we could later backtrace, like the WER algorithm requires us to.
    backtrace = [[0 for inner in range(len(h)+1)] for outer in range(len(r)+1)]
 
    OP_OK = 0
    OP_SUB = 1
    OP_INS = 2
    OP_DEL = 3
    DEL_PENALTY = 1
    INS_PENALTY = 1
    SUB_PENALTY = 1
    
    # First column represents the case where we achieve zero
    # hypothesis words by deleting all reference words.
    for i in range(1, len(r)+1):
        costs[i][0] = DEL_PENALTY*i
        backtrace[i][0] = OP_DEL
    
    # First row represents the case where we achieve the hypothesis
    # by inserting all hypothesis words into a zero-length reference.
    for j in range(1, len(h) + 1):
        costs[0][j] = INS_PENALTY * j
        backtrace[0][j] = OP_INS
    
    # computation
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                costs[i][j] = costs[i-1][j-1] 
                backtrace[i][j] = OP_OK
            else:
                substitutionCost = costs[i-1][j-1] + SUB_PENALTY # penalty is always 1
                insertionCost    = costs[i][j-1] + INS_PENALTY   # penalty is always 1
                deletionCost     = costs[i-1][j] + DEL_PENALTY   # penalty is always 1
                 
                costs[i][j] = min(substitutionCost, insertionCost, deletionCost)
                if costs[i][j] == substitutionCost:
                    backtrace[i][j] = OP_SUB
                elif costs[i][j] == insertionCost:
                    backtrace[i][j] = OP_INS
                else:
                    backtrace[i][j] = OP_DEL
                 
    # back trace though the best route:
    i = len(r)
    j = len(h)
    numSub = 0
    numDel = 0
    numIns = 0
    numCor = 0
    if debug:
        print("OP\tREF\tHYP")
        lines = []
    while i > 0 or j > 0:
        if backtrace[i][j] == OP_OK:
            numCor += 1
            i-=1
            j-=1
            if debug:
                lines.append("OK\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_SUB:
            numSub +=1
            i-=1
            j-=1
            if debug:
                lines.append("SUB\t" + r[i]+"\t"+h[j])
        elif backtrace[i][j] == OP_INS:
            numIns += 1
            j-=1
            if debug:
                lines.append("INS\t" + "****" + "\t" + h[j])
        elif backtrace[i][j] == OP_DEL:
            numDel += 1
            i-=1
            if debug:
                lines.append("DEL\t" + r[i]+"\t"+"****")
    if debug:
        lines = reversed(lines)
        for line in lines:
            print(line)
        print("#cor " + str(numCor))
        print("#sub " + str(numSub))
        print("#del " + str(numDel))
        print("#ins " + str(numIns))
    # return (numSub + numDel + numIns) / (float) (len(r))
    wer_result = round( (numSub + numDel + numIns) / (float) (len(r)), 3)
    return {'WER':wer_result, 'numCor':numCor, 'numSub':numSub, 'numIns':numIns, 'numDel':numDel, "numCount": len(r)}