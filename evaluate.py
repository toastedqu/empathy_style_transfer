import argparse
import pandas as pd
from models.empathy.auto_eval_utterances import Evaluator
from utils import *
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate(args, df):
    sents = df['utterance'].tolist()
    conv_ids = df['conv_id'].tolist()
    turn_ids = df['turn_id'].tolist()
    speakers = df['speaker'].tolist()
    sents = [sents[i:i+10] for i in range(0,len(df),10)]
    conv_ids = [conv_ids[i:i+10] for i in range(0,len(df),10)]
    turn_ids = [turn_ids[i:i+10] for i in range(0,len(df),10)]
    speakers = [speakers[i:i+10] for i in range(0,len(df),10)]
    df = pd.DataFrame(columns=['conv_id','turn_id','speaker','utterance','dialog_act','empathy','emotional_polarity'])
    evaluator = Evaluator()
    for i in range(len(sents)):
        df = evaluator.evaluate_utterances(sents[i],conv_ids[i],turn_ids[i],speakers[i],df)
        df.to_csv(args.path, index=False, encoding='utf-8')
        df = pd.read_csv(args.path)
    return df

def get_da(row):
    tup = eval(row['dialog_act'])
    return tup[0][:2] if tup else 'none'
    
def main(args):
    df = pd.read_csv(args.path, encoding='utf-8')
    df = evaluate(args, df)
    df['da'] = df.apply(lambda row: get_da(row), axis=1)
    df.to_csv(args.path, index=False, encoding='utf-8')
    print('DONE')
    
def da_only(args):
    df = pd.read_csv(args.path, encoding='utf-8')
    # df['dialog_act'] = pd.Series(evaluate_da(df['utterance'].tolist()))
    df['da'] = df.apply(lambda row: get_da(row), axis=1)
    df.to_csv(args.path, encoding="utf-8", index=False)
    print('DA ONLY DONE')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default="data/ec_low_mean_emp_gen_da.csv")
    parser.add_argument('--style', default="empathetic")
    parser.add_argument('--da_only', default='False')
    args = parser.parse_args()
    if eval(args.da_only):
        da_only(args)
    else:
        main(args)