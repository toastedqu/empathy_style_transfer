from models.gpt4api import GPT4Generator
from tqdm import tqdm
import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sent_path', default='data/ec_low_mean_emp.csv')
    parser.add_argument('--output_path', default="data/ec_low_mean_emp_gen_da.csv")
    parser.add_argument('--style', default='empathetic')
    parser.add_argument('--zero', default='False')
    parser.add_argument('--pairwise', default='False')
    parser.add_argument('--use_da', default='True')
    parser.add_argument('--explicit', default='False')
    args = parser.parse_args()

    df = pd.read_csv(args.sent_path)
    l_uttr = df['utterance'].tolist()
    l_da = df['da'].tolist()
    df_new = df[['conv_id','turn_id','speaker']]
    gpt = GPT4Generator(
        style=args.style,
        pairwise=eval(args.pairwise),
        use_da=eval(args.use_da),
        explicit=eval(args.explicit),
        zero=eval(args.zero)
    )
    temp = []
    for i in tqdm(range(len(l_uttr))):
        if eval(args.use_da):
            temp.append(gpt(l_uttr[i], l_da[i]))
        else:
            temp.append(gpt(l_uttr[i]))
    df_new['utterance'] = temp
    df_new.to_csv(args.output_path, index=False, encoding='utf-8')
    with open('temp.txt','w') as f:
        for line in temp:
            f.write(line+'\n')
    print("DONE!")