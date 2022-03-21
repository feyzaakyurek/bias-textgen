
import pandas as pd
import os
import re
import string
from scipy import stats
from config import parse_options
import ipdb

API_KEY = [el for el in open("perspective_key", 'r')][0]

def clean_data(gen_df):
    print("Running cleaning...")
    # remove extra spaces and nonascii
    def clean_line(row):
        rr = re.sub("\s{1,}", " ", row['Generation'])
        rr = ''.join(filter(lambda x: x in string.printable, rr))
        return rr
    gen_df['Generation'] = gen_df.apply(lambda x: clean_line(x), axis=1)
    return gen_df


def compute_hf_sentiment(gen_df):
    print("Running HF classifier...")
    from transformers import pipeline
    classifier = pipeline("sentiment-analysis", device=0)
    name = "Generation" if "Generation" in gen_df.columns else "Text"
    def compute_hf_sentiment_(row):
        result = classifier(row[name])[0]
        if result['label'] == 'NEGATIVE':
            return -1 * result['score']
        else:
            return result['score']

    gen_df['hf_sentiment'] = gen_df.apply(lambda x: compute_hf_sentiment_(x), axis=1)
    return gen_df


def compute_vader(gen_df):
    print("Running VADER...")
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    name = "Generation" if "Generation" in gen_df.columns else "Text"
    vader_sentiments = gen_df.apply(lambda x: an.polarity_scores(x[name]), axis=1)
    temp_vader = pd.DataFrame(vader_sentiments.tolist())
    gen_df = pd.concat([gen_df, temp_vader], axis=1)
    return gen_df
    

def compute_tox_detoxify(gen_df):
    print("Running detoxify...")
    from detoxify import Detoxify
    model = Detoxify('original', device='cuda')
    name = "Generation" if "Generation" in gen_df.columns else "Text"
    toxicities = gen_df.apply(lambda x: model.predict(x[name]), axis=1)
    temp_toxicities = pd.DataFrame(toxicities.tolist())
    gen_df = pd.concat([gen_df, temp_toxicities], axis=1)
    return gen_df


def compute_regard(gen_df, opt):
    print("Running regard...")
    from nlgbias.scripts.run_classifier import regard
    # Set filename for regard, and compute regard
    tst = pd.read_csv(opt.test_file)
    _, tail = os.path.split(opt.test_file)
    tail = tail.strip(".csv")
    newpth = os.path.join(f"nlgbias/data/regard", opt.prompt_domain + "_" + tail + ".tsv")
    tst['Generation'].to_csv(newpth)

    opt.test_file = opt.prompt_domain + "_" + tail + ".tsv"
    preds = regard(opt)[1:]
    gen_df['regard'] = preds
    return gen_df
    
    # Now read predictions back
    # predspth = os.path.join(opt.output_dir, tail+"_predictions.txt")
    # preds = pd.read_csv(predspth, sep='\t', header=None)
    # gen_df['regard'] = preds[0]


def compute_tox_perspective(gen_df):
    client = discovery.build(
      "commentanalyzer",
      "v1alpha1",
      developerKey=API_KEY,
      discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
      static_discovery=False,
    )
    
    def compute_single(text):
    
        analyze_request = {
          'comment': {'text': text} ,
          'requestedAttributes': {'TOXICITY': {}}
        }

        response = client.comments().analyze(body=analyze_request).execute()
        return response['attributeScores']['TOXICITY']['summaryScore']['value']
    
    name = "Generation" if "Generation" in gen_df.columns else "Text"
    gen_df['perspective_api'] = gen_df.apply(lambda x: compute_single(x[name]), axis=1)
    return gen_df
        

def read_file(file):
    if file.endswith(".csv"):
        df = pd.read_csv(file)
    elif file.endswith(".txt"):
        with open(file, 'r') as f:
            lines = [line.rstrip() for line in f]
            df = pd.DataFrame({"Text": lines})
    else:
        raise ValueError()
    print("Finished reading the file.")
    return df
    

if __name__ == "__main__":

    opt = parse_options()
    
    domain = opt.prompt_domain
    cat = opt.category
    cat_var = "" if cat is None else f"{opt.category}_"
    
    gen_df = read_file(opt.test_file) #f"outputs/generations/{opt.prompt_set}_{domain}_{cat}_nosampling_50000_50/gens.csv")

    gen_df = clean_data(gen_df)
    if opt.regard:
        gen_df = compute_regard(gen_df, opt)
    else:
        gen_df = compute_vader(gen_df)
        gen_df = compute_tox_detoxify(gen_df)
        # gen_df = compute_hf_sentiment(gen_df)
        # gen_df = compute_tox_pespective(gen_df)
    
    if opt.summarize:
        mn = gen_df.groupby("Group").mean()
        pvals = []
        for col in mn.columns:
            print(col)
            val = stats.ttest_ind(gen_df.loc[gen_df['Group'] == "American_actors", col],
                                  gen_df.loc[gen_df['Group'] == "American_actresses", col]).pvalue
            print("pvalue: ", val)
            pvals.append(val)
        mn.loc["pvalue"] = pvals
        
        mn.to_csv(os.path.join(opt.save_path, f"{cat_var}sent_tox_summ.csv"), index=False)
        
    gen_df.to_csv(os.path.join(opt.save_path, f"{cat_var}sent_tox.csv"), index=False)
    
    
#     gen_df.to_csv(f"outputs/generations/{opt.prompt_set}_{domain}_{cat}_nosampling_50000_50/{cat}_sent_tox.csv")
#     mn.to_csv(f"outputs/generations/{opt.prompt_set}_{domain}_{cat}_nosampling_50000_50/{cat}_sent_tox_summ.csv")
        
        
