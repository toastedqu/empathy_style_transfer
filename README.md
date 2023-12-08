# Empathy Style Transfer

This repo is used for the experiments conducted in the EMNLP 2023 Findings paper [Conditioning on Dialog Acts Improves Empathy Style Transfer](https://aclanthology.org/2023.findings-emnlp.884.pdf).

# Instruction

To fully replicate the entire experiment, please follow all steps of this instruction. To only reproduce the evaluation results, please follow Step 5.

1. Run ```pip install -r requirements.txt``` in a new virtual environment.
2. Clone the RoBERTa model bundle repository (https://github.com/rguan1/Empathic-Conversations-Auto-Eval/tree/main) to "models/". Rename the cloned folder "empathy". Replace "models/empathy/auto_eval_utterances.py" with "models/auto_eval_utterances.py".
    - (Optional) Download the original EC dataset (https://github.com/wwbp/empathic_reactions). The data has been processed through the following steps and are directly available in "data/ec_low_mean_emp.csv"
        1. Filter out 4 columns only: 
            - *conv_id*: conversation index
            - *turn_id*: turn index
            - *speaker*: speaker ID
            - *utterance*: utterance
        2. Run "auto_eval_utterances.py" from Step 1 on the full dataset by following the instruction on the Github repo.
        3. Use Pandas to calculate the average empathy score of each conversation (grouped by *conv_id*) and filter out 1016 samples (40 conversations) with the lowest average empathy scores. Output that as the data file.
3. Run ```python generate.py --output_path "YOUR_GENERATED_DATA_PATH" --zero True --pairwise False --use_da False --explicit False``` with your choices of arguments. The code should take around 1.5 hours to run, depending on the server status of GPT-4 from OpenAI.
    - *output_path*: the file path of the generated utterances for the specified prompting method. The following suffix mapping was used:
        - Zero-shot: '_zero'
        - Pairwise: '_pair'
        - Target: ''
        - DA-Pairwise: '_pair_da'
        - DA-Target: '_da'
        - DA-Pairwise (explicit): '_pair_da_exp'
        - DA-Target (explicit): '_da_exp'
    - *zero*: If true, use zero-shot prompting (set to false for all few-shot prompting methods)
    - *pairwise*: If true, use pairwise prompting; If false, use target prompting
    - *use_da*: If true, condition on dialog acts; If false, directly prompt
    - *explicit*: If true, explicitly specify dialog acts in prompt; If false, no specification
4. Run ```python evaluate.py --path "YOUR_GENERATED_DATA_PATH"``` with your choice of file path. The code should take around 20 minutes to run, depending on your GPU resources.
5. Open "analyze.ipynb". Run each cell manually and sequentially. Modify the file path to generate analysis results for different prompting methods. Do NOT click "Run All" because the "DA-Specific Analysis" section requires the results for all 7 prompting methods to be available in the "data" folder.

# Description

There are bunch of files already available in the "data" folder. Many of the "csv" files have suffixes that follow the rules of Step 3 in the instruction above, meaning that they have the same structure but were generated by different prompting methods. This section will explain what other parts of the naming mean:
- "ec_emp.csv": the original 11778 utterances from the EC dataset together with their empathy score and dialog act evaluation results.
- "ec_low_mean_emp.csv": the filtered 1016 utterances with the lowest conversation-level average empathy scores.
- "ec_low_mean_emp_gen": a prefix for utterances generated from GPT-4 with designated prompting method, together with their empathy score and dialog act evaluation results.
- "ec_low_mean_emp_diff": a prefix for results of difference between the original and generated utterances, mainly focusing on differences in empathy scores.
- "ec_low_mean_emp_full": a prefix for a concatenation of all results from the original file, the generated file, and the difference file. These files are the ones mainly used for analysis.

On top of that, we have 3 "json" files:
- "da_names.json": a mapping from the dialog act abbreviation to its full name.
- "emp_prompts.json": a set of dialog-act-specific few-shot examples of high empathy.
- "emp_prompts.json": a set of dialog-act-specific few-shot examples of low empathy, the counterparts of "emp_prompts.json".

# Notes

Please note that BLEURT-20 and DeBERTa-xlarge-mnli are huge models. Local GPUs may not be able to run them. Please release GPU memory before running those cells.
