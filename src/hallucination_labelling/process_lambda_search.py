import subprocess
import generate_lambdas
import argparse
import align_data
import entropy_stats
import numpy as np
from entropy_stats import get_hall, get_first_hall

def process_server(log_dir, lambdas):
    split_files = "csplit -f " + log_dir + "xx " + log_dir + "lambda_search.log '/fairseq.tasks.text_to_speech/' '{*}'"
    create_batch_log = "mv " + log_dir + "xx00 " + log_dir + "batch.log"
    delete_lambda_search = "rm " + log_dir + "lambda_search.log"

    subprocess.call(split_files, shell=True)
    subprocess.call(create_batch_log, shell=True)

    delete_target = "rm " + log_dir + "target.detok"
    delete_hypothesis = "rm " + log_dir + "hypothesis.detok"

    for i, lamb in enumerate(lambdas):
        filename = log_dir + "%0.4E.log" % lamb
        create_lamb_log = "mv " + log_dir + "xx%02d " % (i+1) + filename
        subprocess.call(create_lamb_log, shell=True)

    subprocess.call(delete_lambda_search, shell=True)

def convert(raw_data, datatype):
    data = []
    for line in raw_data:
        l = list(map(datatype, line))
        data.append(l)
    return data

def process_local_ref(log_dir, lambdas):
    label_filename = "data/xsum-hallucination/test.label"
    labels = entropy_stats.read_lines(label_filename, int)
    for lamb in lambdas:
        filename = log_dir + "%0.4E.log" % lamb
        save_file = "avg_probs_%0.4E" % lamb
        probs = align_data.read_lines(filename, "P-")
        probs = convert(probs, float)
        hall_probs, non_hall_probs = get_hall(labels, probs)
        first_hall_probs = get_first_hall(labels, probs)
        probs = [item for sublist in probs for item in sublist]

        with open(log_dir + save_file, 'w') as f:
            print(np.mean(probs), np.mean(hall_probs), np.mean(first_hall_probs), file=f)

def process_local_gen(log_dir, lambdas):
    delete_target = "rm " + log_dir + "target.detok"
    delete_hypothesis = "rm " + log_dir + "hypothesis.detok"

    for lamb in lambdas:
        filename = log_dir + "%0.4E.log" % lamb
        save_file = "score_%0.4E" % lamb
        subprocess.run(["src/compute_rouge.sh", log_dir, filename, save_file])
        subprocess.call(delete_target, shell=True)
        subprocess.call(delete_hypothesis, shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('summarization_model', type=int,
                        help='Summarization model number')

    parser.add_argument('--lang_full', action='store_true',
                        help='To use the full language model')

    parser.add_argument('--score_reference', action='store_true',
                        help='To score reference sentences')

    parser.add_argument('--local', action='store_true',
                        help='If running locally, not on the server')

    args = parser.parse_args()

    src_dir = "src/"
    log_dir = "logs/hallucination_labelling/"
    lambdas = generate_lambdas.main()

    experiment_name = "%i" % args.summarization_model

    if args.score_reference:
        experiment_name += "_ref"

    if args.lang_full:
        experiment_name += "_full"

    log_dir += experiment_name + "/"

    if args.local:
        if args.score_reference:
            process_local_ref(log_dir, lambdas)
        else:
            process_local_gen(log_dir, lambdas)
    else:
        process_server(log_dir, lambdas)
