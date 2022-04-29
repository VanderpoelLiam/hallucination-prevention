import subprocess
import generate_lambdas
import argparse
import pyperclip as pc

parser = argparse.ArgumentParser()

parser.add_argument('summarization_model', type=int,
                    help='Summarization model number')

parser.add_argument('--lang_full', action='store_true',
                    help='To use the full language model')

parser.add_argument('--score_reference', action='store_true',
                    help='To score reference sentences')

args = parser.parse_args()

src_dir = "src/hallucination_labelling/"
log_dir = "logs/hallucination_labelling/"
lambdas = generate_lambdas.main()

params = []
experiment_name = "%i" % args.summarization_model

if args.score_reference:
    params.append("-s")
    experiment_name += "_ref"

params.append(str(args.summarization_model))

if args.lang_full:
    params.append("lang_full")
    experiment_name += "_full"
else:
    params.append("lang")

batch_cmd = "bsub -J lambda_search_" + experiment_name
log_dir += experiment_name + "/"
batch_cmd += " -o " + log_dir + "lambda_search.log"
batch_cmd += """ \
-W 1000 -n 4 -R "rusage[mem=2048]" \
-R "rusage[ngpus_excl_p=1]" \
"""

fairseq_cmd = "mkdir -p " + log_dir + "; "
for lamb in lambdas:
    fairseq_cmd += src_dir + "generate.sh " + " ".join(params + [str(lamb)]) + "; "

final_cmd = batch_cmd + '"' + fairseq_cmd + '"'
print("Done.")
pc.copy(final_cmd)
