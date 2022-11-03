import sys
import json

epoch, file = sys.argv[1:]

with open(file, "r") as f:
    res = json.load(f)

disamb_candidate_f1 = res["disamb_candidate_f1"] * 100
object_f1 = res["object_f1"] * 100
act_f1 = res["act_f1"] * 100
slot_f1 = res["slot_f1"] * 100
bleu = res["bleu"] * 100

print("%s\t%.1f\t%.1f\t%.1f\t%.1f\t%.1f" % (epoch, disamb_candidate_f1, object_f1, act_f1, slot_f1, bleu))