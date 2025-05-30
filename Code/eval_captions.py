from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json

# -------------------------
# 📁 File Paths
# -------------------------
ref_path = "/home/pdas4/vitcap_r/data/MSCOCO/annotations/captions_val2014.json"
hyp_path = "/home/pdas4/vitcap_r/Code/generated_captions.json"

# -------------------------
# 📥 Load
# -------------------------
print("📚 Loading ground truth annotations...")
coco = COCO(ref_path)

print("📄 Loading predictions...")
with open(hyp_path, "r") as f:
    results = json.load(f)

pred_ids = [entry["image_id"] for entry in results]
valid_ids = list(set(pred_ids) & set(coco.getImgIds()))

coco_res = coco.loadRes(results)

# -------------------------
# 🎯 Evaluate
# -------------------------
coco_eval = COCOEvalCap(coco, coco_res)
coco_eval.params['image_id'] = valid_ids
coco_eval.evaluate()

# -------------------------
# 📊 Print Scores
# -------------------------
print("\n🎯 Evaluation Metrics:")
for metric, score in coco_eval.eval.items():
    print(f"{metric}: {score:.4f}")
