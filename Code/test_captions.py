import json
from pycocotools.coco import COCO

# Load predictions
with open("/home/pdas4/vitcap_r/Code/generated_captions.json", "r") as f:
    generated = json.load(f)

# Load COCO references
coco = COCO("/home/pdas4/vitcap_r/data/MSCOCO/annotations/captions_val2014.json")

# Show 10 examples
print("🔍 Showing 10 sample captions:")
for i in range(10):
    item = generated[i]
    img_id = item["image_id"]
    gen_caption = item["caption"]

    # Load reference captions
    ann_ids = coco.getAnnIds(imgIds=img_id)
    refs = coco.loadAnns(ann_ids)
    ref_captions = [ref["caption"] for ref in refs]

    print(f"\n🖼️ Image ID: {img_id}")
    print(f"📝 Generated: {gen_caption}")
    print("📚 References:")
    for rc in ref_captions[:3]:  # Show top 3 references
        print(f"  - {rc}")
