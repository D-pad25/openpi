import tensorflow_datasets as tfds

builder = tfds.builder("agrivla_dataset_v3", data_dir="/home/n10813934/data/tfds_datasets")
label_names = builder.info.features["steps"].feature["observation"]["crop_type"].names
print("Label names:", label_names)

ds = builder.as_dataset(split="train")
crops = []

for ep in ds:
    s0 = next(ep["steps"].take(1).as_numpy_iterator())
    crop_id = s0["observation"]["crop_type"]  # integer
    crops.append(label_names[int(crop_id)])

print("Unique crops found:", set(crops))
