import tensorflow_datasets as tfds
ds = tfds.load("agrivla_dataset_v3", data_dir="/home/n10813934/data/tfds_datasets", split="train")
crops = []
for ep in ds:
    s0 = next(ep["steps"].take(1).as_numpy_iterator())
    crops.append(s0["observation"].get("crop_type", b"").decode("utf-8"))
print(set(crops))
