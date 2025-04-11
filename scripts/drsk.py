import pandas as pd
import datasets
import os

_VERSION = datasets.Version("0.0.2")

_DESCRIPTION = ""
_HOMEPAGE = ""
_LICENSE = ""
_CITATION = ""
_BASE_DIR = "/home/ubuntu/datasets/drsk"
_JSONL_PATH = os.path.join(_BASE_DIR, "prompt.json")

_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "conditioning_image": datasets.Image(),
        "text": datasets.Value("string"),
    },
)

_DEFAULT_CONFIG = datasets.BuilderConfig(name="default", version=_VERSION)

class DRSK(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [_DEFAULT_CONFIG]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=_FEATURES,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata_path": _JSONL_PATH,
                    "base_dir": _BASE_DIR,  # Changed to match _generate_examples parameters
                },
            ),
        ]

    def _generate_examples(self, metadata_path, base_dir):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            text = row["prompt"]
            
            # Main image (target)
            image_path = os.path.join(base_dir, row["target"])
            image = open(image_path, "rb").read()
            
            # Conditioning image (source)
            conditioning_image_path = os.path.join(base_dir, row["source"])
            conditioning_image = open(conditioning_image_path, "rb").read()

            # Using target filename as key
            key = os.path.basename(row["target"])
            
            yield key, {
                "text": text,
                "image": {
                    "path": image_path,
                    "bytes": image,
                },
                "conditioning_image": {
                    "path": conditioning_image_path,
                    "bytes": conditioning_image,
                },
            }