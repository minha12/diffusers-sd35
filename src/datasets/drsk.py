import pandas as pd
import datasets
import os
import yaml

class DRSK(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = []
    DEFAULT_CONFIG_NAME = "default"

    def __init__(self, config_path=None, **kwargs):
        # Default config path if not provided
        if config_path is None:
            config_path = os.path.join("./configs", "drsk_config.yaml")
        
        # Load config from YAML
        with open(config_path, "r") as f:
            self.yaml_config = yaml.safe_load(f)
        
        # Get base directory from config
        self.base_dir = self.yaml_config.get("base_dir", "")
        
        # Construct full JSONL path
        self.jsonl_path = os.path.join(
            self.base_dir, 
            self.yaml_config.get("jsonl_path", "")
        )
        
        # Get field names from config using consistent naming
        self.field_names = self.yaml_config.get("field_names", {})
        self.text_field = self.field_names.get("text", "prompt")
        self.image_field = self.field_names.get("image", "target")
        self.conditioning_image_field = self.field_names.get("conditioning_image", "source")
        
        # Create BuilderConfig from YAML config
        config = datasets.BuilderConfig(
            name="default",
            version=datasets.Version(self.yaml_config.get("version", "0.0.1"))
        )
        
        # Override BUILDER_CONFIGS
        DRSK.BUILDER_CONFIGS = [config]
        
        super().__init__(**kwargs)

    def _info(self):
        features = datasets.Features(
            {
                "image": datasets.Image(),
                "conditioning_image": datasets.Image(),
                "text": datasets.Value("string"),
            },
        )
        
        return datasets.DatasetInfo(
            description=self.yaml_config.get("description", ""),
            features=features,
            supervised_keys=None,
            homepage=self.yaml_config.get("homepage", ""),
            license=self.yaml_config.get("license", ""),
            citation=self.yaml_config.get("citation", ""),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "metadata_path": self.jsonl_path,
                    "base_dir": self.base_dir,
                },
            ),
        ]

    def _generate_examples(self, metadata_path, base_dir):
        metadata = pd.read_json(metadata_path, lines=True)

        for _, row in metadata.iterrows():
            text = row[self.text_field]
            
            # Main image
            image_path = os.path.join(base_dir, row[self.image_field])
            image = open(image_path, "rb").read()
            
            # Conditioning image
            conditioning_image_path = os.path.join(base_dir, row[self.conditioning_image_field])
            conditioning_image = open(conditioning_image_path, "rb").read()

            # Using image filename as key
            key = os.path.basename(row[self.image_field])
            
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