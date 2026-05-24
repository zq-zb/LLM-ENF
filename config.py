import os
BASE_DIR = "./movie_book_cdsr_processed"
IMAGE_DIR = "./movie_book_cdsr_processed/image"

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
VAL_CSV = os.path.join(BASE_DIR, "val.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
ITEM_META_LLM = os.path.join(BASE_DIR, "item_meta_llm_refined_clean.csv")
ITEM2ID_PATH = os.path.join(BASE_DIR, "item2id.json")
USER2ID_PATH = os.path.join(BASE_DIR, "user2id.json")
DOMAIN2ID_PATH = os.path.join(BASE_DIR, "domain2id.json")
ITEM_IMAGE_MAP_PATH = os.path.join(BASE_DIR, "item_image_map.json") 

FEATURE_SAVE_DIR = os.path.join(BASE_DIR, "multimodal_features")
os.makedirs(FEATURE_SAVE_DIR, exist_ok=True)

ID_EMBED_DIM = 512   
MODAL_DIM = 512 
CLIP_MODEL = "openai/clip-vit-base-patch32"
LONG_CLIP_MODEL = "zer0int/LongCLIP-GmP-ViT-L-14"