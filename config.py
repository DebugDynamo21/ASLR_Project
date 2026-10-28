"""Central settings. Edit PHRASES to change the vocabulary (folder names under data/raw/)."""

PHRASES = [
    "hello", "thank_you", "please", "sorry", "yes", "no", "help",
    "good_morning", "how_are_you", "i_love_you", "my_name", "what",
    "where", "water", "eat", "stop", "goodbye", "friend", "want",
    "nice_to_meet_you",
]

K_FRAMES = 24          # keyframes kept per sign (same for skeleton + RGB)
IMG_SIZE = 160         # RGB keyframe crop size fed to the CNN
RGB_DIM = 576          # MobileNetV3-small embedding size
MAX_W = 480            # frames are downscaled to this width (speed + consistency)
MP_COMPLEXITY = 1      # MediaPipe Holistic complexity (0 = faster, 1 = more accurate)

RAW_DIR = "data/raw"
PROC_DIR = "data/processed"
MODEL_PATH = "models/sign_model.pt"
