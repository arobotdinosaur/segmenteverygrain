from surrogate_gp import train_model_on_resolutions
from create_synthetic_images import generate_synthetic_images
synthetics_folder = "./synthetic_noisy_images/"
real_folder = "./surrogate_data/reference_noisy_images/"
train_model_on_resolutions(synthetic_folder=synthetics_folder,real_noisy_folder=real_folder,model_name="test_blackbox")
