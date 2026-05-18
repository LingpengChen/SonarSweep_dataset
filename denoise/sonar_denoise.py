import torch
import numpy as np
import cv2
import os
import glob

from denoise.network_scunet import SCUNet as net


class SonarDenoiser:
    """
    Process sonar images with SCUNet and subtract the denoised background.

    The model and denoised background are loaded once during initialization.
    Call process() for each target sonar image.
    """

    def __init__(self, model_path: str, background_image: np.ndarray, mu: float = 15.0, epsilon: float = 0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mu = mu
        self.epsilon = epsilon
        
        self.model = net(in_nc=1, config=[4,4,4,4,4,4,4], dim=64)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        
        # Store the denoised background tensor for reuse.
        with torch.no_grad():
            background_tensor = self._numpy_to_tensor(background_image)
            self.denoised_background_tensor = self.model(background_tensor)
        
        print(f"SonarDenoiser initialized on {self.device}. Background processed.")

    def _numpy_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        Convert an HxW uint8 NumPy image to a 1x1xHxW float tensor in [0, 1].
        """
        img = img.astype(np.float32)
        img /= 255.0
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def process(self, target_image: np.ndarray) -> np.ndarray:
        """
        Denoise one target image and return an 8-bit background-subtracted image.
        """
        target_tensor = self._numpy_to_tensor(target_image)
        denoised_target_tensor = self.model(target_tensor)

        subtracted_tensor = (denoised_target_tensor - self.denoised_background_tensor).clamp_(min=0.0)

        # Optional echo-probability conversion for downstream experiments.
        # ep_tensor = torch.zeros_like(subtracted_tensor)
        # mask = subtracted_tensor >= self.epsilon
        # ep_tensor[mask] = 1.0 / (1.0 + torch.exp(-self.mu * subtracted_tensor[mask]))
        # return ep_tensor.squeeze().cpu().numpy()
        
        float_tensor = subtracted_tensor.squeeze().cpu().numpy()
        final_denoised_image = (float_tensor.clip(0, 1) * 255).astype(np.uint8)
        
        return final_denoised_image


def compute_average_background(background_dir: str = "./background", img_format: str = "png") -> np.ndarray:
    """
    Read all background images in a folder and return their pixelwise average.
    """
    search_pattern = os.path.join(background_dir, f'*.{img_format}')
    background_paths = glob.glob(search_pattern)

    if not background_paths:
        raise FileNotFoundError(f"No background images found with format '{img_format}' in directory '{background_dir}'")

    background_images = []
    print(f"Found {len(background_paths)} background files. Loading to compute average...")

    for path in background_paths:
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image at '{path}'. Skipping.")
                continue
            
            background_images.append(img.astype(np.float32))

        except Exception as e:
            print(f"Warning: Could not process file '{path}': {e}. Skipping.")

    if not background_images:
        raise ValueError("No valid background images could be loaded from the provided paths.")

    average_background = np.mean(np.stack(background_images, axis=0), axis=0)
    
    print("Average background computed successfully.")
    return average_background
