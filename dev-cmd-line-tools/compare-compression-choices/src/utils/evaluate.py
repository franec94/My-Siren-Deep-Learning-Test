from src.utils.diff_operators import gradient, laplace

# skimage
import skimage
import skimage.metrics as skmetrics
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


def eval(model, test_dataloader, device = 'cpu'):
    model.eval()

    test_data, ground_thruth = next(iter(test_dataloader))

    test_data = test_data['coords'].to(device)
    ground_thruth = ground_thruth['img'].to(device)
    predicted_image, predicted_coords = model(test_data)

    # predicted_grad_image = gradient(predicted_image, predicted_coords)
    # predicted_laplacian_image = laplace(predicted_image, predicted_coords)
    
    # return predicted_image, ground_thruth, predicted_image, predicted_grad_image, predicted_laplacian_image
    return predicted_image, ground_thruth, predicted_image, None, None