from utils.libs import *

from utils.custom_argparser import get_custom_argparser
from utils.pictures import show_training_loss_graph
from utils.siren_model import Siren
from utils.training import basic_traininig_loop, plane_traininig_loop


def show_model_informations(model):
    numel_list = [p.numel() for p in model.parameters() if p.requires_grad == True]
    print(model)
    print(sum(numel_list), numel_list)

    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t",name)
    else:
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                print("\t",name)
    pass


def main(args):

    # Set Device
    # ----------------------------------------------------------- #
    device = (torch.device('cuda:0') if torch.cuda.is_available()
        else torch.device('gpu'))
    print(f"Training on device {device}.")
    print(f"# cuda device: {torch.cuda.device_count()}")
    print(f"Id current device: {torch.cuda.current_device()}")


    # Hyper-params
    # ----------------------------------------------------------- #

    # Siren's architecture Hyper-params
    in_features = 2
    out_features = 1
    sidelength = 256
    hidden_layers = 3

    # Training phase's Hyper-params
    learning_rate = args.lr

    # Displaying Images
    steps_til_summary = 10
    
    # Siren's Model Intialization
    # ----------------------------------------------------------- #

    img_siren = Siren(
        in_features = in_features,
        out_features = out_features,
        hidden_features = sidelength, 
        hidden_layers = hidden_layers,
        outermost_linear=True)
    img_siren.cuda()

    model = img_siren.to(device)
    show_model_informations(model)

    # Loss and Optimization Strategies
    # ----------------------------------------------------------- #
    total_steps = 500 # Since the whole image is our dataset, this just means 500 gradient descent steps.

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(
        lr = learning_rate,
        params = img_siren.parameters())

    model_input, ground_truth = next(iter(dataloader))
    model_input, ground_truth = model_input.cuda(), ground_truth.cuda()

    # Training Loop
    # ----------------------------------------------------------- #
    start = time.time()
    # model, history = basic_traininig_loop(
    model, history = plane_traininig_loop(
        optimizer = optim,
        criterion = criterion,
        model = img_siren,
        model_input = model_input,
        ground_truth = ground_truth,
        # total_steps = total_steps, steps_til_summary = steps_til_summary)
        total_steps = total_steps)
    stop = time.time()

    # times = (stop - start) * 1000
    times = (stop - start)
    print('-' * 40)
    # print('Run time takes %d miliseconds' % times) # print('Run time takes %.3f seconds' % times)
    print('Training complete in {:.0f}m {:.0f}s {:.0f}ms'.format(times // 60, times % 60, (times - int((times % 60))) * 1000))

    model_wieghts_path = args.store_weights_path
    torch.save(model.state_dict(), model_wieghts_path + 'siren{}_nn_weights.pt'.format(sidelength))

    return 0


if __name__ == "__main__":

    # Check desired Python's Libraries
    print("PyTorch Version: ",torch.__version__)
    print("Torchvision Version: ",torchvision.__version__)

    # Parse input arguments
    parser = get_custom_argparser()
    args, unknown = parser.parse_known_args()


    # Set seeds for experiments repeatability
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    feature_extract = True
    
    # run main function
    exit_code = main(args)

    return sys.exit(exit_code)
