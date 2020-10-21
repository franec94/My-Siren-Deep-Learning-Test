from utils.libs import *

def create_dir_from_path_by_opt(dir_path, erase_check):
    if os.path.exists(dir_path) and os.path.isdir(dir_path):
        if erase_check:
            shutil.rmtree(dir_path)
            pass
        else: return
        pass
    os.makedirs(dir_path)
    pass

def get_custom_logger(logger_filename):
    # Create a log with the same name as the script that created it
    logger = logging.getLogger('results') 
    logger.setLevel('DEBUG')

    # Create handlers and set their logging level
    filehandler_dbg = logging.FileHandler(f'{logger_filename}', mode='w')
    filehandler_dbg.setLevel('DEBUG')
    
    # Add handlers to logger
    logger.addHandler(filehandler_dbg)
    return logger

def get_input_image(image_file_path: str) -> PIL.Image:
    
    if image_file_path is None:
        image = Image.fromarray(skimage.data.camera())
        return image, 'camera'
    
    if not os.path.exists(image_file_path):
        print(f"Error: '{image_file_path}'", file=sys.stderr)
        sys.exit(-1)
    pass
    image = Image.open(image_file_path)
    image_name = os.path.basename(image_file_path)
    return image, image_name

def get_image_size_as_bits(image: PIL.Image, image_format: str = 'PNG', quality: int = None) -> int:
    with BytesIO() as f:
        if image_format.upper() == 'JPGE':
            image.save(f, format=f'{image_format.upper()}', quality = int(quality))
        else:
            image.save(f, format=f'{image_format.upper()}')
        f.seek(0)
        return int(f.getbuffer().nbytes * 8)

def show_image_characteristics(image: PIL.Image, image_name = None) -> collections.namedtuple:

    # Compute properties to be displayed.
    width, height = image.size
    pixels = width * height
    channels = len(image.mode)
    image_size_bits = get_image_size_as_bits(image)
    bpp = image_size_bits / pixels

    # Create namedtuple for better displaying quantities
    typename = 'ImageProperties'
    fields_name = 'width;height;channels;image_size_bits;bpp;name'.split(";")

    ImageProperties = collections.namedtuple(typename, fields_name)

    # Create instance of such namedtuple
    im_values = [width, height, channels, image_size_bits, bpp, image_name]
    im_prop = ImageProperties._make(im_values)

    # Show results.
    pprint(im_prop._asdict())

    return im_prop