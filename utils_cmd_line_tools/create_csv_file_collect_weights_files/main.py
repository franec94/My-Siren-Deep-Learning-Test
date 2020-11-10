from utils.libs import *


parser = argparse.ArgumentParser()
parser.add_argument("--dir_path", type=str, required=True,
    help="Input dir path where resources are stored."
)
parser.add_argument("--timestamp", type=str, required=True, dest="timestamp",
    help="Timestamp to process."
)
parser.add_argument("--output_dir", type=str, required=True, dest="output_dir",
    help="Dir where to put results."
)
parser.add_argument("--fetch_all_res", required=False, default=False, action="store_true", dest="flag_fetch_all_res",
    help="Whetere to fetch all values computed during training or just the last file, since it own all data."
)


def sort_by_path_piece(df, files_list, df_key = 'archs_names'):
    file_list_sorted = []
    for a_arch in df[f'{df_key}'].values[:]:
        for a_file in files_list[:]:
            found = False
            a_file_tmp = copy.deepcopy(a_file)
            while True:
                a_file_tmp = os.path.split(a_file_tmp)
                # print(a_file_tmp)
                if len(a_file_tmp) == 0: break
                if len(a_file_tmp) == 2:
                    if a_file_tmp[1] == '':
                        break
                dir_name = a_file_tmp[1]
                if dir_name == a_arch:
                    found = True
                    break
                a_file_tmp = a_file_tmp[0]
            if found == True:
                file_list_sorted.append(a_file)
                break
    return file_list_sorted


def main(args):

    single_run_timestamp = args.timestamp
    date_obj = datetime.datetime.utcfromtimestamp(float(single_run_timestamp.replace("-", ".")))
    date_run = str(date_obj.strftime("%d-%m-%Y"))
    single_run_path= os.path.join(args.dir_path, date_run, single_run_timestamp, "train")

    # Fetch files into a list.
    if args.flag_fetch_all_res == False:
        regex_filter = re.compile(r'result_comb_train\.txt$')
    else:
        regex_filter = re.compile(r'result_comb_train(_)?(\d+)?\.txt$')
        pass
    files_list = get_all_files_by_ext(dir_path = single_run_path, ext = 'txt', recursive_search = False, regex_filter = regex_filter)
    res_arr = laod_data_from_files_list(files_list)
    columns = "#params;seed;hl;hf;mse;psnr;ssim;eta".split(";")
    df_res = pd.DataFrame(res_arr, columns = columns).sort_values(by = ['hf', 'hl', 'seed'])

    option_pickle_filename = 'options.pickle'
    pickel_full_path = os.path.join(f'{single_run_path}', f'{option_pickle_filename}')
    if os.path.isfile(pickel_full_path):
        with open(pickel_full_path, "rb") as f:
            opt = pickle.load(f)
            pass
        pass
    else:
        parser_run = argparse.ArgumentParser()
    
        # hf = list(map(int, "45".split(" ")))
        hf = list(sorted(set(df_res['hf'].values)))
        parser_run.add_argument("--hidden_features", default=hf, dest = "hidden_features", required=False)

        # hl = list(map(int, "2 3 4 5 6 7 8 9".split(" ")))
        hl = list(sorted(set(df_res['hl'].values)))
        parser_run.add_argument("--hidden_layers", default=hl, dest = "hidden_layers", required=False)
    
        lr = 0.0001
        parser_run.add_argument("--lr", default=lr, required = False, dest = "lr")
    
        # seeds = list(map(int, "0 42 123 1234 101".split(" ")))
        seeds = list(sorted(set(df_res['seed'].values)))
        parser_run.add_argument("--seeds", default=seeds, dest = "seeds", required=False)
    
        num_attempts = 1
        parser_run.add_argument("--num_attempts", default=num_attempts, dest = "num_attempts", required=False)
    
    
        opt, _ = parser_run.parse_known_args()
        pass

    param_grid = dict(
        timemstamp=[single_run_timestamp],
        hf=opt.hidden_features,
        hl=opt.hidden_layers,
        seeds=opt.seeds,
        lr=[opt.lr]
    )
    data = list(ParameterGrid(param_grid))
    df = pd.DataFrame(data = data)

    archs_names = [f"arch_no_{no}" for no in range(0, df.shape[0])]
    trials_names = [f"trial_no_{no}" for no in range(0, opt.num_attempts)]
    param_grid = dict(
        archs_names=archs_names,
        trials_names=trials_names,
    )
    data = list(ParameterGrid(param_grid))
    data = list(map(list, (map(operator.methodcaller('values'), data))))
    df[['archs_names', 'trials_names']] = data

    regex_filter = re.compile(r'.+\.pth$')
    files_list = get_all_files_by_ext(dir_path = single_run_path, ext = 'pth', recursive_search = True, regex_filter = regex_filter)
    file_list_sorted = sort_by_path_piece(df, files_list, df_key = 'archs_names')
    df['path'] = file_list_sorted

    columns_list = list(df.columns)
    RowTuple = collections.namedtuple('RowTuple', columns_list)
    root_dir_dest = os.path.join(args.output_dir, date_run, single_run_timestamp)
    def map_file_to_copy(a_row, root_dir = root_dir_dest, ext = 'pth'):
        a_row_tuple = RowTuple._make(a_row)
        pieces = [a_row_tuple.timemstamp, a_row_tuple.hf, a_row_tuple.hl, a_row_tuple.archs_names,a_row_tuple.trials_names, ext]
        a_path = '.'.join([str(xx) for xx in pieces])
        return os.path.join(root_dir, a_path)
    df['path_2'] = list(map(map_file_to_copy, df.values))

    try:
        os.makedirs(root_dir_dest)
    except:
        print(f"Dir {root_dir_dest} already exists!")
        pass
    filename_csv = os.path.join(root_dir_dest, 'create_collection.csv')
    df.to_csv(filename_csv)
    for src, dst in df[['path', 'path_2']].values:
        shutil.copyfile(src, dst)
        pass
    
    df_tmp = df[["hf", "hl", "lr", "seeds", "timemstamp", "archs_names", "trials_names"]]
    def map_path(a_path, target = root_dir_dest, update = f'\\content\\{single_run_timestamp}'):
        a_path_updated = a_path.replace(f"{target}", f"{update}").replace("\\", "/")
        return a_path_updated
    df_tmp['path'] = list(map(map_path, df['path_2'].values))
    cropped_heigth = [256] * df_tmp.shape[0]
    df_tmp['cropped_heigth'] = cropped_heigth
    df_tmp['cropped_width'] = cropped_heigth

    columns = "hf,hl,lr,seeds,timestamp,archs_names,trials_names,path,cropped_heigth,cropped_width".split(",")
    df_tmp.columns = columns
    filename_csv = os.path.join(root_dir_dest, f'colab_{single_run_timestamp}.csv')
    df_tmp.to_csv(filename_csv)
    print(df_tmp.head(5))

    if NOTIFICATIONS_ENABLED_VIA_TOAST:
        toaster = win10toast.ToastNotifier()

        toaster.show_toast(
            "Notification", 
            f"Data Prepared!\nDate: {date_run}\nTs: {single_run_timestamp}\nPath: {root_dir_dest}",
            threaded = True, icon_path=None, duration=3)
        pass
    pass


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
    pass