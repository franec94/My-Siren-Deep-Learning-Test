from src.libs import *
import src.libs


def get_data_from_local_list(dir_data_csv_list = None, dir_data_csv = None, data_csv_name = 'result_quant.csv'):
    def get_df(a_path_csv):
        data_df = pd.read_csv(path_data_csv)
        if 'Unnamed: 0' in data_df.columns:
            data_df = data_df.drop(['Unnamed: 0'], axis = 1)
            pass
        return data_df

    if src.libs.FLAG_FETCH_ALL_CSV_FILES == False:
        path_data_csv = os.path.join(dir_data_csv, data_csv_name)
        data_df = get_df(a_path_csv = path_data_csv)
    else:
        data_frames_list = []
        n_skipped = 0
        for dir_data_csv in dir_data_csv_list:
            try:
                path_data_csv = os.path.join(dir_data_csv, data_csv_name)
                data_df_t = get_df(a_path_csv = path_data_csv)
                data_frames_list.append(data_df_t)
            except Exception as _:
                n_skipped += 1
                pass
            pass
        print(f"SKIPPED {n_skipped} | READ {len(dir_data_csv_list) - n_skipped} | TOT {len(dir_data_csv_list)}")
        data_df = pd.concat(data_frames_list)
        pass
    return data_df


def main():
    
    data_df = get_data_from_local_list(dir_data_csv_list = src.libs.DIR_DATA_CSV_LIST, dir_data_csv = None, data_csv_name = 'result_quant.csv')

    def map_to_bpp_by_quant_tech(a_row):
        model_size, quant_tech = a_row
        if quant_tech != 'None':
            return model_size * 8 / 256 / 256
        else:
            return model_size * 32 / 256 / 256

    def map_to_quant_tech_hf(a_row):
        hf, quant_tech = a_row
        return f"{quant_tech}-{hf}"
    data_df['bpp'] = list(map(map_to_bpp_by_quant_tech, data_df[['model_size', 'quant_tech']].values))
    data_df['quant_tech_2'] = list(map(map_to_quant_tech_hf, data_df[['hidden_features', 'quant_tech']].values))

    # --- Run several trials for JPEG compression.
    im = load_target_image(image_file_path = None)
    im_cropped = get_cropped_by_center_image(im, target = 256)
    qualities_arr = np.arange(20, 95+1, dtype = np.int)
    cropped_file_size_bits = None
    with BytesIO() as f:
        im_cropped.save(f, format='PNG')
        cropped_file_size_bits = f.getbuffer().nbytes * 8
        pass
    def map_to_CR_by_quant_tech(a_row, im_size = im_cropped):
        model_size, quant_tech = a_row
        if quant_tech != 'None':
            return im_cropped / (model_size * 8)
        else:
            return im_cropped / (model_size * 32)
    data_df['CR'] = list(map(map_to_CR_by_quant_tech, data_df[['model_size', 'quant_tech']].values))
    result_tuples, _ = \
        calculate_several_jpeg_compression(im_cropped, cropped_file_size_bits, qualities_arr)
    data = list(map(operator.methodcaller('_asdict'), result_tuples))
    jpeg_df = pd.DataFrame(data = data)
    jpeg_df['quant_tech'] = ["jpeg"] *  jpeg_df.shape[0]
    jpeg_df['quant_tech_2'] = ["jpeg"] *  jpeg_df.shape[0]


    siren_columns_for_merge = "mse,psnr,ssim,CR,bpp,quant_tech,quant_tech_2".split(",")  # Here, list siren_df columns for merge purpose.
    jpeg_columns_for_merge = "mse,psnr,ssim,CR,bpp,quant_tech,quant_tech_2".split(",") 
    columns_names_merge = "mse,psnr,ssim,CR,bpp,quant_tech,quant_tech_2".split(",") 

    # Performe merging.
    data_frames_list = [
        data_df[siren_columns_for_merge],
        jpeg_df[jpeg_columns_for_merge],
    ]
    merged_df = pd.concat(data_frames_list, names = columns_names_merge)


    complex_figs_list = []
    y_list = "mse,psnr,ssim,CR".split(",")
    # x = 'bpp'; y = "psnr"; 
    x = 'bpp'; hue='quant_tech'
    for y in y_list:
        fig = px.scatter(data_df, x=f"{x}", y=f"{y}", color=f"{hue}", marginal_y="violin",
               marginal_x="box", trendline="ols", template=DASH_TEMPLATES_LIST[2])
        fig.update_layout(template = DASH_TEMPLATES_LIST[2], title_text=f'{y.upper()} | Groupped by {hue} | siren dataframes')
        complex_figs_list.append(fig)
        pass
    x = 'bpp'; hue='quant_tech_2'
    for y in y_list:
        fig = px.scatter(merged_df, x=f"{x}", y=f"{y}", color=f"{hue}", marginal_y="violin",
           marginal_x="box", trendline="ols", template=DASH_TEMPLATES_LIST[2])
        fig.update_layout(template = DASH_TEMPLATES_LIST[2], title_text=f'{y.upper()} | Groupped by {hue} | siren+jpeg dataframes')
        complex_figs_list.append(fig)
        pass
    complex_figs_dash_list = list(map(lambda fig: dcc.Graph(figure=fig), complex_figs_list))
    
    tab_names = 'Results Siren(MSE,PSNR,SSIM);Results Merged(MSE,PSNR,SSIM)'.split(";")
    app = get_dash_app(
        figs_list = complex_figs_dash_list,
        n_figs = len(y_list),
        tab_names_list = tab_names)
    
    if NOTIFICATIONS_ENABLED_VIA_TOAST:
        from win10toast import ToastNotifier
        toaster = ToastNotifier()
        toaster.show_toast(
            "Notification", f"Data Prepared!\nRunning Dash App\nhttp://localhost:8050",
            threaded = True, icon_path=None, duration=3)
        
        import webbrowser
        webbrowser.open("http://localhost:8050")
        pass
    app.run_server(debug=True, use_reloader=False, host='localhost') 


    pass


if __name__ == "__main__":

    main()
    pass