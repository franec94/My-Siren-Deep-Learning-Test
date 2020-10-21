#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.libs import *


from utils.custom_argparser import get_cmd_line_opts
from utils.functions import get_input_image, get_image_size_as_bits, show_image_characteristics, get_custom_logger
from utils.works import calculate_several_jpeg_compression

from utils.make_graphics import graphics_scatterplot, compute_graph_for_image_by_metrices


def main(opt = None):


    # Get logger for logging reasons.
    logger = get_custom_logger()

    # Get input image fetching by means of opt
    # taken from command input args.
    image, image_name = get_input_image(opt.input_image)

    # Show some infos about input image.
    im_prop_nt: collections.namedtuple = show_image_characteristics(image, image_name)
    logger.info('\n'.join([f"{k}: {v}" for k, v in im_prop_nt._asdict().items()]))

    # Compute compressions and get statistics.
    image_size_bist = get_image_size_as_bits(image)
    qualities_arr = np.arange(20, 95+1)

    result_tuples, failure_qualities = \
        calculate_several_jpeg_compression(image, image_size_bist, qualities_arr)
    if len(failure_qualities) != 0:
        logger.info('Failed qualities:')
        logger.info('\n'.join([str(q) for q in failure_qualities]))
        pass

    # Store data about compressions.
    result_df: pd.DataFrame = None
    if len(result_tuples) != 0:
        data_result = list(map(operator.methodcaller("_asdict"), result_tuples))
        result_df = pd.DataFrame(data = data_result)
        result_df.to_csv('results.csv')
        pass

    # Get big picture.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            g = sns.PairGrid(result_df.drop(['width', 'heigth'], axis = 1), diag_sharey=False)
            g.map_upper(sns.scatterplot, s=15) # 
            g.map_lower(sns.kdeplot)
            g.map_diag(sns.kdeplot, lw=2)
            # plt.savefig(f'scatter_plot_train_no_{train_no}.png')
            plt.savefig(f'big_scatter.png')
        except Exception as err:
            print(str(err))
            pass
        pass

    # Show scatter plot of relevant metrices.
    g = sns.PairGrid(result_df, x_vars = ['bpp', 'file_size_bits'], y_vars = ['psnr', 'ssim', 'CR'])
    _ = g.map(sns.scatterplot) # sns_plot
    g.fig.tight_layout()
    g.fig.subplots_adjust(top=0.9)
    g.fig.suptitle('JPEG Compressions')
    g.fig.savefig('scatter.png')

    grid_shape = "(1, 3)" # @param ["(1, 4)", "(4, 1)", "(2, 2)"]
    grid_shape = eval(grid_shape)

    fig, axes = graphics_scatterplot(
        dataframe = result_df,
        y_axes = ('psnr', 'ssim', 'CR'),
        x_axis = "bpp",
        grid_shape = grid_shape,
        figsize = (20, 10))
    fig.suptitle('Trend MSE and PSNR et al. across archs (grouped by #params).', fontsize=15)
    # plt.savefig(f"scatterplot_mse_psnr_et_al_vs_no_params_train_no_{train_no}.png")
    plt.savefig(f"bpp_vs_others_scatterplot.png")
    # plt.show()

    fig, axes = graphics_scatterplot(
        dataframe = result_df,
        y_axes = ('psnr', 'ssim', 'CR'),
        x_axis = "file_size_bits",
        grid_shape = grid_shape,
        figsize = (20, 10))
    fig.suptitle('Trend MSE and PSNR et al. across archs (grouped by #params).', fontsize=15)
    # plt.savefig(f"scatterplot_mse_psnr_et_al_vs_no_params_train_no_{train_no}.png")
    plt.savefig(f"file_size_bits_vs_others_scatterplot.png")
    # plt.show()

    # Get summary plot about metrices vs bpp and image's size as bits.
    x_axes = "bpp;file_size_bits".split(";")
    y_axes = "psnr;ssim;CR".split(";")

    fig, axes = compute_graph_for_image_by_metrices(
        data_tuples = result_tuples,
        x_axes = x_axes,
        y_axes = y_axes,
        subject = 'jpeg',
        colors = sns.color_palette())
    fig.suptitle(f'JPEG', fontsize=15)
    plt.savefig('summary_plot_metrices_for_jpge_res.png')


    pass


if __name__ == "__main__":
    # Parse input args options from cmd line.
    opt, p = get_cmd_line_opts()
    # Run main.
    main(opt = opt)
    pass