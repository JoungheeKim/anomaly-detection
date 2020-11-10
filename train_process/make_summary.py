from argparse import ArgumentParser
import os
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from bokeh.plotting import figure, output_file, output_notebook, show, save
from bokeh.models import BoxAnnotation, LinearAxis, Range1d



def load_pickle(temp_path):
    with open(temp_path, 'rb') as handle:
        return pickle.load(handle)


def save_pickle(temp_path, data):
    with open(temp_path, 'wb') as handle:
        pickle.dump(data, handle)


def plot_picture(dates, labels, var_values, confidence_values, save_path, name_dict, s_size=0.1):

    ## Fig 생성
    fig = figure(title=name_dict['title'],
               x_axis_label='Timeline',
               x_axis_type='datetime',
               y_axis_label='score',
               plot_width=2000,
               plot_height=500)

    fig.y_range = Range1d(start=0, end=max(var_values))
    fig.line(dates, var_values, line_width=2, color=name_dict['var_color'], legend_label=name_dict['var_name'])

    if labels is not None and len(dates) > 0:
        temp_start = dates[0]
        temp_label = labels[0]

        temp_date = dates[0]
        for xc, value in zip(dates, labels):
            if temp_label != value:
                if temp_label == 1:
                    fig.add_layout(BoxAnnotation(left=temp_start, right=temp_date, fill_alpha=0.2, fill_color='blue'))
                if temp_label == 2:
                    fig.add_layout(BoxAnnotation(left=temp_start, right=temp_date, fill_alpha=0.2, fill_color='orange'))
                temp_start = xc
                temp_label = value
            temp_date = xc

        if temp_label == 1:
            fig.add_layout(BoxAnnotation(left=temp_start, right=xc, fill_alpha=0.2, fill_color='blue'))
        if temp_label == 2:
            fig.add_layout(BoxAnnotation(left=temp_start, right=xc, fill_alpha=0.2, fill_color='orange'))

    if confidence_values is not None:
        fig.extra_y_ranges = {"var": Range1d(start=0, end=max(confidence_values))}
        fig.add_layout(LinearAxis(y_range_name="var"), 'right')
        fig.line(dates, confidence_values, legend_label=name_dict['confidence_name'], line_width=2, y_range_name='var', color=name_dict['confidence_color'], line_alpha=.3)
    output_file(filename=save_path)
    save(fig)


def build_parser():
    parser = ArgumentParser()

    ## Reference 데이터 위치
    parser.add_argument("--reference_file", type=str, default="")

    ## Summary 데이터 위치
    parser.add_argument("--summary_file", type=str, default="")

    ## 저장폴더 위치
    parser.add_argument("--save_path", type=str, required=True, default='그림')

    parser.add_argument("--mse", action="store_true")
    parser.add_argument("--mae", action="store_true")
    parser.add_argument("--mse_list", action="store_true")
    parser.add_argument("--mae_list", action="store_true")
    parser.add_argument("--mse_portion", action="store_true")
    parser.add_argument("--mae_portion", action="store_true")


    config = parser.parse_args()
    return config


def main():
    def _print_config(config):
        import pprint
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))

    ## 설정 불러오기.
    args = build_parser()
    _print_config(args)

    ##저장 폴더 만들기
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)

    df = pd.read_pickle(args.summary_file)

    dates = df['date'].values
    labels = df['label'].values

    name_dict = {
        "var_color": 'green',
        "confidence_color": 'red',
    }

    name_dict.update({
        "var_name": "confidence_score",
        "confidence_name": "confidence",
        'title': "confidence score",
        "var_color": 'black',
        "confidence_color": 'red',
        "var_plot": 'line',
        "confidence_plot": 'line',
    })


    if args.mse:
        target = 'mse'
        save_name = "{}_confidence.html".format(target)
        temp_save_path = os.path.join(args.save_path, save_name)
        plot_picture(dates, labels, df[target].values, None, temp_save_path, name_dict)

    if args.mae:
        target = 'mae'
        save_name = "{}_confidence.html".format(target)
        temp_save_path = os.path.join(args.save_path, save_name)
        plot_picture(dates, labels, df[target].values, None, temp_save_path, name_dict)

    if args.mse_list:
        target = 'mse'
        var_names = [item for item in df.columns if '{}_var_'.format(target) in item]
        for idx, var_name in enumerate(var_names):
            name_dict.update({
                "var_name": "confidence_score",
                "confidence_name": var_name + "_confidence",
                'title': "{} with {}".format("confidence_score", var_name + "_confidence"),
                "var_color": 'black',
                "confidence_color": 'red',
                "var_plot": 'line',
                "confidence_plot": 'line',
            })

            save_name = "{}_{}.html".format(target,var_name)
            temp_save_path = os.path.join(args.save_path, save_name)
            plot_picture(dates, labels, df[target], df[var_name].values, temp_save_path, name_dict)

    if args.mae_list:
        target = 'mae'
        var_names = [item for item in df.columns if '{}_var_'.format(target) in item]
        for idx, var_name in enumerate(var_names):
            name_dict.update({
                "var_name": "confidence_score",
                "confidence_name": var_name + "_confidence",
                'title': "{} with {}".format("confidence_score", var_name + "_confidence"),
                "var_color": 'black',
                "confidence_color": 'red',
                "var_plot": 'line',
                "confidence_plot": 'line',
            })

            save_name = "{}_{}.html".format(target, var_name)
            temp_save_path = os.path.join(args.save_path, save_name)
            plot_picture(dates, labels, df[target], df[var_name].values, temp_save_path, name_dict)

    if args.mse_portion:
        target = 'mse'
        summary_data = {
            "tag_name": [],
            "normal_portion_sum": [],
            "normal_portion_mean": [],
            "progsys_portion_sum": [],
            "progsys_portion_mean": [],
            "abnormal_portion_sum": [],
            "abnormal_portion_mean": [],
        }

        var_names = [item for item in df.columns if '{}_var_'.format(target) in item]
        for var_name in var_names:
            df["{}_portion".format(var_name)] = df[var_name] / df[target]

            summary_data['tag_name'].append(var_name)
            summary_data['normal_portion_sum'].append(
                df[df['state'] == 'normal']["{}_portion".format(var_name)].sum())
            summary_data['normal_portion_mean'].append(
                df[df['state'] == 'normal']["{}_portion".format(var_name)].mean())
            summary_data['progsys_portion_sum'].append(
                df[df['state'] == 'progsys']["{}_portion".format(var_name)].sum())
            summary_data['progsys_portion_mean'].append(
                df[df['state'] == 'progsys']["{}_portion".format(var_name)].mean())
            summary_data['abnormal_portion_sum'].append(
                df[df['state'] == 'abnormal']["{}_portion".format(var_name)].sum())
            summary_data['abnormal_portion_mean'].append(
                df[df['state'] == 'abnormal']["{}_portion".format(var_name)].mean())

        summary_df = pd.DataFrame(summary_data)
        mean_columns = [item for item in summary_df.columns if 'mean' in item]

        ## MEAN 정보 저장
        temp_df = summary_df.set_index("tag_name")
        fig = temp_df[mean_columns].T.plot.bar(figsize=(30, 15), rot=0).get_figure()
        save_name = '{}_{}_mean_portion.png'.format(target, 'all')
        temp_save_path = os.path.join(args.save_path, save_name)
        fig.savefig(temp_save_path, dpi=300)
        plt.close('all')

        for temp_column in mean_columns:
            fig = temp_df[[temp_column]].T.plot.bar(figsize=(30, 15), rot=0).get_figure()
            save_name = '{}_{}_mean_portion.png'.format(target, temp_column)
            temp_save_path = os.path.join(args.save_path, save_name)
            fig.savefig(temp_save_path, dpi=300)
            plt.close('all')

    if args.mae_portion:
        target = 'mae'
        summary_data = {
            "tag_name": [],
            "normal_portion_sum": [],
            "normal_portion_mean": [],
            "progsys_portion_sum": [],
            "progsys_portion_mean": [],
            "abnormal_portion_sum": [],
            "abnormal_portion_mean": [],
        }

        var_names = [item for item in df.columns if '{}_var_'.format(target) in item]
        for var_name in var_names:
            df["{}_portion".format(var_name)] = df[var_name] / df[target]

            summary_data['tag_name'].append(var_name)
            summary_data['normal_portion_sum'].append(
                df[df['state'] == 'normal']["{}_portion".format(var_name)].sum())
            summary_data['normal_portion_mean'].append(
                df[df['state'] == 'normal']["{}_portion".format(var_name)].mean())
            summary_data['progsys_portion_sum'].append(
                df[df['state'] == 'progsys']["{}_portion".format(var_name)].sum())
            summary_data['progsys_portion_mean'].append(
                df[df['state'] == 'progsys']["{}_portion".format(var_name)].mean())
            summary_data['abnormal_portion_sum'].append(
                df[df['state'] == 'abnormal']["{}_portion".format(var_name)].sum())
            summary_data['abnormal_portion_mean'].append(
                df[df['state'] == 'abnormal']["{}_portion".format(var_name)].mean())

        summary_df = pd.DataFrame(summary_data)
        mean_columns = [item for item in summary_df.columns if 'mean' in item]

        ## MEAN 정보 저장
        temp_df = summary_df.set_index("tag_name")
        fig = temp_df[mean_columns].T.plot.bar(figsize=(30, 15), rot=0).get_figure()
        save_name = '{}_{}_mean_portion.png'.format(target, 'all')
        temp_save_path = os.path.join(args.save_path, save_name)
        fig.savefig(temp_save_path, dpi=300)
        plt.close('all')

        for temp_column in mean_columns:
            fig = temp_df[[temp_column]].T.plot.bar(figsize=(30, 15), rot=0).get_figure()
            save_name = '{}_{}_mean_portion.png'.format(target, temp_column)
            temp_save_path = os.path.join(args.save_path, save_name)
            fig.savefig(temp_save_path, dpi=300)
            plt.close('all')








if __name__ == "__main__":
    main()