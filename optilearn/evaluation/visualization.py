import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import pyplot as plt


class Visualization:
    """
    A class for generating visualizations of multi-objective optimization results.
    """

    def __init__(self, n_obj, front=None):
        """
        Initializes a Visualization object.

        Args:
            n_obj (int): The number of objectives.
            front (numpy.ndarray, optional): The true Pareto front. Defaults to None.
        """
        self.front = front
        self.n_obj = n_obj
        if n_obj == 2:
            self.gen_xy_plot = self.gen_xy_plot_2d
        else:
            self.gen_xy_plot = self.gen_xy_plot_nd

    def set_front(self, front):
        """
        Sets the true Pareto front.

        Args:
            front (numpy.ndarray): The true Pareto front.
        """
        self.front = front

    def _gen_plt_2obj(self, fig, u, au, u_itr, au_itr, y, yf, itr, emp_front):
        """
        Generates a 2D plot for two objectives.

        Args:
            fig (matplotlib.figure.Figure): The figure object.
            u (numpy.ndarray): The reference point.
            au (numpy.ndarray): The alternative reference point.
            u_itr (numpy.ndarray): The reference point for the current iteration.
            au_itr (numpy.ndarray): The alternative reference point for the current iteration.
            y (numpy.ndarray): The samples.
            yf (numpy.ndarray): The approximate Pareto front.
            itr (int): The iteration number.
            emp_front (numpy.ndarray): The empirical Pareto front.

        Returns:
            matplotlib.figure.Figure: The updated figure object.
        """
        ax = fig.add_subplot(111)
        picsize = fig.get_size_inches() / 1.3
        fig.set_size_inches(picsize)
        ax.cla()
        if self.front is not None:
            ax.plot(
                self.front[:, 0],
                self.front[:, 1],
                marker="o",
                linestyle="None",
                color="b",
                markersize=8,
                label=r"$Frontier_{True}$",
            )
        if emp_front is not None:
            ax.plot(
                emp_front[:, 0],
                emp_front[:, 1],
                marker="*",
                linestyle="None",
                color="black",
                markersize=8,
                label=r"$Frontier_{Empiric}$",
            )
        ax.plot(
            y[:, 0],
            y[:, 1],
            marker="o",
            linestyle="None",
            color="r",
            markersize=2,
            label=r"$Samples$",
        )
        ax.plot(
            yf[:, 0],
            yf[:, 1],
            marker="o",
            linestyle="None",
            color="g",
            markersize=4,
            label=r"$Frontier_{Approx}$",
        )
        if u is not None:
            ax.plot(u[0], u[1], marker="s", linestyle="None", color="orange", markersize=8)
        if au is not None:
            ax.plot(au[0], au[1], marker="s", linestyle="None", color="orange", markersize=8)
        if u_itr is not None:
            ax.plot(
                u_itr[0],
                u_itr[1],
                marker="x",
                linestyle="None",
                color="orange",
                markersize=8,
            )
        if au_itr is not None:
            ax.plot(
                au_itr[0],
                au_itr[1],
                marker="x",
                linestyle="None",
                color="orange",
                markersize=8,
            )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.title(str(itr))
        return fig

    def _gen_plt_3obj(self, fig, u, au, u_itr, au_itr, y, yf, itr, emp_front):
        """
        Generates a 3D plot for three objectives.

        Args:
            fig (matplotlib.figure.Figure): The figure object.
            u (numpy.ndarray): The reference point.
            au (numpy.ndarray): The alternative reference point.
            u_itr (numpy.ndarray): The reference point for the current iteration.
            au_itr (numpy.ndarray): The alternative reference point for the current iteration.
            y (numpy.ndarray): The samples.
            yf (numpy.ndarray): The approximate Pareto front.
            itr (int): The iteration number.
            emp_front (numpy.ndarray): The empirical Pareto front.

        Returns:
            matplotlib.figure.Figure: The updated figure object.
        """
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        picsize = fig.get_size_inches() / 1.3
        picsize[0] *= 2
        fig.set_size_inches(picsize)
        ax1.cla()
        if self.front is not None:
            ax1.plot(
                self.front[:, 0],
                self.front[:, 1],
                self.front[:, 2],
                marker="o",
                linestyle="None",
                color="b",
                markersize=8,
                label=r"$Frontier_{True}$",
            )
        if emp_front is not None:
            ax1.plot(
                emp_front[:, 0],
                emp_front[:, 1],
                emp_front[:, 2],
                marker="*",
                linestyle="None",
                color="black",
                markersize=8,
                label=r"$Frontier_{Empiric}$",
            )
        ax1.plot(
            y[:, 0],
            y[:, 1],
            y[:, 2],
            marker="o",
            linestyle="None",
            color="r",
            markersize=2,
            label=r"$Samples$",
        )
        ax1.plot(
            yf[:, 0],
            yf[:, 1],
            yf[:, 2],
            marker="o",
            linestyle="None",
            color="g",
            markersize=4,
            label=r"$Frontier_{Approx}$",
        )
        if u is not None:
            ax1.plot(
                u[0],
                u[1],
                u[2],
                marker="s",
                linestyle="None",
                color="orange",
                markersize=8,
            )
        if au is not None:
            ax1.plot(
                au[0],
                au[1],
                au[2],
                marker="s",
                linestyle="None",
                color="orange",
                markersize=8,
            )
        if u_itr is not None:
            ax1.plot(
                u_itr[0],
                u_itr[1],
                u_itr[2],
                marker="s",
                linestyle="None",
                color="orange",
                markersize=8,
            )
        if au_itr is not None:
            ax1.plot(
                au_itr[0],
                au_itr[1],
                au_itr[2],
                marker="x",
                linestyle="None",
                color="orange",
                markersize=8,
            )
        for spine in ax1.spines.values():
            spine.set_visible(False)
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_zlabel("Z")

        ax2.cla()
        ax2.plot(
            y[:, 0],
            y[:, 1],
            y[:, 2],
            marker="o",
            linestyle="None",
            color="r",
            markersize=4,
        )
        ax2.plot(
            yf[:, 0],
            yf[:, 1],
            yf[:, 2],
            marker="o",
            linestyle="None",
            color="g",
            markersize=2,
        )
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax2.view_init(elev=50.0, azim=25)  # change view for second plot
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")
        plt.suptitle(str(itr))
        return fig

    def gen_plt(self, u, au, y, yf, emp_front=None, true_front=None):
        """
        Generates a scatter plot for the objectives.

        Args:
            u (numpy.ndarray): The reference point.
            au (numpy.ndarray): The alternative reference point.
            y (numpy.ndarray): The samples.
            yf (numpy.ndarray): The approximate Pareto front.
            emp_front (numpy.ndarray, optional): The empirical Pareto front. Defaults to None.
            true_front (numpy.ndarray, optional): The true Pareto front. Defaults to None.

        Returns:
            plotly.graph_objects.Figure: The scatter plot figure.
        """
        columns = [f"Obj{i}" for i in range(self.n_obj)]
        type_col = "type"
        marker_col = "marker"
        df_u = pd.DataFrame(data=[u], columns=columns)
        df_u[type_col] = "ref"
        df_u[marker_col] = "ref"
        df_au = pd.DataFrame(data=[au], columns=columns)
        df_au[type_col] = "ref"
        df_au[marker_col] = "ref"
        df_y = pd.DataFrame(data=y, columns=columns)
        df_y[type_col] = "y"
        df_y[marker_col] = "p"
        df_yf = pd.DataFrame(data=yf, columns=columns)
        df_yf[type_col] = "front"
        df_yf[marker_col] = "front"
        df_yef = pd.DataFrame(data=emp_front, columns=columns)
        df_yef[type_col] = "emp_front"
        df_yef[marker_col] = "emp_front"
        df_ytf = pd.DataFrame(data=true_front, columns=columns)
        df_ytf[type_col] = "true_front"
        df_ytf[marker_col] = "true_front"

        colors = ["black", "red", "gray", "blue", "green"]
        symbols = ["circle", "circle-open", "diamond", "square", "cross"]
        df = pd.concat([df_ytf, df_y, df_u, df_au, df_yef, df_yf])

        if self.n_obj == 2:
            fig = px.scatter(
                df,
                x="Obj0",
                y="Obj1",
                color=type_col,
                symbol=marker_col,
                color_discrete_sequence=colors,
                symbol_sequence=symbols,
                labels={type_col: "Type"},
            )
            fig.update_traces(marker_size=12)

        elif self.n_obj == 3:
            fig = px.scatter_3d(
                df,
                x="Obj0",
                y="Obj1",
                z="Obj2",
                color=type_col,
                symbol=marker_col,
                color_discrete_sequence=colors,
                symbol_sequence=symbols,
                labels={type_col: "Type"},
            )
        return fig

    def gen_hist(self, params, itr, name):

        """
        Generates a histogram plot.

        Args:
            params (numpy.ndarray): The parameters.
            itr (int): The iteration number.
            name (str): The name of the histogram.

        Returns:
            matplotlib.figure.Figure: The histogram figure.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        sns.histplot(params, ax=ax)
        plt.title(f"{name} - {itr}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        return fig

    def gen_xy_plot_nd(self, x, y, itr, name, x_front=None, x_front_emp=None):
        """
        Generates a 2D plot for multiple objectives.

        Args:
            x (numpy.ndarray): The x-axis values.
            y (numpy.ndarray): The y-axis values.
            itr (int): The iteration number.
            name (str): The name of the plot.
            x_front (numpy.ndarray, optional): The true Pareto front. Defaults to None.
            x_front_emp (numpy.ndarray, optional): The empirical Pareto front. Defaults to None.

        Returns:
            matplotlib.figure.Figure: The plot figure.
        """
        x_dim = 0
        fig = plt.figure()
        for i in range(self.n_obj):
            plt.plot(
                x[:, x_dim],
                y[:, i],
                marker="o",
                linestyle="None",
                markersize=4,
                label=r"$Objective_{}$".format(i),
            )
            if i == 0:
                if x_front is not None:
                    plt.plot(
                        x[:, x_dim],
                        x_front[:, i],
                        color="black",
                        label=r"$Front_{True}$",
                    )
                if x_front_emp is not None:
                    plt.plot(
                        x[:, x_dim],
                        x_front_emp[:, i],
                        color="gray",
                        label=r"$Front_{Empirical}$",
                        linestyle="-",
                    )
            else:
                if x_front is not None:
                    plt.plot(x[:, x_dim], x_front[:, i], color="black")
                if x_front_emp is not None:
                    plt.plot(x[:, x_dim], x_front_emp[:, i], color="gray")
        plt.title(f"{name} - {itr}")
        plt.ylabel("Reward")
        plt.xlabel("Preference for Obj1")
        plt.legend()
        return fig

    def gen_xy_plot_2d(self, x, y, itr, name, x_front=None, x_front_emp=None):
        """
        Generates a 2D plot for two objectives.

        Args:
            x (numpy.ndarray): The x-axis values.
            y (numpy.ndarray): The y-axis values.
            itr (int): The iteration number.
            name (str): The name of the plot.
            x_front (numpy.ndarray, optional): The true Pareto front. Defaults to None.
            x_front_emp (numpy.ndarray, optional): The empirical Pareto front. Defaults to None.

        Returns:
            matplotlib.figure.Figure: The plot figure.
        """
        x_dim = 0
        fig, ax1 = plt.subplots()
        i = 0
        color = "tab:orange"
        ax1.plot(
            x[:, x_dim],
            y[:, i],
            marker="o",
            linestyle="None",
            markersize=4,
            color=color,
        )
        ax1.set_ylabel(f"Reward Obj{i}", color=color)
        if x_front is not None:
            ax1.plot(x[:, x_dim], x_front[:, i], color=color, linestyle="--")
        if x_front_emp is not None:
            ax1.plot(x[:, x_dim], x_front_emp[:, i], color="gray", linestyle="-")
        ax1.set_xlabel("Preference for Obj0")

        i = 1
        color = "tab:blue"
        ax2 = ax1.twinx()
        ax2.plot(
            x[:, x_dim],
            y[:, i],
            marker="o",
            linestyle="None",
            markersize=4,
            # label=r'$Objective_{}$'.format(i),
            color=color,
        )

        ax2.set_ylabel(f"Reward Obj{i}", color=color)
        # ax2.set_ylabel(r'$Reward Obj_{}$'.format(i), color=color)
        if x_front is not None:
            ax2.plot(
                x[:, x_dim],
                x_front[:, i],
                color="black",
                linestyle="--",
                label="Optimal Rewards",
            )
            ax2.plot(x[:, x_dim], x_front[:, i], color=color, linestyle="--")
        if x_front_emp is not None:
            ax2.plot(x[:, x_dim], x_front_emp[:, i], color="gray")
        plt.title(f"{name} - {itr}")
        plt.legend(loc="lower center")
        plt.tight_layout()
        return fig

    @staticmethod
    def gen_3d_plt(x, y, z):
        """
        Generate a 3D scatter plot.

        Args:
            x (list): List of x-coordinates.
            y (list): List of y-coordinates.
            z (list): List of z-coordinates.

        Returns:
            fig (go.Figure): The generated 3D scatter plot figure.
        """
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="markers",
                    marker=dict(
                        size=12,
                        color=z,
                        # set color to an array/list of desired values
                        colorscale="Viridis",  # choose a colorscale
                        opacity=0.8,
                    ),
                )
            ]
        )
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        return fig

    @staticmethod
    def gen_plot_top_x(top_x, observations):
        """
        Generate a plot displaying the top x predictions and corresponding images.

        Args:
            top_x (DataFrame): A DataFrame containing the top x predictions.
            observations (list): A list of images corresponding to the predictions.

        Returns:
            fig (Figure): The generated matplotlib Figure object.
        """
        fig = plt.figure(figsize=(10, 12))
        for i in range(10):
            pred = int(top_x.iloc[i].prediction)
            pred_value = f"{top_x.iloc[i].prediction_value * 100:.2f}"
            label = int(top_x.iloc[i].label)
            label_value = f"{top_x.iloc[i].label_value * 100:.2f}"
            img_idx = top_x.index[i]
            img = observations[img_idx]
            ax = fig.add_subplot(5, 2, i + 1)
            ax.imshow(img)
            ax.set_title(f"label/pred: {label}({label_value}%)/{pred}({pred_value}%)")

        plt.tight_layout()
        return fig

    @staticmethod
    def gen_plot_bar_category(combined_metrics: pd.DataFrame):
        """
        Generate a bar plot to visualize combined metrics for different categories.

        Args:
            combined_metrics (DataFrame): A DataFrame containing the combined metrics for each category.

        Returns:
            matplotlib.figure.Figure: The generated bar plot figure.
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        categories = combined_metrics.index.values
        bw = combined_metrics["bw"]
        br = combined_metrics["br"]
        tw = combined_metrics["w"]
        # Set the width of the bars
        bar_width = 0.25
        # Calculate the position for each group of bars
        x = np.arange(len(categories))
        ax.bar(x - bar_width, bw, width=bar_width, label="Barely wrong")
        ax.bar(x, br, width=bar_width, label="Barely right")
        ax.bar(x + bar_width, tw, width=bar_width, label="Totally wrong")

        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        plt.tight_layout()
        return fig


if __name__ == "__main__":
    from optilearn.evaluation.neptune_logger import NeptuneLogger
    from optilearn.utils.utils import get_eval_w

    # data_path = '/Users/eikementzendorff/Downloads/results_10000.csv'
    # data = pd.read_csv(data_path, index_col=0)
    # ef_data_path = '/Users/eikementzendorff/Downloads/empiric_frontier_10000.csv'
    # ef_data = pd.read_csv(ef_data_path, index_col=0)
    data = get_eval_w(10, 3)
    df = pd.DataFrame(data=data, columns=[0, 1, 2])

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection="3d")
    ax1.plot(data[:, 0], data[:, 1], data[:, 2])
    plt.show()

    # df = px.data.iris()
    fig = px.scatter_3d(
        df,
        x=0,
        y=1,
        z=2,
        # color='species'
    )
    # fig.show()
    nl = NeptuneLogger()
    nl.start("test", None, run_id="OP-552")
    nl.upload_fig(fig, "test", "test")
    nl.log_fig(fig, "log_test", "test")
    # vis = Visualization(2, ef_data.values)
    # vis.gen_xy_plot_2d(data[['pref_0', 'pref_1']].values, data[['reward_0', 'reward_1']].values, 25500, 'test', x_front_emp=ef_data.values)
    plt.show()
    pass
