import pandas as pd


class DataFrameAnalyzer:
    def __init__(self, labels, probs):
        df_probs = pd.DataFrame(probs)
        df_labels = pd.DataFrame(labels)
        self.dflabels = df_labels
        self.dfactions = df_probs  # todo change to probas
        self.dfTOP10_br, self.dfTOP10_bw, self.dfTOP10_tw, self.df_combined_metrics = self.analyze_data()

    @staticmethod
    # Function to get the second maximum value
    def second_max(row):
        # return row.nlargest(2).iloc[-1], row.nlargest(2).index[-1]
        return pd.Series([row.nlargest(2).iloc[-1], row.nlargest(2).index[-1]], index=["value", "label"])

    def analyze_data(self, threshold=0.2, top_n=10):
        label = self.dflabels.idxmax(axis=1)
        label_value = [self.dfactions.at[i, j] for i, j in zip(label.index, label.values)]
        prediction = self.dfactions.idxmax(axis=1)
        prediction_value = [self.dfactions.at[i, j] for i, j in zip(prediction.index, prediction.values)]
        prediction_2nd = self.dfactions.apply(self.second_max, axis=1)
        df = pd.DataFrame(
            {
                "label": label,
                "label_value": label_value,
                "prediction": prediction,
                "prediction_value": prediction_value,
                "prediction_2nd": prediction_2nd["label"],
                "prediction_2nd_value": prediction_2nd["value"],
            }
        )
        cond = df["label"] == df["prediction"]
        df_r = df.loc[cond].copy()
        df_r.loc[:, "diff"] = df_r["prediction_value"] - df_r["prediction_2nd_value"]
        df_w = df.loc[~cond].copy()
        df_w.loc[:, "diff"] = df_w["prediction_value"] - df_w["label_value"]

        df_top_n_br = df_r.sort_values(by="diff", ascending=True).head(top_n)
        df_top_n_bw = df_w.sort_values(by="diff", ascending=True).head(top_n)
        df_top_n_tw = df_w.sort_values(by="diff", ascending=False).head(top_n)

        df_br = df_r[(df_r["diff"] < threshold)]
        df_bw = df_w[(df_w["diff"] < threshold)]

        # Counting unique values in the 'prediction' column for each DataFrame
        df_br_prediction_counts = df_br["label"].value_counts()
        df_bw_prediction_counts = df_bw["label"].value_counts()
        df_r_prediction_counts = df_r["label"].value_counts()
        df_w_prediction_counts = df_w["label"].value_counts()

        # Combine counts into one DataFrame
        df_combined_metrics = pd.concat(
            [df_br_prediction_counts, df_bw_prediction_counts, df_r_prediction_counts, df_w_prediction_counts],
            axis=1,
            keys=["br", "bw", "r", "w"],
        ).astype(float)
        df_combined_metrics[["br", "bw"]] /= df_combined_metrics[["r", "w"]].values
        df_combined_metrics["w"] /= df_combined_metrics[["r", "w"]].sum(axis=1).values
        df_combined_metrics = df_combined_metrics.drop("r", axis=1)
        df_combined_metrics = df_combined_metrics.reindex([0, 1, 2, 3]).apply(lambda x: round(x * 100, 2))

        return df_top_n_br, df_top_n_bw, df_top_n_tw, df_combined_metrics


if __name__ == "__main__":
    analyzer = DataFrameAnalyzer("test_labels.csv", "test_actions.csv")
    print("1")
