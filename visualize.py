import math
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams["font.sans-serif"] = ["SimSun"]
mpl.rcParams.update({'font.size': 6})


def showFig(df):
    plt.figure()
    plt.subplots_adjust(left=0.03, hspace=1, top=0.96, right=0.99, bottom=0.05)
    count = 0
    for col in df.columns:
        if df[col].dtype == "object" or col == 'star_level':
            bar_value = df[col].value_counts().sort_index()
            # print(bar_value)
            if col != "uid":
                count += 1
                plt.subplot(6, 6, count)
                plt.title(col)
                bar_value.plot.bar()
                plt.xticks(fontsize=6)

                for i in range(len(bar_value)):
                    plt.text(i, bar_value.values[i], bar_value.values[i],
                             ha='center', va='bottom')
        elif not math.isnan(df[col].min()):
            min = df[col].min()
            max = df[col].max()

            list_bin = []

            if min == max:
                list_bin.append(min)
            else:
                step = (max - min) / 10

                for i in range(11):
                    list_bin.append(round(min + i * step, 2))

            count += 1
            plt.subplot(6, 6, count)
            plt.title(col)
            hist = plt.hist(df[col], edgecolor="black")
            plt.xticks(list_bin, rotation=45, fontsize=6)
            for i in range(len(hist[0])):
                plt.text(hist[1][i] + (hist[1][i+1] - hist[1][i])/2,
                         hist[0][i], hist[0][i], ha='center', va='bottom')

    plt.show()
