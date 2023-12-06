import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from decouple import Config, RepositoryEnv
config = Config(RepositoryEnv(".env"))
sys.path.insert(0, config("REPO_ROOT"))

def linear_regression(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c

def plot(baseline_path,
        eval_target_path,
        x_axis,
        y_axis,
        eval_target_label,
        figname,
        title):
    baseline = pd.read_csv(baseline_path)
    eval_target = pd.read_csv(eval_target_path)
    x_span = (min([baseline[x_axis].min(), eval_target[x_axis].min()]),
              max([baseline[x_axis].max(), eval_target[x_axis].max()]))
    x_range = np.linspace(x_span[0], x_span[1], 10)
    x, y = np.array(baseline[x_axis]), np.array(baseline[y_axis])
    m, c = linear_regression(x, y)
    fig = plt.figure()
    plt.scatter(x, y, marker="o", s=20, edgecolors='blue', alpha=0.5, label="baseline")
    plt.plot(x_range, m * x_range + c, linewidth=1, color="green", alpha=0.7, label="baseline fit")
    # drdj adversarial
    x, y = np.array(eval_target[x_axis]), np.array(eval_target[y_axis])
    plt.scatter(x, y, marker="+", color="red", s=80, alpha=0.7, label=eval_target_label)

    plt.legend()
    plt.title(title + f"({eval_target_label})")
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.savefig(os.path.join(config("REPO_ROOT"), f"misc/{figname}.png"))

def main():
    # baseline
    baseline_path ="/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src/misc/ResNet50_baseline_cifar_100_test.csv"
    # drdj adversarial
    drdj_adversarial_path = "/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src/misc/ResNet50_drdj_adversarial_cifar_100_test.csv"
    # drdj vanilla
    drdj_vanilla_path = "/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src/misc/ResNet50_drdj_vanilla_cifar_100_test.csv"
    # drdj vanilla pretrained
    drdj_vanilla_pretrained_path = "/gscratch/jamiemmt/andersonlee/image-distributionally-robust-data-join/src/misc/ResNet50_drdj_vanilla_pretrained_cifar_100_test.csv"

    # regular test vs. easy corruption
    plot(baseline_path=baseline_path,
        eval_target_path=drdj_adversarial_path,
        x_axis="regular_test_accuracy",
        y_axis="easy_corruption_accuracy",
        eval_target_label="drdj adversarial",
        figname="reg_easy_acc_adversarial",
        title="regular vs. easy corruption test acc")
    # regular test vs. hard corruption
    plot(baseline_path=baseline_path,
        eval_target_path=drdj_adversarial_path,
        x_axis="regular_test_accuracy",
        y_axis="hard_corruption_accuracy",
        eval_target_label="drdj adversarial",
        figname="reg_hard_acc_adversarial",
        title="regular vs. hard corruption test acc")
    # regular test vs. easy corruption
    plot(baseline_path=baseline_path,
        eval_target_path=drdj_vanilla_path,
        x_axis="regular_test_accuracy",
        y_axis="easy_corruption_accuracy",
        eval_target_label="drdj vanilla",
        figname="reg_easy_acc_vanilla",
        title="regular vs. easy corruption test acc")
    # regular test vs. hard corruption
    plot(baseline_path=baseline_path,
        eval_target_path=drdj_vanilla_path,
        x_axis="regular_test_accuracy",
        y_axis="hard_corruption_accuracy",
        eval_target_label="drdj vanilla",
        figname="reg_hard_acc_vanilla",
        title="regular vs. hard corruption test acc")
    # regular test vs. easy corruption
    plot(baseline_path=baseline_path,
        eval_target_path=drdj_vanilla_pretrained_path,
        x_axis="regular_test_accuracy",
        y_axis="easy_corruption_accuracy",
        eval_target_label="drdj vanilla pretrained",
        figname="reg_easy_acc_vanilla_pretrained",
        title="regular vs. easy corruption test acc")
    # regular test vs. hard corruption
    plot(baseline_path=baseline_path,
        eval_target_path=drdj_vanilla_pretrained_path,
        x_axis="regular_test_accuracy",
        y_axis="hard_corruption_accuracy",
        eval_target_label="drdj vanilla pretrained",
        figname="reg_hard_acc_vanilla_pretrained",
        title="regular vs. hard corruption test acc")

if __name__ == "__main__":
    main()