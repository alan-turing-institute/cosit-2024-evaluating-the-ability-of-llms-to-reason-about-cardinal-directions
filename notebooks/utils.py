# utils.py

# Utility functions for our COSIT24 / QR2025 evaluation of cardinal
# directions.

from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import scipy.stats as stats
import seaborn as sns
from matplotlib.cm import get_cmap
from matplotlib.colors import to_hex
import colorsys
import distinctipy
import random


def ensure_path(path):
    """
    We always want to expand ~ and allow relative paths.
    """
    return Path(path).expanduser().resolve()


def load_jsonl(filename):
    """
    Load a JSON Lines format file
    """
    data = []
    with open(filename, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def load_questions(filename):
    """
    filename is a path to a JSONL file with id and question.
    """
    filename = ensure_path(filename)
    data = load_jsonl(filename)
    df = pd.DataFrame(data)
    df["id"] = df["id"].astype(str)
    return df


def load_answers(directory, model_name=None):
    """
    directory is a path to an experiment. If model_name is None,
    then load answers for all models.
    """
    directory = ensure_path(directory)
    if model_name:
        file_path = os.path.join(directory, model_name, "answers.jsonl")
        df = pd.read_json(file_path, lines=True)
        df["model"] = model_name
        df["id"] = df["id"].astype(str)
        return df
    else:
        model_names = [
            entry
            for entry in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, entry))
            and os.path.exists(os.path.join(directory, entry, "answers.jsonl"))
        ]
        df = pd.concat(
            [load_answers(directory, model_name) for model_name in model_names],
            ignore_index=True,
        )
        return df


def load_results(directory):
    df1 = load_questions(os.path.join(directory, "questions.jsonl"))
    df1["correctAnswer"] = df1["absoluteAnswer"].apply(clean_answer)
    df2 = load_answers(directory)
    df2["cleanAnswer"] = df2["answer"].apply(clean_answer)
    df3 = pd.merge(df1, df2, on="id", how="inner")
    df3["score"] = df3["correctAnswer"] == df3["cleanAnswer"]
    return df3


def clean_answer(s):

    # Note that we do allow invalid answers to be correct, if say the
    # answer given is "the bandstand is to the north", when all we
    # really wanted was "north".
    answer = s.lower().strip().replace(" ", "-")

    answer = re.sub(
        r"<think>.*?</think>", "", answer, flags=re.DOTALL
    )  # For deepseek on Azure

    answer = "".join([char for char in answer if char.islower() or char == "-"])

    # Substitute non-direction words with an empty string
    pattern = r"\b(?!north\b|south\b|east\b|west\b)\w+\b"
    answer = re.sub(pattern, "", answer)

    # Remove potential leading, trailing, or multiple consecutive hyphens
    answer = re.sub(r"^-+|-+$|(?<=-)-+", "", answer)

    return answer


def prediction_interval(samples, confidence=0.95):
    mean = np.mean(samples)
    n = len(samples)
    std_dev = np.std(samples, ddof=1)  # sample standard deviation n = len(samples)
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin_of_error = t_crit * std_dev * np.sqrt(2 / n)
    return margin_of_error


def plot_model_scores(df, ax=None, width=6, height=8, fontsize=10):
    plt.rcParams.update({"font.size": fontsize})  # Adjust the value as needed

    # Grouping and aggregating the data
    df = df.groupby(["model", "repeat", "colour"])["score"].mean().reset_index()

    df = (
        df.groupby(["model", "colour"])
        .agg(
            mean=("score", "mean"),
            interval=("score", prediction_interval),
            n=("score", "count"),
        )
        .reset_index()
        .sort_values(by="mean", ascending=True)
    )

    # Define color mapping
    colors = df["colour"].apply(lambda x: "#a6cee3" if x == 1 else "#b2df8a")

    # Create a new figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(width, height))
    else:
        fig = ax.figure

    # Models to highlight
    highlight_models = {
        "azure-o1-2024-12-17",
        "azure-o3-mini-2025-01-31",
        "azure-o3-mini-2025-01-31-high",
        "azure-o4-mini-2025-04-16",
        "deepseek-reasoner",
    }

    # Creating the horizontal bar plot with dynamic colors
    bars = ax.barh(
        df["model"],
        df["mean"],
        capsize=5,
        color=colors,
        alpha=0.7,
        edgecolor=[
            "black" if model in highlight_models else "none" for model in df["model"]
        ],
        linewidth=[1 if model in highlight_models else 0 for model in df["model"]],
    )

    ax.set_ylabel("", labelpad=0)
    ax.set_xlabel("Accuracy")
    ax.set_xlim(0, 1.0)  # Assuming the scores are between 0 and 1
    ax.set_ylim(-0.5, len(df["model"]) - 0.5)

    for label in ax.get_yticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment("right")

    split = 0.5
    # Annotating each bar
    for bar, mean, interval, n in zip(bars, df["mean"], df["interval"], df["n"]):
        ax.text(
            mean - 0.03 if mean > split else mean + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{mean:.2f} ± {interval:.3f} (n={n})" if n > 1 else f"{mean:.2f}",
            ha="right" if mean > split else "left",
            va="center",
            fontsize=fontsize - 1,
        )

    ax.axvline(x=0.125, color="red", linestyle="dotted", linewidth=1)


def plot_locomotion(
    df, model, ax=None, show_xlabel=True, show_ylabel=True, show_title=True
):
    df = (
        df[df["model"] == model]
        .groupby("locomotion")
        .agg(
            mean_score=("score", "mean"),
            n_score=("score", "count"),
            interval=("score", prediction_interval),
        )
        .sort_values(by="mean_score", ascending=False)
    )

    df["label"] = df.apply(
        lambda row: f" {row['mean_score']:.3f}±{row['interval']:.3f}",
        axis=1,
    )

    df = df.reset_index()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.barh(df["locomotion"], df["mean_score"], xerr=df["interval"], capsize=5)

    if show_xlabel:
        ax.set_xlabel("Mean Score")
    if show_ylabel:
        ax.set_ylabel("Locomotion")
    if show_title:
        ax.set_title(model)

    split = 0.5
    dx = 0.05

    for bar, mean, ci, label in zip(
        bars, df["mean_score"], df["interval"], df["label"]
    ):
        ax.text(
            mean - dx if mean > split else mean + dx,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="right" if mean > split else "left",
        )

    ax.set_xlim(0, 1)


def plot_direction(
    df, model, ax=None, show_xlabel=True, show_ylabel=True, show_title=True
):
    df = (
        df[df["model"] == model]
        .groupby("correctAnswer")
        .agg(
            mean_score=("score", "mean"),
            interval=("score", prediction_interval),
            n_score=("score", "count"),
        )
        .sort_values(by="mean_score", ascending=False)
    )

    df["label"] = df.apply(
        lambda row: f" {row['mean_score']:.3f}±{row['interval']:.3f}",
        axis=1,
    )

    df = df.reset_index()

    print(df)

    # Create a figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.barh(
        df["correctAnswer"], df["mean_score"], xerr=df["interval"], capsize=5
    )

    if show_xlabel:
        ax.set_xlabel("Mean Score")
    if show_ylabel:
        ax.set_ylabel("Direction")
    if show_title:
        ax.set_title(model)

    split = 0.5
    dx = 0.05

    # Annotate each bar with the mean value
    for bar, mean, interval, label in zip(
        bars, df["mean_score"], df["interval"], df["label"]
    ):
        ax.text(
            mean - dx if mean > split else mean + dx,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="right" if mean > split else "left",
        )

    ax.set_xlim(0, 1)


def plot_direction_all_models(
    df,
    ax=None,
    show_xlabel=True,
    show_ylabel=True,
    show_title=True,
    show_xticklabels=True,
):

    models = sorted(df["model"].unique())  # Alphabetical order

    directions = sorted(df["correctAnswer"].unique(), reverse=True)

    custom_order = [
        "north-west",
        "south-west",
        "south-east",
        "north-east",
        "west",
        "east",
        "south",
        "north",
    ]

    # Convert to list and sort based on index in custom_order
    directions = sorted(
        df["correctAnswer"].unique(), key=lambda x: custom_order.index(x)
    )

    # Prepare plot axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    bar_height = 0.8 / len(models)
    y_positions = np.arange(len(directions))

    colors = plt.cm.tab10.colors
    offsets = np.linspace(-0.4 + bar_height / 2, 0.4 - bar_height / 2, len(models))[
        ::-1
    ]

    for i, model in enumerate(models):
        subdf = (
            df[df["model"] == model]
            .groupby("correctAnswer")
            .agg(mean_score=("score", "mean"))
            .reindex(directions)
        )

        y_offset = y_positions + offsets[i]

        ax.barh(
            y_offset,
            subdf["mean_score"],
            height=bar_height,
            label=model,
            color=colors[i % len(colors)],
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(directions)

    if show_xlabel:
        ax.set_xlabel("Mean Score")
    if show_ylabel:
        ax.set_ylabel("Direction")
    if show_title:
        ax.set_title("Model Comparison by Direction")
    if not show_xticklabels:
        ax.set_xticklabels([])

    ax.set_xlim(0, 1)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Adjust y-axis limits to remove excess whitespace
    min_y = (y_positions + offsets[-1] - bar_height / 2).min()
    max_y = (y_positions + offsets[0] + bar_height / 2).max()
    ax.set_ylim(min_y, max_y)

    return ax  # Return the axis in case further use is needed


def plot_direction_all_models_radar(
    df,
    ax=None,
    show_title=True,
):
    models = sorted(df["model"].unique())

    # Compass directions, ordered clockwise from North
    custom_order = [
        "north",
        "north-east",
        "east",
        "south-east",
        "south",
        "south-west",
        "west",
        "north-west",
    ]
    directions = custom_order
    num_vars = len(directions)

    # Compute angle for each direction (starting at North, going clockwise)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])  # Close the loop

    # Ensure ax is a polar subplot
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 6))

    colors = plt.cm.tab10.colors

    for i, model in enumerate(models):
        subdf = (
            df[df["model"] == model]
            .groupby("correctAnswer")
            .agg(mean_score=("score", "mean"))
            .reindex(directions)
        )

        values = subdf["mean_score"].fillna(0).values
        values = np.concatenate([values, [values[0]]])  # Close the loop

        ax.plot(angles, values, label=model, color=colors[i % len(colors)], linewidth=1)
        # ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.2)

    # Set the direction labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(directions)

    # Set 0 degrees at the top (North), and go clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0.1, 1.1, 0.1))
    ax.set_yticklabels([])

    # Make the grid lines finer
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if show_title:
        ax.set_title("Model Comparison by Direction (Radar Plot)", va="bottom")

    # ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    return ax


def plot_person_all_models_radar(
    df,
    ax=None,
    show_title=True,
):
    models = sorted(df["model"].unique())

    custom_order = [
        "I am",
        "You are",
        "He is",
        "She is",
        "We are",
        "They are",
    ]
    persons = custom_order
    num_vars = len(persons)

    # Compute angle for each direction (starting at North, going clockwise)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])  # Close the loop

    # Ensure ax is a polar subplot
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 6))

    colors = plt.cm.tab10.colors

    for i, model in enumerate(models):
        subdf = (
            df[df["model"] == model]
            .groupby("person")
            .agg(mean_score=("score", "mean"))
            .reindex(persons)
        )

        values = subdf["mean_score"].fillna(0).values
        values = np.concatenate([values, [values[0]]])  # Close the loop

        ax.plot(angles, values, label=model, color=colors[i % len(colors)], linewidth=1)
        # ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.2)

    # Set the direction labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(persons)

    # Set 0 degrees at the top (North), and go clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # ax.set_ylim(0, 1)
    # ax.set_yticklabels([])
    # ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    # ax.xaxis.grid(True, linestyle="--", alpha=0.5)

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0.1, 1.1, 0.1))
    ax.set_yticklabels([])

    # Make the grid lines finer
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if show_title:
        ax.set_title("Model Comparison by Person (Radar Plot)", va="bottom")

    # ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    return ax


def plot_locomotion_all_models_radar(
    df,
    ax=None,
    show_title=True,
):
    models = sorted(df["model"].unique())

    locomotions = sorted(df["locomotion"].unique(), reverse=True)

    num_vars = len(locomotions)

    # Compute angle for each direction (starting at North, going clockwise)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])  # Close the loop

    # Ensure ax is a polar subplot
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 6))

    colors = plt.cm.tab10.colors

    for i, model in enumerate(models):
        subdf = (
            df[df["model"] == model]
            .groupby("locomotion")
            .agg(mean_score=("score", "mean"))
            .reindex(locomotions)
        )

        values = subdf["mean_score"].fillna(0).values
        values = np.concatenate([values, [values[0]]])  # Close the loop

        ax.plot(angles, values, label=model, color=colors[i % len(colors)], linewidth=1)
        # ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.2)

    # Set the direction labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(locomotions)

    # Set 0 degrees at the top (North), and go clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0.1, 1.1, 0.1))
    ax.set_yticklabels([])

    # Make the grid lines finer
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if show_title:
        ax.set_title("Model Comparison by Locomotions (Radar Plot)", va="bottom")

    # ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    return ax


def plot_template_all_models_radar(
    df,
    ax=None,
    show_title=True,
):
    df["template"] = df["template"].str.split().str[0]

    models = sorted(df["model"].unique())

    templates = sorted(df["template"].unique())

    num_vars = len(templates)

    # Compute angle for each direction (starting at North, going clockwise)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])  # Close the loop

    # Ensure ax is a polar subplot
    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(8, 6))

    colors = plt.cm.tab10.colors

    for i, model in enumerate(models):
        subdf = (
            df[df["model"] == model]
            .groupby("template")
            .agg(mean_score=("score", "mean"))
            .reindex(templates)
        )

        values = subdf["mean_score"].fillna(0).values
        values = np.concatenate([values, [values[0]]])  # Close the loop

        ax.plot(angles, values, label=model, color=colors[i % len(colors)], linewidth=1)
        # ax.fill(angles, values, color=colors[i % len(colors)], alpha=0.2)

    # Set the direction labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(templates)

    # Set 0 degrees at the top (North), and go clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # ax.set_ylim(0, 1)
    # ax.set_yticklabels([])

    # ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    # ax.xaxis.grid(True, linestyle="--", alpha=0.5)

    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0.1, 1.1, 0.1))
    ax.set_yticklabels([])

    # Make the grid lines finer
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    if show_title:
        ax.set_title("Model Comparison by Templates (Radar Plot)", va="bottom")

    # ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

    return ax


def plot_person_all_models(
    df,
    ax=None,
    show_xlabel=True,
    show_ylabel=True,
    show_title=True,
    show_xticklabels=True,
):
    models = sorted(df["model"].unique())  # Alphabetical order

    persons = sorted(df["person"].unique(), reverse=True)

    # Prepare plot axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    bar_height = 0.8 / len(models)
    y_positions = np.arange(len(persons))

    colors = plt.cm.tab10.colors
    offsets = np.linspace(-0.4 + bar_height / 2, 0.4 - bar_height / 2, len(models))[
        ::-1
    ]

    for i, model in enumerate(models):
        subdf = (
            df[df["model"] == model]
            .groupby("person")
            .agg(mean_score=("score", "mean"))
            .reindex(persons)
        )

        y_offset = y_positions + offsets[i]

        ax.barh(
            y_offset,
            subdf["mean_score"],
            height=bar_height,
            label=model,
            color=colors[i % len(colors)],
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(persons)

    if show_xlabel:
        ax.set_xlabel("Mean Score")
    if show_ylabel:
        ax.set_ylabel("Person")
    if show_title:
        ax.set_title("Model Comparison by Direction")
    if not show_xticklabels:
        ax.set_xticklabels([])

    ax.set_xlim(0, 1)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Adjust y-axis limits to remove excess whitespace
    min_y = (y_positions + offsets[-1] - bar_height / 2).min()
    max_y = (y_positions + offsets[0] + bar_height / 2).max()
    ax.set_ylim(min_y, max_y)

    return ax  # Return the axis in case further use is needed


def plot_locomotion_all_models(
    df, ax=None, show_xlabel=True, show_ylabel=True, show_title=True
):
    models = sorted(df["model"].unique())  # Alphabetical order

    locomotions = sorted(df["locomotion"].unique(), reverse=True)

    # Prepare plot axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    bar_height = 0.8 / len(models)
    y_positions = np.arange(len(locomotions))

    colors = plt.cm.tab10.colors
    offsets = np.linspace(-0.4 + bar_height / 2, 0.4 - bar_height / 2, len(models))[
        ::-1
    ]

    for i, model in enumerate(models):
        subdf = (
            df[df["model"] == model]
            .groupby("locomotion")
            .agg(mean_score=("score", "mean"))
            .reindex(locomotions)
        )

        y_offset = y_positions + offsets[i]

        ax.barh(
            y_offset,
            subdf["mean_score"],
            height=bar_height,
            label=model,
            color=colors[i % len(colors)],
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(locomotions)

    if show_xlabel:
        ax.set_xlabel("Mean Score")
    if show_ylabel:
        ax.set_ylabel("Locomotion")
    if show_title:
        ax.set_title("Model Comparison by Direction")

    ax.set_xlim(0, 1)
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    # Adjust y-axis limits to remove excess whitespace
    min_y = (y_positions + offsets[-1] - bar_height / 2).min()
    max_y = (y_positions + offsets[0] + bar_height / 2).max()
    ax.set_ylim(min_y, max_y)

    return ax  # Return the axis in case further use is needed


def plot_model_legend(df, ax=None):

    models = sorted(df["model"].unique())  # Alphabetical order
    colors = plt.cm.tab10.colors

    if ax is None:
        fig, ax = plt.subplots()

    # Smaller markers
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=colors[i % len(colors)],
            markersize=8,  # Smaller size
            label=model,
        )
        for i, model in enumerate(models)
    ]

    # Smaller font and tighter spacing
    ax.legend(
        handles=handles,
        loc="center",
        fontsize=8,  # or a specific value like 8
        handlelength=1.0,  # shorten legend handle
        handletextpad=0.5,  # reduce space between handle and text
        borderpad=0.5,  # reduce padding inside legend box
        labelspacing=0.4,  # reduce vertical space between labels
    )
    ax.axis("off")  # Hide axis for legend-only subplot

    return ax


def plot_model_legend(df, ax=None):

    models = sorted(df["model"].unique())  # Alphabetical order
    colors = plt.cm.tab10.colors

    if ax is None:
        fig, ax = plt.subplots()

    # Smaller markers
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=colors[i % len(colors)],
            markersize=8,
            label=model,
        )
        for i, model in enumerate(models)
    ]

    # Center legend vertically in the axis
    ax.legend(
        handles=handles,
        loc="center",  # Anchor legend relative to left center
        bbox_to_anchor=(0.5, 0.3),  # Center of the axes
        fontsize=8,
        handlelength=1.0,
        handletextpad=0.5,
        borderpad=0.5,
        labelspacing=0.4,
    )
    ax.axis("off")  # Hide axis for legend-only subplot

    return ax


import textwrap


def plot_template_bullets(ax=None):
    bullet_points = [
        "T1. You are walking [south] along the [east] shore of a lake; in which direction is the lake?",
        "T2. You are walking [south] along the [east] shore of a lake and then turn around to head back in the direction you came from, in which direction is the lake?",
        "T3. You are walking [south] along the middle of the [east] side of a park; in which direction is the bandstand located in the centre of the park?",
        "T4. You are walking [east] along the [south] side of a road which runs [east to west]. In which direction is the road?",
        "T5. You are walking [south] along the [east] shore of the island. In which direction is the sea?",
        "T6. You are walking [south] along the [east] shore of an island and then turn around to head back in the direction you came from, in which direction is the sea?",
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Set desired wrap width in characters
    wrap_width = 60

    # Wrap each bullet point
    wrapped_points = [
        textwrap.fill(f"{point}", width=wrap_width) for point in bullet_points
    ]
    text = "\n".join(wrapped_points)  # Add spacing between bullets

    ax.axis("off")
    ax.text(0, 1, text, fontsize=8, va="top", ha="left", wrap=True)

    return ax


def plot_person(
    df, model, ax=None, show_xlabel=True, show_ylabel=True, show_title=True
):
    df = (
        df[df["model"] == model]
        .groupby("person")
        .agg(
            mean_score=("score", "mean"),
            n_score=("score", "count"),
            interval=("score", prediction_interval),
        )
        .sort_values(by="mean_score", ascending=False)
    )

    df["label"] = df.apply(
        lambda row: f" {row['mean_score']:.3f}±{row['interval']:.3f}",
        axis=1,
    )

    df = df.reset_index()

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.barh(df["person"], df["mean_score"], xerr=df["interval"], capsize=5)

    if show_xlabel:
        ax.set_xlabel("Accuracy")
    if show_ylabel:
        ax.set_ylabel("Person")
    if show_title:
        ax.set_title(model)

    split = 0.5
    dx = 0.05

    for bar, mean, interval, label in zip(
        bars, df["mean_score"], df["interval"], df["label"]
    ):
        ax.text(
            mean - dx if mean > split else mean + dx,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="right" if mean > split else "left",
        )

    ax.set_xlim(0, 1)


def plot_template(
    df, model, ax=None, show_xlabel=True, show_ylabel=True, show_title=True
):
    df = (
        df[df["model"] == model]
        .groupby("template")
        .agg(
            mean_score=("score", "mean"),
            n_score=("score", "count"),
            interval=("score", prediction_interval),
        )
        .sort_values(by="mean_score", ascending=False)
        .reset_index()
    )

    df["template"] = df["template"].str.split().str[0]

    df["label"] = df.apply(
        lambda row: f" {row['mean_score']:.3f}±{row['interval']:.3f}",
        axis=1,
    )

    df = df.reset_index()

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.barh(df["template"], df["mean_score"], xerr=df["interval"], capsize=5)

    if show_xlabel:
        ax.set_xlabel("Accuracy")
    if show_ylabel:
        ax.set_ylabel("Template")
    if show_title:
        ax.set_title(model)

    split = 0.5
    dx = 0.05

    for bar, mean, interval, label in zip(
        bars, df["mean_score"], df["interval"], df["label"]
    ):
        ax.text(
            mean - dx if mean > split else mean + dx,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="right" if mean > split else "left",
        )

    ax.set_xlim(0, 1)


def my_confusion_matrix(
    true_labels,
    predicted_labels,
    labels=None,
    normalize=None,
    title=None,
    ax=None,
    fontsize=10,
):
    """
    Plots a confusion matrix using matplotlib.

    Parameters:
    true_labels (list): List of true labels
    predicted_labels (list): List of predicted labels
    normalize (str or None): Whether to normalize the confusion matrix.
                             None (default): no normalization,
                             'true': normalize by true labels,
                             'pred': normalize by predicted labels,
                             'all': normalize by all.
    """
    if labels is None:
        labels = sorted(list(set(true_labels) | set(predicted_labels)))

    # Calculate the confusion matrix with specified labels to ensure string labels are used
    cm = confusion_matrix(
        true_labels, predicted_labels, labels=labels, normalize=normalize
    )

    # Create a confusion matrix display object with specified labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # Create a new figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    disp.plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    plt.xticks(rotation=90)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    if title:
        ax.set_title(title)

    # plt.xlabel("Predicted", fontsize=14)
    # plt.ylabel("Actual", fontsize=14)

    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    # plt.show()


def plot_reasoning_tokens(df, model_name, ax=None):

    true_count = df[df["score"] == True].shape[0]
    false_count = df[df["score"] == False].shape[0]

    if ax is None:
        fig, ax = plt.subplots()

    sns.violinplot(x=df["score"], y=df["reasoning_tokens"], order=[False, True], ax=ax)

    ax.set_xticklabels(["Incorrect", "Correct"], fontsize=8)
    ax.set_xlabel("Answer", fontsize=8)
    # ax.set_ylabel("Reasoning Tokens", fontsize=8)
    ax.set_ylabel("", fontsize=8)
    ax.set_title(f"{model_name}", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_ylim(0, 10000)

    # Optional: Add counts with 8pt font
    # ax.text(0, df_model["reasoning_tokens"].max(), f"n={false_count}",
    #         ha='right', va='top', fontsize=8, color='black')
    # ax.text(1, df_model["reasoning_tokens"].max(), f"n={true_count}",
    #         ha='right', va='top', fontsize=8, color='black')

    return ax


def plot_reasoning_tokens(df, model_name, ax=None):

    true_count = df[df["score"] == True].shape[0]
    false_count = df[df["score"] == False].shape[0]

    if ax is None:
        fig, ax = plt.subplots()

    sns.violinplot(x=df["score"], y=df["reasoning_tokens"], order=[False, True], ax=ax)
    ax.set_xticks([0, 1])  # Set tick positions explicitly
    ax.set_xticklabels(["Incorrect", "Correct"], fontsize=8)
    ax.set_xlabel("Answer", fontsize=8)
    # ax.set_ylabel("Reasoning Tokens", fontsize=8)
    ax.set_ylabel("", fontsize=8)
    ax.set_title(f"{model_name}", fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.set_ylim(0, 10000)

    # Optional: Add counts with 8pt font
    # ax.text(0, df_model["reasoning_tokens"].max(), f"n={false_count}",
    #         ha='right', va='top', fontsize=8, color='black')
    # ax.text(1, df_model["reasoning_tokens"].max(), f"n={true_count}",
    #         ha='right', va='top', fontsize=8, color='black')

    return ax


def plot_direction_violin(df, title, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.violinplot(x=df["correctAnswer"], y=df["reasoning_tokens"], ax=ax)
    ax.set_xlabel("CorrectAnswer")
    ax.set_ylabel("Reasoning tokens")
    ax.set_title(title)

    if ax is None:
        plt.show()


def plot_template_violin(df, title, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.violinplot(x=df["template"], y=df["reasoning_tokens"], ax=ax)
    ax.set_xlabel("Template")
    ax.set_ylabel("Reasoning tokens")
    ax.set_title(title)

    if ax is None:
        plt.show()


def get_qualitative_cmap(n_colors):
    """Choose an appropriate colormap based on number of colors needed."""
    if n_colors <= 8:
        cmap = get_cmap("Set2")
        return [to_hex(cmap(i / 8)) for i in range(n_colors)]
    elif n_colors <= 12:
        cmap = get_cmap("Set3")
        return [to_hex(cmap(i / 12)) for i in range(n_colors)]
    else:
        random.seed(10)
        return [
            distinctipy.get_hex(color) for color in distinctipy.get_colors(n_colors)
        ]


def plot_grouped_bar_chart(
    df,
    title="Grouped Bar Chart",
    xlabel=None,
    ylabel="Score",
    figsize=(16, 6),
    bar_width=0.05,
    group_gap=0.2,
    rotate_xticks=False,
):
    labels = df.index.tolist()
    categories = df.columns.tolist()

    x = np.arange(len(labels)) * (1 + group_gap)

    # Get perceptually distinct colors
    colors = get_qualitative_cmap(len(categories))

    fig, ax = plt.subplots(figsize=figsize)

    for i, category in enumerate(categories):
        offset = (i - len(categories) / 2) * bar_width + bar_width / 2
        ax.bar(
            x + offset, df[category], width=bar_width, label=category, color=colors[i]
        )

    ax.set_xlabel(df.index.name if xlabel is None else xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90 if rotate_xticks else 0)  # Apply rotation
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.show()
