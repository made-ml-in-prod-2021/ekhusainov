"""Здесь просто генерируем отчёт в html"""
import pandas as pd
from pandas_profiling import ProfileReport

PATH_TO_DATASET = "../data/raw/heart.csv"
OUTPUT_REPORT_HTML = "profile_report.html"


def main():
    data = pd.read_csv(PATH_TO_DATASET)
    profile = ProfileReport(data)
    profile.to_file(output_file=OUTPUT_REPORT_HTML)


if __name__ == "__main__":
    main()
