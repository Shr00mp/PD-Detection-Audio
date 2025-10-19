import pandas as pd

# demographics contains audio files labeled with HC/PwPD, age and sex
demographics = pd.read_csv("demographic-analysis\Demographics_age_sex.csv")


# Counts the number of PwPD and HC
num_pd = (demographics["Label"] == "PwPD").sum()
num_hc = (demographics["Label"] == "HC").sum()

# Counts number of male and female patients with and w/o PD
num_pd_m = ((demographics["Label"] == "PwPD") & (demographics["Sex"] == "M")).sum()
num_pd_f = ((demographics["Label"] == "PwPD") & (demographics["Sex"] == "F")).sum()
num_hc_m = ((demographics["Label"] == "HC") & (demographics["Sex"] == "M")).sum()
num_hc_f = ((demographics["Label"] == "HC") & (demographics["Sex"] == "F")).sum()

# Create DataFrames grouped by the label for age
age_stats = demographics.groupby("Label")["Age"].agg(["mean", "std"]).round(0)
# Gets mean age and age std for PwPD and HC labels
pd_mean_age = age_stats.loc["PwPD", "mean"]
pd_std_age = age_stats.loc["PwPD", "std"]
hc_mean_age = age_stats.loc["HC", "mean"]
hc_std_age = age_stats.loc["HC", "std"]

# # Print data in a table (pandas dataframe)
# summary_data = {
#     "PD": {
#         "total": num_pd,
#         "mean age": int(pd_mean_age),
#         "age std": int(pd_std_age),
#         "number M": num_pd_m,
#         "number F": num_pd_f
#     },
#     "HC" : {
#         "total": num_hc,
#         "mean age": int(hc_mean_age),
#         "age std": int(hc_std_age),
#         "number M": num_hc_m,
#         "number F": num_hc_f
#     }
# }

# df_summary = pd.DataFrame(summary_data)
# print(df_summary)


# Print data in a table (pandas dataframe)
summary_data = {
    "PD": {
        "total": num_pd,
        "mean age": int(pd_mean_age),
        "age std": int(pd_std_age),
        "percentage M": str(int((num_pd_m/num_pd)*100)) + "%",
        "percentage F": str(int((num_pd_f/num_pd)*100)) + "%"
    },
    "HC" : {
        "total": num_hc,
        "mean age": int(hc_mean_age),
        "age std": int(hc_std_age),
        "percentage M": str(int((num_hc_m/num_hc)*100)) + "%",
        "percentage F": str(int((num_hc_f/num_hc)*100)) + "%"
    }
}

df_summary = pd.DataFrame(summary_data)
print(df_summary)
