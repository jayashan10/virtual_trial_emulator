## Original Task
1. Review our Simulation of clinical trials product

Its the pdf at docs/AI Clinical Trials Final.pdf


3. Understand how we did it

4. Find open source data set of real world clinical trials with verified real results which were achieved by life scientists in the past 

5. Use the input data for the past clinical trials (the same life scientists used for their research) and simulate these clinical trials

6. Compare your simulated outcome with real world outcomes (past outcomes).

So, what we have done till now is choose the dataset. So we have chosen the one here at :
https://data.projectdatasphere.org/projectdatasphere/html/content/149

You can browse the web to get more details aboout this datset.

## Task after brainstorming
For PDS310 (Colorectal Cancer Trial):
  1. Classification Task (like CAMP Study 1):
    • Predict which patients respond to panitumumab vs BSC
    • Classify responders (CR/PR) vs non-responders (SD/PD)
    • Predict adverse event risk (high vs low)
  2. Regression Task (like CAMP Study 2):
    • Predict PFS duration based on patient factors
    • Predict OS based on baseline characteristics
    • Model continuous biomarkers (CEA, LDH, etc.)
  3. Key Features to Include:
    • Demographics: Age, sex, race, ECOG status
    • Disease factors: Time since diagnosis, histological subtype, metastasis sites
    • Labs: Baseline LDH, CEA, albumin, hemoglobin, platelets
    • Molecular: KRAS/NRAS/BRAF mutation status
    • Treatment factors: Number of prior therapies, duration
  4. Potential Models:
    • Survival models (Cox PH, Random Survival Forests) - already in your code!
    • Classification for response prediction
    • Time-to-event for AE occurrence
    • Longitudinal models for biomarker trajectories

## Important things to remember about our codebase:
1. Always use uv as the pacakage manager for installing packages in python. 
2. Use the temp/ folder to create misc docs and summaries unless the directory is specified.
3. You should work honestly. Don't assume anything. Alwasys start by analysing the data by using python, look at how the data is structured, and then plan out on how to implement the features or tasks. 
4. Use the data dictionary, the excel file at allprovidedfiles_310 for pds310.
5. Never use emojis in my codebase .This is important.