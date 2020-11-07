# testes finais
X = dataset.loc[:, ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "Aspartate transaminase"]]
X = dataset.loc[:, ["Proteina C reativa mg/dL", "Urea", "ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Gamma-glutamyltransferase ", "Total Bilirubin"]]
X = dataset.loc[:, ["Sodium", "Alanine transaminase", "Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes"]]
X = dataset.loc[:, ["Patient age quantile", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Alanine transaminase", "Aspartate transaminase", "Hb saturation (arterial blood gases)", "Creatinine", "Potassium"]]

# melhores exames
X = dataset.loc[:, ["Patient age quantile", "Hematocrit", "Proteina C reativa mg/dL", "Platelets", "Proteina C reativa mg/dL", "Red blood Cells", "Monocytes", "Leukocytes", "Eosinophils", "Hemoglobin"]]