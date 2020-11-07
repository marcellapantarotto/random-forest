# testes finais
X = dataset.loc[:, ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "Aspartate transaminase"]]
X = dataset.loc[:, ["Proteina C reativa mg/dL", "Urea", "ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Gamma-glutamyltransferase ", "Total Bilirubin"]]
X = dataset.loc[:, ["Sodium", "Alanine transaminase", "Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes"]]
X = dataset.loc[:, ["Patient age quantile", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Alanine transaminase", "Aspartate transaminase", "Hb saturation (arterial blood gases)", "Creatinine", "Potassium"]]

# melhores exames
X = dataset.loc[:, ["Patient age quantile", "Hematocrit", "Proteina C reativa mg/dL", "Platelets", "Proteina C reativa mg/dL", "Red blood Cells", "Monocytes", "Leukocytes", "Eosinophils", "Hemoglobin"]]

# todas colunas dos testes finais
X = dataset.loc[:, ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "Aspartate transaminase", "Proteina C reativa mg/dL", "Urea", "ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Gamma-glutamyltransferase ", "Total Bilirubin", "Sodium", "Alanine transaminase", "Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes", "Patient age quantile", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Alanine transaminase", "Aspartate transaminase", "Hb saturation (arterial blood gases)", "Creatinine", "Potassium"]]



### TESTANDOS TODAS AS COLUNAS

# colunas com strings
["Urine - Aspect", "Urine - Crystals", "Urine - Color"]    # ERRO

# problemas com NaN
["D-Dimer", "Mycoplasma pneumoniae", "Urine - Sugar", "Partial thromboplastin time (PTT) ", "Prothrombin time (PT), Activity"]



# colunas OK para teste
nomes = ["Patient age quantile", "Patient addmited to regular ward (1=yes, 0=no)", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes"]
nomes = ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Serum Glucose", "Respiratory Syncytial Virus"]
nomes = ["Urine - Leukocytes", "Influenza A", "Influenza B", "Parainfluenza 1", "CoronavirusNL63", "Rhinovirus/Enterovirus",  "Coronavirus HKU1", "Parainfluenza 3", "Chlamydophila pneumoniae", "Adenovirus"]
nomes = ["Parainfluenza 4", "Coronavirus229E", "CoronavirusOC43", "Inf A H1N1 2009", "Bordetella pertussis", "Metapneumovirus", "Parainfluenza 2", "Neutrophils", "Urea", "Proteina C reativa mg/dL"]
nomes = ["Creatinine", "Potassium", "Sodium", "Influenza B, rapid test", "Influenza A, rapid test", "Alanine transaminase", "Aspartate transaminase", "Gamma-glutamyltransferase ", "Total Bilirubin", "Direct Bilirubin"]
nomes = ["Indirect Bilirubin", "Alkaline phosphatase", "Ionized calcium ", "Strepto A", "Magnesium", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "Fio2 (venous blood gas analysis)"]
nomes = ["Total CO2 (venous blood gas analysis)", "pH (venous blood gas analysis)", "HCO3 (venous blood gas analysis)", "Rods #", "Segmented", "Promyelocytes", "Metamyelocytes", "Myelocytes", "Myeloblasts", "Urine - Esterase", "Urine - pH", "Urine - Yeasts", "Urine - Granular cylinders", "Urine - Hyaline cylinders", "Urine - Hemoglobin", "Urine - Bile pigments", "Urine - Ketone Bodies", "Urine - Nitrite", "Urine - Urobilinogen", "Urine - Protein","Urine - Density", "Relationship (Patient/Normal)", "International normalized ratio (INR)", "Urine - Red blood cells", "Lactic Dehydrogenase", "Creatine phosphokinase (CPK) ", "Ferritin"]
nomes = ["Vitamin B12", "Arterial Lactic Acid", "Lipase dosage", "pCO2 (arterial blood gas analysis)", "Base excess (arterial blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)", "pO2 (arterial blood gas analysis)", "Arteiral Fio2", "Phosphor", "ctO2 (arterial blood gas analysis)"]

# todas as colunas para um teste 
nomes = ["Patient age quantile", "Patient addmited to regular ward (1=yes, 0=no)", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes", 
        "Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Serum Glucose", "Respiratory Syncytial Virus", "Influenza A",
        "Influenza B", "Parainfluenza 1", "CoronavirusNL63", "Rhinovirus/Enterovirus",  "Coronavirus HKU1", "Parainfluenza 3", "Chlamydophila pneumoniae", "Adenovirus", "Parainfluenza 4", "Coronavirus229E", "CoronavirusOC43", "Inf A H1N1 2009", "Bordetella pertussis", "Metapneumovirus", 
        "Parainfluenza 2", "Neutrophils", "Urea", "Proteina C reativa mg/dL", "Creatinine", "Potassium", "Sodium", "Influenza B, rapid test", "Influenza A, rapid test", "Alanine transaminase", "Aspartate transaminase", "Gamma-glutamyltransferase ", "Total Bilirubin", "Direct Bilirubin", 
        "Indirect Bilirubin", "Alkaline phosphatase", "Ionized calcium ", "Strepto A", "Magnesium", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "Fio2 (venous blood gas analysis)",
        "Total CO2 (venous blood gas analysis)", "pH (venous blood gas analysis)", "HCO3 (venous blood gas analysis)", "Rods #", "Segmented", "Promyelocytes", "Metamyelocytes", "Myelocytes", "Myeloblasts", "Urine - Esterase", "Urine - pH", "Urine - Yeasts", "Urine - Granular cylinders",
        "Urine - Hyaline cylinders", "Urine - Hemoglobin", "Urine - Bile pigments", "Urine - Ketone Bodies", "Urine - Nitrite", "Urine - Urobilinogen", "Urine - Protein","Urine - Density", "Relationship (Patient/Normal)", "International normalized ratio (INR)", "Urine - Red blood cells",
        "Lactic Dehydrogenase", "Creatine phosphokinase (CPK) ", "Ferritin", "Vitamin B12", "Arterial Lactic Acid", "Lipase dosage", "pCO2 (arterial blood gas analysis)", "Base excess (arterial blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)",
        "HCO3 (arterial blood gas analysis)", "pO2 (arterial blood gas analysis)", "Arteiral Fio2", "Phosphor", "ctO2 (arterial blood gas analysis)", "Urine - Leukocytes"]




        #############################################################################################################################################################################################################################################################################################################################################################################

        # X = dataset.loc[:, ["Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Alanine transaminase", "Aspartate transaminase", "Lactic Dehydrogenase", "Albumin", "Hb saturation (arterial blood gases)", "pCO2 (arterial blood gas analysis)"]]
# X = dataset.loc[:, ["Patient age quantile", "Patient addmited to regular ward (1=yes, 0=no)", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes"]]
# X = dataset.loc[:, ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Serum Glucose", "Respiratory Syncytial Virus"]]
# X = dataset.loc[:, ["Influenza A", "Influenza B", "Parainfluenza 1", "CoronavirusNL63", "Rhinovirus/Enterovirus", "Mycoplasma pneumoniae", "Coronavirus HKU1", "Parainfluenza 3", "Chlamydophila pneumoniae", "Adenovirus"]]    # ERRO
# X = dataset.loc[:, ["Parainfluenza 4", "Coronavirus229E", "CoronavirusOC43", "Inf A H1N1 2009", "Bordetella pertussis", "Metapneumovirus", "Parainfluenza 2", "Neutrophils", "Urea", "Proteina C reativa mg/dL"]]
# X = dataset.loc[:, ["Creatinine", "Potassium", "Sodium", "Influenza B, rapid test", "Influenza A, rapid test", "Alanine transaminase", "Aspartate transaminase", "Gamma-glutamyltransferase ", "Total Bilirubin", "Direct Bilirubin"]]
# X = dataset.loc[:, ["Indirect Bilirubin", "Alkaline phosphatase", "Ionized calcium ", "Strepto A", "Magnesium", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "Fio2 (venous blood gas analysis)"]]
# X = dataset.loc[:, ["Total CO2 (venous blood gas analysis)", "pH (venous blood gas analysis)", "HCO3 (venous blood gas analysis)", "Rods #", "Segmented", "Promyelocytes", "Metamyelocytes", "Myelocytes", "Myeloblasts", "Urine - Esterase"]]
# X = dataset.loc[:, ["Urine - Aspect", "Urine - pH", "Urine - Hemoglobin", "Urine - Bile pigments", "Urine - Ketone Bodies", "Urine - Nitrite", "Urine - Density", "Urine - Urobilinogen", "Urine - Protein", "Urine - Sugar"]]
# X = dataset.loc[:, ["Urine - Leukocytes", "Urine - Crystals", "Urine - Red blood cells", "Urine - Hyaline cylinders", "Urine - Granular cylinders", "Urine - Yeasts", "Urine - Color", "Partial thromboplastin time (PTT) ", "Relationship (Patient/Normal)", "International normalized ratio (INR)"]]
# X = dataset.loc[:, ["Lactic Dehydrogenase", "Prothrombin time (PT), Activity", "Vitamin B12", "Creatine phosphokinase (CPK) ", "Ferritin", "Arterial Lactic Acid", "Lipase dosage", "D-Dimer", "Albumin", "Hb saturation (arterial blood gases)"]]
# X = dataset.loc[:, ["pCO2 (arterial blood gas analysis)", "Base excess (arterial blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)", "pO2 (arterial blood gas analysis)", "Arteiral Fio2", "Phosphor", "ctO2 (arterial blood gas analysis)"]]
# X = dataset.loc[:, ["Neutrophils", "Urea", "Proteina C reativa mg/dL", "Creatinine", "Potassium", "Sodium", "Alanine transaminase", "Aspartate transaminase", "Gamma-glutamyltransferase ", "Total Bilirubin"]]
# X = dataset.loc[:, ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Serum Glucose", "pCO2 (arterial blood gas analysis)"]]
# X = dataset.loc[:, ["Patient age quantile", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes", "Indirect Bilirubin", "Alkaline phosphatase"]]
# X = dataset.loc[:, ["Ionized calcium ", "Magnesium", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "Base excess (arterial blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)"]]
# X = dataset.loc[:, ["pO2 (arterial blood gas analysis)", "ctO2 (arterial blood gas analysis)", "Neutrophils", "Urea", "Proteina C reativa mg/dL", "Creatinine", "Potassium", "Sodium", "Alanine transaminase", "Aspartate transaminase"]]
# X = dataset.loc[:, ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Gamma-glutamyltransferase ", "Total Bilirubin"]]
# X = dataset.loc[:, ["Ionized calcium ", "Magnesium", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)", "Aspartate transaminase"]]
# X = dataset.loc[:, ["ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)", "Aspartate transaminase"]]
# X = dataset.loc[:, ["Urea", "ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "Aspartate transaminase"]]
# X = dataset.loc[:, ["Proteina C reativa mg/dL", "Urea", "ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "Aspartate transaminase"]]
# X = dataset.loc[:, ["Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Alanine transaminase", "Aspartate transaminase", "Lactic Dehydrogenase", "Hb saturation (arterial blood gases)", "Creatinine", "Potassium"]]
# X = dataset.loc[:, ["Sodium", "Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Alanine transaminase", "Aspartate transaminase", "Hb saturation (arterial blood gases)", "Creatinine", "Potassium"]]
# X = dataset.loc[:, ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "Aspartate transaminase"]]
# X = dataset.loc[:, ["Proteina C reativa mg/dL", "Urea", "ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Gamma-glutamyltransferase ", "Total Bilirubin"]]
# X = dataset.loc[:, ["Alanine transaminase", "Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes", "Alkaline phosphatase"]]
# X = dataset.loc[:, ["Sodium", "Alanine transaminase", "Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes"]]
# X = dataset.loc[:, ["Patient age quantile", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Alanine transaminase", "Aspartate transaminase", "Hb saturation (arterial blood gases)", "Creatinine", "Potassium"]]
# X = dataset.loc[:, ["Patient age quantile", "Hematocrit", "Proteina C reativa mg/dL", "Platelets", "Proteina C reativa mg/dL", "Red blood Cells", "Monocytes", "Leukocytes", "Eosinophils", "Hemoglobin"]]
# X = dataset.loc[:, ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "Aspartate transaminase", "Proteina C reativa mg/dL", "Urea", "ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Gamma-glutamyltransferase ", "Total Bilirubin", "Sodium", "Alanine transaminase", "Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes", "Patient age quantile", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Alanine transaminase", "Aspartate transaminase", "Hb saturation (arterial blood gases)", "Creatinine", "Potassium"]]
# X = dataset.loc[:, ["Influenza A", "Influenza B", "Parainfluenza 1", "CoronavirusNL63", "Rhinovirus/Enterovirus",  "Coronavirus HKU1", "Parainfluenza 3", "Chlamydophila pneumoniae", "Adenovirus"]]

# nomes = ["Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Alanine transaminase", "Aspartate transaminase", "Lactic Dehydrogenase", "Albumin", "Hb saturation (arterial blood gases)", "pCO2 (arterial blood gas analysis)"]
# nomes = ["Patient age quantile", "Patient addmited to regular ward (1=yes, 0=no)", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Platelets", "Mean platelet volume", "Red blood Cells", "Lymphocytes"]
# nomes = ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Serum Glucose", "Respiratory Syncytial Virus"]
# nomes = ["Influenza A", "Influenza B", "Parainfluenza 1", "CoronavirusNL63", "Rhinovirus/Enterovirus", "Mycoplasma pneumoniae", "Coronavirus HKU1", "Parainfluenza 3", "Chlamydophila pneumoniae", "Adenovirus"]    # ERRO
# nomes = ["Parainfluenza 4", "Coronavirus229E", "CoronavirusOC43", "Inf A H1N1 2009", "Bordetella pertussis", "Metapneumovirus", "Parainfluenza 2", "Neutrophils", "Urea", "Proteina C reativa mg/dL"]
# nomes = ["Creatinine", "Potassium", "Sodium", "Influenza B, rapid test", "Influenza A, rapid test", "Alanine transaminase", "Aspartate transaminase", "Gamma-glutamyltransferase ", "Total Bilirubin", "Direct Bilirubin"]
# nomes = ["Indirect Bilirubin", "Alkaline phosphatase", "Ionized calcium ", "Strepto A", "Magnesium", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "Fio2 (venous blood gas analysis)"]
# nomes = ["Total CO2 (venous blood gas analysis)", "pH (venous blood gas analysis)", "HCO3 (venous blood gas analysis)", "Rods #", "Segmented", "Promyelocytes", "Metamyelocytes", "Myelocytes", "Myeloblasts", "Urine - Esterase"]    # ERRO
# nomes = ["Urine - Aspect", "Urine - pH", "Urine - Hemoglobin", "Urine - Bile pigments", "Urine - Ketone Bodies", "Urine - Nitrite", "Urine - Density", "Urine - Urobilinogen", "Urine - Protein", "Urine - Sugar"]    # ERRO
# nomes = ["Urine - Leukocytes", "Urine - Crystals", "Urine - Red blood cells", "Urine - Hyaline cylinders", "Urine - Granular cylinders", "Urine - Yeasts", "Urine - Color", "Partial thromboplastin time (PTT) ", "Relationship (Patient/Normal)", "International normalized ratio (INR)"]    # ERRO
# nomes = ["Lactic Dehydrogenase", "Prothrombin time (PT), Activity", "Vitamin B12", "Creatine phosphokinase (CPK) ", "Ferritin", "Arterial Lactic Acid", "Lipase dosage", "D-Dimer", "Albumin", "Hb saturation (arterial blood gases)"]  # ERRO
# nomes = ["pCO2 (arterial blood gas analysis)", "Base excess (arterial blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)", "pO2 (arterial blood gas analysis)", "Arteiral Fio2", "Phosphor", "ctO2 (arterial blood gas analysis)"]
# nomes = ["Neutrophils", "Urea", "Proteina C reativa mg/dL", "Creatinine", "Potassium", "Sodium", "Alanine transaminase", "Aspartate transaminase", "Gamma-glutamyltransferase ", "Total Bilirubin"]
# nomes = ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Serum Glucose", "pCO2 (arterial blood gas analysis)"]
# nomes = ["Patient age quantile", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes", "Indirect Bilirubin", "Alkaline phosphatase"]
# nomes = ["Ionized calcium ", "Magnesium", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "Base excess (arterial blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)"]
# nomes = ["pO2 (arterial blood gas analysis)", "ctO2 (arterial blood gas analysis)", "Neutrophils", "Urea", "Proteina C reativa mg/dL", "Creatinine", "Potassium", "Sodium", "Alanine transaminase", "Aspartate transaminase"]
# nomes = ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Gamma-glutamyltransferase ", "Total Bilirubin"]
# nomes = ["Ionized calcium ", "Magnesium", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)", "Aspartate transaminase"]
# nomes = ["ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "HCO3 (arterial blood gas analysis)", "Aspartate transaminase"]
# nomes = ["Urea", "ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pO2 (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "Aspartate transaminase"]
# nomes = ["Proteina C reativa mg/dL", "Urea", "ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "Aspartate transaminase"]
# nomes = ["Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Alanine transaminase", "Aspartate transaminase", "Lactic Dehydrogenase", "Hb saturation (arterial blood gases)", "Creatinine", "Potassium"]
# nomes = ["Sodium", "Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Alanine transaminase", "Aspartate transaminase", "Hb saturation (arterial blood gases)", "Creatinine", "Potassium"]
# nomes = ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "Aspartate transaminase"]
# nomes = ["Proteina C reativa mg/dL", "Urea", "ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Gamma-glutamyltransferase ", "Total Bilirubin"]
# nomes = ["Alanine transaminase", "Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes", "Alkaline phosphatase"]
# nomes = ["Sodium", "Alanine transaminase", "Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes"]
# nomes = ["Patient age quantile", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Alanine transaminase", "Aspartate transaminase", "Hb saturation (arterial blood gases)", "Creatinine", "Potassium"]
# nomes = ["Patient age quantile", "Hematocrit", "Proteina C reativa mg/dL", "Platelets", "Proteina C reativa mg/dL", "Red blood Cells", "Monocytes", "Leukocytes", "Eosinophils", "Hemoglobin"]
# nomes = ["Mean corpuscular hemoglobin concentration (MCHC)", "Leukocytes", "Basophils", "Mean corpuscular hemoglobin (MCH)", "Eosinophils", "Hb saturation (venous blood gas analysis)", "Base excess (venous blood gas analysis)", "pH (arterial blood gas analysis)", "Total CO2 (arterial blood gas analysis)", "Aspartate transaminase", "Proteina C reativa mg/dL", "Urea", "ctO2 (arterial blood gas analysis)", "Neutrophils", "pCO2 (venous blood gas analysis)", "Mean corpuscular volume (MCV)", "Monocytes", "Red blood cell distribution width (RDW)", "Gamma-glutamyltransferase ", "Total Bilirubin", "Sodium", "Alanine transaminase", "Hemoglobin", "Red blood Cells", "Lymphocytes", "Proteina C reativa mg/dL", "Platelets", "Mean platelet volume ", "Red blood Cells", "Lymphocytes", "Patient age quantile", "Patient addmited to semi-intensive unit (1=yes, 0=no)", "Patient addmited to intensive care unit (1=yes, 0=no)", "Hematocrit", "Hemoglobin", "Alanine transaminase", "Aspartate transaminase", "Hb saturation (arterial blood gases)", "Creatinine", "Potassium"]
# nomes = ["Influenza A", "Influenza B", "Parainfluenza 1", "CoronavirusNL63", "Rhinovirus/Enterovirus",  "Coronavirus HKU1", "Parainfluenza 3", "Chlamydophila pneumoniae", "Adenovirus"]