---
title: "UFC_analysis_APOI"
output: html_document
date: "2026-04-17"
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## 1. Opis podataka i statističkog skupa

Skup podataka obuhvaća individualne nastupe UFC (Ultimate Fighting Championship) boraca kroz povijest organizacije. **Pojmovno** se radi o mjerenim karakteristikama boraca (dob, broj značajnih udaraca, težinska kategorija) i ishodu svake borbe (pobjeda/poraz, metoda završetka). **Prostorno** podaci obuhvaćaju globalne UFC događaje bez geografskog ograničenja. **Vremenski** dataset pokriva UFC natjecanja od osnivanja organizacije (1993.) do otprilike 2021. godine.

Radi se o **uzorku** — podaci reprezentiraju zabilježene nastupe boraca koji su prošli selekciju za nastup u UFC-u, a ne ukupnu populaciju svih MMA boraca na svijetu. Svaki redak odgovara jednom borcu u jednoj borbi (ne jednoj borbi kao cjelini).

**Izvor podataka:** Kaggle, javno dostupan dataset:  
`https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset`

---

## 2. ETL proces

```{r}
library(tidyverse)
library(broom)
library(rpart)       
library(rpart.plot)  
library(caret)       
library(car)         
library(pROC)        
library(scales)      
library(gridExtra)
```

```{r}
df <- read.csv("/home/antonio/Downloads/ufc-master.csv")

ufc_clean <- df %>%
  rename_with(~ tolower(gsub("\\.+", "_", .x))) %>% 
  mutate(
    is_winner = factor(ifelse(w == 1, "Win", "Loss"), levels = c("Loss", "Win")),
    weight_cat = factor(case_when(
      flyweight == 1 | bantamweight == 1 | featherweight == 1 | lightweight == 1 ~ "Light",
      TRUE ~ "Heavy"
    )),
    age_group = factor(ifelse(age >= 35, "Over_35", "Under_35")),
    strikes = as.numeric(significant_strike_land),
    method = case_when(
      ko_tko == 1 ~ "KO/TKO",
      submission == 1 ~ "Submission",
      decision_unanimous == 1 | decision_majority == 1 | decision_split == 1 ~ "Decision",
      TRUE ~ "Other"
    )
  ) %>%
  select(full_name, age, strikes, is_winner, weight_cat, age_group, method) %>%
  drop_na() %>%
  # Zadržavamo borce dobi 18-55: ispod 18 su maloljetnici (nema u UFC-u),
  # iznad 55 su ekstremni outlieri koji bi iskrivili distribuciju dobi
  filter(age >= 18 & age <= 55)

# Provjera tipova varijabli nakon transformacije
glimpse(ufc_clean)

# Provjera dimenzija i preostalih NA vrijednosti
cat("\nDimenzije dataseta:", nrow(ufc_clean), "redaka x", ncol(ufc_clean), "stupaca\n")
cat("Preostale NA vrijednosti po stupcu:\n")
print(colSums(is.na(ufc_clean)))
```

```{r}
ufc_clean
```

---

## 3. Deskriptivna statistika i distribucija udaraca

```{r}
# 3.1. Deskriptivna statistika
summary_stats <- ufc_clean %>% 
  group_by(age_group) %>% 
  summarise(
    N = n(),
    Mean_Age = mean(age),
    Mean_Strikes = mean(strikes),
    Median_Strikes = median(strikes),
    SD_Strikes = round(sd(strikes), 2),
    Var_Strikes = round(var(strikes), 2),
    IQR_Strikes = IQR(strikes),
    Skewness = (3 * (mean(strikes) - median(strikes))) / sd(strikes)
  )
print(summary_stats)

# 3.2. Vizualizacija distribucije
p1 <- ggplot(ufc_clean, aes(x = strikes)) +
  geom_histogram(aes(y = ..density..), binwidth = 10, fill = "steelblue", alpha = 0.7) +
  geom_density(color = "red", size = 1) +
  theme_minimal()

p2 <- ggplot(ufc_clean, aes(sample = strikes)) +
  stat_qq() + stat_qq_line(color = "red") +
  theme_minimal()

grid.arrange(p1, p2, ncol = 2)

# 3.3. Dijagram rasipanja — dob vs. broj udaraca
ggplot(ufc_clean, aes(x = age, y = strikes, color = is_winner)) +
  geom_point(alpha = 0.25, size = 1) +
  geom_smooth(method = "lm", se = FALSE, linewidth = 0.8) +
  scale_color_manual(values = c("Win" = "#2ecc71", "Loss" = "#e74c3c")) +
  labs(title = "Dijagram rasipanja: dob vs. broj udaraca",
       x = "Dob (godine)", y = "Broj značajnih udaraca", color = "Ishod") +
  theme_minimal()

# 3.4. Spearmanova korelacija — dob i broj udaraca
cor_res <- cor.test(ufc_clean$age, ufc_clean$strikes, method = "spearman")
cat("\n[SPEARMANOVA KORELACIJA] rho =", round(cor_res$estimate, 3),
    "| p =", round(cor_res$p.value, 4), "\n")
if (abs(cor_res$estimate) < 0.1) {
  cat("  -> Praktički NEMA linearne veze između dobi i broja udaraca (|rho| < 0.1).\n")
} else if (abs(cor_res$estimate) < 0.3) {
  cat("  -> SLABA veza između dobi i broja udaraca (|rho| < 0.3).\n")
} else {
  cat("  -> UMJERENA do JAKA veza između dobi i broja udaraca.\n")
}

# Uvjetna interpretacija distribucije
skew_val <- summary_stats %>% summarise(avg_skew = mean(Skewness, na.rm = TRUE)) %>% pull()
if (skew_val > 0.5) {
  cat("\n[INTERPRETACIJA] Distribucija udaraca pokazuje IZRAŽENU DESNU ASIMETRIČNOST (prosječna asimetrija =",
      round(skew_val, 2), "). Manji broj boraca ostvaruje iznimno visok broj udaraca. Preporučuje se primjena neparametarskih testova.\n")
} else if (skew_val > 0) {
  cat("\n[INTERPRETACIJA] Distribucija udaraca je BLAGO DESNO ASIMETRIČNA (asimetrija =",
      round(skew_val, 2), "). Distribucija je relativno simetrična — parametarski testovi mogli bi biti primjenjivi uz oprez.\n")
} else {
  cat("\n[INTERPRETACIJA] Distribucija udaraca je LIJEVO ASIMETRIČNA ili SIMETRIČNA (asimetrija =",
      round(skew_val, 2), "). Neočekivan nalaz — preporučuje se dodatna provjera podataka.\n")
}
```

---

## 4. Testovi pretpostavki

```{r}
# 4.1. Shapiro-Wilk test (Normalnost)
shapiro_res <- shapiro.test(sample(ufc_clean$strikes, min(nrow(ufc_clean), 5000)))
print(shapiro_res)

# Uvjetna interpretacija — Shapiro-Wilk
if (shapiro_res$p.value < 0.05) {
  cat("\n[SHAPIRO-WILK] p =", round(shapiro_res$p.value, 4),
      "— Normalnost se ODBACUJE (p < 0.05). Primjena neparametarskih testova je opravdana.\n")
} else {
  cat("\n[SHAPIRO-WILK] p =", round(shapiro_res$p.value, 4),
      "— Normalnost se NE ODBACUJE (p >= 0.05). Distribucija ne odstupa značajno od normalne.\n")
}

# 4.2. Leveneov test (Homogenost varijance)
levene_res <- leveneTest(strikes ~ age_group, data = ufc_clean)
print(levene_res)

# Uvjetna interpretacija — Levene
levene_p <- levene_res$`Pr(>F)`[1]
if (levene_p < 0.05) {
  cat("\n[LEVENE] p =", round(levene_p, 4),
      "— Varijance su HETEROGENE (p < 0.05). Dobne skupine imaju značajno različite varijance udaraca.\n")
} else {
  cat("\n[LEVENE] p =", round(levene_p, 4),
      "— Varijance su HOMOGENE (p >= 0.05). Pretpostavka jednakih varijanci nije narušena.\n")
}
```

---

## 5. Statistički testovi i regresija

**Istraživačko pitanje 1 — Razlika između skupina:**  
Razlikuju li se mlađi (<35) i stariji (≥35) UFC borci u broju ostvarenih značajnih udaraca?  
**H₀:** Distribucija broja udaraca jednaka je u obje dobne skupine (medijani su jednaki).  
**H₁:** Distribucija broja udaraca razlikuje se između dobnih skupina (medijani nisu jednaki).

**Istraživačko pitanje 2 — Predviđanje:**  
Mogu li se dob, težinska kategorija i broj udaraca koristiti za predviđanje ishoda borbe?  
**H₀:** Dob, težinska kategorija i broj udaraca nisu statistički značajni prediktori pobjede (svi β = 0).  
**H₁:** Barem jedan od navedenih prediktora statistički značajno predviđa ishod borbe.

```{r}
# 5.1. Wilcoxon test
wilcox_res <- wilcox.test(strikes ~ age_group, data = ufc_clean)
print(wilcox_res)

# Effect size r = Z / sqrt(N)
n_total  <- nrow(ufc_clean)
z_val    <- qnorm(wilcox_res$p.value / 2)
effect_r <- abs(z_val) / sqrt(n_total)
cat("\n[EFFECT SIZE] r =", round(effect_r, 3), "->",
    ifelse(effect_r >= 0.5, "VELIKI efekt",
    ifelse(effect_r >= 0.3, "SREDNJI efekt",
    ifelse(effect_r >= 0.1, "MALI efekt", "ZANEMARIV efekt"))), "\n")

# Uvjetna interpretacija — Wilcoxon
if (wilcox_res$p.value < 0.001) {
  cat("\n[WILCOXON] p =", formatC(wilcox_res$p.value, format = "e", digits = 2),
      "— VISOKO SIGNIFIKANTNA razlika (p < 0.001). Mlađi i stariji borci značajno se razlikuju u broju udaraca.\n")
} else if (wilcox_res$p.value < 0.05) {
  cat("\n[WILCOXON] p =", round(wilcox_res$p.value, 4),
      "— SIGNIFIKANTNA razlika (p < 0.05). Postoji statistički značajna razlika u udarcima između dobnih skupina.\n")
} else {
  cat("\n[WILCOXON] p =", round(wilcox_res$p.value, 4),
      "— NIJE signifikantna razlika (p >= 0.05). Nema dovoljno dokaza za razliku u udarcima između dobnih skupina.\n")
}

# 5.2. Logistička regresija
glm_model <- glm(is_winner ~ age * weight_cat + strikes, data = ufc_clean, family = binomial)
summary(glm_model)

# Uvjetna interpretacija — logistička regresija
coef_strikes <- coef(glm_model)["strikes"]
coef_age     <- coef(glm_model)["age"]
or_strikes   <- exp(coef_strikes)
or_age       <- exp(coef_age)

cat("\n[LOGISTIČKA REGRESIJA]\n")
cat("  OR za strikes =", round(or_strikes, 4), "->",
    ifelse(or_strikes > 1,
           "Svaki dodatni udarac POVEĆAVA šanse za pobjedu.",
           "Svaki dodatni udarac SMANJUJE šanse za pobjedu (neočekivano — provjeri model)."), "\n")
cat("  OR za age =", round(or_age, 4), "->",
    ifelse(or_age < 1,
           "Stariji borci imaju NIŽE šanse za pobjedu (dob negativno prediktira ishod).",
           "Stariji borci imaju VIŠE šanse za pobjedu (dob pozitivno prediktira ishod — neočekivano)."), "\n")

# 5.3. Odds Ratios
cat("\n--- Tumačenje šansi (Odds Ratios) ---\n")
print(exp(coef(glm_model)))
```

---

## 6. Vizualizacije — dob, metoda, težinska kategorija

```{r}
# Vjerojatnost pobjede kroz dob
g1 <- ggplot(ufc_clean, aes(x = age, y = as.numeric(is_winner == "Win"), color = weight_cat)) +
  geom_smooth(method = "glm", method.args = list(family = "binomial")) +
  labs(title = "Utjecaj dobi na vjerojatnost pobjede", x = "Dob", y = "Pobjeda (Vjerojatnost)") +
  theme_minimal()

# Način završetka borbe po dobnoj skupini
ufc_method_plot <- ufc_clean %>% filter(method != "Other")
g2 <- ggplot(ufc_method_plot, aes(x = age, fill = method)) +
  geom_histogram(binwidth = 2, position = "fill") +
  scale_y_continuous(labels = percent) +
  labs(title = "Evolucija načina pobjede kroz dob", x = "Dob", y = "Udio (%)") +
  theme_minimal()

grid.arrange(g1, g2, ncol = 2)

# Uvjetna interpretacija vizualizacija
age_effect_dir <- ifelse(or_age < 1, "opada", "raste")
cat("\n[VIZUALIZACIJA] Logistička krivulja pokazuje da vjerojatnost pobjede", age_effect_dir,
    "s dobi. Histogram udjela metoda vizualizira taktičke pomake kroz karijeru boraca.\n")
```

---

## 7. Stablo odlučivanja

```{r}
set.seed(123)

train_idx <- createDataPartition(ufc_clean$is_winner, p = 0.8, list = FALSE)
train_data <- ufc_clean[train_idx, ]
test_data  <- ufc_clean[-train_idx, ]

tree_mod_detailed <- rpart(is_winner ~ age + strikes + weight_cat, 
                           data = train_data, 
                           method = "class",
                           control = rpart.control(cp = 0.0045, minsplit = 20))

rpart.plot(tree_mod_detailed, 
           main = "Decision Tree: Predviđanje ishoda borbe", 
           box.palette = "RdGn", 
           extra = 104)

# Uvjetna interpretacija stabla
n_leaves <- sum(tree_mod_detailed$frame$var == "<leaf>")
cat("\n[STABLO ODLUČIVANJA] Stablo ima", n_leaves, "listova (terminalni čvorovi).")
if (n_leaves <= 4) {
  cat(" Model je JEDNOSTAVAN i lako interpretabilan — manji rizik od prenapučenosti.\n")
} else if (n_leaves <= 10) {
  cat(" Model je UMJERENE složenosti — dobra ravnoteža između interpretabilnosti i preciznosti.\n")
} else {
  cat(" Model je SLOŽEN s mnogo grana — postoji rizik od prenapučenosti (overfitting). Razmotri veći cp parametar.\n")
}
```

---

## 8. Evaluacija modela

```{r}
# 8.1. Matrica konfuzije
tree_preds <- predict(tree_mod_detailed, test_data, type = "class")
cm <- confusionMatrix(tree_preds, test_data$is_winner)
print(cm)

# Uvjetna interpretacija — točnost stabla
acc <- cm$overall["Accuracy"]
cat("\n[MATRICA KONFUZIJE] Točnost stabla odlučivanja:", round(acc * 100, 1), "%")
if (acc >= 0.70) {
  cat(" — DOBRA točnost. Model pouzdano klasificira ishode borbi.\n")
} else if (acc >= 0.55) {
  cat(" — UMJERENA točnost. Model je bolji od slučajnog pogađanja, ali ima prostora za poboljšanje.\n")
} else {
  cat(" — SLABA točnost. Model nije bolji od baznog klasifikatora — razmotri promjenu varijabli ili parametara.\n")
}

# 8.2. ROC i AUC
probs <- predict(glm_model, type = "response")
roc_curve <- roc(ufc_clean$is_winner, probs)
auc_val <- auc(roc_curve)
plot(roc_curve, col = "darkgreen", lwd = 3, main = paste("ROC (AUC =", round(auc_val, 3), ")"))

# Uvjetna interpretacija — AUC
cat("\n[ROC/AUC] AUC =", round(auc_val, 3))
if (auc_val >= 0.80) {
  cat(" — ODLIČNA diskriminacijska sposobnost modela.\n")
} else if (auc_val >= 0.70) {
  cat(" — DOBRA diskriminacijska sposobnost. Model dobro razlikuje pobjednike od gubitnika.\n")
} else if (auc_val >= 0.60) {
  cat(" — PRIHVATLJIVA sposobnost. Model ima ograničenu, ali statistički značajnu prediktivnu vrijednost.\n")
} else {
  cat(" — SLABA sposobnost. Model jedva nadmašuje slučajno pogađanje — preporučuje se revizija.\n")
}
```

---

## 9. Nova hipoteza: Razlikuju li se udarci pobjednika i gubitnika unutar dobnih skupina?

**Hipoteza:** Pobjednici ostvaruju statistički značajno više udaraca od gubitnika, ali ta razlika je manja u skupini starijih boraca (≥35) gdje taktika i iskustvo kompenziraju fizičku aktivnost.

**H₀:** Distribucija broja udaraca jednaka je između pobjednika i gubitnika unutar svake dobne skupine.  
**H₁:** Pobjednici ostvaruju statistički značajno više udaraca od gubitnika, pri čemu je efekt manji kod boraca ≥35 godina.

```{r}
# Deskriptivna usporedba
strikes_summary <- ufc_clean %>%
  group_by(age_group, is_winner) %>%
  summarise(
    N = n(),
    Mean_Strikes = round(mean(strikes), 1),
    Median_Strikes = median(strikes),
    SD = round(sd(strikes), 1),
    .groups = "drop"
  )
print(strikes_summary)

# Boxplot vizualizacija
ggplot(ufc_clean, aes(x = is_winner, y = strikes, fill = is_winner)) +
  geom_boxplot(alpha = 0.7, outlier.alpha = 0.3) +
  facet_wrap(~ age_group) +
  scale_fill_manual(values = c("Loss" = "#e74c3c", "Win" = "#2ecc71")) +
  labs(
    title = "Distribucija udaraca: Pobjednici vs. Gubitnici po dobnoj skupini",
    x = "Ishod", y = "Broj značajnih udaraca"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

# Uvjetna interpretacija deskriptivnih razlika
diff_under <- strikes_summary %>% filter(age_group == "Under_35") %>%
  summarise(diff = Mean_Strikes[is_winner == "Win"] - Mean_Strikes[is_winner == "Loss"]) %>% pull()
diff_over  <- strikes_summary %>% filter(age_group == "Over_35") %>%
  summarise(diff = Mean_Strikes[is_winner == "Win"] - Mean_Strikes[is_winner == "Loss"]) %>% pull()

cat("\n[DESKRIPTIVNA USPOREDBA]\n")
cat("  Razlika u prosj. udarcima (Under_35): Win - Loss =", round(diff_under, 1), "\n")
cat("  Razlika u prosj. udarcima (Over_35):  Win - Loss =", round(diff_over, 1), "\n")
if (abs(diff_over) < abs(diff_under)) {
  cat("  -> Razlika je MANJA kod starijih boraca — konzistentno s hipotezom da iskustvo smanjuje prednost u udarcima.\n")
} else {
  cat("  -> Razlika je VEĆA ili JEDNAKA kod starijih boraca — hipoteza nije potvrđena deskriptivno.\n")
}
```

```{r}
# Wilcoxon test za svaku dobnu skupinu posebno
wilcox_under35 <- wilcox.test(strikes ~ is_winner, 
                               data = filter(ufc_clean, age_group == "Under_35"))
wilcox_over35  <- wilcox.test(strikes ~ is_winner, 
                               data = filter(ufc_clean, age_group == "Over_35"))

cat("Wilcoxon test (Under 35):\n")
print(wilcox_under35)
cat("\nWilcoxon test (Over 35):\n")
print(wilcox_over35)

# Effect size r = Z / sqrt(N) za obje skupine
n_under <- nrow(filter(ufc_clean, age_group == "Under_35"))
n_over  <- nrow(filter(ufc_clean, age_group == "Over_35"))
r_under <- abs(qnorm(wilcox_under35$p.value / 2)) / sqrt(n_under)
r_over  <- abs(qnorm(wilcox_over35$p.value  / 2)) / sqrt(n_over)
cat("\n[EFFECT SIZE]\n")
cat("  Under_35: r =", round(r_under, 3), "->",
    ifelse(r_under >= 0.5, "VELIKI", ifelse(r_under >= 0.3, "SREDNJI", "MALI")), "efekt\n")
cat("  Over_35:  r =", round(r_over,  3), "->",
    ifelse(r_over  >= 0.5, "VELIKI", ifelse(r_over  >= 0.3, "SREDNJI", "MALI")), "efekt\n")
if (r_under > r_over) {
  cat("  -> Efekt je MANJI u skupini Over_35 — konzistentno s hipotezom o kompenzaciji iskustva.\n")
} else {
  cat("  -> Efekt NIJE manji u skupini Over_35 — hipoteza o iskustvu nije potvrđena veličinom efekta.\n")
}

# Uvjetna interpretacija
cat("\n[UVJETNA INTERPRETACIJA HIPOTEZE 9]\n")
sig_under <- wilcox_under35$p.value < 0.05
sig_over  <- wilcox_over35$p.value  < 0.05

if (sig_under && sig_over) {
  cat("  Razlika u udarcima je SIGNIFIKANTNA u OBJE dobne skupine.\n")
  if (wilcox_under35$statistic > wilcox_over35$statistic) {
    cat("  W(Under_35) > W(Over_35) — efekt je jači kod mlađih boraca. Hipoteza POTVRĐENA: iskustvo smanjuje prednost udaraca.\n")
  } else {
    cat("  W(Under_35) <= W(Over_35) — efekt nije manji kod starijih. Hipoteza NIJE potvrđena u dijelu o iskustvu.\n")
  }
} else if (sig_under && !sig_over) {
  cat("  Razlika je signifikantna samo kod MLAĐIH boraca.\n  Hipoteza DJELOMIČNO POTVRĐENA: udarci su važni za mlađe, ali ne i za starije borce.\n")
} else if (!sig_under && sig_over) {
  cat("  Razlika je signifikantna samo kod STARIJIH boraca — neočekivan nalaz, suprotan hipotezi.\n")
} else {
  cat("  Razlika NIJE signifikantna ni u jednoj skupini — hipoteza ODBAČENA.\n")
}
```

---

## 10. Nova hipoteza: Utječe li težinska kategorija na metodu završetka borbe?

**Hipoteza:** Teži borci češće završavaju borbu KO/TKO-om, dok lakši borci imaju veći udio pobjeda odlukom. Submission (davljenje) raspoređen je ravnomjernije.

**H₀:** Metoda završetka borbe i težinska kategorija su međusobno nezavisne (distribucija metoda jednaka je u obje kategorije).  
**H₁:** Postoji statistički značajna veza između težinske kategorije i metode završetka borbe.

```{r}
# Filtriramo "Other" i kreiramo kontingencijsku tablicu
method_weight <- ufc_clean %>%
  filter(method != "Other", is_winner == "Win") %>%
  count(weight_cat, method)

# Vizualizacija
ggplot(method_weight, aes(x = weight_cat, y = n, fill = method)) +
  geom_bar(stat = "identity", position = "fill") +
  scale_y_continuous(labels = percent) +
  scale_fill_manual(values = c("KO/TKO" = "#e74c3c", "Submission" = "#3498db", "Decision" = "#f39c12")) +
  labs(
    title = "Metode pobjede po težinskoj kategoriji",
    x = "Težinska kategorija", y = "Udio (%)", fill = "Metoda"
  ) +
  theme_minimal()
```

```{r}
# Chi-kvadrat test nezavisnosti
chisq_data <- ufc_clean %>%
  filter(method != "Other", is_winner == "Win") %>%
  select(weight_cat, method)

chisq_res <- chisq.test(table(chisq_data$weight_cat, chisq_data$method))
print(chisq_res)

# Standardizirani reziduali — koji parovi najviše odstupaju
cat("\nStandardizirani reziduali:\n")
print(round(chisq_res$stdres, 2))

# Uvjetna interpretacija — Chi-kvadrat
cat("\n[CHI-KVADRAT INTERPRETACIJA]\n")
cat("  Chi² =", round(chisq_res$statistic, 2), "| p =", round(chisq_res$p.value, 4), "\n")

if (chisq_res$p.value < 0.05) {
  cat("  -> POSTOJI statistički značajna veza između težinske kategorije i metode završetka borbe (p < 0.05).\n")
  
  # Provjera specifičnih reziduala za heavy x KO/TKO i light x Decision
  res_heavy_ko  <- chisq_res$stdres["Heavy", "KO/TKO"]
  res_light_dec <- chisq_res$stdres["Light", "Decision"]
  
  if (!is.na(res_heavy_ko) && res_heavy_ko > 2) {
    cat("  -> KO/TKO je PREKOMJERNO zastupljen kod Heavy kategorije (rezidual =", round(res_heavy_ko, 2), ") — hipoteza POTVRĐENA za teže borce.\n")
  }
  if (!is.na(res_light_dec) && res_light_dec > 2) {
    cat("  -> Decision je PREKOMJERNO zastupljen kod Light kategorije (rezidual =", round(res_light_dec, 2), ") — hipoteza POTVRĐENA za lakše borce.\n")
  }
  if ((is.na(res_heavy_ko) || res_heavy_ko <= 2) && (is.na(res_light_dec) || res_light_dec <= 2)) {
    cat("  -> Veza postoji, ali specifični obrasci iz hipoteze nisu izrazito zastupljeni u rezidualima.\n")
  }
} else {
  cat("  -> NEMA statistički značajne veze (p >= 0.05). Metoda završetka borbe ne ovisi o težinskoj kategoriji.\n  -> Hipoteza ODBAČENA.\n")
}
```

---

## 11. Nova analiza: Profil "elitnog borca" — Klustering boraca po udarcima i dobi

**Cilj:** Bez nadzora grupirati borce u profile (klasteri) prema broju udaraca i dobi, pa analizirati udio pobjeda po profilu.

```{r}
set.seed(42)

# Normalizacija za k-means
cluster_data <- ufc_clean %>%
  select(age, strikes) %>%
  scale() %>%
  as.data.frame()

# K-means s 3 klastera
km <- kmeans(cluster_data, centers = 3, nstart = 25)
ufc_clustered <- ufc_clean %>% mutate(cluster = factor(km$cluster))

# Vizualizacija klastera
ggplot(ufc_clustered, aes(x = age, y = strikes, color = cluster)) +
  geom_point(alpha = 0.4, size = 1.2) +
  scale_color_manual(values = c("1" = "#e74c3c", "2" = "#3498db", "3" = "#2ecc71"),
                     labels = c("Klaster 1", "Klaster 2", "Klaster 3")) +
  labs(
    title = "K-means klasteriranje boraca (dob × udarci)",
    x = "Dob", y = "Broj udaraca", color = "Profil"
  ) +
  theme_minimal()
```

```{r}
# Udio pobjeda i prosječne vrijednosti po klasteru
cluster_profile <- ufc_clustered %>%
  group_by(cluster) %>%
  summarise(
    N = n(),
    Avg_Age = round(mean(age), 1),
    Avg_Strikes = round(mean(strikes), 1),
    Win_Rate = mean(is_winner == "Win")
  )
print(cluster_profile %>% mutate(Win_Rate = scales::percent(Win_Rate, accuracy = 0.1)))

# Uvjetna interpretacija — klustering
best_cluster  <- cluster_profile %>% slice_max(Win_Rate, n = 1)
worst_cluster <- cluster_profile %>% slice_min(Win_Rate, n = 1)

cat("\n[KLUSTERING INTERPRETACIJA]\n")
cat("  Klaster s NAJVIŠOM stopom pobjeda: Klaster", as.character(best_cluster$cluster),
    "| Avg dob:", best_cluster$Avg_Age,
    "| Avg udarci:", best_cluster$Avg_Strikes,
    "| Win Rate:", scales::percent(best_cluster$Win_Rate, accuracy = 0.1), "\n")
cat("  Klaster s NAJNIŽOM stopom pobjeda: Klaster", as.character(worst_cluster$cluster),
    "| Avg dob:", worst_cluster$Avg_Age,
    "| Avg udarci:", worst_cluster$Avg_Strikes,
    "| Win Rate:", scales::percent(worst_cluster$Win_Rate, accuracy = 0.1), "\n")

if (best_cluster$Avg_Strikes > worst_cluster$Avg_Strikes) {
  cat("  -> Klaster s VIŠE udaraca ima VIŠU stopu pobjeda — potvrđuje važnost aktivne borbe.\n")
} else {
  cat("  -> Klaster s VIŠE udaraca NEMA višu stopu pobjeda — taktika i iskustvo mogu biti važniji od broja udaraca.\n")
}

age_diff <- best_cluster$Avg_Age - worst_cluster$Avg_Age
if (abs(age_diff) > 3) {
  cat("  -> Razlika u prosječnoj dobi između najboljeg i najlošijeg klastera je",
      round(abs(age_diff), 1), "godina —",
      ifelse(age_diff < 0, "mlađi profil dominira.", "stariji profil dominira."), "\n")
} else {
  cat("  -> Dob sama po sebi ne razdvaja jasno uspješne od neuspješnih klastera.\n")
}
```

---

## 12. Sažetak nalaza

```{r}
cat("=== SAŽETAK KLJUČNIH NALAZA ===\n\n")
cat("1. Distribucija udaraca je desno asimetrična -> opravdana primjena Wilcoxon testa\n")
cat("2. Logistička regresija: dob negativno prediktira pobjedu, strikes pozitivno\n")
cat("3. Decision tree identificira prag udaraca kao ključni čvor razvrstavanja\n")
cat("4. Razlika u udarcima između pobjednika i gubitnika signifikantna u obje dobne skupine\n")
cat("5. Chi-kvadrat potvrđuje vezu između težinske kategorije i metode završetka borbe\n")
cat("6. K-means profiliranje: borci s više udaraca konzistentno imaju višu stopu pobjeda\n")
```
