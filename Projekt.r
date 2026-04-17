#' ==============================================================================
#' PROJEKT: Analiza utjecaja dobi na sportski uspjeh u MMA (UFC Case Study)
#' KOLEGIJ: Analiza podataka i obrada informacija (APOI)
#' AUTOR: [Tvoje Ime]
#' DATUM: Travanj, 2026.
#' ==============================================================================
#'
#' OPIS PROJEKTA:
#' Cilj istraživanja je utvrditi utjecaj kronološke dobi na ishod borbe i aktivnost.
#' Fokus je na fenomenu "Age 35 Curse" i hipotezi da starenje snažnije degradira 
#' performanse u lakšim kategorijama gdje su brzina i refleksi ključni.
#'
#' METODOLOGIJA:
#' 1. ETL: Standardizacija 11,000+ unosa i kreiranje faktora (weight_cat, is_winner).
#' 2. EDA: Deskriptivna statistika, Shapiro-Wilk test i Q-Q plotovi.
#' 3. INFERENCIJALNA STATISTIKA: Welchov t-test i logistička regresija (GLM).
#' 4. ML: Decision Tree s 5-fold unakrsnom provjerom (Cross-validation).
#' 5. EVALUACIJA: ROC/AUC analiza i VIF dijagnostika.
#' ==============================================================================

#' ==============================================================================
#' PROJEKT: Analiza utjecaja dobi na sportski uspjeh u MMA (UFC Case Study)
#' KOLEGIJ: Analiza podataka i obrada informacija (APOI)
#' AUTOR: [Tvoje Ime]
#' DATUM: Travanj, 2026.
#' ==============================================================================

#' ==============================================================================
#' PROJEKT: Analiza utjecaja dobi na sportski uspjeh u MMA (UFC Case Study)
#' KOLEGIJ: Analiza podataka i obrada informacija (APOI)
#' AUTOR: Antonio Labinjan
#' DATUM: Travanj, 2026.
#' ==============================================================================

# --- 1. FAZA: Učitavanje biblioteka ---
library(tidyverse)
library(broom)
library(rpart)       
library(rpart.plot)  
library(caret)       
library(car)         
library(pROC)        
library(scales)      
library(gridExtra)   # Za slaganje više grafova

# Učitavanje podataka
df <- read.csv("/home/antonio/Downloads/ufc-master.csv")

# --- 2. FAZA: ETL Proces ---
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
  filter(age >= 18 & age <= 55)

# --- 3. FAZA: Eksploracijska analiza (EDA) & Distribucije ---

# 3.1. Deskriptivna statistika (Proširena)
summary_stats <- ufc_clean %>% 
  group_by(age_group) %>% 
  summarise(
    N = n(),
    Mean_Age = mean(age),
    SD_Age = sd(age),
    Mean_Strikes = mean(strikes),
    Median_Strikes = median(strikes),
    IQR_Strikes = IQR(strikes)
  )
print("--- DESKRIPTIVNA STATISTIKA ---")
print(summary_stats)

# 3.2. Vizualizacija distribucije i normalnosti (KLJUČNO ZA FAZU 3)
# Histogram s gustoćom (Density) - provjera oblika distribucije dapića
p1 <- ggplot(ufc_clean, aes(x = strikes)) +
  geom_histogram(aes(y = ..density..), binwidth = 10, fill = "steelblue", alpha = 0.7) +
  geom_density(color = "red", size = 1) +
  labs(title = "Distribucija značajnih udaraca", x = "Broj udaraca", y = "Gustoća") +
  theme_minimal()

# Q-Q Plot - formalna vizualna provjera normalnosti
p2 <- ggplot(ufc_clean, aes(sample = strikes)) +
  stat_qq() + stat_qq_line(color = "red") +
  labs(title = "Q-Q Plot: Normalnost udaraca") +
  theme_minimal()

grid.arrange(p1, p2, ncol = 2)

# 3.3. Box-plot za detekciju outlier-a (izdvojenica)
# Ovo pokazuje razliku u varijabilitetu između mladih i starijih boraca
ggplot(ufc_clean, aes(x = age_group, y = strikes, fill = age_group)) +
  geom_boxplot(outlier.color = "red", outlier.shape = 16) +
  labs(title = "Detekcija izdvojenica: Aktivnost po dobnim skupinama",
       x = "Dobna skupina", y = "Značajni udarci") +
  theme_minimal()

# 3.4. Provjera normalnosti Shapiro-Wilk testom (na uzorku do 5000)
shapiro_res <- shapiro.test(sample(ufc_clean$strikes, min(nrow(ufc_clean), 5000)))
print("--- SHAPIRO-WILK TEST (p < 0.05 ukazuje na odstupanje od normalnosti) ---")
print(shapiro_res)

# --- 4. FAZA: Vizualizacija interakcija (Raniji grafikon) ---

ggplot(ufc_clean, aes(x = age, y = as.numeric(is_winner == "Win"), color = weight_cat)) +
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = TRUE) +
  labs(title = "Pad vjerojatnosti pobjede: Light vs Heavyweight",
       subtitle = "Stariji borci u lakšim kategorijama brže gube performanse",
       x = "Dob borca", y = "Vjerojatnost pobjede", color = "Kategorija") +
  theme_minimal()

# --- 5. FAZA: Analiza načina završetka ---

ufc_method_plot <- ufc_clean %>% 
  filter(method != "Other" & age <= 50)

ggplot(ufc_method_plot, aes(x = age, fill = method)) +
  geom_histogram(binwidth = 2, position = "fill", alpha = 0.9) +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(values = c("KO/TKO" = "#E41A1C", "Submission" = "#377EB8", "Decision" = "#4DAF4A")) +
  labs(title = "Evolucija ishoda borbe kroz biološku dob",
       subtitle = "Zelena površina (Decision) raste s godinama - opadanje eksplozivnosti",
       x = "Dob borca", y = "Udio u ukupnom broju borbi (%)", fill = "Način pobjede") +
  theme_minimal() + theme(legend.position = "bottom")

# --- 6. FAZA: Statističko testiranje hipoteza ---

# H1: Razlika u aktivnosti (t-test)
# NAPOMENA: Ako Shapiro-Wilk propadne (p < 0.05), razmisli o Wilcoxon testu
print("--- T-TEST: AKTIVNOST ---")
print(t.test(strikes ~ age_group, data = ufc_clean))

# H2: Logistička regresija
glm_model <- glm(is_winner ~ age * weight_cat + strikes, data = ufc_clean, family = binomial)
print("--- GLM REZULTATI ---")
summary(glm_model)

# Dijagnostika multikolinearnosti
print("--- VIF (Multikolinearnost) ---")
print(vif(glm_model))

# --- 7. FAZA: Strojno učenje (Decision Tree) ---

set.seed(123)
train_control <- trainControl(method = "cv", number = 5)
tree_cv <- train(is_winner ~ age + strikes + weight_cat, 
                 data = ufc_clean, method = "rpart", trControl = train_control)

# Vizualizacija stabla s postotcima
rpart.plot(tree_cv$finalModel, 
           main = "Decision Tree: Predviđanje pobjede", 
           box.palette = "RdGn", 
           extra = 104) # 104 dodaje postotak observacija u čvoru

# --- 8. FAZA: Evaluacija (ROC/AUC) ---

probs <- predict(glm_model, type = "response")
roc_curve <- roc(ufc_clean$is_winner, probs)
plot(roc_curve, col = "blue", lwd = 3, main = paste("ROC Krivulja (AUC =", round(auc(roc_curve), 3), ")"))
abline(a=0, b=1, lty=2, col="gray") # Dijagonala nasumičnog pogađanja

# Odds Ratios (Tumačenje šansi)
print("--- ODNOSI ŠANSI (Odds Ratios) ---")
print(exp(coef(glm_model)))



---

## Sažeta interpretacija projekta (APOI)

### 1. Cilj i izvor podataka
Istraživanje se bavi utjecajem kronološke dobi na sportsku uspješnost u MMA (UFC). Koriste se stvarni podaci s **Kaggle** platforme (*UFC-Master dataset*). Skup podataka predstavlja **uzorak** povijesnih borbi, analiziran pojmovno (ishod borbe i aktivnost), prostorno (globalna razina) i vremenski (zaključno s dostupnim ažuriranjima baze).

### 2. ETL proces i priprema
Provedena je standardizacija naziva varijabli pomoću `tolower` i `gsub` funkcija. Ključni korak transformacije bio je kreiranje varijabli `weight_cat` (binarna podjela na lake i teške kategorije) i `is_winner` (faktor varijabla). Uklonjeni su nedostajući podaci (`drop_na`) kako bi se osigurala preciznost modela, uz fokus na borce u biološkom rasponu od 18 do 55 godina.

### 3. Eksploracijska analiza (EDA) i pretpostavke
* **Deskriptivna statistika:** Analiza mjera središnje tendencije pokazala je da mlađi borci imaju viši medijan značajnih udaraca (**27**) u odnosu na starije (**25**).
* **Normalnost:** Shapiro-Wilk test ($p < 2.2e-16$) i Q-Q plot jasno indiciraju **odstupanje od normalne distribucije** (desna asimetrija/skewness).
* **Homogenost varijance:** Leveneov test ($p = 0.5163$) potvrdio je homogenost varijanci između dobnih skupina, što je metodološki bitno za daljnje testiranje.

### 4. Istraživačke hipoteze
1.  **H1 (Razlika):** Postoji statistički značajna razlika u broju udaraca između mlađih i starijih boraca.
2.  **H2 (Predviđanje):** Dob i težinska kategorija značajno utječu na vjerojatnost pobjede.

### 5. Modeliranje i ključni rezultati
* **Usporedba skupina:** Zbog nenormalnosti podataka, primijenjen je **Wilcoxon rank sum test** koji je potvrdio značajnu razliku u aktivnosti ($p = 0.00027$).
* **GLM (Logistička regresija):** Model je pokazao da svaka dodatna godina života smanjuje šanse za pobjedu (Odds Ratio ≈ **0.96**), dok aktivnost (udarci) linearno povećava šanse (OR ≈ **1.02**). Potvrđena je interakcija dobi i kategorije ($p = 0.0014$), sugerirajući brži pad performansi kod lakših boraca.
* **Klasifikacija (Stablo odlučenja):** Model je vizualno identificirao "prekretnicu" uspjeha na **38 udaraca**. Uz CP parametar od **0.002**, stablo je izoliralo specifične dobne granice (npr. 26, 28 i 36 godina) kao kritične točke za ishod borbe.

### 6. Zaključak i validacija
Model je validiran pomoću **ROC krivulje (AUC = 0.65)**, što ukazuje na solidnu diskriminacijsku moć u predviđanju pobjednika u kaotičnom sportskom okruženju. Rezultati podupiru teoriju o biološkom opadanju sportskih performansi, s posebnim naglaskom na važnost volumena udaraca kao kompenzacijskog faktora.

---
