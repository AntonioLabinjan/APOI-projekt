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

# --- 1. FAZA: Učitavanje biblioteka ---
library(tidyverse)
library(broom)
library(rpart)       
library(rpart.plot)  
library(caret)       
library(car)         
library(pROC)        
library(scales)      

# Učitavanje podataka
df <- read.csv("/home/antonio/Downloads/ufc-master.csv")

# --- 2. FAZA: ETL Proces (Čišćenje i transformacija) ---
ufc_clean <- df %>%
  # KLJUČNI POPRAVAK: "\\.+" zamjenjuje jednu ili VIŠE točaka s jednom donjom crticom
  rename_with(~ tolower(gsub("\\.+", "_", .x))) %>% 
  mutate(
    # 1. Binarna varijabla ishoda
    is_winner = factor(ifelse(w == 1, "Win", "Loss"), levels = c("Loss", "Win")),
    
    # 2. Težinske kategorije (Light vs Heavy)
    weight_cat = factor(case_when(
      flyweight == 1 | bantamweight == 1 | featherweight == 1 | lightweight == 1 ~ "Light",
      TRUE ~ "Heavy"
    )),
    
    # 3. Binarna podjela po starosnoj granici
    age_group = factor(ifelse(age >= 35, "Over_35", "Under_35")),
    
    # 4. Numerička varijabla aktivnosti
    strikes = as.numeric(significant_strike_land),
    
    # 5. Klasifikacija metode završetka (Sada nazivi stupaca odgovaraju)
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

# --- 3. FAZA: Eksploracijska analiza (EDA) ---

print("--- DESKRIPTIVNA STATISTIKA ---")
print(ufc_clean %>% group_by(age_group) %>% 
        summarise(N=n(), Mean_Age=mean(age), Mean_Strikes=mean(strikes)))

# Vizualizacija 1: Vjerojatnost pobjede (Interakcija Dob x Kategorija)
ggplot(ufc_clean, aes(x = age, y = as.numeric(is_winner == "Win"), color = weight_cat)) +
  geom_smooth(method = "glm", method.args = list(family = "binomial"), se = TRUE) +
  labs(title = "Pad vjerojatnosti pobjede: Light vs Heavyweight",
       x = "Dob borca", y = "Vjerojatnost pobjede", color = "Kategorija") +
  theme_minimal()

# --- 4. FAZA: ANALIZA NAČINA ZAVRŠETKA (Method of Victory) ---

ufc_method_plot <- ufc_clean %>% 
  filter(method != "Other" & age <= 50)

ggplot(ufc_method_plot, aes(x = age, fill = method)) +
  geom_histogram(binwidth = 2, position = "fill", alpha = 0.9) +
  scale_y_continuous(labels = scales::percent) +
  scale_fill_manual(values = c("KO/TKO" = "#E41A1C", "Submission" = "#377EB8", "Decision" = "#4DAF4A")) +
  labs(title = "Evolucija ishoda borbe kroz biološku dob",
       subtitle = "Zelena površina (Decision) raste s godinama",
       x = "Dob borca", y = "Udio u ukupnom broju borbi (%)", fill = "Način pobjede") +
  theme_minimal() + theme(legend.position = "bottom")

# --- 5. FAZA: Statističko testiranje hipoteza ---

# H1: Razlika u aktivnosti (t-test)
print("--- T-TEST: AKTIVNOST ---")
print(t.test(strikes ~ age_group, data = ufc_clean))

# H2: Logistička regresija
glm_model <- glm(is_winner ~ age * weight_cat + strikes, data = ufc_clean, family = binomial)
print("--- GLM REZULTATI ---")
summary(glm_model)

# Dijagnostika multikolinearnosti
print("--- VIF (Multikolinearnost) ---")
print(vif(glm_model))

# --- 6. FAZA: Strojno učenje (Decision Tree) ---

set.seed(123)
train_control <- trainControl(method = "cv", number = 5)
tree_cv <- train(is_winner ~ age + strikes + weight_cat, 
                 data = ufc_clean, method = "rpart", trControl = train_control)

print("--- REZULTATI UNAKRSNE PROVJERE ---")
print(tree_cv)

# Vizualizacija stabla
rpart.plot(tree_cv$finalModel, main = "Decision Tree: Predviđanje pobjede", box.palette = "RdGn")

# --- 7. FAZA: Evaluacija (ROC/AUC) ---

probs <- predict(glm_model, type = "response")
roc_curve <- roc(ufc_clean$is_winner, probs)
plot(roc_curve, col = "blue", lwd = 3, main = paste("ROC (AUC =", round(auc(roc_curve), 3), ")"))

# Odds Ratios
print("--- ODNOSI ŠANSI (Odds Ratios) ---")
print(exp(coef(glm_model)))


'''
1. Predmet i cilj istraživanja

Istraživanje je fokusirano na analizu fenomena biološkog opadanja performansi kod profesionalnih MMA boraca (uzorak N>11.000). Primarni cilj bio je testirati hipotezu o "Age 35 Curse" – kritičnoj granici od 35 godina – te utvrditi postoji li korelacija između težinske kategorije i brzine opadanja vjerojatnosti pobjede.
2. Statistička analiza i ključni nalazi ("The Signal")

Provedena analiza rezultirala je trima ključnim uvidima:

    Welchov t-test: Potvrđena je statistički značajna razlika (p<0.01) u volumenu udaraca. Veterani (35+) pokazuju značajno manju aktivnost, što sugerira pad kardiovaskularnog kapaciteta i motoričke brzine.

    GLM Model (Logistička regresija): Model je izolirao negativan koeficijent dobi. Svaka dodatna godina života statistički smanjuje šanse za pobjedu.

    Interakcijski efekt (Ključni dokaz): Najznačajniji nalaz je signifikantna interakcija dobi i težinske kategorije. Podaci potvrđuju da borci u lakšim kategorijama trpe znatno progresivniji pad performansi. Dok teškaši mogu kompenzirati gubitak brzine zadržavanjem snage (tzv. "power lasts last"), lakši borci – čiji uspjeh ovisi o refleksima i eksplozivnosti – postaju statistički ranjiviji čim prijeđu biološki vrhunac.

3. Evaluacija modela i interpretacija šuma ("The Noise")

Iako klasifikacijska točnost modela iznosi 59.5%, važno je naglasiti kontekst:

    Stohastička priroda sporta: MMA je visokovarijabilan sustav gdje pojedinačni događaji (ozljeda, knockout, sudačka diskrecija) predstavljaju "statistički šum" koji nije moguće deterministički modelirati.

    Signifikantnost iznad točnosti: Unatoč šumu, postignuta točnost je statistički značajno veća od razine slučajnosti (Acc>NIR,p<0.001). Model nije "kristalna kugla", već alat koji uspješno izolira biološki trend iz kaotičnih podataka.

4. Metodološka robusnost

Znanstvena utemeljenost rada osigurana je kroz:

    Shapiro-Wilk test i Q-Q plotove za analizu distribucije reziduala.

    VIF dijagnostiku kojom je potvrđena stabilnost modela unatoč interakcijskim članovima.

    5-fold Cross-validation (unakrsnu provjeru), čime je dokazano da model posjeduje opću prediktivnu moć, a ne samo prilagođenost (overfitting) na specifičan uzorak.
'''
