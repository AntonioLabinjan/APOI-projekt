# APOI-projekt
https://moodle.srce.hr/2025-2026/course/section.php?id=3277030


Does age significantly predict the probability of winning a UFC fight?

Is the "Age 35 Curse" more prevalent in lighter weight classes compared to Heavyweights?

Klasifikacijski model => supervised learning; definiranje prediktora; 

- Decision tree; prvo gre exploratory analysis
1) GLM; različiti regresijski modeli
2) statističko učenje (ML varijante)
3) uzorak mora bit dovoljno velik za train/test/val split (check if it is big enough)
4) binary pobijedio/nije pobijedio
5) predict => opadanje broja borbi s godinama; potencijalno i opadanje broja pobjeda u borbama
6) međusobne razlike boraca (usporedba godina boraca u trenutku borbe)
7) distribucije udaraca po kategorijama/borcima/something

U skupu odabbrat neke varijable koje će mi dat odgovor na temeljno pitanje
- Opcija 1; razgranat početno pitanje => BOLJA VARIJANTA; odabrat subset podataka koji nas zanimaju; bar 2 pitanja...testiranje hipoteza (postoji li značajna promjena u proporcijama pobjeda/poraza prije/nakon 35); drugo pitanje..dali je moguće nešto od toga predviđati...model!
1) cleaning..Hitit ća stupce koji mi ne trebaju
2) grafikoni
3) sve izračunate brojeve znat objasnit; protumačit..zašto ima smisla
4) min: 2-3 kvantitativne; 1 ili više kvalitativnih
5) temeljem statistike smislit pitanje i dalje istražit
- Opcija 2; nemamo početno pitanje i granamo se dalje
