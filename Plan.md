
# SECT
###### **Security events temporal correlation**

## Intro

**Odhalenie koordinovaných útokov**

Cesnet prevázdkuje rozsiahlu akademickú sieť naprieč českou republikou. Medzi jeho výskumné aktivity patrí okrem iného vývoj rýchlych ethernet kariet, monitorovacích a bezpečnostných riešení. Monitorovacie body na sieti generujú veľké objemy netflow dát, ktoré sú ďalej spracované rôznymi detektormi do bezpečnostných událostí. V skratke, CESNET je schopný na svojej sieti detekovať veľké množstvo bezpečnostných události v čase. Tieto data vcelku určite obsahujú skryté informácie o hrozbách hodné preskúmania. 

Máme za to, že časové korelácie v bezpečnostných udalostiach naznačujú, že ich pôvodcovia môžu byť koordinovaný. Preskúmanie časových korelácii v časových značkách bezpečnostných událostí je jedným z krokov k odhaleniu koordinovaných útokov. Najlepší kandidát na detekciu útok pomocov botnetu.

Táto práca si dáva za cieľ odhaliť koordinované útoky pomocov botnetov.
 
### Predpoklad na overenie
 
Časová korelácie medzi bezpečnostnými udalosťami značí, že povodcovia sú pravdepodobne koordinovaní

Overenie by mohlo prebiehať takto. V dc liberoutri a dohľadadú flow záznamy IP adries ktoré boli korelované do rovnakej skupiny. Zo začiatku by overenie šlo ručne, neskor po zistení na čo presnejšie je treba pozerať na daných datách by šlo proces automatizovať.

## Hlavné problémy

* Najdenie skupín útočiacich IP
    * Výpočet časovej korelácie dvojíc (ok)
    * Výpočet vzájomne súvisiacich skupín (brzdí výpočet)
    
* Artefakty monitorovacej pipeline
* Overenie detekovaných výsledkov

### Aktuálny stav
####Korelačný algoritmus
Momentálne je urobený algoritmus pre koreláciu, na obmedzenej časovej rade funguje ale výpočet skupín beží neprakticky dlho. Problém popíšem presnejšie
až to odladím a výriešim. Zatiaľ mám podozrenie, že artefakty mon. pipeline v časovej značke vytvoria veľké skupiny korelovaných IP a výpočet väcších skupín a celý výpočet sa stáva viac časovo náročnejším.

V korelačnom algoritme som upravil spôsob práce so súborom so snahou prejsť na spracovanie časových značiek, integerov, miesto dátumov, čo mi prišlo rýchlejšie.

####Preprocessing
Vyrobil som skripty pre preprocessing .idea súboru, tento predspracuje eventy na vektory vhodnejšie na strojove učenie. Tiež, pre jednoduchšie experimentovanie je urobená aj verzia preprocesingu pre pandas dataframe kt. umožňuje agregácie filtrovanie a podobné pokročilé operácie. Staré dátové sady nám nestačila, pretože nevieme dohľadať komunikáciu v dc.liberouter, a teda nevieme overiť platnosť korelácií.

####Clustering
Klastrovanie bolo vyskúšané, zatiaľ bez konkrétnych výsledkov ale bolo možné pozorovať zoskupenia. 
Plot feature vektorov [count, duration, inter arrival time [mean median, odchýlka]] bol tiež užitočný a ukázal že v datách existuju artefaktny mon. pipeline. 

#### TLDR 
Je pripravený kód pre predspracovanie aktuálnej sady udalostí pre koreláčný algoritmus. 
Korelačný algoritmus je zatiaľ veľmi časovo náročný, hlavne kvôli redukcii korelovaných skupín.
Bolo vyskúšané klastovanie nad extrahovanými feature z časových rád. 
 
## Celkový plán riešenia [TBD]

Zobrať .idea data o udalostiach predspracovať ich na vektory. Vektory previesť na časovú radu a spustiť časovu koreláciu. Výsledkom analýzy sú skupiny IP adries a časová rada udalostí v skupine.

Podľa výsledkov v dc.liberoutri ručne preskúmať podobnosť komunikácie v skupinách.

Vstpnými datami pre ďalšie spracovanie sú textové súbory vo formáte json so špecifickou štruktúrou. Pre zjednodušenie a generalizáciu korelácie sa vyberie podmnožina z dostupných informácií, a to následovne. 

|Pole||||
|---|---|---|---|
|Časová značka udalosti|
|IP adresa zdroja|
|Detektor kt. udalosť generoval|
|Typ udalosti|


| |Features | |
|---|---|---|
|**Pole**|**Typ**|**Účel**|
|Bitfield typov udalosti|Binary|Podobné utoky budú mať podobné odtlačky|

---
Zišlo by sa ešte, pre možnosť dalšej práce s výsledkami:

* Riadok z kt. bol feature vygenerovaný
* Agregačné okno
* Závažnosť


Je možne, že detektory ktoré hlásia udalosti v pravidelných intervaloch vnesú do dát tzv. artefakty. Tieto spôsobia korelácie tam kde, by inak nemali nastať. Toto je nežiadúce. Zatiaľ neviem ako by som toto uspokojivo riešil. 

### TLDR

## Plán na tento týžden
Dostať prvé korelačné výsledky, pozrieť skupiny ip v dc.liberouter. 

###Odstránenie artefaktov
Bude vyžadovať úpravu preprocesingu, pridanie príznakov o evente, znova spustenie vizualizácie feature časových rád. Pak spustim koreláciu na prečistených datách, aspoň nad hodinovým intervalom. 

###Dočasné riešenie náročnosti korelácií
Ako bolo spomenuté, je výsoká naročnosť tvorby skúpín z korelovaných dvojíc. Aktuálne vidím dva prístupy ako sa s tým vysporiadať:

* Upraviť data [možnosti]
    * Predradiť klastovanie
       * Natrénovať model vždy na danom časovom intervale a korelovať len v rámci klastrov
    * Prefiltrovať data
       * Predfiltrovať IP add na počet výšší ako aspoň x eventov za hodinu

* Upraviť algoritmus [...]
    ? Imo zatial nema plán

Zahladiť kód a zjednodušiť pridanie/odobratie feature pre klastrovanie

## Ďalšie možné spôsoby

* Lepšie features
* Lepšie datove struktury pre filtrovanie - tabuľka optimalizovaná pre časovú dimenziu ako index (kvôli prístupu)
* Hierarchické klastovanie pomocov DTW
* Aplikácia rekurentných sieti

## Thought
 Nebolo by odveci mať obecný korelačný algoritmus pre riedke časové rady. 
 Pak by sa dali zjemnovať korelácie
 Nad eventmi => Nad flow záznamami => Potenciálne nad paketmi