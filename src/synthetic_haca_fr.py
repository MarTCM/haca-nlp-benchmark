"""
French HACA-Sent — synthetic broadcast-register French, authored by Claude.

NO LLM API. Every utterance below is hand-written by Claude (the assistant) in this
project, to fit report/ANNOTATION_RUBRIC_V3.md and the register of Moroccan French-language
broadcast (news / explainer / debate — e.g. 2M, Médi1, French coverage of Morocco).

Why this file exists: there is no HACA-style French training pool (the only real French SRT,
emission_francaise.srt, is tiny and noisy). So the synthetic set IS the French training data
— unlike the Arabic flow, there is no MAC/real pool to mix in. We therefore keep the three
classes reasonably balanced (instead of the neu-heavy broadcast prior) so a fresh French head
can learn all three. These rows are flagged synthetic=true and used for TRAINING ONLY — never
for the frozen gold test (data/test_sets/francais_haca_gold.csv is hand-labelled real SRT).

Labelling rubric (same as the Arabic v3):
  * pos — the content reports something good (success, reform that helps, growth, opportunity);
  * neg — the content reports something bad (failure, loss, shortage, harm, corruption);
  * neu — procedural / definitional / descriptive, no recoverable valence.

Each entry is (text, topic).

Usage:
    python src/synthetic_haca_fr.py        # writes data/test_sets/synthetic_haca_fr.csv
"""

import csv
import os

OUT_CSV = "data/test_sets/synthetic_haca_fr.csv"

# ── POSITIVE (content reports something good) ────────────────────────────────
POS = [
    # economy / investment / jobs
    ("Les investissements étrangers au Maroc ont atteint un niveau record cette année, signe de la confiance grandissante des marchés dans l'économie nationale.", "economy"),
    ("Une nouvelle usine de composants aéronautiques verra le jour près de Tanger et emploiera plusieurs milliers de jeunes ingénieurs marocains.", "economy"),
    ("Les exportations automobiles du Royaume ont battu tous les records et le secteur est devenu le premier pourvoyeur de devises du pays.", "economy"),
    ("Le taux de chômage a reculé de façon notable grâce aux programmes d'emploi des jeunes lancés par l'État.", "economy"),
    ("Le Maroc s'est imposé comme un véritable pôle industriel en Afrique, et de nombreuses multinationales ont choisi d'y implanter leurs sites de production.", "economy"),
    ("Les petites et moyennes entreprises ont bénéficié d'un nouveau dispositif de soutien qui leur a permis de croître et de recruter davantage.", "economy"),
    ("Le port de Tanger Med est devenu l'un des plus grands d'Afrique et de Méditerranée, créant des milliers d'emplois directs et indirects.", "economy"),
    ("La croissance du produit intérieur brut a dépassé les prévisions cette année malgré un contexte international difficile.", "economy"),
    ("Le secteur des énergies renouvelables attire des capitaux considérables et place le pays parmi les leaders régionaux du domaine.", "economy"),
    ("Un salon économique international s'est tenu à Casablanca et a ouvert de nouvelles perspectives de partenariat et d'exportation pour les entreprises locales.", "economy"),
    # reforms that help people
    ("La réforme de l'impôt sur le revenu allégera la charge fiscale de millions de ménages de la classe moyenne, qui conserveront un pouvoir d'achat plus élevé.", "reform"),
    ("La couverture médicale obligatoire s'étend désormais à des millions de Marocains qui n'étaient pas assurés auparavant, un acquis social majeur.", "reform"),
    ("Le gouvernement a décidé d'exonérer les médicaments essentiels de la TVA, ce qui réduira leur prix pour les patients.", "reform"),
    ("L'aide directe aux familles démunies commence à atteindre ses bénéficiaires et soulage de nombreux foyers en difficulté.", "reform"),
    ("Le salaire minimum a été revalorisé, améliorant concrètement le pouvoir d'achat des salariés et des ouvriers.", "reform"),
    ("La nouvelle loi sur la protection du consommateur renforce les droits des citoyens et oblige les entreprises à plus de transparence.", "reform"),
    ("Le programme d'aide au logement a permis à des milliers de familles d'accéder à la propriété à un prix abordable.", "reform"),
    ("La généralisation de la protection sociale progresse et inclut chaque année de nouvelles catégories de la population.", "reform"),
    # health
    ("Un nouveau centre hospitalier universitaire a été inauguré dans une région longtemps sous-dotée et desservira des millions d'habitants.", "health"),
    ("Le Maroc a réussi sa campagne nationale de vaccination en atteignant un taux de couverture élevé, protégeant ainsi la santé publique.", "health"),
    ("Une équipe médicale marocaine a réalisé avec succès une opération chirurgicale complexe qui ne se pratiquait jusque-là qu'à l'étranger.", "health"),
    ("Le nombre de médecins diplômés a doublé après l'augmentation des capacités d'accueil des facultés de médecine.", "health"),
    ("Des caravanes médicales ont atteint les zones enclavées et soigné des milliers de patients qui n'avaient pas accès aux soins.", "health"),
    ("L'industrie pharmaceutique nationale a commencé à exporter ses médicaments vers plusieurs pays africains.", "health"),
    # education
    ("Le taux de réussite au baccalauréat a connu une nette amélioration cette année, fruit des efforts conjugués des enseignants et des élèves.", "education"),
    ("Une université marocaine est entrée pour la première fois dans le classement mondial des meilleurs établissements, une fierté pour l'enseignement supérieur.", "education"),
    ("Le programme de bourses a permis à des milliers d'étudiants issus de familles modestes de poursuivre leurs études supérieures.", "education"),
    ("La formation professionnelle offre désormais aux jeunes des compétences recherchées sur le marché du travail et facilite leur insertion.", "education"),
    ("Le transport scolaire a été étendu au monde rural, réduisant nettement l'abandon scolaire, en particulier chez les filles.", "education"),
    ("Des élèves marocains ont décroché des médailles aux Olympiades internationales de mathématiques et ont honoré leur pays.", "education"),
    # infrastructure / energy / water
    ("Le Maroc a inauguré l'une des plus grandes centrales solaires au monde et s'est imposé comme un pionnier des énergies propres.", "infra"),
    ("La nouvelle ligne à grande vitesse a rapproché les villes et facilité les déplacements de millions de voyageurs.", "infra"),
    ("Des projets de dessalement de l'eau de mer vont sécuriser l'approvisionnement des régions touchées par la sécheresse.", "infra"),
    ("Le Maroc a raccordé à l'électricité la quasi-totalité de ses villages, atteignant l'un des meilleurs taux d'électrification rurale de la région.", "infra"),
    ("Un nouveau barrage permettra de stocker des millions de mètres cubes d'eau destinés à l'irrigation et à la consommation.", "infra"),
    ("Le tramway a fluidifié la circulation dans les grandes villes et réduit la pollution et les embouteillages.", "infra"),
    # sports
    ("L'équipe nationale a réalisé un exploit historique en atteignant les demi-finales de la Coupe du monde, à la grande joie des Marocains du monde entier.", "sport"),
    ("Le Maroc coorganisera la Coupe du monde 2030 avec l'Espagne et le Portugal, un événement majeur qui fera rayonner le pays.", "sport"),
    ("Une athlète marocaine a remporté la médaille d'or d'une compétition internationale et hissé haut les couleurs nationales.", "sport"),
    ("Un club marocain a remporté la Ligue des champions africaine et ses supporters ont laissé éclater leur joie dans les rues.", "sport"),
    ("La sélection féminine s'est qualifiée pour la première fois à la Coupe du monde, un acquis majeur pour le sport féminin.", "sport"),
    # culture / heritage / tourism
    ("L'UNESCO a classé un site historique marocain au patrimoine mondial, une reconnaissance de la richesse de la civilisation du pays.", "culture"),
    ("Un film marocain a remporté un prix prestigieux dans un festival international et fait connaître la créativité nationale.", "culture"),
    ("Le Festival international du film de Marrakech a réuni des stars du monde entier et conforté le Maroc comme destination culturelle.", "culture"),
    ("Le nombre de touristes ayant visité le Maroc a atteint un record cette année, dynamisant le secteur et l'économie locale.", "tourism"),
    ("La médina a été restaurée et a retrouvé tout son éclat, devenant une source de revenus pour les artisans.", "culture"),
    ("L'artisanat marocain est de plus en plus demandé sur les marchés internationaux et génère des revenus importants pour les artisans.", "culture"),
    # agriculture
    ("La campagne agricole a été excellente cette année grâce à des précipitations favorables et a donné une récolte céréalière abondante.", "agri"),
    ("Le programme d'appui aux petits agriculteurs les a aidés à moderniser leur production et à augmenter leurs revenus.", "agri"),
    ("Les exportations agricoles de fruits et légumes ont fortement progressé et ouvert de nouveaux marchés à l'étranger.", "agri"),
    ("Les projets d'irrigation au goutte-à-goutte ont permis d'économiser l'eau et d'accroître les rendements malgré la rareté des pluies.", "agri"),
    # diplomacy / national
    ("Une grande puissance a reconnu la souveraineté du Maroc sur son Sahara, une victoire diplomatique importante pour le Royaume.", "diplomacy"),
    ("Le Maroc a accueilli un sommet international d'envergure, signe de la place qu'il occupe désormais sur la scène mondiale.", "diplomacy"),
    ("Un nouvel accord de partenariat avec l'Union européenne ouvrira de nouveaux marchés aux produits marocains.", "diplomacy"),
    ("Les transferts de la diaspora marocaine ont atteint un niveau record et soutenu fortement l'économie nationale.", "diplomacy"),
    # technology / startups / youth / social
    ("Une start-up marocaine spécialisée dans la technologie a levé des fonds importants et va recruter de jeunes ingénieurs.", "tech"),
    ("De jeunes Marocains ont conçu une application au service des agriculteurs et remporté un prix international de l'innovation.", "tech"),
    ("La numérisation de l'administration permet désormais aux citoyens d'effectuer leurs démarches en ligne sans se déplacer.", "tech"),
    ("Le taux de scolarisation des filles en milieu rural a fortement augmenté, une étape importante vers l'égalité.", "social"),
    ("Des coopératives féminines exportent désormais leurs produits et ont amélioré le revenu de centaines de femmes.", "social"),
    ("L'Initiative nationale pour le développement humain a amélioré les conditions de vie de milliers de familles dans les régions défavorisées.", "social"),
    ("Le microcrédit a permis à de nombreuses personnes modestes de lancer des activités génératrices de revenus et de sortir de la pauvreté.", "opportunity"),
    ("Désormais, tout petit entrepreneur peut soumissionner aux marchés publics et bénéficier du budget considérable que l'État dépense chaque année.", "opportunity"),
]

# ── NEGATIVE (content reports something bad) ─────────────────────────────────
NEG = [
    ("Les prix des produits alimentaires de base ont flambé ce mois-ci et de nombreuses familles peinent à subvenir à leurs besoins.", "prices"),
    ("La sécheresse qui a frappé le pays cette année a ruiné les récoltes des agriculteurs et aggravé la détresse du monde rural.", "drought"),
    ("Une pénurie d'eau potable dans certaines régions a contraint les habitants à réclamer des solutions d'urgence.", "water"),
    ("L'hôpital public souffre d'un manque criant de médecins et d'équipements, et les patients attendent des mois pour un rendez-vous.", "health"),
    ("L'émigration des médecins et des compétences à l'étranger s'est accentuée, laissant le système de santé exsangue.", "braindrain"),
    ("Le chômage des jeunes a atteint des niveaux préoccupants et des centaines de milliers de diplômés restent sans emploi.", "unemployment"),
    ("Un nouveau scandale de corruption a secoué l'opinion publique et englouti des milliards de dirhams d'argent public.", "corruption"),
    ("Un grave accident de la route a fait des morts et des blessés, relançant la question de la sécurité routière dans le pays.", "accident"),
    ("La pollution industrielle a dégradé l'environnement et les habitants de la région souffrent désormais de graves maladies respiratoires.", "pollution"),
    ("La classe moyenne se plaint du poids des impôts et de la cherté de la vie et a le sentiment de supporter seule tout le fardeau.", "prices"),
    ("Le projet de développement a pris des années de retard à cause d'une mauvaise gestion et les fonds se sont volatilisés sans bénéfice pour le citoyen.", "governance"),
    ("La surcharge des classes et le manque d'enseignants nuisent à la qualité de l'enseignement et compromettent l'avenir des élèves.", "education"),
    ("La hausse des prix des carburants s'est répercutée sur tout et a accru la pression sur le budget des ménages modestes.", "prices"),
    ("De nombreux villages restent privés de routes, de dispensaires et d'écoles, et leurs habitants vivent dans des conditions très difficiles.", "rural"),
    ("Des infrastructures vétustes n'ont pas résisté aux inondations : de nombreuses maisons ont été submergées et les biens des habitants endommagés.", "disaster"),
    ("La corruption dans certaines administrations continue d'entraver le citoyen, contraint de payer pour obtenir ce qui lui est dû.", "corruption"),
    ("La fermeture d'une usine a laissé des centaines d'ouvriers sans emploi et aggravé le chômage dans la ville.", "unemployment"),
    ("Les longues listes d'attente dans les hôpitaux ont conduit des patients à mourir avant d'avoir pu être soignés.", "health"),
    ("La flambée des loyers dans les grandes villes empêche les jeunes de se loger et de commencer leur vie de manière autonome.", "prices"),
    ("Des manifestations ont éclaté pour réclamer une réforme du secteur de la santé après la dégradation des services et la multiplication des plaintes.", "protest"),
    ("L'évasion fiscale de certains gros contribuables prive le Trésor de recettes importantes destinées aux services publics.", "corruption"),
    ("Le retard dans le versement des indemnités aux sinistrés a aggravé leur détresse et nourri un sentiment d'injustice et d'abandon.", "governance"),
    ("Le taux de pauvreté reste élevé dans certaines régions, où des familles vivent sans revenu stable ni couverture médicale.", "poverty"),
    ("Des coupures d'eau répétées dans certains quartiers ont éprouvé les habitants, surtout pendant les chaudes journées d'été.", "water"),
    ("La faiblesse du réseau internet dans les zones reculées prive les élèves de l'enseignement à distance et de toute chance d'apprendre.", "rural"),
    ("La hausse de l'endettement du pays inquiète les économistes et pèse lourdement sur le budget de l'État.", "economy"),
    ("La dégradation du pouvoir d'achat a contraint de nombreux ménages à réduire leurs dépenses, jusque sur les produits de première nécessité.", "prices"),
    ("La surpopulation des hôpitaux et le manque de lits ont contraint des malades à être soignés dans les couloirs, dans des conditions indignes.", "health"),
    ("L'abandon scolaire en milieu rural demeure élevé, en particulier chez les filles qui quittent l'école très tôt.", "education"),
    ("La désertification et l'avancée des sables menacent les terres agricoles et poussent les habitants à quitter leurs régions.", "drought"),
    ("La mauvaise gestion des déchets dans certaines villes a créé des problèmes environnementaux et sanitaires dont se plaignent les habitants.", "pollution"),
    ("Les prix de l'immobilier ont grimpé de façon vertigineuse et ont éloigné le rêve d'accéder à la propriété pour la classe moyenne.", "prices"),
    ("Le transport public, vétuste et délabré, met en danger la vie des passagers qui s'en plaignent quotidiennement.", "transport"),
    ("Le chômage des titulaires de diplômes supérieurs augmente chaque année et pousse les compétences à émigrer.", "unemployment"),
    ("Les inondations ont emporté les routes et isolé les villages, causant des pertes dans les biens et les récoltes.", "disaster"),
    ("La faiblesse du contrôle des marchés a permis à certains commerçants de gonfler les prix et de spéculer sur le dos des consommateurs.", "prices"),
    ("La pénurie de ressources hydriques, après plusieurs années de sécheresse, a imposé des restrictions d'irrigation et frappé l'agriculture.", "drought"),
    ("Le déficit d'investissement public dans certaines régions les a maintenues dans le sous-développement et creusé l'écart avec les grandes villes.", "governance"),
    ("La concurrence déloyale a poussé de petits producteurs à la faillite, incapables de tenir face à des prix cassés.", "economy"),
    ("La pénurie de main-d'œuvre qualifiée freine certaines entreprises et limite leur capacité à se développer.", "economy"),
]

# ── NEUTRAL (procedural / definitional / descriptive — no recoverable valence) ──
NEU = [
    ("La loi de finances est un document qui fixe les recettes et les dépenses de l'État pour une année entière et qui est voté par le Parlement.", "definition"),
    ("La taxe sur la valeur ajoutée s'applique à la plupart des biens et services et est acquittée par le consommateur final.", "definition"),
    ("La Cour des comptes est une institution chargée de contrôler l'emploi des deniers publics et de publier des rapports annuels.", "definition"),
    ("Un marché public passe par plusieurs étapes, de la définition des besoins à l'annonce du soumissionnaire retenu, selon un cahier des charges.", "procedure"),
    ("Pour soumissionner à un marché public, il faut créer un compte sur le portail dédié et disposer d'une signature électronique.", "procedure"),
    ("Le système fiscal marocain répartit les revenus en catégories, chacune disposant de ses propres règles de calcul de l'impôt.", "definition"),
    ("La Bourse est un marché organisé où s'achètent et se vendent des actions et des obligations sous le contrôle d'une autorité spécialisée.", "definition"),
    ("La motion de censure est un mécanisme constitutionnel qui permet à l'opposition de mettre en cause la responsabilité du gouvernement devant le Parlement.", "definition"),
    ("La déclaration de patrimoine est une procédure à laquelle se soumettent périodiquement les responsables qui gèrent des fonds publics.", "procedure"),
    ("La couverture médicale obligatoire est financée par les cotisations des salariés et des employeurs ainsi que par l'appui de l'État.", "definition"),
    ("La séparation des pouvoirs est un principe constitutionnel qui répartit les compétences entre les pouvoirs législatif, exécutif et judiciaire.", "definition"),
    ("Le produit intérieur brut est un indicateur qui mesure la valeur totale des biens et services produits par un pays.", "definition"),
    ("Les obligations sont un type de dette émis par l'État ou les entreprises pour emprunter des fonds en contrepartie d'intérêts.", "definition"),
    ("Les élections législatives sont organisées tous les cinq ans pour que les citoyens choisissent leurs représentants au Parlement.", "definition"),
    ("L'histoire de la région a connu la succession de nombreux États et dynasties sur plusieurs siècles.", "history"),
    ("La campagne agricole débute à l'automne et sa réussite dépend largement de la quantité des précipitations.", "description"),
    ("Le décret sur les marchés publics définit sept types de marchés selon la nature des besoins.", "definition"),
    ("Le régime de retraite fonctionne sur le principe de la cotisation : l'adhérent cotise régulièrement pour percevoir ensuite une pension.", "definition"),
    ("La régionalisation avancée confère aux régions des compétences plus larges dans la gestion de leurs affaires locales.", "definition"),
    ("Le présent reportage revient sur l'organisation administrative du pays et sur le rôle de ses différentes institutions.", "description"),
    ("Nous recevons aujourd'hui notre invité pour évoquer le projet de réforme et ses principales étapes.", "description"),
    ("Dans un instant, nous verrons comment se déroule concrètement la procédure et quelles pièces sont demandées au citoyen.", "description"),
    ("Le rapport présente d'abord le contexte, puis les chiffres, avant de détailler les recommandations formulées.", "description"),
    ("Le programme se compose de trois volets : un état des lieux, une analyse et une série de propositions.", "description"),
    ("L'organisme en question est chargé de la régulation du secteur et veille au respect du cahier des charges par les opérateurs.", "definition"),
    ("La séance s'est ouverte par la lecture de l'ordre du jour, suivie de la présentation des différents points à examiner.", "procedure"),
    ("Le texte distingue plusieurs catégories de bénéficiaires et précise les conditions d'éligibilité pour chacune d'elles.", "definition"),
    ("Le recensement de la population est réalisé périodiquement afin d'actualiser les données démographiques du pays.", "procedure"),
    ("La commission a auditionné plusieurs responsables avant de rédiger ses conclusions, conformément à son règlement intérieur.", "procedure"),
    ("Le journaliste rappelle le calendrier des prochaines échéances et les modalités d'inscription sur les listes.", "description"),
    ("La monnaie nationale est émise par la banque centrale, qui veille à la stabilité des prix et à la régulation du crédit.", "definition"),
    ("Le contrat de travail précise la durée, la rémunération et les obligations respectives de l'employeur et du salarié.", "definition"),
    ("Le budget de l'État se divise en un volet de fonctionnement et un volet d'investissement, examinés séparément.", "definition"),
    ("La séance de questions orales permet aux parlementaires d'interroger les ministres sur la politique de leur département.", "definition"),
    ("Le document retrace les grandes étapes du chantier, depuis l'appel d'offres jusqu'à la réception des travaux.", "description"),
    ("La région se caractérise par un relief montagneux et un climat qui varie sensiblement d'une saison à l'autre.", "description"),
    ("Le rapporteur a présenté les amendements article par article avant le passage au vote.", "procedure"),
    ("L'autorité de régulation publie chaque année un bilan d'activité détaillant le nombre de dossiers traités.", "definition"),
    ("Le dispositif prévoit un guichet unique où le demandeur dépose son dossier et suit l'avancement de sa demande.", "procedure"),
    ("Le panel réunit aujourd'hui un économiste, un juriste et un représentant du secteur pour débattre du sujet.", "description"),
]


def build_rows():
    rows = []
    i = 0
    for items, label in ((POS, "pos"), (NEG, "neg"), (NEU, "neu")):
        for text, topic in items:
            i += 1
            rows.append({
                "utterance_id": f"synthfr_{i:04d}",
                "file": "synthetic_fr",
                "fmt": "synthetic",
                "detected_lang": "francais",
                "quality": "clean",
                "text": " ".join(text.split()).strip(),
                "label": label,
                "label_source": "claude-synth",
                "synthetic": True,
                "topic": topic,
            })
    return rows


def main():
    rows = build_rows()
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    fields = ["utterance_id", "file", "fmt", "detected_lang", "quality", "text",
              "label", "label_source", "synthetic", "topic"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    dist = Counter(r["label"] for r in rows)
    print(f"Wrote {len(rows)} synthetic French rows → {OUT_CSV}")
    print(f"  distribution: {dict(dist)}")


if __name__ == "__main__":
    main()
