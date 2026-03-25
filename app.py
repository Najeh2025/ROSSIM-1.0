# =============================================================================
# ROSSim Online — v1.0
# Application Web complète de Dynamique des Rotors basée sur ROSS
# Conforme au Cahier des Charges (CdC) — Sprints S1→S4
# =============================================================================
# Modules couverts :
#   M1 — Constructeur de Rotor
#   M2 — Analyses Statiques & Modales
#   M3 — Campbell & Stabilité
#   M4 — Balourd & Réponse Fréquentielle
#   M5 — Réponse Temporelle & Défauts (Crack, Misalignment, Rubbing)
#   Tutoriels officiels ROSS : Part 1, 2.1, 2.2, Part 4 (MultiRotor)
#   ROSS GPT — Assistant IA contextuel (Anthropic Claude)
# =============================================================================

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json
import io
import traceback

try:
    import ross as rs
    ROSS_AVAILABLE = True
    ROSS_VERSION = getattr(rs, '__version__', 'unknown')
except ImportError:
    ROSS_AVAILABLE = False
    ROSS_VERSION = "non installé"

# =============================================================================
# CONFIGURATION GLOBALE
# =============================================================================
st.set_page_config(
    page_title="ROSSim Online",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "ROSSim Online — Application de Dynamique des Rotors • Basée sur ROSS"}
)

# ── CSS Global ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] { height: 44px; font-weight: 600; font-size: 13px; border-radius: 8px 8px 0 0; }
/* Cards */
.card { background:#F0F4FF; border-left:5px solid #1F5C8B; border-radius:8px; padding:16px 20px; margin:8px 0; }
.card-green  { background:#F0FFF4; border-left:5px solid #22863A; border-radius:8px; padding:14px 18px; margin:6px 0; }
.card-orange { background:#FFF8E1; border-left:5px solid #C55A11; border-radius:8px; padding:14px 18px; margin:6px 0; }
.card-red    { background:#FFE6E6; border-left:5px solid #C00000; border-radius:8px; padding:14px 18px; margin:6px 0; }
/* Badges */
.badge { display:inline-block; padding:4px 14px; border-radius:20px; font-size:12px; font-weight:700; margin:2px; }
.badge-gold   { background:#FFD700; color:#7A5700; }
.badge-silver { background:#C0C0C0; color:#3A3A3A; }
.badge-bronze { background:#CD7F32; color:#fff; }
.badge-blue   { background:#1F5C8B; color:#fff; }
/* Module badges */
.mod-badge { display:inline-block; padding:3px 10px; border-radius:12px; font-size:11px;
             font-weight:600; margin:2px; background:#EBF4FB; color:#1F5C8B; border:1px solid #AED6F1; }
/* Code blocks */
.code-box { background:#1E1E1E; color:#D4D4D4; padding:12px 16px; border-radius:6px;
            font-family:monospace; font-size:12px; margin:8px 0; white-space:pre; overflow-x:auto; }
/* Status */
.status-ok   { background:#E6FFE6; border:1px solid #22863A; border-radius:6px; padding:8px 14px; display:inline-block; }
.status-warn { background:#FFF8E1; border:1px solid #F9A825; border-radius:6px; padding:8px 14px; display:inline-block; }
.status-err  { background:#FFE6E6; border:1px solid #C00000; border-radius:6px; padding:8px 14px; display:inline-block; }
/* Progress */
div[data-testid="stProgress"] > div > div { background:#1F5C8B !important; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CACHE MODULE-LEVEL (objets ROSS non sérialisables — CdC §2.12)
# =============================================================================
if "_CACHE" not in st.session_state:
    st.session_state["_CACHE"] = {}
_CACHE = st.session_state["_CACHE"]

# =============================================================================
# MATÉRIAUX
# =============================================================================
MAT_STEEL = None
if ROSS_AVAILABLE:
    try:
        MAT_STEEL = rs.Material(name="Steel", rho=7810, E=211e9, G_s=81.2e9)
    except Exception:
        pass

MATERIALS_DB = {
    "Acier standard (AISI 1045)": {"rho": 7810.0, "E": 211e9, "G_s": 81.2e9},
    "Acier inoxydable (316L)":     {"rho": 7990.0, "E": 193e9, "G_s": 74.0e9},
    "Aluminium (7075-T6)":         {"rho": 2810.0, "E":  72e9, "G_s": 27.0e9},
    "Titane (Ti-6Al-4V)":          {"rho": 4430.0, "E": 114e9, "G_s": 44.0e9},
    "Inconel 718":                 {"rho": 8220.0, "E": 200e9, "G_s": 77.0e9},
    "Personnalisé":                {"rho": 7810.0, "E": 211e9, "G_s": 81.2e9},
}

BEARING_PRESETS = {
    "Roulement à billes":         {"kxx":1e7,"kyy":1e7,"kxy":0.0,"cxx":500.0,"cyy":500.0},
    "Palier lisse hydrodynamique":{"kxx":1e7,"kyy":5e6,"kxy":2e6,"cxx":2000.0,"cyy":2000.0},
    "Support souple (amortisseur)":{"kxx":1e6,"kyy":1e6,"kxy":0.0,"cxx":5000.0,"cyy":5000.0},
    "Palier rigide (encastrement)":{"kxx":1e9,"kyy":1e9,"kxy":0.0,"cxx":100.0,"cyy":100.0},
    "Personnalisé":               {"kxx":1e7,"kyy":1e7,"kxy":0.0,"cxx":500.0,"cyy":500.0},
}

# =============================================================================
# CATALOGUE TUTORIELS (CdC §2.4)
# =============================================================================
TUTORIALS = {
    "T1": {
        "title": "Part 1 — Création du Modèle (Modeling)",
        "level": "🟢 Débutant",
        "duration": "~15 min",
        "api": ["Material", "ShaftElement", "DiskElement", "BearingElement", "Rotor", "plot_rotor()"],
        "steps": [
            {"id": "T1_S1", "title": "Définir un matériau",
             "theory": "Le matériau définit les propriétés physiques de l'arbre : densité ρ (kg/m³), module d'Young E (Pa) et module de cisaillement G_s (Pa). Ces valeurs déterminent la masse et la rigidité du modèle.",
             "objective": "Créer un matériau Acier avec ρ=7810 kg/m³, E=211 GPa, G_s=81.2 GPa",
             "code": "import ross as rs\nsteel = rs.Material(name='Steel', rho=7810, E=211e9, G_s=81.2e9)\nprint(steel)"},
            {"id": "T1_S2", "title": "Créer les éléments d'arbre",
             "theory": "L'arbre est discrétisé en éléments de poutre de Timoshenko. Chaque ShaftElement est défini par sa longueur L (m), son diamètre intérieur idl et extérieur odl. Les nœuds sont numérotés automatiquement de 0 à N.",
             "objective": "Créer 5 éléments d'arbre de L=0.2m, Ø50mm",
             "code": "shaft = [rs.ShaftElement(L=0.25, idl=0.0, odl=0.05, material=steel)\n         for _ in range(6)]"},
            {"id": "T1_S3", "title": "Ajouter un disque",
             "theory": "Les disques sont modélisés comme des corps rigides. from_geometry() calcule automatiquement la masse et les inerties depuis la géométrie (largeur, diamètre intérieur/extérieur).",
             "objective": "Ajouter un disque Ø250mm au nœud central",
             "code": "disk = rs.DiskElement.from_geometry(\n    n=2, material=steel,\n    width=0.07, i_d=0.05, o_d=0.25\n)"},
            {"id": "T1_S4", "title": "Définir les paliers",
             "theory": "Les paliers sont modélisés comme des éléments ressort-amortisseur linéaires. kxx, kyy = raideurs directes (N/m) ; kxy = raideur croisée ; cxx, cyy = amortissements (N·s/m).",
             "objective": "Créer 2 paliers kxx=kyy=1e7 N/m aux extrémités",
             "code": "bear0 = rs.BearingElement(n=0, kxx=1e7, kyy=1e7, cxx=500, cyy=500)\nbear5 = rs.BearingElement(n=5, kxx=1e7, kyy=1e7, cxx=500, cyy=500)"},
            {"id": "T1_S5", "title": "Assembler et visualiser le rotor",
             "theory": "rs.Rotor() assemble les éléments en construisant les matrices globales M, K, C, G par superposition des contributions élémentaires. plot_rotor() affiche la géométrie 3D interactive.",
             "objective": "Assembler rotor et afficher : masse totale + géométrie 3D",
             "code": "rotor = rs.Rotor(shaft, [disk], [bear0, bear5])\nprint(f'Masse : {rotor.m:.2f} kg')\nrotor.plot_rotor()"},
        ]
    },
    "T2_1": {
        "title": "Part 2.1 — Analyses Statiques & Modales",
        "level": "🔵 Intermédiaire",
        "duration": "~20 min",
        "api": ["run_static()", "run_modal()", "run_campbell()", "run_critical_speed()", "plot_mode_3d()"],
        "steps": [
            {"id": "T21_S1", "title": "Analyse statique (gravité)",
             "theory": "run_static() calcule la déflexion de l'arbre sous son propre poids et celui des disques. Les réactions aux paliers, le diagramme de moment fléchissant et de cisaillement sont automatiquement calculés.",
             "objective": "Calculer et afficher la déflexion statique du rotor",
             "code": "static = rotor.run_static()\nstatic.plot_deflected_shape()\n# Attributs : static.shaft_deflection, static.disk_forces"},
            {"id": "T21_S2", "title": "Analyse modale",
             "theory": "run_modal() résout le problème aux valeurs propres : det(K + jωC − ω²M) = 0. Les valeurs propres complexes donnent les fréquences naturelles (wn, wd) et le décrément logarithmique (log_dec). La direction de précession (avant/arrière) est aussi calculée.",
             "objective": "Calculer les 6 premiers modes à vitesse nulle",
             "code": "modal = rotor.run_modal(speed=0)\nprint('fn (Hz):', modal.wn[:6] / (2*np.pi))\nprint('Log Dec:', modal.log_dec[:6])\nmodal.plot_mode_3d(mode=0)"},
            {"id": "T21_S3", "title": "Diagramme de Campbell",
             "theory": "Le diagramme de Campbell trace les fréquences des modes en fonction de la vitesse de rotation. L'effet gyroscopique sépare les modes de précession avant (FW) et arrière (BW). Les intersections avec la droite 1X donnent les vitesses critiques synchrones.",
             "objective": "Tracer le Campbell de 0 à 10 000 RPM",
             "code": "speeds = np.linspace(0, 10000*np.pi/30, 100)\ncamp = rotor.run_campbell(speeds)\ncamp.plot()"},
            {"id": "T21_S4", "title": "Vitesses critiques",
             "theory": "run_critical_speed() calcule les vitesses critiques non amorties (intersections des modes avec la droite 1X). La marge API 684 exige que la plage opérationnelle soit à ±15% de toute vitesse critique.",
             "objective": "Identifier les vitesses critiques et vérifier la marge API 684",
             "code": "# Vitesses critiques depuis analyse modale\nfn = modal.wn / (2*np.pi)  # Hz\nvc_rpm = fn * 60  # RPM\nprint('Vitesses critiques (RPM):', vc_rpm[:4].round(0))"},
        ]
    },
    "T2_2": {
        "title": "Part 2.2 — Analyses Temporelles & Fréquentielles",
        "level": "🔵 Intermédiaire",
        "duration": "~25 min",
        "api": ["run_unbalance_response()", "run_freq_response()", "run_time_response()", "Probe", "plot_orbit()", "plot_dfft()"],
        "steps": [
            {"id": "T22_S1", "title": "Configuration des sondes (Probe)",
             "theory": "Une sonde virtuelle (Probe) est placée sur un nœud à un angle donné. Elle simule un capteur de déplacement (eddy current probe). Les résultats sont exprimés dans le repère de la sonde : composantes directe et en quadrature.",
             "objective": "Définir 2 sondes à 45° sur le nœud 2",
             "code": "from ross import Probe\nprobe1 = Probe(2, 45.0)   # nœud 2, angle 45°\nprobe2 = Probe(2, 135.0)  # nœud 2, angle 135°"},
            {"id": "T22_S2", "title": "Réponse au balourd",
             "theory": "Un balourd (déséquilibre de masse) génère une force tournante F = m·e·ω². run_unbalance_response() calcule la réponse en fréquence. Le DAF (Dynamic Amplification Factor) = Amplitude_max / Amplitude_statique mesure l'amplification à la résonance.",
             "objective": "Analyser la réponse à un balourd de 0.001 kg·m au nœud 2",
             "code": "resp = rotor.run_unbalance_response(\n    node=[2],\n    magnitude=[0.001],  # kg.m\n    phase=[0.0],\n    frequency_range=np.linspace(0, 5000, 500)\n)\nresp.plot_magnitude(probe=[2, 0])\nresp.plot_phase(probe=[2, 0])"},
            {"id": "T22_S3", "title": "Réponse fréquentielle H(jω)",
             "theory": "run_freq_response() calcule la fonction de transfert H(jω) entre un DDL d'excitation et un DDL de réponse. Le diagramme de Bode (magnitude en dB + phase en degrés) identifie les modes et leurs facteurs d'amortissement.",
             "objective": "Calculer H(jω) entre le DDL 0 (excitation nœud 0, direction X) et le DDL 8 (nœud 2, direction X)",
             "code": "freq_resp = rotor.run_freq_response(\n    inp=0, out=8,\n    frequency_range=np.linspace(0, 5000, 500)\n)\nfreq_resp.plot_bode(inp=0, out=8)"},
            {"id": "T22_S4", "title": "Réponse temporelle et orbites",
             "theory": "run_time_response() intègre l'équation du mouvement dans le domaine temporel (méthode de Newmark). Les orbites décrivent la trajectoire du centre de l'arbre dans le plan XY. La DFFT donne le spectre de vibration.",
             "objective": "Simuler le transitoire de démarrage et afficher les orbites au nœud 2",
             "code": "# Force balourd tournante\nt = np.linspace(0, 2, 1000)\nomega = 3000 * np.pi/30\nF = np.zeros((rotor.ndof, len(t)))\n# Appliquer force au nœud 2\ntime_resp = rotor.run_time_response(\n    speed=omega, force=F, time_range=t\n)\ntime_resp.plot_orbit(node=2)\ntime_resp.plot_dfft(probe=[2, 0], rpm=3000)"},
        ]
    },
    "T4": {
        "title": "Part 4 — Système Multi-Rotors (MultiRotor & GearElement)",
        "level": "🔴 Avancé",
        "duration": "~30 min",
        "api": ["GearElement", "MultiRotor", "couplage latéral-torsionnel"],
        "steps": [
            {"id": "T4_S1", "title": "Création des deux rotors",
             "theory": "Un système engrenage connecte deux rotors tournant à des vitesses différentes. Chaque rotor est créé indépendamment. Le rapport de réduction est déterminé par le nombre de dents.",
             "objective": "Créer le rotor moteur (4 éléments) et le rotor récepteur (3 éléments)",
             "code": "# Rotor 1 (moteur)\nshaft1 = [rs.ShaftElement(L=0.25, idl=0, odl=0.05, material=steel)\n          for _ in range(4)]\n# Rotor 2 (récepteur)\nshaft2 = [rs.ShaftElement(L=0.25, idl=0, odl=0.04, material=steel)\n          for _ in range(3)]"},
            {"id": "T4_S2", "title": "Définir l'élément engrenage",
             "theory": "GearElement modélise le couplage latéral-torsionnel via la ligne d'action des dents. Les paramètres clés sont : pitch_diameter (diamètre de tête), pressure_angle (angle de pression typiquement 20° ou 22.5°) et helix_angle (engrenage hélicoïdal).",
             "objective": "Créer un engrenage droit (pression 20°, Ø100mm) au nœud 2 du rotor 1",
             "code": "gear = rs.GearElement(\n    n=2,\n    pitch_diameter=0.1,\n    pressure_angle=np.radians(20),\n    helix_angle=0.0\n)"},
            {"id": "T4_S3", "title": "Assembler le système MultiRotor",
             "theory": "MultiRotor couple les deux rotors via l'engrenage. Les matrices globales incluent le couplage cinématique imposé par la ligne d'action. L'analyse modale révèle les modes couplés latéraux-torsionnels.",
             "objective": "Assembler le MultiRotor et analyser ses modes couplés",
             "code": "# Note: API MultiRotor dépend de la version ROSS\n# Cf. documentation Part 4 du tutoriel officiel\nrotor1 = rs.Rotor(shaft1, disks1, bears1)\nrotor2 = rs.Rotor(shaft2, disks2, bears2)\n# Couplage via GearElement au nœud spécifié"},
            {"id": "T4_S4", "title": "Analyse Campbell du système couplé",
             "theory": "Le diagramme de Campbell d'un système engrenage montre les modes latéraux de chaque rotor, les modes torsionnels et les modes couplés. Les fréquences d'engrènement apparaissent comme des excitations supplémentaires.",
             "objective": "Tracer le Campbell couplé 0–10 000 RPM (référencé sur rotor 1)",
             "code": "# Campbell du système couplé\nspeeds = np.linspace(0, 10000*np.pi/30, 80)\n# Référencer les vitesses sur le rotor primaire\ncampbell_coupled = multi_rotor.run_campbell(speeds)\ncampbell_coupled.plot()"},
        ]
    },
}

# =============================================================================
# CLASSES MÉTIER
# =============================================================================

class RotorBuilder:
    """M1 — Constructeur de rotor ROSS avec validation complète (CdC §2.3)."""

    def __init__(self):
        self._shaft: List = []
        self._disks: List = []
        self._bears: List = []
        self._errors: List[str] = []
        self._warnings: List[str] = []
        self.material = MAT_STEEL

    def set_material(self, name: str) -> "RotorBuilder":
        if not ROSS_AVAILABLE:
            return self
        props = MATERIALS_DB.get(name, MATERIALS_DB["Acier standard (AISI 1045)"])
        try:
            self.material = rs.Material(name=name.replace(" ", "_"), rho=props["rho"], E=props["E"], G_s=props["G_s"])
        except Exception as e:
            self._errors.append(f"Matériau invalide : {e}")
        return self

    def add_shaft_from_df(self, df: pd.DataFrame) -> "RotorBuilder":
        if not ROSS_AVAILABLE:
            return self
        self._shaft.clear()
        for i, row in df.iterrows():
            try:
                L, idl, odl = float(row.get("L (m)", 0.2)), float(row.get("id (m)", 0.0)), float(row.get("od (m)", 0.05))
                if L <= 0:
                    self._errors.append(f"Élément {i+1} : L doit être > 0")
                    continue
                if idl >= odl:
                    self._errors.append(f"Élément {i+1} : id doit être < od")
                    continue
                self._shaft.append(rs.ShaftElement(L=L, idl=idl, odl=odl, material=self.material))
            except Exception as e:
                self._errors.append(f"Élément d'arbre {i+1} : {e}")
        return self

    def add_disk(self, node: int, od: float, width: float, id_: float = 0.0) -> "RotorBuilder":
        if not ROSS_AVAILABLE:
            return self
        n_nodes = len(self._shaft) + 1
        if node < 0 or node >= n_nodes:
            self._errors.append(f"Nœud disque {node} invalide — arbre a {n_nodes} nœuds (0→{n_nodes-1})")
            return self
        if id_ >= od:
            self._errors.append(f"Disque nœud {node} : diamètre intérieur ≥ extérieur")
            return self
        try:
            self._disks.append(rs.DiskElement.from_geometry(
                n=node, material=self.material, width=width, i_d=id_, o_d=od))
        except Exception as e:
            self._errors.append(f"Disque nœud {node} : {e}")
        return self

    def add_bearing(self, node: int, kxx: float, kyy: float,
                    kxy: float = 0.0, cxx: float = 500.0, cyy: float = 500.0) -> "RotorBuilder":
        if not ROSS_AVAILABLE:
            return self
        n_nodes = len(self._shaft) + 1
        if node < 0 or node >= n_nodes:
            self._errors.append(f"Nœud palier {node} invalide — arbre a {n_nodes} nœuds (0→{n_nodes-1})")
            return self
        if kxx <= 0 or kyy <= 0:
            self._warnings.append(f"Palier nœud {node} : raideur très faible (< 0)")
        try:
            self._bears.append(rs.BearingElement(
                n=node, kxx=kxx, kyy=kyy, kxy=kxy, kyx=-kxy, cxx=cxx, cyy=cyy))
        except Exception as e:
            self._errors.append(f"Palier nœud {node} : {e}")
        return self

    def build(self):
        if self._errors:
            return None
        if not self._shaft:
            self._errors.append("Aucun élément d'arbre défini")
            return None
        if not self._bears:
            self._errors.append("Aucun palier défini")
            return None
        try:
            return rs.Rotor(self._shaft, self._disks, self._bears)
        except Exception as e:
            self._errors.append(f"Assemblage impossible : {e}")
            return None

    @property
    def errors(self): return self._errors
    @property
    def warnings(self): return self._warnings
    @property
    def n_nodes(self): return len(self._shaft) + 1


class SimulationEngine:
    """Moteur de simulation avec fallbacks multi-version ROSS (CdC §2.12)."""

    def __init__(self, rotor):
        self.rotor = rotor
        self._err = ""

    def run_modal(self, speed_rpm=0.0):
        try:
            return self.rotor.run_modal(speed=float(speed_rpm) * np.pi / 30)
        except Exception as e:
            self._err = str(e); return None

    def run_campbell(self, vmax_rpm=8000.0, n=100):
        try:
            sp = np.linspace(0, float(vmax_rpm) * np.pi / 30, int(n))
            return self.rotor.run_campbell(sp)
        except Exception as e:
            self._err = str(e); return None

    def run_static(self):
        try:
            return self.rotor.run_static()
        except Exception as e:
            self._err = str(e); return None

    def run_unbalance(self, nodes, mags, phases, fmax, n=500):
        freqs = np.linspace(0, float(fmax), int(n))
        
        # Extraction des valeurs (ROSS récent préfère des scalaires)
        n_val = nodes[0] if isinstance(nodes, list) else nodes
        m_val = mags[0] if isinstance(mags, list) else mags
        p_val = phases[0] if isinstance(phases, list) else phases

        # Tentative 1 : ROSS récent (frequency)
        try:
            return self.rotor.run_unbalance_response(
                node=n_val, unbalance_magnitude=m_val, unbalance_phase=p_val, frequency=freqs)
        except TypeError:
            pass

        # Tentative 2 : ROSS intermédiaire (frequency_range)
        try:
            return self.rotor.run_unbalance_response(
                node=n_val, magnitude=m_val, phase=p_val, frequency_range=freqs)
        except TypeError:
            pass
            
        # Tentative 3 : ROSS ancien (speed_range)
        try:
            return self.rotor.run_unbalance_response(
                node=n_val, magnitude=m_val, phase=p_val, speed_range=freqs * 2 * np.pi)
        except Exception as e:
            self._err = f"Échec unbalance : {str(e)}"
            return None

    def run_freq_response(self, inp, out, fmax, n=500):
        # Les arguments inp/out sont ignorés ici car ROSS ne les utilise que pour le tracé (plot)
        freqs = np.linspace(0, float(fmax), int(n))
        
        # Tentative 1 : frequency
        try:
            return self.rotor.run_freq_response(frequency=freqs)
        except TypeError:
            pass
            
        # Tentative 2 : frequency_range
        try:
            return self.rotor.run_freq_response(frequency_range=freqs)
        except TypeError:
            pass
            
        # Tentative 3 : speed_range
        try:
            return self.rotor.run_freq_response(speed_range=freqs * 2 * np.pi)
        except Exception as e:
            self._err = f"Échec freq_response : {str(e)}"
            return None

    def run_time_response(self, speed_rpm, F, t):
        speed_rad = float(speed_rpm) * np.pi / 30
        
        last_err = ""
        # On teste la matrice F telle quelle, puis sa transposée F.T
        for force_mat in [F, F.T]:
            # Tentative 1 : API ROSS récente ('F' et 't')
            try:
                return self.rotor.run_time_response(speed=speed_rad, F=force_mat, t=t)
            except ValueError as e:
                last_err = str(e)
                if "same number of rows" in str(e):
                    continue  # Si SciPy râle sur les dimensions, on essaie la transposée
            except TypeError:
                pass
                
            # Tentative 2 : Ancienne API ROSS ('force' et 'time_range')
            try:
                return self.rotor.run_time_response(speed=speed_rad, force=force_mat, time_range=t)
            except ValueError as e:
                last_err = str(e)
                if "same number of rows" in str(e):
                    continue
            except TypeError:
                pass
                
            # Tentative 3 : Arguments positionnels
            try:
                return self.rotor.run_time_response(speed_rad, force_mat, t)
            except ValueError as e:
                last_err = str(e)
                if "same number of rows" in str(e):
                    continue
            except Exception as e:
                self._err = f"Échec temporel : {str(e)}"
                return None
                
        self._err = f"Échec de la réponse temporelle : {last_err}"
        return None

    def run_crack(self, **kwargs):
        speed = kwargs.get('speed', 150.0)
        crack_node = kwargs.get('crack_node', 1)
        crack_depth = kwargs.get('crack_depth', 0.1)
        
        # Paramètres d'excitation par défaut pour ROSS récent
        t_arr = np.linspace(0, 1.0, 500)
        
        try:
            # Tentative 1 : Nouvelle API (avec balourd d'excitation intégré)
            return self.rotor.run_crack(
                n=crack_node, 
                depth_ratio=crack_depth, 
                speed=speed,
                node=crack_node, 
                unbalance_magnitude=1e-4, 
                unbalance_phase=0.0, 
                t=t_arr
            )
        except TypeError:
            try:
                # Tentative 2 : Ancienne API ROSS
                return self.rotor.run_crack(**kwargs)
            except Exception as e:
                self._err = str(e); return None
        except Exception as e:
            self._err = str(e); return None

    def run_misalignment(self, **kwargs):
        speed = kwargs.get('speed', 150.0)
        n = kwargs.get('n', 1)
        misalignment = kwargs.get('misalignment', 0.001)
        
        t_arr = np.linspace(0, 1.0, 500)
        
        try:
            return self.rotor.run_misalignment(
                n=n, 
                misalignment=misalignment, 
                speed=speed,
                node=n, 
                unbalance_magnitude=1e-4, 
                unbalance_phase=0.0, 
                t=t_arr
            )
        except TypeError:
            try:
                return self.rotor.run_misalignment(**kwargs)
            except Exception as e:
                self._err = str(e); return None
        except Exception as e:
            self._err = str(e); return None

    def run_rubbing(self, **kwargs):
        speed = kwargs.get('speed', 150.0)
        n = kwargs.get('n', 1)
        distance = kwargs.get('radial_clearance', 0.0001)
        k_contact = kwargs.get('contact_stiffness', 1e7)
        
        t_arr = np.linspace(0, 1.0, 500)
        
        try:
            return self.rotor.run_rubbing(
                n=n, 
                contact_stiffness=k_contact, 
                distance=distance, 
                contact_damping=0.0,      # Nouvel argument requis
                friction_coeff=0.1,       # Nouvel argument requis
                speed=speed,
                node=n, 
                unbalance_magnitude=1e-4, 
                unbalance_phase=0.0, 
                t=t_arr
            )
        except TypeError:
            try:
                return self.rotor.run_rubbing(**kwargs)
            except Exception as e:
                self._err = str(e); return None
        except Exception as e:
            self._err = str(e); return None

    @property
    def last_error(self): return self._err


class ReportGenerator:
    """Export HTML, CSV, code Python (CdC §2.8)."""

    def __init__(self, username="Utilisateur"):
        self.username = username
        self.ts = datetime.now().strftime("%Y-%m-%d %H:%M")

    def html_report(self, title, params, sections):
        body_html = ""
        for s in sections:
            body_html += f"<h2 style='color:#C55A11'>{s['title']}</h2>"
            if "table" in s:
                df = s["table"]
                body_html += df.to_html(index=False, border=0, classes="data-table")
            if "text" in s:
                body_html += f"<p>{s['text']}</p>"
        params_rows = "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
                               for k, v in params.items())
        return f"""<!DOCTYPE html>
<html><head><meta charset='UTF-8'>
<title>ROSSim — {title}</title>
<style>
  body {{ font-family:Arial,sans-serif; max-width:960px; margin:40px auto; color:#333; }}
  h1 {{ color:#1F5C8B; border-bottom:3px solid #1F5C8B; padding-bottom:8px; }}
  h2 {{ color:#C55A11; }}
  table {{ width:100%; border-collapse:collapse; margin:12px 0; }}
  th {{ background:#1F5C8B; color:#fff; padding:8px 10px; text-align:left; }}
  td {{ padding:6px 10px; border:1px solid #ddd; }}
  tr:nth-child(even) {{ background:#F8F8F8; }}
  .data-table th {{ background:#C55A11; }}
  .footer {{ margin-top:40px; color:#999; font-size:.85em; border-top:1px solid #eee; padding-top:10px; }}
</style></head><body>
<h1>⚙️ ROSSim Online — {title}</h1>
<p><b>Généré par :</b> {self.username} &nbsp;|&nbsp; <b>Date :</b> {self.ts}</p>
<h2>Paramètres</h2>
<table><tr><th>Paramètre</th><th>Valeur</th></tr>{params_rows}</table>
{body_html}
<div class='footer'>ROSSim Online — Basé sur ROSS (Rotordynamic Open-Source Software) · ross.readthedocs.io</div>
</body></html>"""

    def python_script(self, rotor_params: dict) -> str:
        lines = [
            "# Script ROSS généré par ROSSim Online",
            f"# Date : {self.ts}",
            "import ross as rs",
            "import numpy as np",
            "",
            "# ── Matériau ──",
            f"mat = rs.Material(name='Steel', rho={rotor_params.get('rho',7810)}, "
            f"E={rotor_params.get('E',211e9):.2e}, G_s={rotor_params.get('G_s',81.2e9):.2e})",
            "",
            "# ── Arbre ──",
        ]
        for i, el in enumerate(rotor_params.get("shaft", [])):
            lines.append(f"# Élément {i+1}")
        lines += [
            "",
            "# ── Assemblage ──",
            "rotor = rs.Rotor(shaft, disks, bearings)",
            "print(f'Masse : {rotor.m:.2f} kg | Nœuds : {rotor.nodes}')",
            "",
            "# ── Analyse modale ──",
            "modal = rotor.run_modal(speed=0)",
            "print('fn (Hz):', (modal.wn / (2*np.pi))[:6].round(2))",
            "print('Log Dec:', modal.log_dec[:6].round(4))",
            "",
            "# ── Campbell ──",
            "speeds = np.linspace(0, 8000*np.pi/30, 100)",
            "camp = rotor.run_campbell(speeds)",
            "camp.plot()",
        ]
        return "\n".join(lines)


# =============================================================================
# HELPERS UI
# =============================================================================

def _badge(badge_type, label):
    classes = {"gold":"badge-gold","silver":"badge-silver","bronze":"badge-bronze","info":"badge-blue"}
    return f"<span class='badge {classes.get(badge_type,'badge-blue')}'>{label}</span>"

def _card(content, style=""):
    return f"<div class='card{style}'>{content}</div>"

def _modal_table(modal) -> pd.DataFrame:
    fn  = modal.wn / (2 * np.pi)
    ld  = getattr(modal, 'log_dec', np.zeros(len(fn)))
    n   = min(8, len(fn))
    stab = []
    for v in ld[:n]:
        if v > 0.3: stab.append("✅ Très stable")
        elif v > 0.1: stab.append("🟡 Stable")
        elif v > 0: stab.append("⚠️ Peu amorti")
        else: stab.append("❌ INSTABLE")
    return pd.DataFrame({
        "Mode": range(1, n+1),
        "fn (Hz)": [f"{v:.3f}" for v in fn[:n]],
        "ωn (rad/s)": [f"{v:.2f}" for v in modal.wn[:n]],
        "Log Dec": [f"{v:.4f}" for v in ld[:n]],
        "Stabilité": stab,
    })

def _build_quick_rotor(n_el=5, L=0.2, od=0.05, disk_node=2, disk_od=0.25,
                        disk_w=0.07, kxx=1e7, cxx=500.0, kxy=0.0, kyy=None):
    if kyy is None: kyy = kxx
    b = RotorBuilder()
    b.add_shaft_from_df(pd.DataFrame([{"L (m)":L,"id (m)":0.0,"od (m)":od}
                                        for _ in range(n_el)]))
    b.add_disk(min(disk_node, n_el), disk_od, disk_w)
    b.add_bearing(0, kxx, kyy, kxy, cxx)
    b.add_bearing(n_el, kxx, kyy, kxy, cxx)
    for e in b.errors: st.error(e)
    return b.build()

def _plot_campbell_fallback(camp, vmax_rpm, n_pts):
    """Tracé Campbell manuel si camp.plot() échoue."""
    try:
        spd = np.linspace(0, vmax_rpm, n_pts)
        fig = go.Figure()
        if hasattr(camp, 'wd') and camp.wd is not None:
            fn_mat = camp.wd / (2*np.pi)
        elif hasattr(camp, 'wn'):
            fn_mat = camp.wn / (2*np.pi)
        else:
            st.warning("Données Campbell non disponibles"); return
        colors = ["#1F5C8B","#22863A","#C55A11","#7B1FA2","#117A8B","#C00000"]
        for i in range(min(6, fn_mat.shape[1])):
            fig.add_trace(go.Scatter(x=spd, y=fn_mat[:,i], name=f"Mode {i+1}",
                                     line=dict(color=colors[i%len(colors)])))
        fig.add_trace(go.Scatter(x=spd, y=spd/60, name="1X (synchrone)",
                                  line=dict(color="red", dash="dash")))
        fig.add_trace(go.Scatter(x=spd, y=spd/30, name="2X",
                                  line=dict(color="orange", dash="dot")))
        fig.update_layout(xaxis_title="Vitesse (RPM)", yaxis_title="Fréquence (Hz)",
                           title="Diagramme de Campbell", height=480)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Tracé Campbell impossible : {e}")

def _extract_unbal(res, probe_node, probe_dof):
    """Extrait les données en fournissant correctement l'argument 'probe' exigé par ROSS."""
    
    # 1. Extraction des fréquences
    if hasattr(res, 'speed_range'):
        freqs = np.array(res.speed_range) / (2 * np.pi)
    elif hasattr(res, 'frequency'):
        freqs = np.array(res.frequency)
    elif hasattr(res, 'frequency_range'):
        freqs = np.array(res.frequency_range)
    else:
        freqs = np.linspace(0, 100, 100) # Fallback

    freqs = np.atleast_1d(freqs)
    
    # Format attendu par ROSS pour la sonde : [nœud, direction]
    probe_target = [probe_node, probe_dof]

    # 2. PLAN A : API ROSS Moderne (utilisation des méthodes avec 'probe')
    if hasattr(res, 'data_magnitude') and callable(res.data_magnitude):
        try:
            # On demande directement à ROSS les données filtrées pour CETTE sonde
            mag = np.array(res.data_magnitude(probe=probe_target))
            ph = np.array(res.data_phase(probe=probe_target))
            
            amps = np.atleast_1d(mag).flatten()
            phases = np.atleast_1d(ph).flatten()
            
            min_len = min(len(amps), len(freqs))
            return freqs[:min_len], amps[:min_len], phases[:min_len]
        except Exception:
            pass # Si ça échoue (ex: format de probe refusé), on passe au Plan B

    # 3. PLAN B : Ancienne API (extraction manuelle depuis la matrice globale)
    if hasattr(res, 'forced_resp'):
        arr = np.array(res.forced_resp)
    elif hasattr(res, 'response'):
        arr = np.array(res.response)
    else:
        raise AttributeError("Données vibratoires introuvables (ni data_magnitude, ni forced_resp).")

    mag = np.abs(arr)
    ph = np.angle(arr)

    mag = np.atleast_1d(mag)
    ph = np.atleast_1d(ph)

    # Pivot de la matrice si nécessaire
    if mag.ndim >= 2:
        if mag.shape[0] == len(freqs) and mag.shape[1] != len(freqs):
            mag = mag.T
            ph = ph.T

    # Calcul du DDL global
    dof = probe_node * 4 + probe_dof
    safe_dof = 0
    if mag.ndim > 1:
        safe_dof = min(dof, mag.shape[0] - 1)

    if mag.ndim == 3:
        amps = mag[safe_dof, 0, :]
        phases = ph[safe_dof, 0, :]
    elif mag.ndim == 2:
        amps = mag[safe_dof, :]
        phases = ph[safe_dof, :]
    else:
        amps = mag; phases = ph

    amps = np.atleast_1d(amps).flatten()
    phases = np.atleast_1d(phases).flatten()

    min_len = min(len(amps), len(freqs))
    return freqs[:min_len], amps[:min_len], phases[:min_len]
    raise AttributeError("Impossible d'extraire les données avec les attributs disponibles.")

def _plot_bode_unbal(res, probe_node, probe_dof, freq_max, modal=None):
    """Diagramme de Bode pour la réponse au balourd (100% Custom Plotly)."""
    try:
        freqs, amps, phases = _extract_unbal(res, probe_node, probe_dof)
        amps_um = amps * 1e6
        
        # --- ALERTE PHYSIQUE ---
        if np.max(amps_um) < 1e-6:
            st.warning("⚠️ L'amplitude mesurée est nulle ou presque. Vérifiez que votre sonde (Nœud probe) n'est pas placée sur un palier rigide !")
        # -----------------------
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=["Amplitude (µm)","Phase (°)"],
                             vertical_spacing=0.1)
                             
        fig.add_trace(go.Scatter(x=freqs, y=amps_um, line=dict(color="#1F5C8B", width=2),
                                  name="Amplitude"), row=1, col=1)
        fig.add_trace(go.Scatter(x=freqs, y=np.degrees(phases),
                                  line=dict(color="#C55A11", width=2), name="Phase"), row=2, col=1)
                                  
        if modal is not None:
            for i, f in enumerate(modal.wn[:4] / (2*np.pi)):
                for row in [1, 2]:
                    fig.add_vline(x=f, line_dash="dot", line_color="#22863A", opacity=0.6,
                                   annotation_text=f"M{i+1}" if row==1 else "", row=row, col=1)
                                   
        idx_max = int(np.argmax(amps_um))
        a_max = amps_um[idx_max]
        f_res = freqs[idx_max]
        a_stat = amps_um[1] if len(amps_um) > 1 and amps_um[1] > 0 else 1e-12
        daf = a_max / a_stat
        
        fig.update_xaxes(title_text="Fréquence (Hz)", row=2, col=1)
        fig.update_yaxes(title_text="µm", row=1, col=1)
        fig.update_yaxes(title_text="°", row=2, col=1)
        fig.update_layout(height=480, showlegend=False, title="Diagramme de Bode — Balourd")
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Fréquence résonance", f"{f_res:.1f} Hz")
        col2.metric("Amplitude max", f"{a_max:.2f} µm")
        col3.metric("DAF", f"{daf:.1f}")
        
    except Exception as e:
        st.error(f"Visualisation balourd impossible : {e}")

def _plot_freq_resp(res, inp_dof, out_dof, fmax, modal=None):
    """Diagramme de Bode pour H(jω)."""
    for m in ["plot_bode", "plot_magnitude"]:
        if hasattr(res, m):
            try:
                st.plotly_chart(getattr(res, m)(inp=inp_dof, out=out_dof),
                                 use_container_width=True)
                return
            except TypeError:
                try:
                    st.plotly_chart(getattr(res, m)(), use_container_width=True)
                    return
                except Exception:
                    pass
    try:
        for attr in ("freq_resp", "response", "H"):
            if hasattr(res, attr):
                H_raw = np.array(getattr(res, attr))
                break
        else:
            st.warning("Structure FreqResponse non reconnue"); return
        H = H_raw[out_dof, inp_dof, :] if H_raw.ndim == 3 else H_raw
        for attr in ("frequency_range", "speed_range"):
            if hasattr(res, attr):
                freqs = np.array(getattr(res, attr))
                if "speed" in attr: freqs /= (2*np.pi)
                break
        else:
            freqs = np.linspace(0, fmax, len(H))
        mag_db = 20*np.log10(np.abs(H) + 1e-30)
        phase  = np.degrees(np.unwrap(np.angle(H)))
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                             subplot_titles=["Magnitude (dB)","Phase (°)"],
                             vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=freqs, y=mag_db, line=dict(color="#1F5C8B",width=2),
                                  name="Magnitude"), row=1, col=1)
        fig.add_trace(go.Scatter(x=freqs, y=phase, line=dict(color="#C55A11",width=2),
                                  name="Phase"), row=2, col=1)
        if modal is not None:
            for i, f in enumerate(modal.wn[:6] / (2*np.pi)):
                for row in [1,2]:
                    fig.add_vline(x=f, line_dash="dot", line_color="#22863A", opacity=0.5,
                                   annotation_text=f"M{i+1}" if row==1 else "", row=row, col=1)
        fig.update_layout(height=480, showlegend=False,
                           title=f"H(jω) — DDL {inp_dof}→{out_dof}")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Impossible d'afficher H(jω) : {e}")

# =============================================================================
# PAGE : TABLEAU DE BORD
# =============================================================================
def render_dashboard():
    uname = st.session_state.get("user_name", "Utilisateur")
    st.markdown(f"""
    <div style='text-align:center; padding:28px 0 10px'>
      <h1 style='color:#1F5C8B; font-size:2.6em; margin:0'>⚙️ ROSSim Online</h1>
      <p style='color:#555; font-size:1.15em; margin:4px 0'>
        Application de Dynamique des Rotors • Basée sur ROSS Open-Source
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not ROSS_AVAILABLE:
        st.error("⚠️ ROSS non installé — `pip install ross-rotordynamics`")
    else:
        st.success(f"✅ ROSS {ROSS_VERSION} opérationnel")

    # ── Progression ──────────────────────────────────────────────────────────
    badges = st.session_state.get("badges", {})
    tut_done = st.session_state.get("tut_done", set())
    n_done = len(tut_done); total = len(TUTORIALS)
    pct = n_done / total if total else 0

    col_prog, col_stat = st.columns([2, 1])
    with col_prog:
        st.markdown(f"### 📊 Progression — {uname}")
        st.progress(pct)
        st.caption(f"{n_done}/{total} tutoriels complétés ({pct*100:.0f}%)")
        if badges:
            bh = "".join(_badge("gold" if v=="gold" else "silver" if v=="silver" else "bronze",
                                 f"{'🥇' if v=='gold' else '🥈' if v=='silver' else '🥉'} {k}")
                          for k, v in badges.items())
            st.markdown(bh, unsafe_allow_html=True)
        else:
            st.info("🏅 Complétez les tutoriels pour débloquer vos badges !")

    with col_stat:
        st.markdown("### 🔧 Statut")
        sim_count = st.session_state.get("sim_count", 0)
        st.metric("Simulations lancées", sim_count)
        rotor_ok = _CACHE.get("free_rotor") is not None
        st.metric("Rotor en mémoire", "✅ Oui" if rotor_ok else "❌ Non")

    st.markdown("---")
    # ── Modules ──────────────────────────────────────────────────────────────
    st.markdown("### 🚀 Accès Rapide aux Modules")
    cols = st.columns(3)
    modules = [
        ("M1 🏗️", "Constructeur de Rotor", "Créer arbre, disques, paliers — bibliothèque matériaux", "#22863A"),
        ("M2 📊", "Statique & Modal", "Déflexion statique, fréquences propres, déformées 3D", "#1F5C8B"),
        ("M3 📈", "Campbell & Stabilité", "Diagramme de Campbell, vitesses critiques, API 684", "#1F5C8B"),
        ("M4 🌀", "Balourd & H(jω)", "Réponse au balourd, Bode, Nyquist, DAF", "#C55A11"),
        ("M5 ⏱️", "Temporel & Défauts", "Transitoires, orbites, fissure, désalignement, frottement", "#C00000"),
        ("M6 🎲", "Stochastique", "Incertitudes, Monte Carlo, intervalles de confiance", "#7B1FA2"),
    ]
    for i, (badge, title, desc, color) in enumerate(modules):
        with cols[i % 3]:
            st.markdown(f"""
            <div class='card' style='border-left-color:{color}'>
              <b style='color:{color}'>{badge} — {title}</b><br>
              <small>{desc}</small>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    # ── Tutoriels rapides ─────────────────────────────────────────────────────
    st.markdown("### 🎓 Tutoriels Officiels ROSS")
    tcols = st.columns(4)
    for i, (tid, tdata) in enumerate(TUTORIALS.items()):
        done = tid in tut_done
        with tcols[i]:
            status = "✅" if done else "▶️"
            st.markdown(f"""
            <div class='card{"" if not done else ""}'
                 style='border-left-color:{"#22863A" if done else "#1F5C8B"}'>
              {status} <b>{tdata["level"]}</b><br>
              <small><b>{tdata["title"][:40]}</b></small><br>
              <small>⏱ {tdata["duration"]}</small>
            </div>""", unsafe_allow_html=True)

    # ── Exemples industriels ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🏭 Exemple Industriel ROSS (Compresseur Centrifuge)")
    if ROSS_AVAILABLE:
        if st.button("🔄 Charger rs.compressor_example()", key="load_compressor"):
            try:
                rotor = rs.compressor_example()
                _CACHE["free_rotor"] = rotor
                st.session_state["sim_count"] = st.session_state.get("sim_count",0) + 1
                st.success(f"✅ Compresseur chargé — {len(rotor.nodes)} nœuds, {rotor.m:.1f} kg")
                st.info("👉 Allez dans 🔬 Mode Simulation pour analyser ce rotor industriel")
            except Exception as e:
                st.error(f"Erreur chargement compresseur : {e}")

# =============================================================================
# PAGE : MODE PÉDAGOGIQUE — TUTORIELS
# =============================================================================
def render_tutorial_mode():
    st.title("🎓 Mode Pédagogique — Tutoriels ROSS Officiels")
    st.caption("Tous les tutoriels de ross.readthedocs.io/en/stable/ intégrés interactivement")

    tut_id = st.sidebar.selectbox(
        "Tutoriel :", list(TUTORIALS.keys()),
        format_func=lambda x: f"{TUTORIALS[x]['level']} {TUTORIALS[x]['title'][:35]}..."
    )
    tut = TUTORIALS[tut_id]

    # En-tête tutoriel
    api_badges = "".join(f"<span class='mod-badge'>{a}</span>" for a in tut["api"])
    st.markdown(f"""
    <div class='card'>
      <h2 style='color:#1F5C8B; margin:0'>{tut['title']}</h2>
      <p>{tut['level']} &nbsp;|&nbsp; ⏱ {tut['duration']}</p>
      <p><b>API ROSS utilisée :</b> {api_badges}</p>
    </div>""", unsafe_allow_html=True)

    # Steps
    steps = tut["steps"]
    n_steps = len(steps)
    done_key = f"tut_step_{tut_id}"
    current = st.session_state.get(done_key, 0)

    # Progress bar des étapes
    st.progress(current / n_steps if n_steps else 0)
    st.caption(f"Étape {min(current+1, n_steps)} / {n_steps}")

    # Sidebar navigation étapes
    step_idx = st.sidebar.radio(
        "Étapes :",
        range(n_steps),
        index=min(current, n_steps-1),
        format_func=lambda i: f"{'✅' if i < current else '▶️' if i == current else '⬜'} Étape {i+1}: {steps[i]['title']}"
    )

    step = steps[step_idx]
    st.markdown(f"## Étape {step_idx+1} — {step['title']}")

    tab_th, tab_sim, tab_code = st.tabs(["📖 Théorie", "🔬 Simulation", "💻 Code"])

    with tab_th:
        st.info(step["theory"])
        st.markdown(f"**🎯 Objectif :** {step['objective']}")

    with tab_code:
        st.markdown("**Code de référence ROSS :**")
        st.code(step["code"], language="python")
        st.download_button("⬇️ Télécharger ce snippet",
                           data=step["code"].encode(),
                           file_name=f"ross_{step['id']}.py",
                           mime="text/plain")

    with tab_sim:
        _run_tutorial_step(tut_id, step_idx, step, tut)

    # Navigation
    col_prev, col_next, col_done = st.columns([1, 1, 2])
    with col_prev:
        if step_idx > 0:
            if st.button("⬅️ Précédent"):
                st.session_state[done_key] = max(0, step_idx - 1)
                st.rerun()
    with col_next:
        if step_idx < n_steps - 1:
            if st.button("Suivant ➡️", type="primary"):
                st.session_state[done_key] = step_idx + 1
                st.rerun()
    with col_done:
        if step_idx == n_steps - 1:
            if st.button("🏆 Terminer ce tutoriel", type="primary"):
                tut_done = st.session_state.get("tut_done", set())
                tut_done.add(tut_id)
                st.session_state["tut_done"] = tut_done
                # Badge
                badges = st.session_state.get("badges", {})
                badges[tut_id] = "gold"
                st.session_state["badges"] = badges
                st.balloons()
                st.success(f"🥇 Tutoriel {tut['title'][:30]} complété ! Badge Or débloqué.")


def _run_tutorial_step(tut_id, step_idx, step, tut):
    """Simulation interactive pour chaque étape de tutoriel."""
    sid = step["id"]

    # ── Tutoriel 1 ────────────────────────────────────────────────────────────
    if tut_id == "T1":
        if step_idx == 0:  # Matériau
            mat_name = st.selectbox("Choisir un matériau :", list(MATERIALS_DB.keys()))
            props = MATERIALS_DB[mat_name]
            st.json(props)
            if st.button("✅ Créer le matériau", key=f"btn_{sid}"):
                if ROSS_AVAILABLE:
                    try:
                        m = rs.Material(name=mat_name, rho=props["rho"],
                                         E=props["E"], G_s=props["G_s"])
                        _CACHE["tut_mat"] = m
                        st.success(f"✅ Matériau '{mat_name}' créé")
                    except Exception as e: st.error(str(e))

        elif step_idx == 1:  # Arbre
            col1, col2 = st.columns(2)
            with col1:
                n_el = st.slider("Nombre d'éléments", 3, 10, 5, key=f"sl_{sid}")
                L    = st.number_input("L par élément (m)", 0.05, 1.0, 0.25, key=f"L_{sid}")
                od   = st.number_input("Diamètre (m)", 0.02, 0.2, 0.05, key=f"od_{sid}")
            with col2:
                st.metric("Longueur totale", f"{n_el*L:.3f} m")
                mat = _CACHE.get("tut_mat", MAT_STEEL)
                if mat and ROSS_AVAILABLE:
                    try:
                        shaft = [rs.ShaftElement(L=L, idl=0, odl=od, material=mat)
                                  for _ in range(n_el)]
                        _CACHE["tut_shaft"] = shaft
                        _CACHE["tut_nel"] = n_el
                        st.success(f"✅ {n_el} éléments d'arbre prêts")
                        st.metric("Nœuds créés", n_el + 1)
                    except Exception as e: st.error(str(e))

        elif step_idx == 2:  # Disque
            n_el = _CACHE.get("tut_nel", 5)
            dn  = st.slider("Nœud du disque", 0, n_el, n_el//2, key=f"dn_{sid}")
            dod = st.number_input("Diamètre disque (m)", 0.1, 0.6, 0.25, key=f"dod_{sid}")
            dw  = st.number_input("Largeur disque (m)", 0.02, 0.3, 0.07, key=f"dw_{sid}")
            mat = _CACHE.get("tut_mat", MAT_STEEL)
            if st.button("✅ Créer le disque", key=f"btn_{sid}") and mat and ROSS_AVAILABLE:
                try:
                    disk = rs.DiskElement.from_geometry(n=dn, material=mat,
                                                         width=dw, i_d=0.05, o_d=dod)
                    _CACHE["tut_disk"] = disk
                    st.success(f"✅ Disque au nœud {dn} — masse estimée calculée")
                except Exception as e: st.error(str(e))

        elif step_idx == 3:  # Paliers
            preset = st.selectbox("Preset palier :", list(BEARING_PRESETS.keys()), key=f"bp_{sid}")
            p = BEARING_PRESETS[preset]
            col1, col2 = st.columns(2)
            with col1:
                kxx = st.number_input("Kxx (N/m)", 1e4, 1e9, float(p["kxx"]), format="%.2e", key=f"kxx_{sid}")
                cxx = st.number_input("Cxx (N·s/m)", 10.0, 10000.0, float(p["cxx"]), key=f"cxx_{sid}")
            n_el = _CACHE.get("tut_nel", 5)
            if st.button("✅ Créer les 2 paliers", key=f"btn_{sid}") and ROSS_AVAILABLE:
                try:
                    b0 = rs.BearingElement(n=0, kxx=kxx, kyy=kxx, cxx=cxx, cyy=cxx)
                    bn = rs.BearingElement(n=n_el, kxx=kxx, kyy=kxx, cxx=cxx, cyy=cxx)
                    _CACHE["tut_bears"] = [b0, bn]
                    st.success(f"✅ Paliers aux nœuds 0 et {n_el}")
                except Exception as e: st.error(str(e))

        elif step_idx == 4:  # Assemblage
            shaft = _CACHE.get("tut_shaft")
            disk  = _CACHE.get("tut_disk")
            bears = _CACHE.get("tut_bears")
            if not all([shaft, disk, bears]):
                st.warning("⚠️ Complétez les étapes 1–4 d'abord")
                return
            if st.button("🚀 Assembler le rotor", type="primary", key=f"btn_{sid}"):
                try:
                    rotor = rs.Rotor(shaft, [disk], bears)
                    _CACHE["tut_rotor"] = rotor
                    _CACHE["free_rotor"] = rotor  # aussi disponible en Mode Simulation
                    st.success(f"✅ Rotor assemblé — {len(rotor.nodes)} nœuds | Masse : {rotor.m:.2f} kg")
                    st.session_state["sim_count"] = st.session_state.get("sim_count",0)+1
                    try:
                        st.plotly_chart(rotor.plot_rotor(), use_container_width=True)
                    except Exception:
                        st.info("Visualisation 3D non disponible dans cette version ROSS")
                except Exception as e:
                    st.error(f"Assemblage impossible : {e}")

    # ── Tutoriel 2.1 ─────────────────────────────────────────────────────────
    elif tut_id == "T2_1":
        rotor = _CACHE.get("tut_rotor") or _CACHE.get("free_rotor")
        if rotor is None:
            st.warning("⚠️ Aucun rotor — Complétez d'abord le Tutoriel Part 1")
            if ROSS_AVAILABLE and st.button("Charger exemple compresseur", key=f"cex_{sid}"):
                try:
                    rotor = rs.compressor_example()
                    _CACHE["tut_rotor"] = rotor
                    _CACHE["free_rotor"] = rotor
                    st.rerun()
                except Exception as e: st.error(str(e))
            return

        eng = SimulationEngine(rotor)

        if step_idx == 0:  # Statique
            if st.button("📏 Lancer l'analyse statique", key=f"btn_{sid}", type="primary"):
                with st.spinner("Calcul statique..."):
                    static = eng.run_static()
                if static:
                    _CACHE["tut_static"] = static
                    try: st.plotly_chart(static.plot_deflected_shape(), use_container_width=True)
                    except Exception:
                        try: st.plotly_chart(static.plot(), use_container_width=True)
                        except Exception: st.info("Visualisation statique non disponible")
                else:
                    st.error(f"Erreur : {eng.last_error}")

        elif step_idx == 1:  # Modal
            speed_rpm = st.slider("Vitesse de rotation (RPM)", 0, 10000, 0, key=f"sp_{sid}")
            if st.button("📊 Calculer les modes propres", key=f"btn_{sid}", type="primary"):
                with st.spinner("Calcul modal..."):
                    modal = eng.run_modal(speed_rpm)
                if modal:
                    _CACHE["tut_modal"] = modal
                    st.dataframe(_modal_table(modal), use_container_width=True, hide_index=True)
                else:
                    st.error(f"Erreur : {eng.last_error}")
            modal = _CACHE.get("tut_modal")
            if modal:
                n_modes = min(6, len(modal.wn)//2 if len(modal.wn) > 2 else 1)
                mode_i = st.selectbox("Mode à visualiser :", range(n_modes),
                                       format_func=lambda x: f"Mode {x+1} — {modal.wn[x]/(2*np.pi):.2f} Hz",
                                       key=f"mi_{sid}")
                for m in ["plot_mode_3d", "plot_mode_shape"]:
                    if hasattr(modal, m):
                        try:
                            st.plotly_chart(getattr(modal, m)(mode=mode_i),
                                             use_container_width=True)
                            break
                        except Exception: continue

        elif step_idx == 2:  # Campbell
            vmax = st.slider("Vitesse max (RPM)", 2000, 20000, 10000, key=f"vm_{sid}")
            npts = st.slider("Résolution (points)", 30, 150, 100, key=f"np_{sid}")
            if st.button("📈 Tracer le Campbell", key=f"btn_{sid}", type="primary"):
                with st.spinner("Calcul Campbell (peut prendre quelques secondes)..."):
                    camp = eng.run_campbell(vmax, npts)
                if camp:
                    _CACHE["tut_camp"] = camp
                    try: st.plotly_chart(camp.plot(), use_container_width=True)
                    except Exception: _plot_campbell_fallback(camp, vmax, npts)
                else:
                    st.error(f"Erreur : {eng.last_error}")

        elif step_idx == 3:  # Vitesses critiques
            modal = _CACHE.get("tut_modal")
            if modal is None:
                st.warning("Calculez d'abord les modes (étape 2)")
                return
            fn = modal.wn / (2*np.pi)
            vc_rpm = fn * 60
            op_rpm = st.number_input("Vitesse opérationnelle (RPM)", 500.0, 15000.0, 3000.0,
                                      key=f"op_{sid}")
            zone_l, zone_h = op_rpm * 0.85, op_rpm * 1.15
            df_vc = pd.DataFrame({
                "Mode": range(1, len(fn[:6])+1),
                "fn (Hz)": [f"{f:.2f}" for f in fn[:6]],
                "Vitesse critique (RPM)": [f"{v:.0f}" for v in vc_rpm[:6]],
                "Zone interdite API": ["❌ OUI" if zone_l <= v <= zone_h else "✅ NON"
                                        for v in vc_rpm[:6]],
            })
            st.dataframe(df_vc, use_container_width=True, hide_index=True)
            st.markdown(f"🔴 **Zone interdite API 684 :** [{zone_l:.0f} – {zone_h:.0f}] RPM")
            ok_modes = sum(1 for v in vc_rpm[:6] if not (zone_l <= v <= zone_h))
            score_api = ok_modes / min(6, len(fn[:6])) * 100
            color = "#22863A" if score_api >= 100 else "#C00000"
            st.markdown(f"<h3 style='color:{color}'>Conformité API 684 : {score_api:.0f}%</h3>",
                         unsafe_allow_html=True)

    # ── Tutoriel 2.2 ─────────────────────────────────────────────────────────
    elif tut_id == "T2_2":
        rotor = _CACHE.get("tut_rotor") or _CACHE.get("free_rotor")
        if rotor is None:
            st.warning("⚠️ Aucun rotor — Complétez d'abord le Tutoriel Part 1"); return
        eng = SimulationEngine(rotor)
        n_nodes = len(rotor.nodes) - 1

        if step_idx == 0:  # Probe
            st.markdown("""
            Une **sonde (Probe)** simule un capteur de déplacement sur l'arbre.
            Elle est définie par son nœud et son angle d'installation.
            """)
            col1, col2 = st.columns(2)
            with col1:
                p1n = st.slider("Sonde 1 — Nœud", 0, n_nodes, min(2, n_nodes), key=f"p1n_{sid}")
                p1a = st.slider("Sonde 1 — Angle (°)", 0, 360, 45, key=f"p1a_{sid}")
            with col2:
                p2n = st.slider("Sonde 2 — Nœud", 0, n_nodes, min(2, n_nodes), key=f"p2n_{sid}")
                p2a = st.slider("Sonde 2 — Angle (°)", 0, 360, 135, key=f"p2a_{sid}")
            _CACHE["tut_probe"] = {"n1":p1n,"a1":p1a,"n2":p2n,"a2":p2a}
            st.success(f"Sondes configurées : Sonde1@nœud{p1n}({p1a}°) | Sonde2@nœud{p2n}({p2a}°)")

        elif step_idx == 1:  # Balourd
            probe = _CACHE.get("tut_probe", {"n1":2,"n2":2})
            col1, col2 = st.columns(2)
            with col1:
                un  = st.slider("Nœud du balourd", 0, n_nodes, min(2,n_nodes), key=f"un_{sid}")
                mag = st.number_input("Magnitude (kg·m)", 1e-5, 0.1, 0.001, format="%.5f", key=f"mag_{sid}")
                ph  = st.slider("Phase (°)", 0, 360, 0, key=f"ph_{sid}")
            with col2:
                fmax = st.slider("Fréquence max (Hz)", 100, 5000, 2000, key=f"fm_{sid}")
            if st.button("🌀 Calculer la réponse au balourd", key=f"btn_{sid}", type="primary"):
                with st.spinner("Calcul en cours..."):
                    res = eng.run_unbalance(
                        nodes=[un], mags=[mag], phases=[np.deg2rad(ph)], fmax=float(fmax))
                if res:
                    _CACHE["tut_unbal"] = res
                    _CACHE["tut_probe_node"] = probe["n1"]
                    modal = _CACHE.get("tut_modal")
                    _plot_bode_unbal(res, probe["n1"], 0, float(fmax), modal)
                else:
                    st.error(f"Erreur : {eng.last_error}")
            elif _CACHE.get("tut_unbal"):
                modal = _CACHE.get("tut_modal")
                _plot_bode_unbal(_CACHE["tut_unbal"], probe.get("n1",2), 0, 2000.0, modal)

        elif step_idx == 2:  # Freq Response
            n_dof = len(rotor.nodes) * 4
            col1, col2 = st.columns(2)
            with col1:
                inp_n = st.slider("DDL excitation (inp)", 0, min(n_dof-1, 20), 0, key=f"inp_{sid}")
            with col2:
                out_n = st.slider("DDL réponse (out)", 0, min(n_dof-1, 20), min(8, n_dof-1), key=f"out_{sid}")
                fmax  = st.slider("Fréquence max (Hz)", 100, 5000, 2000, key=f"fm2_{sid}")
            if st.button("📡 Calculer H(jω)", key=f"btn_{sid}", type="primary"):
                with st.spinner("Calcul H(jω)..."):
                    fr = eng.run_freq_response(inp_n, out_n, float(fmax))
                if fr:
                    _CACHE["tut_freq"] = fr
                    modal = _CACHE.get("tut_modal")
                    _plot_freq_resp(fr, inp_n, out_n, float(fmax), modal)
                else:
                    st.error(f"Erreur : {eng.last_error}")

        elif step_idx == 3:  # Temporel
            speed_rpm = st.slider("Vitesse rotation (RPM)", 500, 10000, 3000, key=f"sp_{sid}")
            t_end     = st.slider("Durée simulation (s)", 0.5, 5.0, 2.0, key=f"te_{sid}")
            n_t       = st.slider("Points temporels", 200, 2000, 500, key=f"nt_{sid}")
            if st.button("⏱️ Simuler la réponse temporelle", key=f"btn_{sid}", type="primary"):
                t = np.linspace(0, t_end, n_t)
                F = np.zeros((rotor.ndof, n_t))
                with st.spinner("Intégration temporelle (Newmark)..."):
                    tr = eng.run_time_response(speed_rpm, F, t)
                if tr:
                    _CACHE["tut_time"] = tr
                    try:
                        node_obs = min(2, n_nodes)
                        st.plotly_chart(tr.plot_orbit(node=node_obs),
                                         use_container_width=True)
                    except Exception:
                        try:
                            fig_t = go.Figure()
                            disp = np.array(tr.response)
                            idx = min(node_obs*4, disp.shape[0]-1)
                            fig_t.add_trace(go.Scatter(x=t, y=disp[idx,:]*1e6,
                                                        name=f"Nœud {node_obs} X (µm)"))
                            fig_t.update_layout(xaxis_title="Temps (s)",
                                                 yaxis_title="Déplacement (µm)")
                            st.plotly_chart(fig_t, use_container_width=True)
                        except Exception as e:
                            st.info(f"Visualisation temporelle non disponible : {e}")
                else:
                    st.error(f"Erreur : {eng.last_error}")

    # ── Tutoriel 4 ────────────────────────────────────────────────────────────
    elif tut_id == "T4":
        st.info("Le tutoriel MultiRotor nécessite une configuration avancée. "
                "Voici un guide pas à pas avec le code de référence.")
        step_theory = {
            0: "Création de 2 rotors indépendants avec des géométries différentes",
            1: "GearElement modélise le couplage par ligne d'action des dents",
            2: "MultiRotor assemble les deux systèmes avec la contrainte cinématique d'engrenage",
            3: "L'analyse Campbell révèle les modes latéraux, torsionnels et couplés"
        }
        st.markdown(f"**Concept :** {step_theory.get(step_idx, '')}")
        if ROSS_AVAILABLE:
            if step_idx == 0:
                n_el1 = st.slider("Éléments Rotor 1", 3, 8, 4, key=f"nel1_{sid}")
                n_el2 = st.slider("Éléments Rotor 2", 3, 8, 3, key=f"nel2_{sid}")
                if st.button("Créer les 2 rotors", key=f"btn_{sid}"):
                    mat = MAT_STEEL or _CACHE.get("tut_mat")
                    if mat:
                        sh1 = [rs.ShaftElement(L=0.25, idl=0, odl=0.05, material=mat)
                                for _ in range(n_el1)]
                        sh2 = [rs.ShaftElement(L=0.25, idl=0, odl=0.04, material=mat)
                                for _ in range(n_el2)]
                        _CACHE["t4_shaft1"], _CACHE["t4_shaft2"] = sh1, sh2
                        _CACHE["t4_nel1"],   _CACHE["t4_nel2"]   = n_el1, n_el2
                        st.success(f"✅ Rotor 1 : {n_el1+1} nœuds | Rotor 2 : {n_el2+1} nœuds")
            elif step_idx == 1:
                pd_val = st.number_input("Pitch diameter (m)", 0.05, 0.5, 0.1, key=f"pd_{sid}")
                pa_deg = st.number_input("Angle de pression (°)", 14.5, 30.0, 20.0, key=f"pa_{sid}")
                st.info(f"GearElement(pitch_diameter={pd_val}, pressure_angle={np.radians(pa_deg):.4f} rad)")
                _CACHE["t4_gear_params"] = {"pd": pd_val, "pa_rad": np.radians(pa_deg)}
                st.success("✅ Paramètres d'engrenage configurés")
            elif step_idx in [2, 3]:
                sh1 = _CACHE.get("t4_shaft1")
                sh2 = _CACHE.get("t4_shaft2")
                if sh1 and sh2:
                    st.code("# Assemblage MultiRotor (voir documentation ROSS Part 4)\n"
                             "# Disponible si MultiRotor est dans votre version ROSS\n"
                             "try:\n"
                             "    from ross import MultiRotor\nexcept ImportError:\n"
                             "    print('MultiRotor non disponible — vérifiez la version ROSS')",
                             language="python")
                else:
                    st.warning("Complétez l'étape 1 d'abord")


# =============================================================================
# PAGE : MODE SIMULATION — M1 à M5
# =============================================================================
def render_simulation_mode():
    st.title("🔬 Mode Simulation — Analyses Complètes")

    module = st.sidebar.selectbox("Module :", [
        "M1 🏗️ Constructeur",
        "M2 📊 Statique & Modal",
        "M3 📈 Campbell & Stabilité",
        "M4 🌀 Balourd & H(jω)",
        "M5 ⏱️ Temporel & Défauts",
    ])

    if "M1" in module:   _render_m1()
    elif "M2" in module: _render_m2()
    elif "M3" in module: _render_m3()
    elif "M4" in module: _render_m4()
    elif "M5" in module: _render_m5()


# ── M1 — Constructeur ─────────────────────────────────────────────────────────
def _render_m1():
    st.subheader("🏗️ M1 — Constructeur de Rotor")
    st.caption("Bibliothèque de matériaux · Validation temps réel · Export TOML/Python")

    # Matériau
    with st.expander("🧱 Matériau", expanded=True):
        mat_name = st.selectbox("Matériau :", list(MATERIALS_DB.keys()), key="m1_mat")
        props = MATERIALS_DB[mat_name]
        if mat_name == "Personnalisé":
            col1, col2, col3 = st.columns(3)
            with col1: props["rho"] = st.number_input("ρ (kg/m³)", 500.0, 20000.0, float(props["rho"]))
            with col2: props["E"]   = st.number_input("E (GPa)", 10.0, 500.0, float(props["E"])/1e9) * 1e9
            with col3: props["G_s"] = st.number_input("G_s (GPa)", 5.0, 200.0, float(props["G_s"])/1e9) * 1e9
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("ρ (kg/m³)", f"{props['rho']:.0f}")
            col2.metric("E (GPa)", f"{props['E']/1e9:.1f}")
            col3.metric("G_s (GPa)", f"{props['G_s']/1e9:.1f}")

    # Arbre
    with st.expander("🔩 Arbre (ShaftElement)", expanded=True):
        st.caption("Éléments de poutre Timoshenko — chaque ligne = 1 section")
        df_s = pd.DataFrame([{"L (m)":0.2,"id (m)":0.0,"od (m)":0.05} for _ in range(5)])
        ed_s = st.data_editor(df_s, num_rows="dynamic", key="m1_shaft")

    # Disques
    with st.expander("💿 Disques (DiskElement)"):
        df_d = pd.DataFrame([{"nœud":2,"id (m)":0.05,"od (m)":0.25,"largeur (m)":0.07}])
        ed_d = st.data_editor(df_d, num_rows="dynamic", key="m1_disk")

    # Paliers
    with st.expander("🔗 Paliers (BearingElement)", expanded=True):
        preset = st.selectbox("Preset :", list(BEARING_PRESETS.keys()), key="m1_bp")
        p = BEARING_PRESETS[preset]
        n_el_est = max(1, len(ed_s) - 1)
        df_b = pd.DataFrame([
            {"nœud":0, "kxx":p["kxx"],"kyy":p["kyy"],"kxy":p["kxy"],"cxx":p["cxx"],"cyy":p["cyy"]},
            {"nœud":n_el_est, "kxx":p["kxx"],"kyy":p["kyy"],"kxy":p["kxy"],"cxx":p["cxx"],"cyy":p["cyy"]},
        ])
        ed_b = st.data_editor(df_b, num_rows="dynamic", key="m1_bear")

    # Build
    if st.button("🚀 Assembler le rotor", type="primary", key="m1_build"):
        if not ROSS_AVAILABLE:
            st.error("ROSS non disponible"); return
        try:
            mat = rs.Material(name=mat_name.replace(" ", "_"), rho=props["rho"], E=props["E"], G_s=props["G_s"])
            shaft = [rs.ShaftElement(L=float(r[1]), idl=float(r[2]), odl=float(r[3]), material=mat)
                     for r in ed_s.itertuples()]
            disks = [rs.DiskElement.from_geometry(n=int(r[1]), material=mat,
                      width=float(r[4]), i_d=float(r[2]), o_d=float(r[3]))
                     for r in ed_d.itertuples()]
            bears = [rs.BearingElement(n=int(r[1]), kxx=float(r[2]), kyy=float(r[3]),
                      kxy=float(r[4]), kyx=-float(r[4]), cxx=float(r[5]), cyy=float(r[6]))
                     for r in ed_b.itertuples()]
            rotor = rs.Rotor(shaft, disks, bears)
            _CACHE["free_rotor"] = rotor
            _CACHE["free_mat_props"] = props.copy()
            st.session_state["sim_count"] = st.session_state.get("sim_count",0)+1
            st.success(f"✅ Rotor assemblé — {len(rotor.nodes)} nœuds | Masse : {rotor.m:.2f} kg")
            col1, col2, col3 = st.columns(3)
            col1.metric("Masse totale", f"{rotor.m:.2f} kg")
            col2.metric("Nœuds", len(rotor.nodes))
            col3.metric("Longueur", f"{sum(float(r[1]) for r in ed_s.itertuples()):.3f} m")
            try: st.plotly_chart(rotor.plot_rotor(), use_container_width=True)
            except Exception: st.info("Visualisation 3D — non disponible dans cette version ROSS")
        except Exception as e:
            st.error(f"❌ Erreur d'assemblage : {e}")
            st.code(traceback.format_exc(), language="text")

    # Export
    rotor = _CACHE.get("free_rotor")
    if rotor:
        st.markdown("---")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            rep = ReportGenerator(st.session_state.get("user_name",""))
            script = rep.python_script({"rho": props["rho"], "E": props["E"], "G_s": props["G_s"]})
            st.download_button("💾 Export code Python ROSS", data=script.encode(),
                                file_name="mon_rotor_ross.py", mime="text/plain")
        with col_dl2:
            params_info = {
                "Matériau": mat_name,
                "Masse (kg)": f"{rotor.m:.3f}",
                "Nœuds": str(len(rotor.nodes)),
            }
            html = ReportGenerator().html_report("Modèle Rotor", params_info,
                [{"title":"Masse et géométrie",
                  "text":f"Masse totale : {rotor.m:.2f} kg | {len(rotor.nodes)} nœuds"}])
            st.download_button("📄 Export rapport HTML", data=html.encode(),
                                file_name="rapport_rotor.html", mime="text/html")


# ── M2 — Statique & Modal ─────────────────────────────────────────────────────
def _render_m2():
    st.subheader("📊 M2 — Analyses Statiques & Modales")
    rotor = _CACHE.get("free_rotor")
    if rotor is None:
        st.warning("⚠️ Aucun rotor — Construisez d'abord un rotor dans M1 ou chargez un exemple")
        if ROSS_AVAILABLE and st.button("Charger rs.compressor_example()"):
            try:
                rotor = rs.compressor_example()
                _CACHE["free_rotor"] = rotor
                st.rerun()
            except Exception as e: st.error(str(e))
        return

    eng = SimulationEngine(rotor)
    tab_stat, tab_modal = st.tabs(["📏 Analyse Statique", "🎵 Analyse Modale"])

    with tab_stat:
        st.info("Calcule la déflexion par gravité + réactions aux paliers + diagramme de moment fléchissant")
        if st.button("📏 Lancer l'analyse statique", type="primary", key="m2_static"):
            with st.spinner("Calcul statique..."):
                static = eng.run_static()
            if static:
                _CACHE["free_static"] = static
                for m in ["plot_deflected_shape", "plot_bending_moment", "plot"]:
                    if hasattr(static, m):
                        try: st.plotly_chart(getattr(static, m)(), use_container_width=True); break
                        except Exception: continue
            else:
                st.error(f"Erreur : {eng.last_error}")

    with tab_modal:
        col1, col2 = st.columns([2,1])
        with col1:
            speed_rpm = st.slider("Vitesse (RPM) pour l'analyse modale", 0, 15000, 0, key="m2_speed")
        with col2:
            st.metric("Vitesse (rad/s)", f"{speed_rpm*np.pi/30:.1f}")

        if st.button("📊 Calculer les modes propres", type="primary", key="m2_modal"):
            with st.spinner("Calcul modal..."):
                modal = eng.run_modal(speed_rpm)
            if modal:
                _CACHE["free_modal"] = modal
                st.dataframe(_modal_table(modal), use_container_width=True, hide_index=True)
            else:
                st.error(f"Erreur : {eng.last_error}")

        modal = _CACHE.get("free_modal")
        if modal:
            n_m = min(6, max(1, len(modal.wn)//2))
            mode_i = st.selectbox("Mode à visualiser :", range(n_m),
                                   format_func=lambda x: f"Mode {x+1} — {modal.wn[x]/(2*np.pi):.2f} Hz",
                                   key="m2_mode")
            for m in ["plot_mode_3d", "plot_mode_shape"]:
                if hasattr(modal, m):
                    try:
                        st.plotly_chart(getattr(modal, m)(mode=mode_i), use_container_width=True)
                        break
                    except Exception: continue
            # Export CSV
            df_modal = _modal_table(modal)
            st.download_button("📥 Export CSV fréquences",
                                data=df_modal.to_csv(index=False).encode(),
                                file_name="frequences_modales.csv", mime="text/csv")


# ── M3 — Campbell & Stabilité ─────────────────────────────────────────────────
def _render_m3():
    st.subheader("📈 M3 — Diagramme de Campbell & Stabilité")
    rotor = _CACHE.get("free_rotor")
    if rotor is None:
        st.warning("⚠️ Aucun rotor — M1 d'abord"); return

    eng = SimulationEngine(rotor)
    tab_camp, tab_stab, tab_api = st.tabs(["📈 Campbell", "📉 Stabilité (Log Dec)", "🔧 Vérification API 684"])

    with tab_camp:
        col1, col2 = st.columns(2)
        with col1:
            vmax = st.slider("Vitesse max (RPM)", 2000, 30000, 10000, key="m3_vmax")
        with col2:
            npts = st.slider("Résolution (points)", 50, 200, 100, key="m3_npts")
        if st.button("📈 Tracer le Campbell", type="primary", key="m3_camp"):
            with st.spinner(f"Calcul Campbell ({npts} points)..."):
                camp = eng.run_campbell(vmax, npts)
            if camp:
                _CACHE["free_camp"] = camp
                _CACHE["free_camp_vmax"] = vmax
                _CACHE["free_camp_npts"] = npts
                try: st.plotly_chart(camp.plot(), use_container_width=True)
                except Exception: _plot_campbell_fallback(camp, vmax, npts)
                modal_0 = eng.run_modal(0)
                if modal_0:
                    fn = modal_0.wn / (2*np.pi)
                    df_vc = pd.DataFrame({
                        "Mode": range(1, len(fn[:6])+1),
                        "fn (Hz)": [f"{v:.2f}" for v in fn[:6]],
                        "Vitesse critique (RPM)": [f"{v*60:.0f}" for v in fn[:6]],
                    })
                    st.markdown("**⚡ Vitesses critiques (intersections 1X) :**")
                    st.dataframe(df_vc, use_container_width=True, hide_index=True)
            else:
                st.error(f"Erreur : {eng.last_error}")

    with tab_stab:
        camp = _CACHE.get("free_camp")
        vmax = _CACHE.get("free_camp_vmax", 10000)
        npts = _CACHE.get("free_camp_npts", 100)
        if camp is None:
            st.info("Calculez d'abord le Campbell (onglet précédent)"); return
        st.info("Log Dec < 0 → Instabilité | Log Dec ≥ 0.1 → Conforme API 684")
        fig_s = go.Figure()
        try:
            ld = camp.log_dec
            spd = np.linspace(0, vmax, npts)
            colors = ["#1F5C8B","#22863A","#C55A11","#7B1FA2","#117A8B","#C00000"]
            for i in range(min(6, ld.shape[1])):
                fig_s.add_trace(go.Scatter(x=spd, y=ld[:,i], name=f"Mode {i+1}",
                                            line=dict(color=colors[i%len(colors)])))
            fig_s.add_hline(y=0, line_dash="dash", line_color="red",
                             annotation_text="Seuil instabilité (0)")
            fig_s.add_hline(y=0.1, line_dash="dot", line_color="orange",
                             annotation_text="Seuil API 684 (0.1)")
            fig_s.update_layout(xaxis_title="Vitesse (RPM)", yaxis_title="Log Décrément",
                                  title="Stabilité des modes vs vitesse", height=450)
            st.plotly_chart(fig_s, use_container_width=True)
        except Exception as e:
            st.warning(f"Log Dec non disponible : {e}")

        # Graphique instabilité par Kxy
        st.markdown("---")
        st.markdown("**🎛️ Influence de la raideur croisée Kxy sur la stabilité**")
        kxy_val = st.slider("Kxy (N/m)", 0, int(1e7), 0, step=int(5e5), key="m3_kxy")
        if st.button("Recalculer avec ce Kxy", key="m3_kxy_btn"):
            rotor_kxy = _build_quick_rotor(kxy=float(kxy_val))
            if rotor_kxy:
                camp_kxy = SimulationEngine(rotor_kxy).run_campbell(vmax, 50)
                if camp_kxy:
                    fig_kxy = go.Figure()
                    try:
                        for i in range(min(4, camp_kxy.log_dec.shape[1])):
                            ld_v = camp_kxy.log_dec[:,i]
                            fig_kxy.add_trace(go.Scatter(
                                x=np.linspace(0,vmax,50), y=ld_v, name=f"Mode {i+1}"))
                        fig_kxy.add_hline(y=0, line_dash="dash", line_color="red")
                        fig_kxy.update_layout(title=f"Stabilité avec Kxy = {kxy_val:.0e} N/m",
                                               xaxis_title="Vitesse (RPM)", yaxis_title="Log Dec")
                        st.plotly_chart(fig_kxy, use_container_width=True)
                        if kxy_val > 5e6:
                            st.markdown("<div class='card-red'>❌ Kxy élevé — risque d'instabilité détecté !</div>",
                                         unsafe_allow_html=True)
                        elif kxy_val > 2e6:
                            st.markdown("<div class='card-orange'>⚠️ Kxy modéré — surveillez le Log Dec</div>",
                                         unsafe_allow_html=True)
                        else:
                            st.markdown("<div class='card-green'>✅ Kxy faible — système stable</div>",
                                         unsafe_allow_html=True)
                    except Exception: st.warning("Log Dec non disponible pour cette configuration")

    with tab_api:
        modal_0 = _CACHE.get("free_modal")
        if modal_0 is None:
            eng2 = SimulationEngine(rotor)
            modal_0 = eng2.run_modal(0)
        if modal_0 is None:
            st.info("Lancez l'analyse modale (M2) d'abord"); return
        fn = modal_0.wn / (2*np.pi)
        vc_rpm = fn * 60
        ld = getattr(modal_0, 'log_dec', np.zeros(len(fn)))
        op_rpm = st.number_input("Vitesse opérationnelle (RPM)", 500.0, 20000.0, 3000.0, key="m3_op")
        zl, zh = op_rpm * 0.85, op_rpm * 1.15
        results_api = []
        for i, (vc, log_d) in enumerate(zip(vc_rpm[:6], ld[:6])):
            in_zone = zl <= vc <= zh
            ok = not in_zone and log_d >= 0.1
            results_api.append({
                "Mode": i+1,
                "fn (Hz)": f"{fn[i]:.2f}",
                "Vitesse critique (RPM)": f"{vc:.0f}",
                "Zone interdite": "❌ OUI" if in_zone else "✅ NON",
                "Log Dec ≥ 0.1": "✅" if log_d >= 0.1 else "❌",
                "Conforme API 684": "✅" if ok else "❌",
            })
        st.dataframe(pd.DataFrame(results_api), use_container_width=True, hide_index=True)
        st.markdown(f"**Zone interdite API 684 :** [{zl:.0f} – {zh:.0f}] RPM")
        n_ok = sum(1 for r in results_api if r["Conforme API 684"] == "✅")
        score = n_ok / max(len(results_api), 1) * 100
        color = "#22863A" if score >= 100 else "#C55A11" if score >= 67 else "#C00000"
        st.markdown(f"<h3 style='color:{color}'>Score conformité API 684 : {score:.0f}%</h3>",
                     unsafe_allow_html=True)
        # Export
        df_api = pd.DataFrame(results_api)
        rep = ReportGenerator(st.session_state.get("user_name",""))
        html = rep.html_report("Rapport API 684",
            {"Vitesse opérationnelle (RPM)": f"{op_rpm:.0f}",
             "Zone interdite": f"[{zl:.0f} – {zh:.0f}] RPM",
             "Score conformité": f"{score:.0f}%"},
            [{"title": "Résultats API 684", "table": df_api}])
        st.download_button("📄 Export Rapport API 684 (HTML)", data=html.encode(),
                            file_name="rapport_api684.html", mime="text/html")


# ── M4 — Balourd & Réponse Fréquentielle ─────────────────────────────────────
def _render_m4():
    st.subheader("🌀 M4 — Réponse au Balourd & Réponse Fréquentielle H(jω)")
    rotor = _CACHE.get("free_rotor")
    if rotor is None:
        st.warning("⚠️ Aucun rotor — M1 d'abord"); return

    eng = SimulationEngine(rotor)
    n_nodes = len(rotor.nodes) - 1
    tab_bal, tab_freq = st.tabs(["🌀 Balourd", "📡 H(jω) Fréquentielle"])

    with tab_bal:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🔩 Balourd**")
            un  = st.slider("Nœud", 0, n_nodes, min(2, n_nodes), key="m4_un")
            mag = st.number_input("Magnitude (kg·m)", 1e-5, 0.5, 0.001, format="%.5f", key="m4_mag")
            ph  = st.slider("Phase (°)", 0, 360, 0, key="m4_ph")
        with col2:
            st.markdown("**📡 Sonde**")
            pn  = st.slider("Nœud probe", 0, n_nodes, min(2, n_nodes), key="m4_pn")
            pd  = st.radio("Direction", ["X (0)","Y (1)"], horizontal=True, key="m4_pd")
            pdf = 0 if "X" in pd else 1
            fm  = st.slider("Fmax (Hz)", 100, 5000, 2000, key="m4_fm")

        if st.button("🌀 Calculer réponse au balourd", type="primary", key="m4_bal"):
            with st.spinner("Calcul balourd..."):
                res = eng.run_unbalance(nodes=[un], mags=[mag],
                                         phases=[np.deg2rad(ph)], fmax=float(fm))
            if res:
                _CACHE["m4_unbal"] = res
                _CACHE["m4_probe"] = (pn, pdf, float(fm))
                modal = _CACHE.get("free_modal")
                vtab1, vtab2, vtab3 = st.tabs(["📊 Bode", "🎯 Polaire", "📈 + Campbell"])
                with vtab1:
                    _plot_bode_unbal(res, pn, pdf, float(fm), modal)
                with vtab2:
                    _plot_polar_unbal(res, pn, pdf)
                with vtab3:
                    _plot_camp_unbal(res, rotor, pn, pdf, float(fm))
            else:
                st.error(f"Erreur : {eng.last_error}")

    with tab_freq:
        n_dof = len(rotor.nodes) * 4
        col1, col2 = st.columns(2)
        with col1:
            inp_n = st.slider("DDL excitation (inp)", 0, min(n_dof-1,31), 0, key="m4_inp")
            st.caption(f"Nœud {inp_n//4}, DDL {inp_n%4} ({'XY'[inp_n%4//2]})")
        with col2:
            out_n = st.slider("DDL réponse (out)", 0, min(n_dof-1,31), min(8,n_dof-1), key="m4_out")
            fm2   = st.slider("Fmax (Hz)", 100, 5000, 2000, key="m4_fm2")
        if st.button("📡 Calculer H(jω)", type="primary", key="m4_freq"):
            with st.spinner("Calcul H(jω)..."):
                fr = eng.run_freq_response(inp_n, out_n, float(fm2))
            if fr:
                _CACHE["m4_freq"] = fr
                modal = _CACHE.get("free_modal")
                _plot_freq_resp(fr, inp_n, out_n, float(fm2), modal)
                with st.expander("🔄 Diagramme de Nyquist"):
                    _plot_nyquist(fr, inp_n, out_n)
            else:
                st.error(f"Erreur : {eng.last_error}")


def _plot_polar_unbal(res, probe_node, probe_dof):
    for m, kw in [("plot_polar_bode", {"probe":[probe_node,probe_dof]}),
                   ("plot_polar_bode", {"probe": probe_node})]:
        if hasattr(res, m):
            try:
                st.plotly_chart(getattr(res, m)(**kw), use_container_width=True)
                return
            except Exception: continue
    try:
        freqs, amps, phases = _extract_unbal(res, probe_node, probe_dof)
        x_re = amps * np.cos(phases) * 1e6
        y_im = amps * np.sin(phases) * 1e6
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x_re, y=y_im, mode="lines+markers",
            marker=dict(size=4, color=freqs, colorscale="Viridis",
                        colorbar=dict(title="Hz"), showscale=True),
            line=dict(color="#1F5C8B", width=1.5), name="Trajectoire"))
        idx_max = int(np.argmax(amps))
        fig.add_trace(go.Scatter(x=[x_re[idx_max]], y=[y_im[idx_max]],
            mode="markers+text", marker=dict(size=14, color="#C00000", symbol="star"),
            text=[f"{freqs[idx_max]:.0f} Hz"], textposition="top center", name="Résonance"))
        fig.update_layout(title="Diagramme Polaire de Bode",
                           xaxis_title="Re (µm)", yaxis_title="Im (µm)",
                           yaxis_scaleanchor="x", height=480)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(f"Diagramme polaire non disponible : {e}")


def _plot_camp_unbal(res, rotor, probe_node, probe_dof, freq_max):
    try:
        freqs, amps, _ = _extract_unbal(res, probe_node, probe_dof)
        speeds = np.linspace(0, freq_max*2*np.pi, 40)
        camp   = rotor.run_campbell(speeds)
        spd_rpm = speeds * 30 / np.pi
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        if hasattr(camp, 'wd') and camp.wd is not None:
            fn_m = camp.wd / (2*np.pi)
            for i in range(min(4, fn_m.shape[1])):
                fig.add_trace(go.Scatter(x=spd_rpm, y=fn_m[:,i], name=f"Mode {i+1}",
                    line=dict(dash="dot"), opacity=0.6), secondary_y=False)
        fig.add_trace(go.Scatter(x=spd_rpm, y=spd_rpm/60, name="1X",
            line=dict(color="red", dash="dash"), opacity=0.5), secondary_y=False)
        fig.add_trace(go.Scatter(x=freqs*60, y=amps*1e6, name=f"Balourd nœud {probe_node}",
            line=dict(color="#FF6B00", width=3), fill="tozeroy",
            fillcolor="rgba(255,107,0,0.1)"), secondary_y=True)
        fig.update_layout(title="Campbell + Réponse au balourd superposés", height=480)
        fig.update_yaxes(title_text="Fréquence (Hz)", secondary_y=False)
        fig.update_yaxes(title_text="Amplitude (µm)", secondary_y=True, showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("💡 Les pics de la courbe orange coïncident avec les vitesses critiques (intersections 1X)")
    except Exception as e:
        st.info(f"Superposition non disponible : {e}")


def _plot_nyquist(freq_resp, inp_dof, out_dof):
    try:
        for attr in ("freq_resp","response","H"):
            if hasattr(freq_resp, attr):
                H_raw = np.array(getattr(freq_resp, attr)); break
        else: return
        H = H_raw[out_dof,inp_dof,:] if H_raw.ndim==3 else H_raw
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=H.real, y=H.imag, mode="lines",
            line=dict(color="#1F5C8B", width=2), name="H(jω)"))
        fig.update_layout(title="Nyquist — H(jω)", xaxis_title="Re",
                           yaxis_title="Im", yaxis_scaleanchor="x", height=420)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"Nyquist non disponible : {e}")


# ── M5 — Temporel & Défauts ───────────────────────────────────────────────────
def _render_m5():
    st.subheader("⏱️ M5 — Réponse Temporelle & Analyse de Défauts")
    rotor = _CACHE.get("free_rotor")
    if rotor is None:
        st.warning("⚠️ Aucun rotor — M1 d'abord"); return

    eng = SimulationEngine(rotor)
    n_nodes = len(rotor.nodes) - 1

    tab_time, tab_crack, tab_mis, tab_rub = st.tabs([
        "⏱️ Temporel & Orbites", "🔧 Fissure (Crack)", "↔️ Désalignement", "🔥 Frottement"
    ])

    with tab_time:
        col1, col2 = st.columns(2)
        with col1:
            speed_rpm = st.slider("Vitesse (RPM)", 500, 10000, 3000, key="m5_sp")
            t_end     = st.slider("Durée (s)", 0.2, 5.0, 1.0, key="m5_te")
        with col2:
            n_t    = st.slider("Points temporels", 200, 2000, 500, key="m5_nt")
            node_o = st.slider("Nœud d'observation", 0, n_nodes, min(2,n_nodes), key="m5_no")
        if st.button("⏱️ Simuler la réponse temporelle", type="primary", key="m5_time"):
            t = np.linspace(0, t_end, n_t)
            # Force balourd tournante
            omega = speed_rpm * np.pi / 30
            F = np.zeros((rotor.ndof, n_t))
            dof_x = node_o * 4
            if dof_x < rotor.ndof:
                F[dof_x, :] = 0.001 * omega**2 * np.cos(omega * t)
                if dof_x+1 < rotor.ndof:
                    F[dof_x+1, :] = 0.001 * omega**2 * np.sin(omega * t)
                    
            with st.spinner("Intégration temporelle (peut prendre du temps)..."):
                tr = eng.run_time_response(speed_rpm, F, t)
                
            if tr:
                _CACHE["m5_time"] = tr
                try:
                    # 1. Extraction robuste des données
                    t_arr = getattr(tr, 'time', getattr(tr, 't', t))
                    resp = getattr(tr, 'yout', getattr(tr, 'response', getattr(tr, 'disp', None)))
                    
                    if resp is None:
                        raise ValueError("Données introuvables dans l'objet de réponse.")
                        
                    t_arr = np.array(t_arr)
                    resp = np.array(resp)
                    
                    # 2. Transposition si matrice inversée
                    if resp.ndim >= 2 and resp.shape[0] == len(t_arr) and resp.shape[1] != len(t_arr):
                        resp = resp.T
                        
                    # 3. Extraction DDL X et Y pour la sonde
                    safe_x = min(dof_x, resp.shape[0] - 1)
                    safe_y = min(dof_x + 1, resp.shape[0] - 1)
                    x_um = resp[safe_x, :] * 1e6
                    y_um = resp[safe_y, :] * 1e6
                    
                    # 4. Affichage Plotly Sur Mesure
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        fig_t = go.Figure()
                        fig_t.add_trace(go.Scatter(x=t_arr, y=x_um, name="X", line=dict(color="#1F5C8B", width=1.5)))
                        fig_t.add_trace(go.Scatter(x=t_arr, y=y_um, name="Y", line=dict(color="#C55A11", width=1.5)))
                        fig_t.update_layout(title=f"Réponse Temporelle (Nœud {node_o})", 
                                            xaxis_title="Temps (s)", yaxis_title="Déplacement (µm)", height=400)
                        st.plotly_chart(fig_t, use_container_width=True)
                        
                    with col_p2:
                        fig_o = go.Figure()
                        fig_o.add_trace(go.Scatter(x=x_um, y=y_um, mode="lines", name="Orbite", 
                                                   line=dict(color="#22863A", width=2)))
                        fig_o.update_layout(title=f"Orbite (Nœud {node_o})", 
                                            xaxis_title="X (µm)", yaxis_title="Y (µm)", 
                                            yaxis_scaleanchor="x", height=400)
                        st.plotly_chart(fig_o, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"Le calcul a réussi, mais l'extraction graphique est impossible : {e}")
            else:
                st.error(f"Erreur : {eng.last_error}")
    with tab_crack:
        st.info("Modèle de fissure transversale (Gasch/Mayes) — variation de raideur oscillante → harmonique 2X caractéristique")
        col1, col2 = st.columns(2)
        with col1:
            crack_depth = st.slider("Profondeur de fissure (α = a/R)", 0.0, 0.9, 0.3, key="m5_cd",
                                     help="Ratio profondeur/rayon de l'arbre")
            crack_node  = st.slider("Nœud de la fissure", 1, max(1,n_nodes-1), max(1,n_nodes//2), key="m5_cn")
        with col2:
            model_type = st.radio("Modèle de fissure", ["Gasch", "Mayes"], key="m5_cm")
            speed_rpm_c = st.slider("Vitesse (RPM)", 500, 5000, 1500, key="m5_csp")
        if st.button("🔧 Simuler la fissure", type="primary", key="m5_crack"):
            try:
                crack_res = eng.run_crack(
                    crack_depth=crack_depth,
                    crack_node=crack_node,
                    speed=float(speed_rpm_c)*np.pi/30,
                    model=model_type.lower()
                )
                if crack_res:
                    for m in ["plot_dfft", "plot_orbit", "plot"]:
                        if hasattr(crack_res, m):
                            try:
                                st.plotly_chart(getattr(crack_res, m)(), use_container_width=True)
                                break
                            except Exception: continue
                else:
                    st.error(f"Erreur : {eng.last_error}")
            except Exception as e:
                st.warning(f"run_crack non disponible dans cette version ROSS : {e}")
                st.info("**Simulation de référence :** Une fissure de profondeur α=0.3 sur un arbre Ø50mm "
                         "génère une composante 2X dans le spectre de vibration. L'amplitude 2X/1X augmente "
                         "avec la profondeur et atteint un maximum à la demi-vitesse critique.")

    with tab_mis:
        st.info("Désalignement parallèle/angulaire d'un accouplement — génère des harmoniques 2X, 3X")
        col1, col2 = st.columns(2)
        with col1:
            mis_type = st.radio("Type désalignement", ["Parallèle", "Angulaire"], key="m5_mt")
            mis_val  = st.slider("Magnitude (mm ou °)", 0.1, 2.0, 0.5, key="m5_mv")
        with col2:
            mis_node = st.slider("Nœud accouplement", 1, max(1,n_nodes-1), max(1,n_nodes//2), key="m5_mn")
            speed_rpm_m = st.slider("Vitesse (RPM)", 500, 8000, 3000, key="m5_msp")
        if st.button("↔️ Simuler le désalignement", type="primary", key="m5_mis"):
            try:
                mis_res = eng.run_misalignment(
                    mis_type=mis_type.lower(),
                    misalignment=mis_val * (1e-3 if mis_type=="Parallèle" else np.pi/180),
                    n=mis_node,
                    speed=float(speed_rpm_m)*np.pi/30
                )
                if mis_res:
                    for m in ["plot_dfft", "plot_orbit", "plot"]:
                        if hasattr(mis_res, m):
                            try:
                                st.plotly_chart(getattr(mis_res, m)(), use_container_width=True)
                                break
                            except Exception: continue
                else:
                    st.error(f"Erreur : {eng.last_error}")
            except Exception as e:
                st.warning(f"run_misalignment non disponible dans cette version ROSS : {e}")

    with tab_rub:
        st.info("Contact rotor-stator — non-linéaire, peut générer des sous-harmoniques et du chaos")
        col1, col2 = st.columns(2)
        with col1:
            clearance = st.slider("Jeu rotor-stator (µm)", 10, 500, 100, key="m5_rc")
            stiffness = st.number_input("Raideur contact (N/m)", 1e5, 1e9, 1e7, format="%.2e", key="m5_rs")
        with col2:
            rub_node  = st.slider("Nœud contact", 1, max(1,n_nodes-1), max(1,n_nodes//2), key="m5_rn")
            speed_rpm_r = st.slider("Vitesse (RPM)", 500, 8000, 3000, key="m5_rsp")
        if st.button("🔥 Simuler le frottement", type="primary", key="m5_rub"):
            try:
                rub_res = eng.run_rubbing(
                    radial_clearance=float(clearance)*1e-6,
                    contact_stiffness=float(stiffness),
                    n=rub_node,
                    speed=float(speed_rpm_r)*np.pi/30
                )
                if rub_res:
                    for m in ["plot_dfft", "plot_orbit", "plot"]:
                        if hasattr(rub_res, m):
                            try:
                                st.plotly_chart(getattr(rub_res, m)(), use_container_width=True)
                                break
                            except Exception: continue
                else:
                    st.error(f"Erreur : {eng.last_error}")
            except Exception as e:
                st.warning(f"run_rubbing non disponible dans cette version ROSS : {e}")
                st.info("Quand le rotor touche le stator (jeu < amplitude vibratoire), "
                         "le contact génère des sous-harmoniques (0.5X, 0.33X) et peut "
                         "conduire à un comportement chaotique.")


# =============================================================================
# PAGE : BIBLIOTHÈQUE ROSS
# =============================================================================
def render_library():
    st.title("📚 Bibliothèque — Exemples Officiels ROSS & Documentation")

    tab_ex, tab_theory, tab_api, tab_norms = st.tabs([
        "🏭 Exemples ROSS", "📐 Théorie", "🛠️ API ROSS", "📏 Normes"
    ])

    with tab_ex:
        st.markdown("### Exemples officiels de la documentation ROSS")
        examples = {
            "🏭 Compresseur centrifuge (rs.compressor_example())": {
                "desc": "Rotor industriel complet — 7 sections, 2 paliers hydrodynamiques, 5 disques/roues. Analyse Campbell 0–10 000 RPM.",
                "func": "compressor_example"
            },
        }
        for name, info in examples.items():
            with st.expander(name):
                st.info(info["desc"])
                if ROSS_AVAILABLE and st.button(f"Charger cet exemple", key=f"load_{info['func']}"):
                    try:
                        rotor = getattr(rs, info["func"])()
                        _CACHE["free_rotor"] = rotor
                        st.success(f"✅ Chargé — {len(rotor.nodes)} nœuds | {rotor.m:.1f} kg")
                        st.info("👉 Allez dans 🔬 Mode Simulation pour l'analyser")
                    except Exception as e:
                        st.error(f"Erreur : {e}")
        # Codes de tutoriels de référence
        st.markdown("---")
        st.markdown("### 📓 Codes de Référence des Tutoriels")
        for tid, tdata in TUTORIALS.items():
            with st.expander(f"{tdata['level']} {tdata['title']}"):
                st.markdown(f"**API utilisée :** {', '.join(tdata['api'])}")
                for step in tdata["steps"]:
                    st.markdown(f"**{step['title']}**")
                    st.code(step["code"], language="python")

    with tab_theory:
        st.markdown("""
        ## Fondements Théoriques

        ### Modèle de Timoshenko
        L'arbre est discrétisé en éléments de **poutre de Timoshenko** à 4 DDL/nœud :
        translations u, v et rotations β, γ dans les plans XZ et YZ.

        L'énergie cinétique et l'énergie de déformation intègrent :
        - Cisaillement transversal (facteur κ)
        - Inertie rotatoire des sections

        ### Équation du mouvement
        """)
        st.latex(r"[M]\{\ddot{q}\} + ([C]+[G])\{\dot{q}\} + [K]\{q\} = \{F(t)\}")
        st.markdown("""
        Où **G** = matrice gyroscopique (antisymétrique, proportionnelle à Ω).

        ### Analyse Modale Complexe
        """)
        st.latex(r"\det([K] + j\omega[C] - \omega^2[M]) = 0 \Rightarrow \lambda_k = -\xi_k\omega_{nk} \pm j\omega_{dk}")
        st.latex(r"\delta_k = \frac{2\pi\xi_k}{\sqrt{1-\xi_k^2}} \quad \text{(Décrément logarithmique)}")
        st.markdown("""
        ### Diagramme de Campbell
        Trace les fréquences des modes en fonction de la vitesse Ω.
        **Effet gyroscopique** : sépare les modes FW (Forward Whirl) et BW (Backward Whirl).
        **Vitesses critiques** : intersections avec les droites nX (n=1,2,3...).

        ### Réponse au Balourd
        Force tournante : **F = m·e·ω²** — réponse dans le domaine fréquentiel via :
        """)
        st.latex(r"\{q(\omega)\} = [H(\omega)]\{F_{balourd}\} = ([K] + j\omega[C] - \omega^2[M])^{-1}\{F\}")
        st.latex(r"\text{DAF} = \frac{A_{max}}{A_{statique}} = \frac{1}{2\xi}\quad\text{(cas non-amorti)}")

    with tab_api:
        st.markdown("""
        ## API ROSS — Référence Complète

        | Méthode | Analyse | Sorties principales |
        |---------|---------|---------------------|
        | `run_static()` | Statique | `.plot_deflected_shape()`, `.plot_bending_moment()` |
        | `run_modal(speed)` | Modal | `.wn`, `.wd`, `.log_dec`, `.plot_mode_3d()` |
        | `run_campbell(speeds)` | Campbell | `.plot()`, `.log_dec`, `.wd`, `.whirl` |
        | `run_critical_speed()` | Vitesses critiques | `.plot()`, valeurs numériques |
        | `run_unbalance_response()` | Balourd | `.plot_magnitude()`, `.plot_phase()`, `.plot_polar_bode()` |
        | `run_freq_response(inp,out)` | H(jω) | `.plot_bode()`, `.plot_magnitude()` |
        | `run_time_response()` | Temporel | `.plot_orbit()`, `.plot_dfft()` |
        | `run_misalignment()` | Désalignement | Orbites, DFFT |
        | `run_rubbing()` | Frottement | Réponse non-linéaire |
        | `run_crack()` | Fissure | Signature 2X |

        **Référence :** Timbó et al. (2020), JOSS 5(48):2120
        **Documentation :** https://ross.readthedocs.io/en/stable/
        """)
        st.code("""
# Exemple complet ROSS
import ross as rs
import numpy as np

steel = rs.Material(name='Steel', rho=7810, E=211e9, G_s=81.2e9)
shaft = [rs.ShaftElement(L=0.25, idl=0, odl=0.05, material=steel) for _ in range(6)]
disk  = rs.DiskElement.from_geometry(n=3, material=steel, width=0.07, i_d=0.05, o_d=0.30)
b0    = rs.BearingElement(n=0, kxx=1e7, kyy=1e7, cxx=500, cyy=500)
b6    = rs.BearingElement(n=6, kxx=1e7, kyy=1e7, cxx=500, cyy=500)
rotor = rs.Rotor(shaft, [disk], [b0, b6])

# Analyses
modal  = rotor.run_modal(speed=0)
speeds = np.linspace(0, 10000*np.pi/30, 100)
camp   = rotor.run_campbell(speeds)
unbal  = rotor.run_unbalance_response(node=[3], magnitude=[0.001],
             phase=[0.0], frequency_range=np.linspace(0, 5000, 500))
        """, language="python")

    with tab_norms:
        st.markdown("""
        ## Normes Industrielles de Référence

        ### API 684 — 2ème Édition
        Standard pour la rotordynamique des turbomachines industrielles :
        1. **Marge de vitesse critique ≥ 15%** — pas de vitesse critique dans [0.85·Nop, 1.15·Nop]
        2. **Log Dec ≥ 0.1** pour tous les modes dans la plage opérationnelle
        3. **Réponse au balourd** — amplitudes < limites ISO 7919
        4. **Analyse de stabilité** — calcul du niveau d'instabilité résiduelle

        ### ISO 1925 — Terminologie Vibrations
        Définitions officielles : balourd, déséquilibre résiduel, qualité G, rayon d'excentricité.

        ### ISO 7919-3 — Vibrations des machines
        Limites de déplacement de vibration pour turbines et compresseurs.

        ### Références Académiques
        - Friswell, M.I. et al. (2010) *Dynamics of Rotating Machines* — Cambridge University Press
        - Timbó et al. (2020) JOSS 5(48):2120 — *An Open Source Code for Rotordynamics*
        - Santos et al. (2025) — *An Open-Source Software for Rotordynamics (ROSS 2.0)*
        """)


# =============================================================================
# PAGE : ROSS GPT — Assistant IA
# =============================================================================
def render_ross_gpt():
    st.title("💬 ROSS GPT — Assistant Virtuel Spécialisé")
    st.caption("Propulsé par Anthropic Claude • Contexte ROSS injecté automatiquement")

    st.markdown("""
    <div class='card'>
    <b>🤖 ROSS GPT</b> est un assistant IA spécialisé dans la bibliothèque ROSS.<br>
    Il peut : générer du code Python ROSS, expliquer les résultats, suggérer des optimisations,
    déboguer votre modèle et répondre à vos questions de rotordynamique.
    </div>
    """, unsafe_allow_html=True)

    # Quick prompts contextuels
    st.markdown("### 💡 Questions rapides")
    quick_prompts = [
        "Comment créer un rotor simple avec ROSS ?",
        "Explique-moi le diagramme de Campbell",
        "Pourquoi le Log Dec peut-il être négatif ?",
        "Comment améliorer la stabilité de mon rotor ?",
        "Quelle est la différence entre précession avant et arrière ?",
        "Comment calculer le DAF (Dynamic Amplification Factor) ?",
        "Comment modéliser un défaut de fissure avec ROSS ?",
        "Comment vérifier la conformité API 684 ?",
    ]
    cols = st.columns(4)
    for i, qp in enumerate(quick_prompts):
        with cols[i % 4]:
            if st.button(qp[:40] + ("..." if len(qp)>40 else ""), key=f"qp_{i}"):
                st.session_state["gpt_input"] = qp

    st.markdown("---")

    # Contexte du modèle courant
    rotor = _CACHE.get("free_rotor")
    modal = _CACHE.get("free_modal")
    context_json = {"rotor_loaded": rotor is not None}
    if rotor:
        context_json.update({
            "n_nodes": len(rotor.nodes),
            "mass_kg": round(float(rotor.m), 3),
        })
    if modal:
        fn = modal.wn / (2*np.pi)
        ld = getattr(modal, 'log_dec', [])
        context_json["modal"] = {
            "fn_hz": [round(float(v),2) for v in fn[:4]],
            "log_dec": [round(float(v),4) for v in ld[:4]],
        }

    with st.expander("🔍 Contexte injecté automatiquement"):
        st.json(context_json)

    # Interface chat
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Posez votre question ROSS...",
                                key="gpt_chat_input")
    if not user_input:
        user_input = st.session_state.pop("gpt_input", None)

    if user_input:
        st.session_state["chat_history"].append({"role":"user","content":user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("ROSS GPT réfléchit..."):
                response = _call_ross_gpt(user_input, context_json,
                                           st.session_state["chat_history"][:-1])
            st.markdown(response)
            st.session_state["chat_history"].append({"role":"assistant","content":response})

    if st.session_state.get("chat_history"):
        if st.button("🗑️ Effacer la conversation", key="gpt_clear"):
            st.session_state["chat_history"] = []
            st.rerun()


def _call_ross_gpt(user_msg: str, context: dict, history: list) -> str:
    """Appel à l'API Anthropic Claude avec contexte ROSS spécialisé."""
    try:
        import anthropic
        client = anthropic.Anthropic()

        system_prompt = f"""Tu es ROSS GPT, un expert spécialisé dans la bibliothèque Python ROSS (Rotordynamic Open-Source Software).

Tu maîtrises parfaitement :
- La modélisation rotordynamique : ShaftElement (Timoshenko), DiskElement, BearingElement, GearElement, CouplingElement, Material, Probe
- Toutes les analyses ROSS : run_static(), run_modal(), run_campbell(), run_critical_speed(), run_unbalance_response(), run_freq_response(), run_time_response(), run_crack(), run_misalignment(), run_rubbing()
- Les packages complémentaires : ROSS-FluidFlow, ROSS-Stochastic
- La théorie de la dynamique des rotors : MEF, matrices M/K/C/G, Campbell, modes propres, Log Dec, DAF, précession avant/arrière
- Les normes : API 684, ISO 7919, ISO 1925

Contexte du modèle actuellement chargé dans ROSSim Online :
{json.dumps(context, ensure_ascii=False, indent=2)}

Réponds en français. Fournis du code Python ROSS fonctionnel quand c'est pertinent.
Sois concis, précis et pédagogique. Indique toujours les unités physiques dans tes réponses."""

        messages = []
        for h in history[-6:]:  # Contexte des 6 derniers échanges
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_msg})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

    except ImportError:
        return _fallback_ross_gpt(user_msg, context)
    except Exception as e:
        if "authentication" in str(e).lower() or "api_key" in str(e).lower():
            return (_fallback_ross_gpt(user_msg, context) +
                    "\n\n*Note : Pour activer ROSS GPT complet, configurez votre clé API Anthropic.*")
        return f"❌ Erreur ROSS GPT : {e}\n\n" + _fallback_ross_gpt(user_msg, context)


def _fallback_ross_gpt(user_msg: str, context: dict) -> str:
    """Réponses intelligentes pré-programmées sans API (mode offline)."""
    msg_lower = user_msg.lower()
    rotor_loaded = context.get("rotor_loaded", False)

    if any(k in msg_lower for k in ["créer", "créer un rotor", "premier rotor", "modéliser"]):
        return """## Créer un rotor avec ROSS

```python
import ross as rs
import numpy as np

# 1. Matériau
steel = rs.Material(name='Steel', rho=7810, E=211e9, G_s=81.2e9)

# 2. Arbre (6 éléments de 250mm, Ø50mm)
shaft = [rs.ShaftElement(L=0.25, idl=0.0, odl=0.05, material=steel)
         for _ in range(6)]

# 3. Disque au nœud central
disk = rs.DiskElement.from_geometry(n=3, material=steel,
                                      width=0.07, i_d=0.05, o_d=0.25)

# 4. Paliers aux extrémités
b0 = rs.BearingElement(n=0, kxx=1e7, kyy=1e7, cxx=500, cyy=500)
b6 = rs.BearingElement(n=6, kxx=1e7, kyy=1e7, cxx=500, cyy=500)

# 5. Assemblage
rotor = rs.Rotor(shaft, [disk], [b0, b6])
print(f"Masse : {rotor.m:.2f} kg | Nœuds : {rotor.nodes}")
rotor.plot_rotor()
```
**Conseils :** Les nœuds sont numérotés de 0 à N (N = nombre d'éléments). Les paliers doivent être placés sur des nœuds existants."""

    if "campbell" in msg_lower:
        return """## Diagramme de Campbell

Le diagramme de Campbell trace les **fréquences des modes** en fonction de la **vitesse de rotation** Ω.

**Interprétation :**
- Chaque courbe = un mode de vibration
- **FW (Forward Whirl)** = précession dans le sens de rotation → fréquence augmente avec Ω
- **BW (Backward Whirl)** = précession inverse → fréquence diminue avec Ω
- **Droite 1X** (synchrone) : intersections = **vitesses critiques**
- **Droite 2X, 3X** : excitations super-synchrones (engrenages, défauts)

```python
speeds = np.linspace(0, 10000*np.pi/30, 100)  # 0 à 10 000 RPM
camp = rotor.run_campbell(speeds)
camp.plot()  # Graphique Plotly interactif

# Accéder aux données numériques
print("Fréquences amorties (rad/s) :", camp.wd[:5, :4])
print("Log Dec :", camp.log_dec[:5, :4])
```
**Norme API 684 :** Les vitesses critiques doivent être à ±15% de la vitesse opérationnelle."""

    if "log dec" in msg_lower or "instabilit" in msg_lower or "stabilité" in msg_lower:
        return """## Log Décrément et Stabilité

Le **Log Dec (δ)** quantifie l'amortissement de chaque mode :

δ = 2π·ξ / √(1−ξ²)

**Interprétation :**
- δ > 0.3 : ✅ Très bien amorti (stable)
- 0.1 < δ < 0.3 : 🟡 Correctement amorti (conforme API 684)
- 0 < δ < 0.1 : ⚠️ Peu amorti (limite API 684 = 0.1)
- **δ < 0 : ❌ INSTABLE** → l'amplitude croît exponentiellement

**Cause principale d'instabilité :** raideur croisée Kxy des paliers hydrodynamiques.

```python
modal = rotor.run_modal(speed=3000*np.pi/30)
print("Log Dec :", modal.log_dec[:6])
# Identifier les modes instables
for i, ld in enumerate(modal.log_dec[:6]):
    status = "✅" if ld > 0.1 else "⚠️" if ld > 0 else "❌ INSTABLE"
    print(f"Mode {i+1}: δ={ld:.4f} {status}")
```"""

    if "daf" in msg_lower or "balourd" in msg_lower or "déséquilibre" in msg_lower:
        return """## Réponse au Balourd et DAF

```python
resp = rotor.run_unbalance_response(
    node=[2],            # Nœud du balourd
    magnitude=[0.001],   # 0.001 kg·m = 1g × 1m
    phase=[0.0],         # Phase initiale (rad)
    frequency_range=np.linspace(0, 5000, 500)
)
resp.plot_magnitude(probe=[2, 0])  # probe = [nœud, direction]
resp.plot_phase(probe=[2, 0])
resp.plot_polar_bode(probe=[2, 0])
```

**DAF (Dynamic Amplification Factor) = A_max / A_statique**
- Pour un système 1-DDL non amorti : DAF = 1/(2ξ)
- À la résonance : amplification maximale
- Réduction du DAF : augmenter l'amortissement cxx/cyy des paliers"""

    if "api 684" in msg_lower or "norme" in msg_lower or "conformité" in msg_lower:
        return """## Vérification API 684

La norme API 684 impose pour les turbomachines industrielles :

1. **Marge de vitesse critique ≥ 15%**
   - Pas de vitesse critique dans [0.85×Nop, 1.15×Nop]
2. **Log Dec ≥ 0.1** pour tous les modes
3. **Réponse au balourd** < limites ISO 7919

```python
op_rpm = 3000  # Vitesse opérationnelle
modal = rotor.run_modal(speed=0)
fn_hz = modal.wn / (2*np.pi)
vc_rpm = fn_hz * 60

zone_low  = op_rpm * 0.85  # 2550 RPM
zone_high = op_rpm * 1.15  # 3450 RPM

for i, (vc, ld) in enumerate(zip(vc_rpm[:6], modal.log_dec[:6])):
    in_zone = zone_low <= vc <= zone_high
    ok = not in_zone and ld >= 0.1
    print(f"Mode {i+1}: Vc={vc:.0f} RPM | δ={ld:.3f} | {'✅ OK' if ok else '❌ NON CONFORME'}")
```"""

    if "fissure" in msg_lower or "crack" in msg_lower:
        return """## Simulation de Fissure Transversale (ROSS)

```python
# Modèle de Gasch ou Mayes
crack_res = rotor.run_crack(
    crack_depth=0.3,       # α = profondeur/rayon = 0 à 0.9
    crack_node=3,          # Nœud de la fissure
    speed=1500*np.pi/30,   # Vitesse en rad/s
    model='gasch'          # 'gasch' ou 'mayes'
)
crack_res.plot_dfft(probe=[3, 0], rpm=1500)
crack_res.plot_orbit(node=3)
```

**Signature vibratoire d'une fissure :**
- Harmonique **2X** caractéristique (variation de raideur deux fois par tour)
- Augmentation de l'amplitude **1X** à la vitesse critique
- Pic marqué à **N_crit/2** (demi-vitesse critique)"""

    # Réponse générique contextualisée
    ctx_info = ""
    if rotor_loaded:
        ctx_info = (f"\n\n*Votre rotor actuel : {context.get('n_nodes','?')} nœuds, "
                    f"{context.get('mass_kg','?')} kg*")
    return (f"""Je suis ROSS GPT, votre assistant spécialisé en dynamique des rotors avec ROSS.

Votre question : **{user_msg}**

Je peux vous aider avec :
- 🏗️ Création de modèles ROSS (ShaftElement, DiskElement, BearingElement)
- 📊 Analyses modales, Campbell, réponse au balourd, temporelle
- 🔧 Simulation de défauts (fissure, désalignement, frottement)
- 📐 Normes API 684, ISO 7919
- 💡 Optimisation et débogage de modèles

Posez une question plus précise ou utilisez les boutons de questions rapides ci-dessus.{ctx_info}
""")


# =============================================================================
# NAVIGATION PRINCIPALE — MAIN
# =============================================================================
def main():
    # ── Session State ──────────────────────────────────────────────────────────
    for key, default in [
        ("user_name", "Utilisateur"),
        ("badges", {}),
        ("tut_done", set()),
        ("sim_count", 0),
        ("chat_history", []),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ ROSSim Online")
        st.session_state["user_name"] = st.text_input(
            "👤 Votre nom :", st.session_state["user_name"])
        st.markdown("---")

        page = st.radio("🗺️ Navigation :", [
            "🏠 Tableau de Bord",
            "🎓 Mode Pédagogique",
            "🔬 Mode Simulation",
            "📚 Bibliothèque",
            "💬 ROSS GPT",
        ])

        # Progression
        if st.session_state["badges"]:
            st.markdown("---")
            st.markdown("**🏅 Mes Badges :**")
            for tid, btype in st.session_state["badges"].items():
                icon = {"gold":"🥇","silver":"🥈","bronze":"🥉"}.get(btype,"🏅")
                st.markdown(f"{icon} {tid}")

        # Statut ROSS
        st.markdown("---")
        if ROSS_AVAILABLE:
            st.success(f"✅ ROSS {ROSS_VERSION}")
        else:
            st.error("❌ ROSS non installé")
            st.code("pip install ross-rotordynamics", language="bash")

        # Rotor en mémoire
        if _CACHE.get("free_rotor"):
            r = _CACHE["free_rotor"]
            st.markdown("---")
            st.markdown("**🔧 Rotor actif :**")
            st.caption(f"  {len(r.nodes)} nœuds | {r.m:.2f} kg")

        st.caption("ROSSim Online v1.0 • ROSS Open-Source")

    # ── Routing ────────────────────────────────────────────────────────────────
    if "Tableau" in page:
        render_dashboard()
    elif "Pédagogique" in page:
        render_tutorial_mode()
    elif "Simulation" in page:
        render_simulation_mode()
    elif "Bibliothèque" in page:
        render_library()
    elif "GPT" in page:
        render_ross_gpt()


if __name__ == "__main__":
    main()
