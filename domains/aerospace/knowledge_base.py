"""Aerospace research knowledge base - curated documents for RAG."""

from __future__ import annotations

from typing import List

from rag_master.models import Document

AEROSPACE_DOCUMENTS: List[Document] = [
    # --- Propulsion Systems ---
    Document(
        doc_id="prop_001",
        content=(
            "Ion propulsion systems use electrically charged atoms (ions) to create thrust. "
            "The NASA Evolutionary Xenon Thruster (NEXT) achieves a specific impulse of 4,190 seconds, "
            "far exceeding chemical rockets (typically 300-450s). Ion engines produce very low thrust "
            "(0.5 N for NEXT) but can operate continuously for years, making them ideal for deep space "
            "missions. The Dawn spacecraft used ion propulsion to orbit both Vesta and Ceres. "
            "Key challenges include electrode erosion, power supply mass, and thrust-to-weight ratio "
            "limitations for planetary departure maneuvers."
        ),
        source="NASA Technical Reports Server",
        metadata={"topic": "propulsion", "subtopic": "ion_propulsion", "difficulty": "medium"},
    ),
    Document(
        doc_id="prop_002",
        content=(
            "Rotating detonation engines (RDEs) represent a paradigm shift in aerospace propulsion. "
            "Unlike conventional pulse detonation engines, RDEs maintain a continuous detonation wave "
            "traveling around an annular combustion chamber at speeds of 1,500-2,500 m/s. This design "
            "eliminates the need for repeated ignition cycles. AFRL tests in 2023 demonstrated "
            "a 15% increase in thermodynamic efficiency over conventional turbofan combustors. "
            "Challenges include thermal management of chamber walls (temperatures exceed 3,000K), "
            "fuel injection timing, and acoustic/vibration coupling effects."
        ),
        source="AIAA Journal of Propulsion and Power",
        metadata={"topic": "propulsion", "subtopic": "detonation_engines", "difficulty": "hard"},
    ),
    Document(
        doc_id="prop_003",
        content=(
            "Nuclear thermal propulsion (NTP) uses a nuclear reactor to heat hydrogen propellant to "
            "extreme temperatures (2,500-3,000K), achieving specific impulse of 850-1,000 seconds — "
            "roughly double that of chemical rockets. NASA's DRACO program, in partnership with DARPA, "
            "aims to demonstrate NTP technology in orbit by 2027. The reactor uses HALEU "
            "(high-assay low-enriched uranium) fuel to address proliferation concerns. NTP could "
            "reduce Mars transit time from 9 months to approximately 4 months, significantly "
            "reducing crew radiation exposure and consumable requirements."
        ),
        source="NASA DRACO Program Documentation",
        metadata={"topic": "propulsion", "subtopic": "nuclear_thermal", "difficulty": "hard"},
    ),

    # --- Materials Science ---
    Document(
        doc_id="mat_001",
        content=(
            "Ceramic Matrix Composites (CMCs) are revolutionizing turbine engine design. Silicon carbide "
            "fiber-reinforced silicon carbide (SiC/SiC) CMCs operate at temperatures up to 1,316°C — "
            "about 200°C higher than nickel superalloys — while being one-third the weight. GE Aviation's "
            "CMC turbine shrouds in the LEAP engine reduced fuel consumption by 15%. CMC manufacturing "
            "involves chemical vapor infiltration (CVI) of SiC matrix into woven fiber preforms, "
            "followed by environmental barrier coating (EBC) application to prevent moisture attack. "
            "Challenges include long manufacturing cycles (weeks per part) and non-destructive inspection."
        ),
        source="Materials Science and Engineering Journal",
        metadata={"topic": "materials", "subtopic": "ceramic_composites", "difficulty": "medium"},
    ),
    Document(
        doc_id="mat_002",
        content=(
            "Ultra-high temperature ceramics (UHTCs) based on hafnium diboride (HfB2) and zirconium "
            "diboride (ZrB2) can withstand temperatures exceeding 3,000°C, making them candidates for "
            "hypersonic vehicle leading edges and rocket nozzle throats. Recent research at "
            "the German Aerospace Center (DLR) demonstrated HfB2-SiC composites maintaining structural "
            "integrity at 2,800°C in oxidizing environments for over 600 seconds. Additive manufacturing "
            "of UHTCs using selective laser sintering enables complex geometries previously impossible. "
            "Oxidation resistance remains the primary challenge for sustained hypersonic flight."
        ),
        source="Journal of the European Ceramic Society",
        metadata={"topic": "materials", "subtopic": "ultra_high_temp", "difficulty": "hard"},
    ),

    # --- Orbital Mechanics ---
    Document(
        doc_id="orb_001",
        content=(
            "Low-energy transfer trajectories exploit the gravitational dynamics of the Sun-Earth-Moon "
            "system's Lagrange points. The weak stability boundary (WSB) transfer to lunar orbit, "
            "first used by Japan's Hiten spacecraft, requires 25-30% less delta-v than Hohmann transfers "
            "at the cost of longer transit times (3-4 months vs. 3 days). The three-body problem "
            "dynamics create invariant manifolds — 'tubes' in phase space — that connect equilibrium "
            "regions. Applications include station-keeping at Earth-Moon L1/L2 (Artemis Gateway) "
            "and low-cost asteroid rendezvous missions."
        ),
        source="Celestial Mechanics and Dynamical Astronomy",
        metadata={"topic": "orbital_mechanics", "subtopic": "low_energy_transfer", "difficulty": "hard"},
    ),
    Document(
        doc_id="orb_002",
        content=(
            "Satellite constellation design involves optimizing orbital parameters to achieve global "
            "or regional coverage. Walker Delta constellations use i:t/p/f notation where t = total "
            "satellites, p = orbital planes, f = phasing factor. SpaceX's Starlink uses a Shell "
            "structure: Shell 1 has 1,584 satellites at 550 km (53° inclination), providing near-global "
            "coverage between ±53° latitude. The inter-satellite links operate at 100 Gbps using laser "
            "communication terminals. Orbital debris mitigation requires deorbit capability within "
            "5 years of mission end, driving propulsion system requirements."
        ),
        source="IEEE Transactions on Aerospace and Electronic Systems",
        metadata={"topic": "orbital_mechanics", "subtopic": "constellations", "difficulty": "medium"},
    ),

    # --- Thermal Protection ---
    Document(
        doc_id="therm_001",
        content=(
            "NASA's PICA-X (Phenolic Impregnated Carbon Ablator) heat shield, developed by SpaceX "
            "for Dragon capsules, can withstand heat fluxes up to 200 W/cm² during Earth reentry. "
            "The ablative material works by controlled charring — as the surface heats, pyrolysis "
            "gases are released that create a cool boundary layer, reducing convective heating by "
            "up to 50%. PICA-X is approximately 10x cheaper to manufacture than NASA's original "
            "PICA material used on Stardust. For Mars return missions, heat shield requirements "
            "increase to 300+ W/cm² due to higher entry velocities (12+ km/s)."
        ),
        source="NASA Ames Research Center",
        metadata={"topic": "thermal_protection", "subtopic": "ablative", "difficulty": "medium"},
    ),

    # --- Life Support ---
    Document(
        doc_id="life_001",
        content=(
            "The Environmental Control and Life Support System (ECLSS) on the International Space Station "
            "recycles approximately 90% of crew water. The Water Recovery System (WRS) uses vapor "
            "compression distillation (VCD) for urine processing and a catalytic oxidation reactor "
            "for removing organic contaminants. The Oxygen Generation System (OGS) electrolyzes water "
            "to produce O2 at 5.4 kg/day for a 6-person crew. For Mars missions, NASA targets "
            "98% water recovery and integration of Sabatier reactors to convert CO2 and H2 into "
            "methane and water, closing the carbon loop."
        ),
        source="NASA ECLSS Technical Reference",
        metadata={"topic": "life_support", "subtopic": "water_recycling", "difficulty": "medium"},
    ),
    Document(
        doc_id="life_002",
        content=(
            "Bioregenerative life support systems (BLSS) use plant cultivation to supplement or replace "
            "physicochemical air and water processing. The MELiSSA (Micro-Ecological Life Support "
            "Alternative) project by ESA demonstrates a closed-loop system using five compartments: "
            "thermophilic bacteria for waste liquefaction, photoautotrophic bacteria for CO2 fixation, "
            "nitrifying bacteria for nitrogen conversion, higher plants for food production and "
            "O2 generation, and the crew compartment. Current efficiency is approximately 70% mass "
            "closure. Integration with hydroponic systems can provide 25-50% of crew caloric needs."
        ),
        source="ESA MELiSSA Project Documentation",
        metadata={"topic": "life_support", "subtopic": "bioregenerative", "difficulty": "hard"},
    ),

    # --- Autonomy and AI ---
    Document(
        doc_id="auto_001",
        content=(
            "Autonomous spacecraft operations use model-based planning systems. NASA's AEGIS "
            "(Autonomous Exploration for Gathering Increased Science) on Mars rovers enables "
            "autonomous target selection for ChemCam laser spectroscopy, increasing science return "
            "by 20-30%. The system uses onboard image classification (trained on terrestrial analog "
            "datasets) to identify scientifically interesting rock targets during communication "
            "blackout periods. Fault detection, isolation, and recovery (FDIR) systems employ "
            "Bayesian networks and model-based diagnosis to maintain operations when ground contact "
            "is unavailable for up to 20 minutes (Mars) or hours (outer solar system)."
        ),
        source="NASA JPL Autonomous Systems Division",
        metadata={"topic": "autonomy", "subtopic": "autonomous_ops", "difficulty": "medium"},
    ),

    # --- Hypersonic Flight ---
    Document(
        doc_id="hyp_001",
        content=(
            "Scramjet (supersonic combustion ramjet) engines operate in the Mach 5-15 regime by "
            "compressing incoming air through the vehicle's geometry rather than mechanical compressors. "
            "The X-51A Waverider achieved Mach 5.1 for 210 seconds using JP-7 fuel in 2013. "
            "Key aerothermodynamic challenges include shock-boundary layer interactions, thermal choking "
            "at low Mach numbers, and fuel-air mixing in millisecond residence times. The scramjet "
            "isolator manages pressure rise through a series of oblique shocks. Combined cycle "
            "propulsion (turbine-based for M0-4, scramjet for M4-10, rocket above M10) is the "
            "leading architecture for reusable hypersonic space access."
        ),
        source="AIAA Hypersonics Conference Proceedings",
        metadata={"topic": "hypersonics", "subtopic": "scramjet", "difficulty": "hard"},
    ),

    # --- Space Debris ---
    Document(
        doc_id="deb_001",
        content=(
            "The Kessler syndrome describes a cascading collision scenario where space debris density "
            "reaches a critical threshold, making certain orbital regions unusable. As of 2024, "
            "ESA tracks over 36,500 objects >10 cm in LEO. The most concerning debris bands are at "
            "750-1,000 km altitude. Active debris removal (ADR) technologies include electrodynamic "
            "tethers (generating Lorentz force drag), laser ablation (ground-based or orbital), "
            "and robotic capture (ESA's ClearSpace-1 mission). The economic cost of debris "
            "mitigation is estimated at $500M-$1B per year for major space operators."
        ),
        source="ESA Space Debris Office",
        metadata={"topic": "space_debris", "subtopic": "kessler_syndrome", "difficulty": "medium"},
    ),

    # --- Entry, Descent, Landing ---
    Document(
        doc_id="edl_001",
        content=(
            "Mars entry, descent, and landing (EDL) faces the 'Mars paradox': the atmosphere is thick "
            "enough to generate extreme heating (peak heat flux 100-200 W/cm²) but too thin for "
            "effective aerodynamic deceleration of heavy payloads (>2 metric tons). NASA's Supersonic "
            "Retropropulsion (SRP), demonstrated by SpaceX Falcon 9, is the baseline for Mars human "
            "missions requiring 20-40 MT landed mass. The sequence: aerocapture → aerobraking → "
            "hypersonic entry → supersonic retropropulsion → terminal landing. Terrain-relative "
            "navigation using LIDAR (as on Perseverance) provides 40m landing accuracy."
        ),
        source="NASA Mars Architecture Study",
        metadata={"topic": "edl", "subtopic": "mars_landing", "difficulty": "hard"},
    ),

    # --- Radiation ---
    Document(
        doc_id="rad_001",
        content=(
            "Galactic cosmic radiation (GCR) poses the greatest health risk for deep space missions. "
            "GCR consists of high-energy nuclei (85% protons, 14% alpha, 1% HZE ions) with energies "
            "up to 10^20 eV. Annual dose in interplanetary space is approximately 600-1,000 mSv, "
            "compared to 0.3 mSv/year on Earth. Shielding with polyethylene (rich in hydrogen) "
            "reduces dose by 30-40% for a 20 g/cm² thickness. Active magnetic shielding concepts "
            "using superconducting coils could reduce dose by 50-70% but require significant "
            "mass and power. NASA's permissible exposure limit is 600 mSv career dose."
        ),
        source="NASA Space Radiation Analysis Group",
        metadata={"topic": "radiation", "subtopic": "gcr_shielding", "difficulty": "hard"},
    ),

    # --- Manufacturing ---
    Document(
        doc_id="mfg_001",
        content=(
            "In-space manufacturing (ISM) using additive techniques enables on-demand production of "
            "replacement parts and tools. NASA's In-Space Manufacturing project demonstrated FDM "
            "3D printing on ISS, producing a functional ratchet wrench. Electron beam free-form "
            "fabrication (EBF3) can manufacture large metallic structures in microgravity using "
            "titanium and aluminum alloys. The Redwire Regolith Print project demonstrated printing "
            "with simulated lunar regolith, enabling in-situ resource utilization (ISRU) for "
            "construction. Estimated cost savings: 50-80% reduction in launch mass for "
            "spare parts inventory on long-duration missions."
        ),
        source="NASA In-Space Manufacturing Project",
        metadata={"topic": "manufacturing", "subtopic": "in_space_manufacturing", "difficulty": "medium"},
    ),
]


def get_aerospace_documents() -> List[Document]:
    """Return all aerospace research documents."""
    return AEROSPACE_DOCUMENTS
