# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# _entity_poly.type
PROTEIN = "polypeptide(L)"
DNA = "polydeoxyribonucleotide"
RNA = "polyribonucleotide"
LIGAND = "ligand"  # non-polymer
POLYMER = [PROTEIN, DNA, RNA]

CRYSTALLIZATION_AIDS = (
    "SO4", "GOL", "EDO", "PO4", "ACT", "PEG", "DMS", "TRS", "PGE", "PG4",
    "FMT", "EPE", "MPD", "MES", "CD", "IOD",
)  # fmt:skip

CRYSTALLIZATION_METHODS = {
    "X-RAY DIFFRACTION",
    "NEUTRON DIFFRACTION",
    "ELECTRON CRYSTALLOGRAPHY",
    "POWDER CRYSTALLOGRAPHY",
    "FIBER DIFFRACTION",
}


PRO_STD_RESIDUES = (
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'UNK'
)  # fmt:skip

RNA_STD_RESIDUES = ("A", "G", "C", "U", "N")
DNA_STD_RESIDUES = ("DA", "DG", "DC", "DT", "DN")

STD_RESIDUES = PRO_STD_RESIDUES + RNA_STD_RESIDUES + DNA_STD_RESIDUES

PRO_STD_RESIDUES_ONE_LETTER = tuple("ARNDCQEGHILKMFPSTWYVBZX")


# In the CCD CIF file, there is only one atom with a CCD Code.
IONS = {'0BE', '3CO', '3NI', '4MO', '4PU', '4TI', '6MO', 'AG', 'AL', 'AM',
        'AR', 'ARS', 'AU', 'AU3', 'BA', 'BR', 'BRO', 'BS3', 'CA', 'CD', 
        'CE', 'CF', 'CL', 'CLO', 'CO', 'CR', 'CS', 'CU', 'CU1', 'CU3', 
        'D8U', 'DUM', 'DY', 'ER3', 'EU', 'EU3', 'F', 'FE', 'FE2', 'FLO', 
        'GA', 'GD', 'GD3', 'H', 'HG', 'HO', 'HO3', 'IDO', 'IN', 'IOD', 
        'IR', 'IR3', 'K', 'KR', 'LA', 'LI', 'LU', 'MG', 'MN', 'MN3', 
        'MO', 'NA', 'ND', 'NGN', 'NI', 'O', 'OS', 'OS4', 'OX', 'OXO', 
        'PB', 'PD', 'PR', 'PT', 'PT4', 'QTR', 'RB', 'RE', 'RH', 'RH3',
        'RHF', 'RU', 'S', 'SB', 'SE', 'SM', 'SR', 'TA0', 'TB', 'TE', 
        'TH', 'TL', 'U1', 'UNX', 'V', 'W', 'XE', 'Y1', 'YB', 'YB2', 
        'YT3', 'ZCM', 'ZN', 'ZN2', 'ZR'}  # fmt: skip

# AlphaFold3 SI Tabel 11
GLYCANS = {'79J', 'LXZ', 'KO1', 'Z57', 'XDX', '8OQ', 'G0S', '14T', 
           'ZB3', '9PG', 'BGL', 'GYU', 'AHG', 'SUC', 'ADA', 'NGR', 
           '4R1', 'EBQ', 'GAF', 'NAA', 'GYP', 'NDG', 'U2D', 'ISL', 
           '9GP', 'KDM', 'HSX', 'NYT', 'V3P', '4NN', 'Z3L', 'ZCZ', 
           'D5E', 'RIP', '3LR', 'GL1', 'K99', 'MQG', 'RAM', 'TUP', 
           'KDB', 'SIO', 'Z5L', 'GUL', 'GU2', 'EQV', '0V4', 'ABD', 
           'RY7', '5II', 'GAL', '2GL', 'DR5', '4RS', 'MNA', 'DFX', 
           '0WK', 'HTG', 'RP5', 'A1Q', 'B1N', 'GUF', 'NGA', 'TMR', 
           'C3X', '9S7', 'XLS', 'MAG', 'RST', 'SDY', 'HSH', 'GN4', 
           'GTR', 'KBA', '6YR', 'CKB', 'DDA', 'RHC', 'OPM', 'SIZ', 
           'GE3', 'TS8', 'Z6W', 'BZD', '56N', 'RIB', 'GL6', '8GA', 
           'GLC', 'TAG', 'QIF', 'TA6', 'UAP', 'TVY', 'GC1', 'ARW', 
           'GU3', 'LBS', 'KDD', 'NPF', '49V', 'CDR', '12E', '6LA', 
           '2M4', 'SA0', 'HNW', 'AOG', 'G8Z', '8LR', 'GPH', 'XXX', 
           'GPM', 'MTT', 'JFZ', 'LOG', 'LMO', '5TH', '8I4', 'GUP', 
           '5KQ', 'R2G', 'SSG', 'P8E', 'RF5', 'TOC', 'CT3', '2FL', 
           '73E', 'VJ4', '0H0', 'ERI', 'AMG', '3GR', 'BO1', 'AFD', 
           'FYJ', 'IDF', 'NBY', 'DOM', 'MBF', 'QDK', 'TDG', '6GR', 
           'MAV', '1X4', 'AF1', 'EEN', 'ZB1', 'Z2D', '445', 'KHP', 
           'LKS', '10M', '491', 'OTU', 'BNG', 'AY9', 'KDR', 'LEC', 
           'FFX', 'AFO', 'SGA', '16F', 'X34', 'SEJ', 'LAG', 'DNO', 
           '6PZ', 'LBT', 'OSU', '3BU', '6K3', 'SFU', 'YDR', 'SIA', 
           '2WP', '25E', 'SMD', 'NBG', 'DO8', 'LGU', 'S81', 'Z3Q', 
           'TWA', 'G6S', '2WS', 'G6D', '18D', 'IN1', '64K', 'QPS', 
           'PTQ', 'FX1', 'RVM', '8GP', 'NLC', 'FCA', 'JLT', 'AH8', 
           'MFB', 'RRJ', 'SOL', 'TM5', 'TCB', 'GU5', 'TWY', 'ETT', 
           '8YV', 'SG6', 'XMM', '17T', 'BGC', 'MLR', 'Z6J', '9SJ', 
           'R2B', 'BBK', 'BEM', 'LTG', '0NZ', 'DKZ', '3YW', 'ASO', 
           'FUB', '4GL', 'GLT', 'KTU', 'CBF', 'ARI', 'FIF', 'LCN', 
           'SG5', 'AC1', 'SUP', 'ZMR', 'GU8', 'YYH', 'XKJ', 'JSV', 
           'DQR', 'M6D', 'FBP', 'AFP', 'F6P', 'GLG', 'JZR', 'DLG', 
           '9C1', 'AAL', 'RRY', 'ZDC', 'TVS', 'B1H', 'XXM', '8B7', 
           'RCD', 'UBO', '7D1', 'XYT', 'WZ2', 'X1X', 'LRH', 'GDA', 
           'GLS', 'G6P', '49A', 'NM9', 'DVC', 'MG5', 'SCR', 'MAF', 
           '149', 'LFC', 'FMF', 'FRU', 'BG8', 'GP4', 'GU1', 'XXR', 
           '4V5', 'MA2', '293', '6KH', 'GAA', 'MXY', 'QV4', 'MSX', 
           'GU6', '95Z', 'Z9M', 'ARB', 'FNY', 'H1S', 'VG1', 'VTB', 
           'Z61', 'H6Z', '7K3', 'XGP', 'SOE', 'Z6H', 'GYV', 'MLB', 
           'DR3', 'ISD', 'BGN', 'AXR', 'SCG', 'Z8T', '6UD', 'KDF', 
           'GLA', 'BNX', '3MG', 'BDP', 'KFN', 'Z9N', '2FG', 'PNA', 
           'MUB', 'ZDO', '9WJ', 'GMB', 'LER', 'TVM', '89Y', 'Z4Y', 
           '9SM', 'NGS', 'LAO', 'KGM', 'FKD', 'M1F', 'BG6', 'LAK', 
           '8GG', '6LS', 'GBH', 'CEG', 'BDR', 'RR7', 'SOG', 'AZC', 
           'AMU', 'BS7', '3S6', 'MXZ', 'Z3U', 'MDP', '6MJ', 'M3M', 
           'DT6', 'PRP', 'TUG', 'Z16', 'IDG', 'TUR', 'Z4S', 'GM0', 
           'A0K', 'GCN', 'ZEE', 'UEA', 'HVC', 'CE5', 'FUD', 'NAG', 
           'GPO', '22S', '3J4', 'DKX', 'FMO', 'BXP', 'NSQ', '50A', 
           'MAT', '5TM', '0MK', '9OK', 'RI2', 'SZZ', 'IDS', 'JRV', 
           '18O', '1CF', 'RAO', 'P53', '27C', 'Z3K', 'Z4U', 'Z4R', 
           'B4G', '6KU', 'HBZ', '07E', 'KBG', '98U', 'GFP', 'LFR', 
           'G2F', '51N', 'FUF', 'LGC', '6S2', 'E3M', 'G7P', 'OTN', 
           'MVP', 'TVD', 'BBV', 'E5G', 'MJJ', 'IEM', 'FSA', 'CE8', 
           'U1Y', '1FT', 'HTM', 'DLD', 'YO5', 'W9T', '5N6', 'PNG', 
           'NGY', 'DSR', 'M3N', 'GP0', '3MK', 'RBL', 'GTM', 'FSW', 
           '4JA', 'YYM', 'Z4V', '3HD', '2DR', 'AIG', 'GL0', 'BND', 
           'TM6', 'TUJ', 'DAN', '5GF', '4QY', '3FM', '6KW', 'LNV', 
           '289', 'BFN', 'PSG', 'U9J', 'YX0', 'EQP', 'YZ0', '0BD', 
           'GAT', 'LVZ', 'FUL', '22O', 'DLF', 'MA1', 'BXY', 'C3G', 
           'CR6', 'GNS', 'EEQ', 'IDY', 'FFC', 'NBX', 'SID', '9KJ', 
           '9WZ', 'M2F', 'FK9', 'SSH', 'TWG', 'RVG', 'BXX', '24S', 
           'FSM', 'GDL', 'F1X', '3R3', 'ALX', '4GC', 'GL2', 'DL6', 
           'GS1', 'AMV', 'TVV', '2DG', 'RGG', 'TFU', '1GN', 'N3U', 
           'SOR', 'MA3', 'GCT', 'H1M', '16G', '49T', 'BCD', 'GPW', 
           'DAG', 'GN1', 'IAB', 'EBG', 'GPU', '38J', '1LL', 'DR2', 
           'YIO', 'YKR', '15L', 'WZ1', 'BTG', 'GPK', '5MM', '26O', 
           'AMN', 'DEL', 'CTT', '83Y', 'GMT', 'CTO', 'MBE', '1SD', 
           '6ZC', 'AXP', 'OX2', '5LT', 'MRH', '6BG', 'MDA', 'SG7', 
           '045', 'GC4', 'LDY', 'YYJ', '07Y', 'KDO', 'GP1', 'BHG', 
           'DPC', 'BM3', 'GU4', 'ISX', 'P6P', 'GPQ', '1S4', '475', 
           'GYE', 'CBK', 'CEZ', 'SGD', 'TH1', 'V3M', 'RWI', 'RM4', 
           'U9M', 'U2A', '7GP', '05L', 'Z0F', 'GLO', 'LXB', 'TGA', 
           '61J', 'GYG', 'GCU', 'GE1', 'F1P', 'GLP', 'CTR', 'AHR', 
           '3LJ', 'FUY', 'JVA', 'LAT', 'NHF', 'RB5', 'XYS', 'LXC', 
           'SLT', 'U8V', 'GMH', 'EAG', 'GCV', 'B6D', 'IDU', 'KG1', 
           'BDF', 'NTP', 'IXD', 'RZM', 'PH5', 'SHB', 'X6Y', 'B16', 
           'Z9E', '9VP', 'LAH', 'H2P', 'TNX', '5GO', 'TGY', '5SP', 
           'RHA', '5KV', 'GTK', 'SUS', 'DAF', '6DM', '8S0', '6MN', 
           'G4D', 'NT1', 'XYF', '5TJ', '46Z', '9AM', '7K2', '6C2', 
           'WIA', '9YW', 'G4S', '46D', 'Z9W', 'ABL', 'XYZ', 'G3I', 
           'S7P', 'GC9', 'GQ1', 'GCO', 'M6P', 'WUN', 'U63', 'ZB2', 
           'GLD', 'T6P', 'ZEL', '145', '2OS', 'BGP', 'C4W', 'IDX', 
           'MUR', '3SA', 'CR1', '34V', 'DEG', 'F55', 'L0W', 'TYV', 
           'CJB', 'TW7', 'DDL', '5L3', 'NGC', 'ACX', 'JVS', 'NA1', 
           'GAD', '7JZ', 'BOG', 'GCW', 'BDG', 'Z15', '0LP', 'ABE', 
           'RG1', 'DGU', 'N1L', 'NGE', 'PUF', 'B9D', '49S', '5LS', 
           '4N2', '23V', 'RUU', 'B0D', 'RTV', '42D', 'M1P', 'MAB', 
           '2F8', 'TQY', 'L6S', 'V71', '2H5', 'M8C', 'NTF', 'H3S', 
           'LM2', 'MN0', 'JV4', '9WN', 'U9G', 'LZ0', 'X0X', 'TXB', 
           '3DO', 'SG4', 'IDR', '8B9', 'TOA', 'CRA', 'HSJ', '0HX', 
           'FDQ', 'FUC', 'ABF', 'ALL', 'G20', 'GL9', 'IDC', 'LOX', 
           'Z2T', 'RP6', '2HA', 'AHM', 'DRI', 'EMZ', 'GMZ', 'HD4', 
           'GU9', 'L1L', 'PNW', 'PPC', 'MMA', 'CE6', '5KS', 'MGC', 
           'XLF', 'KO2', 'RUG', 'HSG', 'SF6', 'IPT', 'TF0', 'GCD', 
           'B8D', '0YT', 'GRX', 'HNV', 'FVQ', 'RV7', 'J5B', 'ERE', 
           'DFR', 'LVO', '4GP', 'BQY', 'BMA', 'KDA', 'ARA', 'KDN', 
           'ZCD', 'A5C', 'T68', 'XYL', 'YJM', 'NM6', '9CD', 'CNP', 
           'U97', '9T1', 'C5X', 'R1X', 'BW3', '09X', 'GNX', 'PDX', 
           'Z9D', 'DGO', 'SLM', '66O', '4CQ', 'X6X', 'RTG', 'HSY', 
           '20X', 'GCB', 'EUS', 'FNG', '1S3', 'EGA', 'MQT', 'NXD', 
           '5TK', 'Z9K', 'TGR', '9MR', 'M7P', 'PA1', 'MFU', 'UBH', 
           'CBI', 'TMX', 'T6D', '32O', 'JHM', 'X2F', '4SG', '3DY', 
           'SGC', 'PAV', 'A2G', 'LAI', '0UB', 'BXF', '3J3', '9T7', 
           'T6T', 'OI7', 'ANA', '9QG', 'K5B', 'KOT', 'GIV', 'MGL', 
           'GL4', '9SP', 'FDP', 'GPV', '6KS', 'GXV', 'NFG', 'M7B', 
           'DG0', '57S', 'GUZ', '96O', 'GCS', 'MAN', 'YYB', 'TWD', 
           'MGS', 'TT7', 'PNJ', 'GXL', 'TRE', 'G28', '7NU', '8PK', 
           'LKA', 'ASG', 'SF9', '2M8', '1GL', '5KT', 'BWG', 'OTG', 
           'VJ1', 'ZGE', '40J', 'Z4K', 'F58', 'KME', 'SR1', 'ZB0', 
           'UDC', '6KL', '6LW', '8EX', 'D1M', '62I', 'H6Q', 'RAE', 
           'SHD', 'AGL', 'DGS', 'VKN', 'TWJ', 'MRP', 'TGK', 'HSQ', 
           'ASC', 'F8X', '6GB', '0XY', 'BMX', 'SN5', 'Z5J', 'ZD0', 
           'DJB', 'KDE', 'TEU', 'M55', 'YYQ', 'DK4', 'D6G', 'KD5', 
           'AH2', '4AM', 'RER', '16O', 'C3B', 'G1P', 'NG6', 'MBG', 
           'Z4W', 'MAW', '147', 'NGK', 'CKP', 'DJE', 'GL5', 'TVG', 
           'PKM', 'L6T', 'XS2', '2GS', 'BTU', 'G16', 'PSV', 'AQA', 
           'MCU', 'SNG', '2M5', 'SLB', 'BM7', 'H53', 'MA8', 'OAK', 
           'GRF', 'BGS', 'NTO', 'YYK', 'EPG', '6GP', 'MYG', 'FCT', 
           'Z9H', 'GL7', '48Z', '4UZ', '7CV', 'DYM', 'GLF', 'GU0', 
           'CGF', 'STZ', '44S', 'LB2', 'TU4', 'Z8H', '5QP', 'A6P', 
           'XYP', 'B2G', 'U9A', 'SWE', 'NGZ', 'SGN', 'B7G', 'MAL', 
           '291', 'FSI', 'R1P', 'ACR', 'PZU', 'X2Y', 'Z9L', 'STW', 
           'U9D', 'X1P', 'TTV', 'GS9', 'QKH', 'SHG', 'N9S', 'NNG', 
           'RP3', 'G3F', 'YX1', 'EMP', 'XIL', '08U', 'WOO', 'FCB', 
           'NG1', 'TRV', '20S', 'RAF', 'GZL', 'C4B', '9SG', 'GAC'}  # fmt: skip
