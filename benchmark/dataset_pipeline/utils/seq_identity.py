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


from biotite.sequence import NucleotideSequence, ProteinSequence
from biotite.sequence.align import align_optimal, SubstitutionMatrix

from pxmeter.constants import PRO_STD_RESIDUES_ONE_LETTER

BLOSUM62 = SubstitutionMatrix.std_protein_matrix()
NUC_STD = SubstitutionMatrix.std_nucleotide_matrix()

# Unknown/ambiguous residue sets for mismatch handling
PROT_UNKNOWN = set("X")  # treated as mismatches for proteins
NUCL_UNKNOWN = set("N")  # treated as mismatches for nucleic acids


def smith_waterman_identity(
    query: str,
    subject: str,
    gap_open: int = -11,
    gap_extend: int = -1,
    is_nucleic: bool = False,
) -> tuple[float, int, str, str]:
    """
    Compute sequence identity as defined in the referenced paper:
    number of exact residue matches after Smith-Waterman alignment,
    divided by the original length of the query sequence.

    Alignment settings:
      - Algorithm: Smith-Waterman (local alignment)
      - Protein:   BLOSUM62; unknown residues X/B/Z/J/U/O count as mismatches
      - Nucleic:   standard nucleotide matrix; 'N' counts as mismatch
      - Affine gap penalty: open = -11, extend = -1 (can be changed via args)

    Args:
        query (str): The query sequence. Its original length is used as
            the denominator.
        subject (str): The subject (target) sequence.
        gap_open (int, optional): Gap opening penalty for affine gaps.
            Defaults to -11.
        gap_extend (int, optional): Gap extension penalty for affine gaps.
            Defaults to -1.
        is_nucleic (bool, optional): If True, perform nucleotide alignment
            (treat 'N' as unknown). If False, perform protein alignment
            (treat 'X' as unknown). Defaults to False.

    Returns:
        Tuple[float, str, str]:
            - identity (float): Exact-match count / len(query), in [0.0, 1.0].
            - exact_matches (int): Number of exact residue matches in the
              optimal local alignment.
            - aligned_query (str): Gapped query sequence from the optimal
              local alignment.
            - aligned_subject (str): Gapped subject sequence from the optimal
              local alignment.
    """

    def _std_prot_seq(prot_seq):
        std_seq = ""
        for i in prot_seq:
            if i in PRO_STD_RESIDUES_ONE_LETTER:
                std_seq += i
            elif i == "U":
                # U -> SEC -> CYS -> C
                std_seq += "C"
            else:
                std_seq += "X"
        return std_seq

    def _std_nuc_seq(nuc_seq):
        std_seq = ""
        for i in nuc_seq:
            if i in {"A", "T", "C", "G"}:
                std_seq += i
            elif i == "U":
                std_seq += "T"
            else:
                std_seq += "N"
        return std_seq

    # Choose sequence class and substitution matrix based on sequence type
    if is_nucleic:
        q = NucleotideSequence(_std_nuc_seq(query))
        s = NucleotideSequence(_std_nuc_seq(subject))
        subst = NUC_STD
        unknown = NUCL_UNKNOWN
    else:
        q = ProteinSequence(_std_prot_seq(query))
        s = ProteinSequence(_std_prot_seq(subject))
        subst = BLOSUM62
        unknown = PROT_UNKNOWN

    # Run Smith-Waterman (local) with affine gap penalties
    aln = align_optimal(q, s, subst, gap_penalty=(gap_open, gap_extend), local=True)[
        0
    ]  # best-scoring alignment

    # Extract gapped strings for counting
    aligned_q, aligned_s = aln.get_gapped_sequences()
    aligned_q = str(aligned_q)
    aligned_s = str(aligned_s)

    # Count exact matches:
    # - skip gap columns
    # - treat unknown residues as mismatches
    exact_matches = 0
    for a, b in zip(aligned_q, aligned_s):
        if a == "-" or b == "-":
            continue
        if a in unknown or b in unknown:
            continue
        if a == b:
            exact_matches += 1

    identity = exact_matches / len(query) if len(query) > 0 else 0.0
    return identity, exact_matches, aligned_q, aligned_s
