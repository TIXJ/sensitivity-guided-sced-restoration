"""Map all original review items to necessary edits, not cosmetic rewriting."""
from pathlib import Path
import json
import re
import sys

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from revision_precise_marking import strip_marks

REASONS={
    'AE.1':'Add relevant literature; retain the existing problem statement and review structure.',
    'AE.2':'Confirm completed language review in the response; no further blanket prose rewrite.',
    'AE.3':'Expand abbreviations in required front/back matter; retain other original wording.',
    'AE.4':'Use the corrected bibliography style and specific citations; retain supported references.',
    'AE.5':'Retain the revised scope-limited title, also required by R1.4.',
    'AE.6':'Provide the existing three editable highlights; no added main-text section.',
    'AE.7':'Retain the abstract opening and method wording; update necessary results and scope.',
    'AE.8':'Keep concise captions and retain metric definitions in text or table labels.',
    'AE.9':'Keep physical units; distinguish benchmark cost units and exchange assumptions.',
    'AE.10':'One concise conclusion paragraph, reusing accurate submitted result sentences.',
    'AE.11':'Qualify the contribution using matched-basis evidence; avoid rewriting the method.',
    'AE.12':'Add specifically relevant studies without replacing the existing review.',
    'AE.13':'Retain analytical formulation and add the completed numerical validation.',
    'AE.14':'Existing IEEE-derived scope is international; explain this in the response.',
    'R1.1':'Keep conditional results and add all-method unfiltered comparisons.',
    'R1.2':'Report 44/75 in abstract, main results and conclusion; retain conditional 27/27.',
    'R1.3':'Retain completed seeds and uncertainty summaries; no further training.',
    'R1.4':'Keep title/scope corrections and explicit outage audits.',
    'R1.5':'Keep reward controls and qualify workflow applicability, not policy transfer.',
    'R1.6':'Existing reasonable-request statement remains; describe actual materials in the reply.',
    'R2.1':'Explicitly distinguish full-population and screened success.',
    'R2.2':'Keep screening disclosure and common unfiltered evaluations.',
    'R2.3':'Keep base-case/contingency boundary and explicit outage diagnostics.',
    'R2.4':'Keep exclusion rationale and the line-outage scope.',
    'R2.5':'Keep measured flow-error and actual-flow-anchor diagnostics.',
    'R2.6':'Keep external-source definition and paired signed-exchange costs.',
    'R2.7':'Keep voltage-limit provenance, tighter-band results/time and materials response.',
    'R2.8':'Keep completed varied-condition evaluations and their actual populations.',
    'R2.9':'Keep all seed outcomes, including failed runs and conditional-cost definitions.',
    'R2.10':'Keep basis/dimension/budget controls; qualify mixed-basis claims.',
    'R2.11':'Restrict the subspace claim to pre-recovery actions; retain projection diagnostics.',
    'R2.12':'Keep matched initialization comparisons and distinguish their objectives.',
    'R2.13':'Keep feedback-frequency results and observed check counts.',
    'R3.1':'Keep the added result figures; avoid replacing original figures unnecessarily.',
    'R3.2':'Keep demand-wise results, screening explanation and overlapping failure counts.',
    'R3.3':'Keep the completed multi-seed comparison and variability.',
    'R3.4':'Keep the explicit linear-versus-AC contingency distinction and audit.',
    'R3.5':'Keep common small-system controls; minimally replace transferability wording.',
    'R3.6':'Keep matched final timing, economics and conditional speed conclusions.',
    'R3.7':'Retain relevant added citations and explain broader suggested papers in the reply.',
    'R4.1':'Keep unfiltered results and distinguish original policy from optimization fallback.',
    'R4.2':'Keep the full AC security boundary and unfavorable outage results.',
    'R4.3':'Keep all completed seeds and their observed variability.',
}


def main():
    evidence=json.loads((HERE/'response_evidence.json').read_text(encoding='utf-8'))
    assert set(REASONS)==set(evidence['comments']) and len(REASONS)==43
    original=(HERE/'GAT_SAC_PTDF_LODF_SCED_Restoration.tex').read_text(encoding='utf-8')
    revised=(HERE/'Manuscript_Revised_Marked.tex').read_text(encoding='utf-8')
    section=lambda t:t.split(r'\section{Conclusions}',1)[1].split(r'\printcredits',1)[0]
    conclusion=section(revised)
    records=[]
    for key,reason in REASONS.items():
        e=evidence['comments'][key]
        scope=('response_only' if key in ('AE.2','AE.14') else
               'response_and_materials' if key=='R1.6' else
               'separate_submission_file' if key=='AE.6' else 'targeted_manuscript_change')
        records.append({'comment':key,'scope':scope,'reason':reason,
                        'original_comment_sha256':e['original_comment_sha256'],
                        'manuscript_locations':e['location_labels']})
    audit={'baseline':'Original submitted manuscript, not the previous revised version.',
           'comments_reviewed':43,'review_items':records,
           'conclusion_words':{'submitted':len(strip_marks(section(original)).split()),
                               'revised':len(strip_marks(conclusion).split())},
           'this_pass':'Restore accurate submitted wording in six introduction/method/scope edits and the conclusion. Preserve all completed experiments and qualifications required by reviewers.',
           'marking':'Sentence-aligned comparison retains unchanged wording in black even after sentence reordering. New results remain blue.',
           'acceptance_guaranteed':False}
    (HERE/'Minimal_Revision_Audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({'reviewed':43,'conclusion_words':audit['conclusion_words']},indent=2))


if __name__=='__main__':main()
