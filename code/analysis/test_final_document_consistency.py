"""Regression checks for the specific final-review omissions."""
import json
from pathlib import Path
import re
import unittest

HERE=Path(__file__).resolve().parents[1]


class FinalDocumentTests(unittest.TestCase):
    def test_literature_summary_uses_fixed_local_paragraph_spacing(self):
        for name in ('Manuscript_Revised_Marked.tex','Manuscript_Revised_Clean.tex'):
            text=(HERE/name).read_text(encoding='utf-8')
            start=text.index(r'\begin{enumerate}')
            end=text.index(r'\begin{table*}',start)
            self.assertTrue(text[:start].rstrip().endswith(r'\setlength{\parskip}{0pt}'))
            self.assertIn('parsep=0pt,partopsep=0pt',text[start:end])
            self.assertTrue(text[start:end].rstrip().endswith(r'\par\endgroup'))
            self.assertIn(r'\flushbottom',text)

    def test_second_page_columns_fill_without_stretched_summary_gaps(self):
        import fitz
        for name in ('Manuscript_Revised_Marked.pdf','Manuscript_Revised_Clean.pdf'):
            with fitz.open(HERE/name) as doc:
                lines=[line for block in doc[1].get_text('dict')['blocks'] if 'lines' in block
                       for line in block['lines'] if line['bbox'][1]<750]
                left=[line for line in lines if line['bbox'][0]<300]
                right=[line for line in lines if line['bbox'][0]>=300]
                self.assertLess(abs(max(x['bbox'][3] for x in left)
                                    -max(x['bbox'][3] for x in right)),15)
                summary=next(line for line in left if ''.join(s['text'] for s in line['spans'])
                             .startswith('These studies motivate'))
                preceding=max(x['bbox'][3] for x in left if x['bbox'][3]<summary['bbox'][1])
                self.assertLess(summary['bbox'][1]-preceding,10)

    def test_added_results_do_not_use_run_in_microheadings(self):
        for name in ('Manuscript_Revised_Marked.tex','Manuscript_Revised_Clean.tex'):
            text=(HERE/name).read_text(encoding='utf-8')
            self.assertIsNone(re.search(r'\\paragraph\{\\rev\{',text))

    def test_fourth_literature_topic_precedes_the_summary(self):
        from revision_precise_marking import strip_marks
        source=(HERE/'Manuscript_Revised_Marked.tex').read_text(encoding='utf-8')
        text=strip_marks(source)
        self.assertIn('grouped into four directions',text)
        topic=text.split(r'\begin{enumerate}',1)[1].split(r'\end{enumerate}',1)[0]
        self.assertEqual(len(re.findall(r'\\item\b',topic)),4)
        self.assertIn('Safe and physics-informed learning.',topic)
        fourth=re.split(r'\\item\b',topic)[-1].strip()
        self.assertNotIn('\n\n',fourth)
        for key in ('Su2025SafeReview','Wu2025SafeOPF','Lopez2025Typed','Liu2026RefineDC'):
            self.assertIn(r'\cite{'+key+'}',topic)
        self.assertLess(source.index(r'\label{reply:quote_related_correction}'),source.index(r'\end{enumerate}'))
        self.assertGreater(source.index('These studies motivate two design questions'),source.index(r'\end{enumerate}'))
        baseline=strip_marks((HERE/'GAT_SAC_PTDF_LODF_SCED_Restoration.tex').read_text(encoding='utf-8'))
        original_topic=baseline.split(r'\begin{enumerate}',1)[1].split(r'\end{enumerate}',1)[0]
        original_items=re.split(r'\\item\b',original_topic)[1:]
        current_items=re.split(r'\\item\b',topic)[1:4]
        self.assertEqual([re.sub(r'\s+',' ',x).strip() for x in original_items],
                         [re.sub(r'\s+',' ',x).strip() for x in current_items])

    def test_minimal_revision_audit_and_conclusion_marking(self):
        audit=json.loads((HERE/'Minimal_Revision_Audit.json').read_text(encoding='utf-8'))
        self.assertEqual(audit['comments_reviewed'],43)
        self.assertEqual(len(audit['review_items']),43)
        text=(HERE/'Manuscript_Revised_Marked.tex').read_text(encoding='utf-8')
        conclusion=text.split(r'\section{Conclusions}',1)[1].split(r'\printcredits',1)[0]
        self.assertIn(r'\rev{\label{reply:quote_conclusion}Two conclusions follow.',conclusion)
        self.assertIn('the policy accepts 44 of 75',conclusion)
        self.assertIn('The 27/27 figure applies only to the screened subset.',conclusion)

    def test_all_figure_text_uses_times_and_boxplot_scales_are_explicit(self):
        import fitz
        text=(HERE/'Manuscript_Revised_Clean.tex').read_text(encoding='utf-8')
        paths=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text)
        self.assertEqual(len(paths),12)
        for name in paths:
            with fitz.open(HERE/name) as doc:
                fonts={s['font'] for p in doc for b in p.get_text('dict')['blocks'] if 'lines' in b
                       for line in b['lines'] for s in line['spans']}
                self.assertTrue(fonts and all('Times' in font for font in fonts),(name,fonts))
        self.assertIn('uses separate vertical scales',text)
        self.assertIn('Outliers are omitted visually but retained in the reported statistics',text)

    def test_small_system_section_has_no_forced_float_clear(self):
        text=(HERE/'Manuscript_Revised_Marked.tex').read_text(encoding='utf-8')
        before=text.split(r'\subsection{Additional IEEE 39-bus and 118-bus Experiments}',1)[0]
        self.assertFalse(before.rstrip().endswith(r'\FloatBarrier'))

    def test_response_tone_and_quantitative_preservation(self):
        items=json.loads((HERE/'revision_responses_resolved.json').read_text(encoding='utf-8'))
        text=' '.join(item['response'] for item in items.values())
        for phrase in ('falsely attributed','fabricate a priced comparison','We agree that screening'):
            self.assertNotIn(phrase,text)
        audit=json.loads((HERE/'Response_Tone_Audit.json').read_text(encoding='utf-8'))
        self.assertGreater(len(audit),15)
        self.assertTrue(all(row['numeric_tokens_unchanged'] for row in audit))
        self.assertIn('[-0.003, 0.088] s includes zero',items['R3.6']['response'])
        self.assertRegex(items['R4.2']['response'],r'None of the three (?:cases|operating points) checked against all 253 valid outages passes every outage')

    def test_conclusion_retains_unfiltered_and_conditional_counts(self):
        text=(HERE/'Manuscript_Revised_Marked.tex').read_text(encoding='utf-8')
        conclusion=text.split(r'\section{Conclusions}',1)[1].split(r'\appendix',1)[0]
        from revision_precise_marking import strip_marks
        conclusion=strip_marks(conclusion)
        self.assertIn('the policy accepts 44 of 75 candidates',conclusion)
        self.assertIn('27/27 figure applies only to the screened subset',conclusion)
        self.assertIn('Two conclusions follow.',conclusion)
        self.assertNotRegex(conclusion.split(r'\printcredits')[0],r'\b(?:SAC|SCED)\b')

    def test_restored_environment_and_current_ae_caption_response(self):
        text=(HERE/'Manuscript_Revised_Marked.tex').read_text(encoding='utf-8')
        self.assertIn('Both final timing comparisons were run on the same workstation',text)
        self.assertIn('Intel Core Ultra~9 285 CPU',text)
        self.assertIn('NVIDIA RTX PRO~2000 Blackwell GPU',text)
        self.assertIn('pandapower/PYPOWER interior-point optimizer',text)
        self.assertIn('no explicit solver or BLAS thread cap',text)
        items=json.loads((HERE/'revision_responses_resolved.json').read_text(encoding='utf-8'))
        self.assertNotIn('cost-time figure',items['AE.8']['response'])
        self.assertIn('60 jointly accepted historical cases',items['AE.8']['response'])
        for key in ('AE.3','AE.5','R1.4'):
            self.assertNotIn('retain the submitted title',items[key]['response'])

    def test_security_locations_are_not_economic_section(self):
        evidence=json.loads((HERE/'response_evidence.json').read_text(encoding='utf-8'))['comments']
        for key in ('AE.13','R2.3','R2.5','R3.4','R4.2'):
            self.assertIn('reply:conditions',evidence[key]['location_labels'])
            self.assertNotIn('reply:optimization',evidence[key]['location_labels'])
        self.assertIn('reply:optimization',evidence['R3.6']['location_labels'])

    def test_deployment_definition_is_in_methods(self):
        text=(HERE/'Manuscript_Revised_Marked.tex').read_text(encoding='utf-8')
        method=text.split(r'\subsection{Evaluation Protocol}',1)[1].split(r'\section{Numerical Results}',1)[0]
        self.assertIn('eight-step cap',method)
        self.assertIn('local analytic AC derivative',method)
        self.assertIn('not an equivalent implementation',method)
        self.assertIn('already accepted point skips correction',method)

    def test_abstract_and_reproduction_entry_points(self):
        text=(HERE/'Manuscript_Revised_Marked.tex').read_text(encoding='utf-8')
        abstract=re.search(r'\\begin\{abstract\}.*?\n(.*?)\\end\{abstract\}',text,re.S).group(1)
        self.assertGreaterEqual(len(abstract.split()),150)
        self.assertLessEqual(len(abstract.split()),250)
        guide=(HERE/'Reproduce_Primary_Results.md').read_text(encoding='utf-8')
        self.assertIn('remeasure_verified_dual.py',guide)
        self.assertIn('remeasure_spatial_dual.py',guide)
        self.assertIn('previously examined',guide)

    def test_final_populations_share_one_table_not_sequence_rows(self):
        text=(HERE/'Manuscript_Revised_Marked.tex').read_text(encoding='utf-8')
        table=next(m.group() for m in re.finditer(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',text,re.S)
                   if r'\label{tab:review_followup_runtime}' in m.group())
        self.assertIn('Historical (75)',table)
        self.assertIn('Spatial (100)',table)
        self.assertIn('60/75',table)
        self.assertIn('93/100',table)
        self.assertIn('[-0.003, 0.088]',table)
        self.assertNotIn('Sequence',table)
        self.assertNotIn(r'\label{tab:final_spatial_dual}',text)
        self.assertIn(r'\label{tab:sequence_runtime}',text)

    def test_primary_plot_and_provenance_are_explicit(self):
        text=(HERE/'Manuscript_Revised_Marked.tex').read_text(encoding='utf-8')
        self.assertNotIn(r'\appendix',text)
        self.assertIn(r'\label{fig:final_paired_timing}',text)
        self.assertIn(r'\label{fig:revision_tradeoff}',text)
        self.assertIn('not a cross-panel speed comparison',text)
        self.assertIn('no documented standards-based calibration',text)
        self.assertIn('AC-anchor preparation failures remain in the denominator',text)
        self.assertIn('contributing 0.182 s',text)

    def test_replies_are_final_answers_not_chronological_appendices(self):
        items=json.loads((HERE/'revision_responses_resolved.json').read_text(encoding='utf-8'))
        for key,limit in (('R2.7',300),('R3.6',350)):
            self.assertLess(len(items[key]['response'].split()),limit)
            self.assertIn(items[key]['response'].count('\n\n'),(1,2))
        self.assertIn('no documented standards-based derivation',items['R2.7']['response'])
        for key in ('R2.8','R3.6'):
            self.assertTrue(any(x in items[key]['response'] for x in ('same 93 of the 100 cases in every repeat','same 93 cases','both workflows accept 93/100')))
            self.assertNotIn('93/93',items[key]['response'])

    def test_tighter_voltage_response_covers_materials_and_timing(self):
        items=json.loads((HERE/'revision_responses_resolved.json').read_text(encoding='utf-8'))
        reply=items['R2.7']['response']
        self.assertIn('Response R1.6',reply)
        self.assertIn('upon reasonable request',reply)
        self.assertIn('not been publicly deposited',reply)
        self.assertIn('1.207 s over all 75 cases, including failures',reply)

    def test_innovation_excerpt_is_a_complete_representation_paragraph(self):
        evidence=json.loads((HERE/'response_evidence.json').read_text(encoding='utf-8'))
        ids=evidence['comments']['AE.11']['excerpt_ids']
        self.assertIn('redispatch_contribution',ids)
        self.assertNotIn('scope',ids)
        value=evidence['verbatim_excerpts']['redispatch_contribution']
        self.assertTrue(value.startswith('Direct active-power control requires'))
        self.assertIn('lower-dimensional set of balanced redispatch directions',value)
        self.assertTrue(value.endswith('local generator adjustment.'))

    def test_primary_table_is_legible_and_keeps_statistical_notes(self):
        manuscript=(HERE/'Manuscript_Revised_Marked.tex').read_text(encoding='utf-8')
        table=next(m.group() for m in re.finditer(r'\\begin\{table\}.*?\\end\{table\}',manuscript,re.S)
                   if r'\label{tab:review_followup_runtime}' in m.group())
        self.assertIn(r'\footnotesize',table)
        self.assertNotIn(r'\scriptsize',table)
        self.assertIn('Final-configuration performance on historical and spatially perturbed cases.',table)
        self.assertIn(r'\begin{tabularx}{\linewidth}',table)
        self.assertNotIn('minipage',table)
        for phrase in ('Times include preparation, failures and all acceptance work',
                       'identical IDs within each population and repeat','inner OPF solves per repeat','case-bootstrap 95'):
            self.assertIn(phrase,manuscript)

    def test_primary_notes_do_not_attribute_policy_to_direct_baseline(self):
        for name in ('Manuscript_Revised_Marked.tex','Response_to_Associate_Editor_and_Reviewers.tex'):
            text=(HERE/name).read_text(encoding='utf-8')
            self.assertIn('direct optimization does not invoke the policy.',text.lower())
            self.assertTrue('Both workflows share the optimizer, limits and acceptance checks' in text or
                            'using the same dual-initialization optimizer, operating limits and acceptance checks' in text or
                            'using the same dual-initialization optimizer, limits and acceptance checks' in text)
            self.assertNotIn('Both arms use the same final short-policy/dual-initialization configuration',text)

    def test_configuration_caption_limits_weight_sharing_to_policy_workflows(self):
        for name in ('Manuscript_Revised_Marked.tex','Manuscript_Revised_Clean.tex'):
            text=(HERE/name).read_text(encoding='utf-8')
            self.assertIn('Policy-based configurations use the same frozen policy weights; all configurations share the same operating acceptance limits.',text)
            self.assertNotIn('All use the same frozen policy weights and operating acceptance limits.',text)

    def test_scope_limited_title_and_column_width_added_tables(self):
        from build_revision_documents import TITLE, ORIGINAL_TITLE
        self.assertNotEqual(TITLE,ORIGINAL_TITLE)
        self.assertEqual(TITLE,'Sensitivity-Guided Low-Dimensional Redispatch for Base-Case Alternating-Current Feasibility and Linear N-1 Contingency Screening')
        text=(HERE/'Manuscript_Revised_Marked.tex').read_text(encoding='utf-8')
        from revision_precise_marking import strip_marks
        self.assertIn(r'\title[mode=title]{'+TITLE+'}',strip_marks(text))
        tables=[m.group() for m in re.finditer(r'\\begin\{table\*?\}.*?\\end\{table\*?\}',text,re.S)
                if r'\ifrevisionmarked\color{blue}\fi' in m.group()]
        self.assertGreater(len(tables),15)
        for table in tables:
            self.assertNotIn(r'\begin{table*}',table)
            self.assertIn(r'\begin{tabularx}{\linewidth}',table)
            self.assertNotIn(r'\textit{Notes.}',table)

    def test_author_confirmed_funding_identifier(self):
        for name in ('Manuscript_Revised_Marked.tex','Manuscript_Revised_Clean.tex'):
            text=(HERE/name).read_text(encoding='utf-8')
            self.assertIn('ZDKJXM20250028',text)
            self.assertNotIn('ZDKJXM20250026',text)

    def test_cadence_answer_retains_numbers_and_pointer_without_table_copy(self):
        evidence=json.loads((HERE/'response_evidence.json').read_text(encoding='utf-8'))['comments']['R2.13']
        self.assertEqual(evidence['reproduced_tables'],[])
        self.assertIn('tab:revision_cadence',evidence['location_labels'])
        items=json.loads((HERE/'revision_responses_resolved.json').read_text(encoding='utf-8'))
        for value in ('1725/1725','603/1773','416/1736','1.341','0.921','0.464'):
            self.assertIn(value,items['R2.13']['response'])


if __name__=='__main__':
    unittest.main()
